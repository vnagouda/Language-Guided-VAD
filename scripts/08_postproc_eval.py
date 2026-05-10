"""Post-Processing Evaluation: Gaussian Smoothing + Z-Score Normalisation.

Applies two zero-cost post-processing steps to the V4 model's raw segment scores:
  1. Temporal Gaussian smoothing (sigma sweep)
  2. Video-level Z-score normalisation

Both are applied AFTER inference — no retraining required.
Tests each combination and reports the best frame-AUROC improvement.

Usage:
    python scripts/08_postproc_eval.py \\
        --config configs/config_v4_sota.yaml \\
        --checkpoint checkpoints_v4/best_model_framelevel.pth
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vad_architecture import LanguageGuidedVAD
from utils.video_utils import load_config
from utils.metrics import interpolate_scores


# ---------------------------------------------------------------------------
# Annotation Parser
# ---------------------------------------------------------------------------

def load_annotations(annotation_file: str) -> Dict[str, List[int]]:
    """Parse UCF-Crime temporal annotations (full protocol including normals).

    Returns:
        Dict: video_stem -> [start1, end1, start2, end2].
              Normal videos: [-1, -1, -1, -1] -> all-zero GT.
    """
    ann: Dict[str, List[int]] = {}
    with open(annotation_file, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            name = parts[0].replace(".mp4", "")
            try:
                ann[name] = [int(x) for x in parts[-4:]]
            except ValueError:
                warnings.warn(f"Skipping malformed annotation at line {lineno}: {line.strip()!r}")
    return ann


# ---------------------------------------------------------------------------
# Score Post-Processing
# ---------------------------------------------------------------------------

def gaussian_smooth(scores: np.ndarray, sigma: float) -> np.ndarray:
    """Apply 1D Gaussian smoothing to segment scores.

    Args:
        scores: (T,) raw anomaly scores per segment.
        sigma:  Gaussian standard deviation (in segments).

    Returns:
        (T,) smoothed scores.
    """
    if sigma <= 0:
        return scores
    return gaussian_filter1d(scores.astype(np.float64), sigma=sigma).astype(np.float32)


def zscore_normalise(scores: np.ndarray) -> np.ndarray:
    """Normalise scores to zero mean, unit variance per video, then shift to [0,1].

    Removes per-video bias (some videos inherently score higher regardless of content).

    Args:
        scores: (T,) raw anomaly scores.

    Returns:
        (T,) normalised scores in [0, 1].
    """
    mu  = scores.mean()
    std = scores.std()
    if std < 1e-6:
        return np.zeros_like(scores)
    z = (scores - mu) / std
    # Shift from z-space to [0,1] via sigmoid
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_postprocessing(
    config_path: str,
    checkpoint_path: str,
    t_proxy: int = 16,
) -> None:
    """Sweep post-processing parameters and report best AUROC.

    Tests the following combinations:
      - Baseline (no post-processing)
      - Gaussian only (sigma in 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
      - Z-score only
      - Z-score + Gaussian (sigma in 0.5, 1.0, 1.5, 2.0, 2.5)

    Args:
        config_path:     Path to YAML config.
        checkpoint_path: Path to model checkpoint (.pth).
        t_proxy:         Proxy frames per segment (T * t_proxy).
    """
    warnings.warn(
        "This script sweeps post-processing parameters on the test set. "
        "Results are optimistically biased — use a validation split for tuning.",
        stacklevel=2,
    )
    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[POSTPROC] Device: {device}")

    # Load model
    model = LanguageGuidedVAD.from_config(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[POSTPROC] Loaded: {checkpoint_path}")

    T            = cfg["model"]["num_segments"]
    features_dir = Path(cfg["data"]["features_dir"]) / "Test"
    annotations  = load_annotations(cfg["data"]["annotation_file"])
    all_vis      = sorted(features_dir.glob("*_visual.pt"))
    test_videos  = [f.stem.replace("_visual", "") for f in all_vis]

    # -------------------------------------------------------
    # Step 1: collect raw segment scores for all test videos
    # -------------------------------------------------------
    raw_scores: Dict[str, np.ndarray] = {}
    for video in tqdm(test_videos, desc="[POSTPROC] Inference"):
        vis_p  = features_dir / f"{video}_visual.pt"
        txt_p  = features_dir / f"{video}_text.pt"
        flow_p = features_dir / f"{video}_flow.pt"
        if not vis_p.exists() or not txt_p.exists():
            continue
        vis  = torch.load(vis_p,  weights_only=True).unsqueeze(0).to(device)
        txt  = torch.load(txt_p,  weights_only=True).unsqueeze(0).to(device)
        flow = (
            torch.load(flow_p, weights_only=True).unsqueeze(0).to(device)
            if flow_p.exists() else torch.zeros(1, T, device=device)
        )
        scores, _, _ = model(vis, txt, flow)
        raw_scores[video] = scores.squeeze(0).cpu().numpy()   # (T,)

    print(f"[POSTPROC] Videos scored: {len(raw_scores)}")

    # -------------------------------------------------------
    # Step 2: build ground truth (full protocol)
    # -------------------------------------------------------
    def build_frame_arrays(
        scores_dict: Dict[str, np.ndarray],
        smooth_fn=None,
    ):
        """Apply optional smooth_fn then build concatenated preds + labels."""
        preds_list, labels_list = [], []
        for video, seg_scores in scores_dict.items():
            ann = annotations.get(video) or annotations.get(video.replace("_x264", ""))
            if ann is None:
                continue
            processed = smooth_fn(seg_scores) if smooth_fn else seg_scores
            N          = T * t_proxy
            frame_pred = interpolate_scores(processed, N)
            lbl        = np.zeros(N, dtype=np.int32)
            s1, e1, s2, e2 = ann
            if s1 >= 0 and e1 >= 0:
                lbl[min(s1, N - 1):min(e1, N)] = 1
            if s2 >= 0 and e2 >= 0:
                lbl[min(s2, N - 1):min(e2, N)] = 1
            preds_list.append(frame_pred)
            labels_list.append(lbl)
        return np.concatenate(preds_list), np.concatenate(labels_list)

    # -------------------------------------------------------
    # Step 3: sweep post-processing parameters
    # -------------------------------------------------------
    results = []

    # Baseline
    preds, labels = build_frame_arrays(raw_scores)
    baseline_auroc = roc_auc_score(labels, preds)
    results.append(("Baseline (none)", 0.0, False, baseline_auroc))
    print(f"\n[POSTPROC] Baseline AUROC: {baseline_auroc:.4f}")

    sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    # Gaussian only
    print("\n[POSTPROC] Sweeping Gaussian sigma (no z-score)...")
    for sigma in sigmas:
        fn = lambda s, sg=sigma: gaussian_smooth(s, sg)
        preds, labels = build_frame_arrays(raw_scores, fn)
        auroc = roc_auc_score(labels, preds)
        results.append((f"Gaussian(sigma={sigma})", sigma, False, auroc))
        marker = " <-- BEST" if auroc > max(r[3] for r in results[:-1]) else ""
        print(f"  sigma={sigma:.1f}  ->  AUROC={auroc:.4f}{marker}")

    # Z-score only
    preds, labels = build_frame_arrays(raw_scores, zscore_normalise)
    auroc = roc_auc_score(labels, preds)
    results.append(("Z-score only", 0.0, True, auroc))
    print(f"\n[POSTPROC] Z-score only AUROC: {auroc:.4f}")

    # Z-score + Gaussian
    print("\n[POSTPROC] Sweeping Z-score + Gaussian sigma...")
    for sigma in sigmas:
        def fn(s, sg=sigma):
            return gaussian_smooth(zscore_normalise(s), sg)
        preds, labels = build_frame_arrays(raw_scores, fn)
        auroc = roc_auc_score(labels, preds)
        results.append((f"Zscore+Gaussian(sigma={sigma})", sigma, True, auroc))
        marker = " <-- BEST" if auroc > max(r[3] for r in results[:-1]) else ""
        print(f"  sigma={sigma:.1f}  ->  AUROC={auroc:.4f}{marker}")

    # -------------------------------------------------------
    # Step 4: Report
    # -------------------------------------------------------
    best = max(results, key=lambda x: x[3])
    print(f"\n{'='*60}")
    print(f"[POSTPROC] RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline                 : {baseline_auroc:.4f}")
    print(f"  Best post-processing     : {best[3]:.4f}  ({best[0]})")
    gain = best[3] - baseline_auroc
    print(f"  Gain                     : {gain:+.4f}  ({gain * 100:+.2f}%)")
    print(f"{'='*60}\n")

    # Full table
    print("[POSTPROC] Full results:")
    for name, sigma, zscore, auroc in sorted(results, key=lambda x: -x[3]):
        print(f"  {auroc:.4f}  |  {name}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-processing sweep: Gaussian smoothing + Z-score normalisation"
    )
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--t_proxy",    type=int, default=16)
    args = parser.parse_args()

    evaluate_postprocessing(
        config_path     = args.config,
        checkpoint_path = args.checkpoint,
        t_proxy         = args.t_proxy,
    )
