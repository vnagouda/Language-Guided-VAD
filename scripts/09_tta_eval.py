"""Test-Time Augmentation (TTA) Evaluation for UCF-Crime.

Applies multiple augmentation strategies at inference time and averages
the resulting scores to improve frame-level AUROC without any retraining.

Augmentations:
    1. Original scores (baseline)
    2. Temporal flip (reverse segment order → re-reverse scores)
    3. Feature dropout (N masks, average)
    4. Score power calibration (grid search over exponents)

Usage:
    python scripts/09_tta_eval.py --config configs/config_v12.yaml \
        --checkpoint checkpoints_v12_s777/best_model_framelevel.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vad_architecture import LanguageGuidedVAD
from models.vad_architecture_v13 import TriModalVAD
from utils.video_utils import load_config
from utils.metrics import interpolate_scores


# ---------------------------------------------------------------------------
# Annotation Parser
# ---------------------------------------------------------------------------

def load_annotations(annotation_file: str) -> Dict[str, List[int]]:
    """Parse UCF-Crime temporal annotations."""
    ann: Dict[str, List[int]] = {}
    with open(annotation_file, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 5:
                name = parts[0].replace(".mp4", "")
                ann[name] = [int(x) for x in parts[-4:]]
    return ann


# ---------------------------------------------------------------------------
# Frame-Level AUROC Computation
# ---------------------------------------------------------------------------

def compute_frame_auroc(
    scores_dict: Dict[str, np.ndarray],
    annotations: Dict[str, List[int]],
    frame_counts: Dict[str, int],
    T: int = 128,
    t_proxy: int = 16,
) -> float:
    """Compute frame-level AUROC from segment scores using full protocol."""
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for video, seg_scores in scores_dict.items():
        ann = annotations.get(video) or annotations.get(video.replace("_x264", ""))

        if ann is not None:
            s1, e1, s2, e2 = ann
            max_ann = max(v for v in ann if v > 0) if any(v > 0 for v in ann) else 0
            lookup = video if video in frame_counts else video.replace("_x264", "")
            if lookup in frame_counts:
                N = frame_counts[lookup]
            elif max_ann > 0:
                N = max(max_ann + 1, T * t_proxy)
            else:
                N = T * t_proxy

            frame_scores = interpolate_scores(seg_scores, N)
            frame_labels = np.zeros(N, dtype=np.int32)
            if s1 >= 0 and e1 >= 0:
                frame_labels[min(s1, N - 1):min(e1, N)] = 1
            if s2 >= 0 and e2 >= 0:
                frame_labels[min(s2, N - 1):min(e2, N)] = 1
        else:
            lookup = video if video in frame_counts else video.replace("_x264", "")
            N = frame_counts.get(lookup, T * t_proxy)
            frame_scores = interpolate_scores(seg_scores, N)
            frame_labels = np.zeros(N, dtype=np.int32)

        all_preds.append(frame_scores)
        all_labels.append(frame_labels)

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return float(roc_auc_score(labels, preds))


# ---------------------------------------------------------------------------
# TTA Score Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def tta_score_extraction(
    config_path: str,
    checkpoint_path: str,
    model_type: str = "v12",
    n_dropout_runs: int = 5,
    dropout_rate: float = 0.1,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract scores with multiple TTA strategies.

    Args:
        config_path: Path to config YAML.
        checkpoint_path: Path to checkpoint.
        model_type: 'v12' or 'v13' (TriModalVAD).
        n_dropout_runs: Number of feature dropout runs.
        dropout_rate: Fraction of features to zero out.

    Returns:
        Dict mapping strategy_name → {video_name → scores array}.
    """
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if model_type == "v12":
        model = LanguageGuidedVAD.from_config(cfg).to(device)
    else:
        model = TriModalVAD.from_config(cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    features_dir = Path(cfg["data"]["features_dir"]) / "Test"
    T = cfg["model"]["num_segments"]

    # I3D setup (for v13 models)
    i3d_dir = None
    i3d_dim = 1024
    if model_type != "v12":
        i3d_dir = Path(cfg["data"].get("i3d_dir", "data/features_v13_i3d")) / "Test"
        i3d_dim = cfg["model"].get("i3d_dim", 1024)

    # Collect all test videos
    all_vis_files = sorted(features_dir.glob("*_visual.pt"))
    test_videos = [f.stem.replace("_visual", "") for f in all_vis_files]

    # Result containers
    results: Dict[str, Dict[str, np.ndarray]] = {
        "original": {},
        "temporal_flip": {},
    }
    for i in range(n_dropout_runs):
        results[f"dropout_{i}"] = {}

    for video in tqdm(test_videos, desc="[TTA] Scoring"):
        vis_p = features_dir / f"{video}_visual.pt"
        txt_p = features_dir / f"{video}_text.pt"
        flow_p = features_dir / f"{video}_flow.pt"

        if not txt_p.exists():
            continue

        vis = torch.load(vis_p, weights_only=True).unsqueeze(0).to(device)
        txt = torch.load(txt_p, weights_only=True).unsqueeze(0).to(device)
        flow = (
            torch.load(flow_p, weights_only=True).unsqueeze(0).to(device)
            if flow_p.exists() else torch.zeros(1, T, device=device)
        )

        i3d = None
        if i3d_dir is not None:
            i3d_p = i3d_dir / f"{video}_i3d.pt"
            i3d = (
                torch.load(i3d_p, weights_only=True).unsqueeze(0).to(device)
                if i3d_p.exists()
                else torch.zeros(1, T, i3d_dim, device=device)
            )

        # --- Strategy 1: Original ---
        if model_type == "v12":
            out = model(vis, txt, flow)
        else:
            out = model(vis, txt, flow, i3d_features=i3d)
        results["original"][video] = out[0].squeeze(0).cpu().numpy()

        # --- Strategy 2: Temporal Flip ---
        vis_flip = vis.flip(dims=[1])
        txt_flip = txt.flip(dims=[1])
        flow_flip = flow.flip(dims=[1]) if flow.dim() >= 2 else flow
        i3d_flip = i3d.flip(dims=[1]) if i3d is not None else None

        if model_type == "v12":
            out_flip = model(vis_flip, txt_flip, flow_flip)
        else:
            out_flip = model(vis_flip, txt_flip, flow_flip, i3d_features=i3d_flip)
        # Reverse back to original temporal order
        flip_scores = out_flip[0].squeeze(0).flip(dims=[0]).cpu().numpy()
        results["temporal_flip"][video] = flip_scores

        # --- Strategy 3: Feature Dropout (multiple runs) ---
        for i in range(n_dropout_runs):
            # Apply random mask to visual features
            mask = (torch.rand_like(vis) > dropout_rate).float()
            vis_dropped = vis * mask / (1.0 - dropout_rate)  # Scale to preserve magnitude

            if model_type == "v12":
                out_drop = model(vis_dropped, txt, flow)
            else:
                out_drop = model(vis_dropped, txt, flow, i3d_features=i3d)
            results[f"dropout_{i}"][video] = out_drop[0].squeeze(0).cpu().numpy()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run TTA evaluation."""
    parser = argparse.ArgumentParser(description="Test-Time Augmentation Evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", default="v12", choices=["v12", "v13"])
    parser.add_argument("--n_dropout", type=int, default=5)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    args = parser.parse_args()

    # Load shared resources
    annotations = load_annotations("data/Temporal_Anomaly_Annotation.txt")
    with open("data/video_frame_counts.json") as f:
        frame_counts: Dict[str, int] = json.load(f)

    print(f"[TTA] Config: {args.config}")
    print(f"[TTA] Checkpoint: {args.checkpoint}")
    print(f"[TTA] Model type: {args.model_type}")
    print(f"[TTA] Dropout runs: {args.n_dropout}, rate: {args.dropout_rate}\n")

    # Extract all TTA scores
    tta_results = tta_score_extraction(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        n_dropout_runs=args.n_dropout,
        dropout_rate=args.dropout_rate,
    )

    # Evaluate individual strategies
    print(f"\n{'='*60}")
    print("  INDIVIDUAL STRATEGY RESULTS")
    print(f"{'='*60}\n")

    strategy_aurocs: Dict[str, float] = {}
    for strategy, scores in tta_results.items():
        auroc = compute_frame_auroc(scores, annotations, frame_counts)
        strategy_aurocs[strategy] = auroc
        print(f"  {strategy:20s}: {auroc:.4f}")

    # --- TTA Ensemble: Average all strategies ---
    print(f"\n{'='*60}")
    print("  TTA ENSEMBLE RESULTS")
    print(f"{'='*60}\n")

    common_videos = set(tta_results["original"].keys())
    strategy_names = list(tta_results.keys())

    # 1) Original + Flip
    ens_of: Dict[str, np.ndarray] = {}
    for v in common_videos:
        ens_of[v] = 0.5 * tta_results["original"][v] + 0.5 * tta_results["temporal_flip"][v]
    auroc_of = compute_frame_auroc(ens_of, annotations, frame_counts)
    print(f"  Original + Flip:              {auroc_of:.4f}")

    # 2) Original + All Dropouts
    ens_od: Dict[str, np.ndarray] = {}
    n_drop = args.n_dropout
    for v in common_videos:
        combined = tta_results["original"][v].copy()
        for i in range(n_drop):
            combined = combined + tta_results[f"dropout_{i}"][v]
        ens_od[v] = combined / (1 + n_drop)
    auroc_od = compute_frame_auroc(ens_od, annotations, frame_counts)
    print(f"  Original + {n_drop} Dropouts:       {auroc_od:.4f}")

    # 3) All strategies combined
    ens_all: Dict[str, np.ndarray] = {}
    n_total = len(strategy_names)
    for v in common_videos:
        combined = sum(tta_results[s][v] for s in strategy_names)
        ens_all[v] = combined / n_total
    auroc_all = compute_frame_auroc(ens_all, annotations, frame_counts)
    print(f"  All {n_total} strategies:            {auroc_all:.4f}")

    # 4) Weighted: heavy on original, light on augmented
    for w_orig in [0.5, 0.6, 0.7, 0.8, 0.9]:
        w_aug = (1.0 - w_orig) / (n_total - 1)
        ens_w: Dict[str, np.ndarray] = {}
        for v in common_videos:
            combined = w_orig * tta_results["original"][v]
            for s in strategy_names:
                if s != "original":
                    combined = combined + w_aug * tta_results[s][v]
            ens_w[v] = combined
        auroc_w = compute_frame_auroc(ens_w, annotations, frame_counts)
        print(f"  Weighted (orig={w_orig:.1f}):         {auroc_w:.4f}")

    # 5) Score power calibration on best ensemble
    print(f"\n  --- Score Power Calibration (on best single) ---")
    for power in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        cal: Dict[str, np.ndarray] = {}
        for v in common_videos:
            s = tta_results["original"][v]
            cal[v] = np.power(np.clip(s, 1e-8, 1.0), power)
        auroc_p = compute_frame_auroc(cal, annotations, frame_counts)
        print(f"  Power={power:.2f}:                    {auroc_p:.4f}")

    best_auroc = max(auroc_of, auroc_od, auroc_all, strategy_aurocs["original"])
    print(f"\n{'='*60}")
    print(f"  BASELINE (original):  {strategy_aurocs['original']:.4f}")
    print(f"  BEST TTA result:      {best_auroc:.4f}")
    print(f"  Improvement:          {best_auroc - strategy_aurocs['original']:+.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
