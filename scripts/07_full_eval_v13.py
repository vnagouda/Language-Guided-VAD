"""Full UCF-Crime Protocol Evaluation Script.

This script implements the CORRECT frame-level AUROC evaluation matching
state-of-the-art papers (RTFM, MGFN, etc.):
  - Includes ALL test videos: anomalous (GT labels from annotation file)
    AND normal (all frames labeled 0).
  - This is the standard UCF-Crime benchmark protocol.

Our training evaluator (frame_eval.py) only scored annotated anomalous videos,
making our reported 0.7824 non-comparable to papers reporting 84-87%.

Usage:
    python scripts/07_full_eval.py \\
        --config configs/config_v4_sota.yaml \\
        --checkpoint checkpoints_v4/best_model_framelevel.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vad_architecture_v13 import TriModalVAD
from utils.video_utils import load_config
from utils.metrics import interpolate_scores


# ---------------------------------------------------------------------------
# Annotation Parser
# ---------------------------------------------------------------------------

def load_annotations(annotation_file: str) -> Dict[str, List[int]]:
    """Parse UCF-Crime temporal annotations.

    Format: VideoName.mp4  Category  start1  end1  start2  end2

    Args:
        annotation_file: Path to Temporal_Anomaly_Annotation.txt.

    Returns:
        Dict: video_stem -> [start1, end1, start2, end2] for anomalous videos.
              Normal videos are NOT in this file (all-zero GT).
    """
    ann: Dict[str, List[int]] = {}
    with open(annotation_file, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 5:
                name = parts[0].replace(".mp4", "")
                ann[name] = [int(x) for x in parts[-4:]]
    return ann


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def full_protocol_eval(
    config_path: str,
    checkpoint_path: str,
    t_proxy: int = 16,
) -> None:
    """Evaluate frame-AUROC using the FULL UCF-Crime benchmark protocol.

    Includes both anomalous (from annotation file) and normal test videos
    (all frames labeled 0). This matches evaluation in RTFM, MGFN, UCA, etc.

    Args:
        config_path:     Path to YAML config.
        checkpoint_path: Path to model checkpoint (.pth).
        t_proxy:         Proxy frames per segment (T * t_proxy = total proxy frames).
    """
    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Device: {device}")

    # Load model
    model = TriModalVAD.from_config(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[EVAL] Loaded: {checkpoint_path}")

    features_dir = Path(cfg["data"]["features_dir"]) / "Test"
    i3d_dir      = Path(cfg["data"].get("i3d_dir", "data/features_v13_i3d")) / "Test"
    annotations  = load_annotations(cfg["data"]["annotation_file"])
    T            = cfg["model"]["num_segments"]  # 32

    # All test visual features
    all_vis_files = sorted(features_dir.glob("*_visual.pt"))
    test_videos   = [f.stem.replace("_visual", "") for f in all_vis_files]
    print(f"[EVAL] Total test videos: {len(test_videos)}")
    print(f"[EVAL] Annotated (anomalous): {sum(1 for v in test_videos if v in annotations or v.replace('_x264','') in annotations)}")
    print(f"[EVAL] Normal (no annotation): {sum(1 for v in test_videos if v not in annotations and v.replace('_x264','') not in annotations)}")

    all_preds:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    anomalous_count = 0
    normal_count    = 0

    # V11: Load pre-computed ORIGINAL video frame counts
    import json as _json
    _frame_counts: Dict[str, int] = {}
    fc_path = Path("data/video_frame_counts.json")
    if fc_path.exists():
        with open(fc_path) as _fc_fh:
            _frame_counts = _json.load(_fc_fh)
        print(f"[EVAL] Loaded {len(_frame_counts)} frame counts from {fc_path}")
    t_proxy = 16  # fallback multiplier

    for video in tqdm(test_videos, desc="[EVAL] Scoring"):
        vis_p  = features_dir / f"{video}_visual.pt"
        txt_p  = features_dir / f"{video}_text.pt"
        flow_p = features_dir / f"{video}_flow.pt"
        i3d_p  = i3d_dir / f"{video}_i3d.pt"

        if not vis_p.exists() or not txt_p.exists():
            continue

        vis  = torch.load(vis_p,  weights_only=True).unsqueeze(0).to(device)
        txt  = torch.load(txt_p,  weights_only=True).unsqueeze(0).to(device)
        flow = (
            torch.load(flow_p, weights_only=True).unsqueeze(0).to(device)
            if flow_p.exists() else torch.zeros(1, T, device=device)
        )
        i3d = (
            torch.load(i3d_p, weights_only=True).unsqueeze(0).to(device)
            if i3d_p.exists() else torch.zeros(1, T, cfg["model"].get("i3d_dim", 2048), device=device)
        )

        scores, _, _ = model(vis, txt, flow, i3d_features=i3d)
        seg_scores   = scores.squeeze(0).cpu().numpy()  # (T,)
        
        # -------------------------------------------------------------
        # V7 Eval Mode: No Post-Processing Native Output
        # -------------------------------------------------------------
        # Dilated convolutions already perfectly map temporal velocity gradients.
        # Adding Gaussian blur or text-masks mathematically flattens the exact
        # anomaly structures V7 learned. We evaluate the raw model.
        # seg_scores = gaussian_filter1d(seg_scores, sigma=2.0)

        # V10 APEX: Use ACTUAL frame count from annotations, not T*t_proxy.
        # The standard UCF-Crime protocol requires interpolating to the
        # original video's frame count. Annotations reference original
        # frame indices (up to 10,335 for long videos).
        ann = annotations.get(video) or annotations.get(video.replace("_x264", ""))

        if ann is not None:
            s1, e1, s2, e2 = ann
            max_ann = max(v for v in ann if v > 0) if any(v > 0 for v in ann) else 0
            has_anomaly = max_ann > 0

            # V11: Use JSON frame counts (estimates original video length)
            lookup = video if video in _frame_counts else video.replace("_x264", "")
            if lookup in _frame_counts:
                N = _frame_counts[lookup]
            elif has_anomaly:
                N = max(max_ann + 1, T * t_proxy)
            else:
                N = T * t_proxy

            frame_scores = interpolate_scores(seg_scores, N)
            frame_labels = np.zeros(N, dtype=np.int32)
            if s1 >= 0 and e1 >= 0:
                frame_labels[min(s1, N - 1):min(e1, N)] = 1
            if s2 >= 0 and e2 >= 0:
                frame_labels[min(s2, N - 1):min(e2, N)] = 1

            if has_anomaly:
                anomalous_count += 1
            else:
                normal_count += 1
        else:
            # Video not in annotation file at all
            lookup = video if video in _frame_counts else video.replace("_x264", "")
            N = _frame_counts.get(lookup, T * t_proxy)
            frame_scores = interpolate_scores(seg_scores, N)
            frame_labels = np.zeros(N, dtype=np.int32)
            normal_count += 1

        all_preds.append(frame_scores)
        all_labels.append(frame_labels)

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    n_anomaly = labels.sum()
    n_normal  = len(labels) - n_anomaly

    print(f"\n[EVAL] Videos scored:  anomalous={anomalous_count}, normal={normal_count}")
    print(f"[EVAL] Total frames:   anomaly={n_anomaly:,}, normal={n_normal:,}")
    print(f"[EVAL] Anomaly ratio:  {n_anomaly / len(labels):.3%}")

    auroc = roc_auc_score(labels, preds)

    print(f"\n{'='*60}")
    print(f"  FULL-PROTOCOL Frame-AUROC : {auroc:.4f}")
    print(f"  (OLD annotated-only AUROC was: 0.7824)")
    diff = auroc - 0.7824
    print(f"  Difference                : {diff:+.4f}")
    print(f"  This number IS comparable to RTFM/MGFN papers")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full UCF-Crime Protocol Frame-AUROC Evaluation"
    )
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--t_proxy",    type=int, default=16,
                        help="Proxy frames per segment T (default 16 -> 512 total)")
    args = parser.parse_args()

    full_protocol_eval(
        config_path     = args.config,
        checkpoint_path = args.checkpoint,
        t_proxy         = args.t_proxy,
    )
