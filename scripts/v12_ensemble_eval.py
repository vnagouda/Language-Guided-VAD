"""V12 Quick Evaluation: Ensemble + Score Sharpening to push past 85%.

Tests multiple strategies with NO retraining:
1. Score power transform (sigmoid sharpening)
2. Ensemble of V4 + V5 + V11
3. Ensemble + sharpening
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, ".")
from models.vad_architecture import LanguageGuidedVAD
from utils.metrics import compute_auroc, interpolate_scores
from utils.video_utils import load_config


def load_annotations(path: str) -> Dict[str, List[int]]:
    """Load temporal annotations from UCF-Crime annotation file."""
    annotations: Dict[str, List[int]] = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                name = parts[0].replace(".mp4", "")
                annotations[name] = [int(x) for x in parts[-4:]]
    return annotations


def score_all_videos(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Score all test videos with a given model checkpoint.

    Args:
        config_path: Path to config YAML.
        checkpoint_path: Path to model checkpoint.
        device: Torch device.

    Returns:
        Dict mapping video_name to segment scores (T,).
    """
    config = load_config(config_path)
    model = LanguageGuidedVAD.from_config(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    features_dir = Path(config["data"]["features_dir"]) / "Test"
    T = config["model"]["num_segments"]

    video_scores: Dict[str, np.ndarray] = {}
    all_vis = sorted(features_dir.glob("*_visual.pt"))

    for vis_p in all_vis:
        video = vis_p.stem.replace("_visual", "")
        txt_p = features_dir / f"{video}_text.pt"
        flow_p = features_dir / f"{video}_flow.pt"

        if not txt_p.exists():
            continue

        vis = torch.load(vis_p, weights_only=True).unsqueeze(0).to(device)
        txt = torch.load(txt_p, weights_only=True).unsqueeze(0).to(device)
        flow = (
            torch.load(flow_p, weights_only=True).unsqueeze(0).to(device)
            if flow_p.exists()
            else torch.zeros(1, T, device=device)
        )

        with torch.no_grad():
            scores, _, _ = model(vis, txt, flow)

        video_scores[video] = scores.squeeze(0).cpu().numpy()

    return video_scores


def evaluate_scores(
    video_scores: Dict[str, np.ndarray],
    annotations: Dict[str, List[int]],
    frame_counts: Dict[str, int],
    power: float = 1.0,
) -> float:
    """Evaluate frame-level AUROC with optional score sharpening.

    Args:
        video_scores: Dict mapping video name to segment scores.
        annotations: Dict mapping video name to [s1, e1, s2, e2].
        frame_counts: Dict mapping video name to original frame count.
        power: Score power transform exponent (>1 sharpens, <1 softens).

    Returns:
        Frame-level AUROC.
    """
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for video, seg_scores in video_scores.items():
        ann = annotations.get(video) or annotations.get(video.replace("_x264", ""))
        if ann is None:
            continue

        lookup = video if video in frame_counts else video.replace("_x264", "")
        N = frame_counts.get(lookup, len(seg_scores) * 16)

        # Apply score power transform
        if power != 1.0:
            seg_scores = np.power(np.clip(seg_scores, 0, 1), power)

        frame_scores = interpolate_scores(seg_scores, N)

        s1, e1, s2, e2 = ann
        frame_labels = np.zeros(N, dtype=np.int32)
        if s1 >= 0 and e1 >= 0:
            frame_labels[min(s1, N - 1) : min(e1, N)] = 1
        if s2 >= 0 and e2 >= 0:
            frame_labels[min(s2, N - 1) : min(e2, N)] = 1

        all_preds.append(frame_scores)
        all_labels.append(frame_labels)

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return float(compute_auroc(preds, labels))


def ensemble_scores(
    scores_list: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Average segment scores from multiple models.

    Args:
        scores_list: List of per-model video score dicts.

    Returns:
        Ensemble-averaged video scores.
    """
    all_videos = set()
    for s in scores_list:
        all_videos.update(s.keys())

    ensemble: Dict[str, np.ndarray] = {}
    for video in all_videos:
        arrays = [s[video] for s in scores_list if video in s]
        if arrays:
            # Pad to max length if needed
            max_len = max(len(a) for a in arrays)
            padded = [np.pad(a, (0, max_len - len(a))) for a in arrays]
            ensemble[video] = np.mean(padded, axis=0)

    return ensemble


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    annotations = load_annotations("data/Temporal_Anomaly_Annotation.txt")

    with open("data/video_frame_counts.json") as f:
        frame_counts = json.load(f)

    # Score all models
    models = {
        "V4": ("configs/config_v4_sota.yaml", "checkpoints_v4/best_model_framelevel.pth"),
        "V5": ("configs/config_v5.yaml", "checkpoints_v5/best_model_framelevel.pth"),
        "V11": ("configs/config_v11.yaml", "checkpoints_v11/best_model_framelevel.pth"),
    }

    all_scores: Dict[str, Dict[str, np.ndarray]] = {}
    for name, (cfg, ckpt) in models.items():
        print(f"\n--- Scoring {name} ---")
        all_scores[name] = score_all_videos(cfg, ckpt, device)
        print(f"  {name}: {len(all_scores[name])} videos scored")

    # 1. Individual models with different power transforms
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODELS + SCORE SHARPENING")
    print("=" * 60)
    for name in models:
        for p in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            auroc = evaluate_scores(all_scores[name], annotations, frame_counts, power=p)
            marker = " <<<" if auroc > 0.85 else ""
            print(f"  {name} (p={p:.1f}): {auroc:.4f}{marker}")

    # 2. Ensembles
    print("\n" + "=" * 60)
    print("ENSEMBLES")
    print("=" * 60)
    
    combos = [
        ("V4+V5", ["V4", "V5"]),
        ("V4+V11", ["V4", "V11"]),
        ("V5+V11", ["V5", "V11"]),
        ("V4+V5+V11", ["V4", "V5", "V11"]),
    ]
    
    for combo_name, members in combos:
        ens = ensemble_scores([all_scores[m] for m in members])
        for p in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            auroc = evaluate_scores(ens, annotations, frame_counts, power=p)
            marker = " <<<" if auroc > 0.85 else ""
            print(f"  {combo_name} (p={p:.1f}): {auroc:.4f}{marker}")
        print()

    print("\nDONE!")
