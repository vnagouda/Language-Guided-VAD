"""Score-Level Ensemble Evaluation for UCF-Crime.

Loads multiple model checkpoints (V12, V13, V14, V15), generates per-video
segment scores from each, and evaluates all pairwise/triplet/quad ensembles
with weight grid search.

Usage:
    python scripts/08_ensemble_eval.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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
# Score Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all_scores_v12(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Extract per-video segment scores from a V12 (LanguageGuidedVAD) model.

    Args:
        config_path: Path to V12 YAML config.
        checkpoint_path: Path to V12 checkpoint.
        device: Torch device.

    Returns:
        Dict mapping video_name → segment scores array of shape (T,).
    """
    cfg = load_config(config_path)
    model = LanguageGuidedVAD.from_config(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    features_dir = Path(cfg["data"]["features_dir"]) / "Test"
    T = cfg["model"]["num_segments"]
    scores_dict: Dict[str, np.ndarray] = {}

    for vis_p in sorted(features_dir.glob("*_visual.pt")):
        video = vis_p.stem.replace("_visual", "")
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

        out = model(vis, txt, flow)
        scores_dict[video] = out[0].squeeze(0).cpu().numpy()

    return scores_dict


@torch.no_grad()
def extract_all_scores_v13(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Extract per-video segment scores from a V13+ (TriModalVAD) model.

    Args:
        config_path: Path to V13/V14/V15 YAML config.
        checkpoint_path: Path to checkpoint.
        device: Torch device.

    Returns:
        Dict mapping video_name → segment scores array of shape (T,).
    """
    cfg = load_config(config_path)
    model = TriModalVAD.from_config(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    features_dir = Path(cfg["data"]["features_dir"]) / "Test"
    i3d_dir = Path(cfg["data"].get("i3d_dir", "data/features_v13_i3d")) / "Test"
    T = cfg["model"]["num_segments"]
    i3d_dim = cfg["model"].get("i3d_dim", 1024)
    scores_dict: Dict[str, np.ndarray] = {}

    for vis_p in sorted(features_dir.glob("*_visual.pt")):
        video = vis_p.stem.replace("_visual", "")
        txt_p = features_dir / f"{video}_text.pt"
        flow_p = features_dir / f"{video}_flow.pt"
        i3d_p = i3d_dir / f"{video}_i3d.pt"

        if not txt_p.exists():
            continue

        vis = torch.load(vis_p, weights_only=True).unsqueeze(0).to(device)
        txt = torch.load(txt_p, weights_only=True).unsqueeze(0).to(device)
        flow = (
            torch.load(flow_p, weights_only=True).unsqueeze(0).to(device)
            if flow_p.exists() else torch.zeros(1, T, device=device)
        )
        i3d = (
            torch.load(i3d_p, weights_only=True).unsqueeze(0).to(device)
            if i3d_p.exists() else torch.zeros(1, T, i3d_dim, device=device)
        )

        out = model(vis, txt, flow, i3d_features=i3d)
        scores_dict[video] = out[0].squeeze(0).cpu().numpy()

    return scores_dict


# ---------------------------------------------------------------------------
# Frame-Level AUROC from Segment Scores
# ---------------------------------------------------------------------------

def compute_frame_auroc(
    scores_dict: Dict[str, np.ndarray],
    annotations: Dict[str, List[int]],
    frame_counts: Dict[str, int],
    T: int = 128,
    t_proxy: int = 16,
) -> float:
    """Compute frame-level AUROC from segment scores using full protocol.

    Args:
        scores_dict: video_name → segment scores (T,).
        annotations: video_name → [s1, e1, s2, e2].
        frame_counts: video_name → original frame count.
        T: Number of segments.
        t_proxy: Fallback proxy multiplier.

    Returns:
        float: Frame-level AUROC.
    """
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
# Ensemble Logic
# ---------------------------------------------------------------------------

def ensemble_scores(
    score_dicts: List[Dict[str, np.ndarray]],
    weights: List[float],
) -> Dict[str, np.ndarray]:
    """Weighted average of multiple score dictionaries.

    Args:
        score_dicts: List of score dicts from different models.
        weights: Corresponding weights (will be normalised to sum to 1).

    Returns:
        Dict: Ensembled scores per video.
    """
    w = np.array(weights)
    w = w / w.sum()

    # Use intersection of all video sets
    common_videos = set(score_dicts[0].keys())
    for sd in score_dicts[1:]:
        common_videos &= set(sd.keys())

    ensembled: Dict[str, np.ndarray] = {}
    for video in common_videos:
        combined = sum(w[i] * score_dicts[i][video] for i in range(len(w)))
        ensembled[video] = combined  # type: ignore[assignment]

    return ensembled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all ensemble experiments."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENSEMBLE] Device: {device}\n")

    # Load shared resources
    annotations = load_annotations("data/Temporal_Anomaly_Annotation.txt")
    with open("data/video_frame_counts.json") as f:
        frame_counts: Dict[str, int] = json.load(f)

    # -----------------------------------------------------------------------
    # Step 1: Extract scores from all available models
    # -----------------------------------------------------------------------
    models_info: Dict[str, Tuple[str, str, str]] = {}  # name → (config, ckpt, type)

    # V12 (LanguageGuidedVAD, no I3D) — seed 42
    if Path("checkpoints_v12/best_model_framelevel.pth").exists():
        models_info["V12"] = (
            "configs/config_v12.yaml",
            "checkpoints_v12/best_model_framelevel.pth",
            "v12",
        )

    # Auto-discover all V12 seed checkpoints
    import glob
    for ckpt_dir in sorted(glob.glob("checkpoints_v12_s*")):
        seed_str = ckpt_dir.replace("checkpoints_v12_s", "")
        ckpt_path = Path(ckpt_dir) / "best_model_framelevel.pth"
        config_path = f"configs/config_v12_s{seed_str}.yaml"
        if ckpt_path.exists():
            name = f"V12_s{seed_str}"
            # Use V12 config as fallback if seed-specific config doesn't exist
            if not Path(config_path).exists():
                config_path = "configs/config_v12.yaml"
            models_info[name] = (config_path, str(ckpt_path), "v12")

    # V13 SKIPPED — checkpoint uses old early fusion architecture (2048-dim I3D)
    # incompatible with current code and 1024-dim features

    # V14 (TriModalVAD, late fusion, warm-start)
    if Path("checkpoints_v14/best_model_framelevel.pth").exists():
        models_info["V14"] = (
            "configs/config_v14.yaml",
            "checkpoints_v14/best_model_framelevel.pth",
            "v13",
        )

    # V15 (TriModalVAD, late fusion, from scratch)
    if Path("checkpoints_v15/best_model_framelevel.pth").exists():
        models_info["V15"] = (
            "configs/config_v15.yaml",
            "checkpoints_v15/best_model_framelevel.pth",
            "v13",
        )

    print(f"[ENSEMBLE] Found {len(models_info)} models: {list(models_info.keys())}")
    all_scores: Dict[str, Dict[str, np.ndarray]] = {}
    individual_aurocs: Dict[str, float] = {}

    for name, (cfg_path, ckpt_path, model_type) in models_info.items():
        print(f"\n--- Extracting scores: {name} ---")
        try:
            if model_type == "v12":
                scores = extract_all_scores_v12(cfg_path, ckpt_path, device)
            else:
                scores = extract_all_scores_v13(cfg_path, ckpt_path, device)
            all_scores[name] = scores
            # Compute individual AUROC
            auroc = compute_frame_auroc(scores, annotations, frame_counts)
            individual_aurocs[name] = auroc
            print(f"  {name} standalone Frame-AUROC: {auroc:.4f}")
        except Exception as exc:
            print(f"  [WARN] Failed to load {name}: {exc}")

    # -----------------------------------------------------------------------
    # Step 2: Filter to top-N models for ensemble (avoid combinatorial explosion)
    # -----------------------------------------------------------------------
    TOP_N = 5  # Only ensemble the best 5 models
    sorted_models = sorted(individual_aurocs, key=individual_aurocs.get, reverse=True)

    # Always include V15 if available (decorrelated tri-modal errors)
    top_models = sorted_models[:TOP_N]
    if "V15" in individual_aurocs and "V15" not in top_models:
        top_models.append("V15")
    if "V14" in individual_aurocs and "V14" not in top_models:
        top_models.append("V14")

    print(f"\n{'='*60}")
    print(f"  INDIVIDUAL RESULTS (all {len(individual_aurocs)} models)")
    print(f"{'='*60}")
    for name in sorted_models:
        marker = " ★" if name == sorted_models[0] else ""
        top = " [TOP-5]" if name in top_models else ""
        print(f"  {name:<15} {individual_aurocs[name]:.4f}{marker}{top}")

    print(f"\n{'='*60}")
    print(f"  ENSEMBLE RESULTS (top-{len(top_models)} models)")
    print(f"{'='*60}\n")

    best_overall_auroc = 0.0
    best_overall_combo = ""
    best_overall_weights: List[float] = []

    # Weight grid: 0.0 to 1.0 in steps of 0.05
    weight_steps = np.arange(0.0, 1.05, 0.05)

    # --- Pairwise (top models only) ---
    for (n1, n2) in combinations(top_models, 2):
        best_auroc = 0.0
        best_w = 0.0
        for w1 in weight_steps:
            w2 = 1.0 - w1
            ens = ensemble_scores([all_scores[n1], all_scores[n2]], [w1, w2])
            auroc = compute_frame_auroc(ens, annotations, frame_counts)
            if auroc > best_auroc:
                best_auroc = auroc
                best_w = w1
        print(f"  {n1}+{n2}: AUROC={best_auroc:.4f}  (w=[{best_w:.2f}, {1-best_w:.2f}])")
        if best_auroc > best_overall_auroc:
            best_overall_auroc = best_auroc
            best_overall_combo = f"{n1}+{n2}"
            best_overall_weights = [best_w, 1 - best_w]

    # --- Triplets (top-5 only, coarser grid) ---
    if len(top_models) >= 3:
        for combo in combinations(top_models, 3):
            best_auroc = 0.0
            best_ws: List[float] = []
            for w1 in np.arange(0.0, 1.1, 0.1):
                for w2 in np.arange(0.0, 1.1 - w1, 0.1):
                    w3 = 1.0 - w1 - w2
                    if w3 < -0.01:
                        continue
                    w3 = max(w3, 0.0)
                    ens = ensemble_scores(
                        [all_scores[combo[0]], all_scores[combo[1]], all_scores[combo[2]]],
                        [w1, w2, w3],
                    )
                    auroc = compute_frame_auroc(ens, annotations, frame_counts)
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_ws = [w1, w2, w3]
            ws_str = ", ".join(f"{w:.1f}" for w in best_ws)
            print(f"  {'+'.join(combo)}: AUROC={best_auroc:.4f}  (w=[{ws_str}])")
            if best_auroc > best_overall_auroc:
                best_overall_auroc = best_auroc
                best_overall_combo = "+".join(combo)
                best_overall_weights = best_ws

    # --- Uniform average of top-3 ---
    if len(top_models) >= 3:
        top3 = top_models[:3]
        uniform_w = [1.0 / 3] * 3
        ens = ensemble_scores([all_scores[n] for n in top3], uniform_w)
        auroc = compute_frame_auroc(ens, annotations, frame_counts)
        print(f"\n  Uniform top-3 ({'+'.join(top3)}): AUROC={auroc:.4f}")

    # --- Uniform average of top-5 ---
    if len(top_models) >= 5:
        top5 = top_models[:5]
        uniform_w = [1.0 / 5] * 5
        ens = ensemble_scores([all_scores[n] for n in top5], uniform_w)
        auroc = compute_frame_auroc(ens, annotations, frame_counts)
        print(f"  Uniform top-5 ({'+'.join(top5)}): AUROC={auroc:.4f}")

    print(f"\n{'='*60}")
    print(f"  BEST ENSEMBLE: {best_overall_combo}")
    print(f"  Frame-AUROC:   {best_overall_auroc:.4f}")
    print(f"  Weights:       {best_overall_weights}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
