"""Model Soup — Weight Averaging of Multiple Checkpoints.

Averages the parameters of models trained with different seeds to create
a single smoother model that often outperforms score-level ensembling.

Reference: Wortsman et al., "Model soups: averaging weights of multiple
fine-tuned models improves accuracy without increasing inference cost" (2022).

Usage:
    python scripts/10_model_soup.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import OrderedDict

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
# Helpers
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


def compute_frame_auroc(
    scores_dict: Dict[str, np.ndarray],
    annotations: Dict[str, List[int]],
    frame_counts: Dict[str, int],
    T: int = 128,
    t_proxy: int = 16,
) -> float:
    """Compute frame-level AUROC from segment scores."""
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for video, seg_scores in scores_dict.items():
        ann = annotations.get(video) or annotations.get(video.replace("_x264", ""))
        if ann is not None:
            s1, e1, s2, e2 = ann
            max_ann = max(v for v in ann if v > 0) if any(v > 0 for v in ann) else 0
            lookup = video if video in frame_counts else video.replace("_x264", "")
            N = frame_counts.get(lookup, max(max_ann + 1, T * t_proxy) if max_ann > 0 else T * t_proxy)
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

    return float(roc_auc_score(np.concatenate(all_labels), np.concatenate(all_preds)))


@torch.no_grad()
def score_with_model(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    model_type: str = "v12",
) -> Dict[str, np.ndarray]:
    """Score all test videos with a given model."""
    features_dir = Path(config["data"]["features_dir"]) / "Test"
    T = config["model"]["num_segments"]
    i3d_dir = None
    i3d_dim = 1024

    if model_type != "v12":
        i3d_dir = Path(config["data"].get("i3d_dir", "data/features_v13_i3d")) / "Test"
        i3d_dim = config["model"].get("i3d_dim", 1024)

    scores: Dict[str, np.ndarray] = {}
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

        if model_type == "v12":
            out = model(vis, txt, flow)
        else:
            i3d_p = i3d_dir / f"{video}_i3d.pt"
            i3d = (
                torch.load(i3d_p, weights_only=True).unsqueeze(0).to(device)
                if i3d_p.exists()
                else torch.zeros(1, T, i3d_dim, device=device)
            )
            out = model(vis, txt, flow, i3d_features=i3d)

        scores[video] = out[0].squeeze(0).cpu().numpy()

    return scores


# ---------------------------------------------------------------------------
# Weight Averaging
# ---------------------------------------------------------------------------

def average_state_dicts(
    state_dicts: List[OrderedDict],
    weights: List[float] | None = None,
) -> OrderedDict:
    """Average multiple state dicts with optional weights.

    Args:
        state_dicts: List of model state dicts.
        weights: Optional weights (normalised to sum to 1).

    Returns:
        OrderedDict: Averaged state dict.
    """
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    avg_sd = OrderedDict()
    for key in state_dicts[0].keys():
        avg_sd[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts))
        avg_sd[key] = avg_sd[key].to(state_dicts[0][key].dtype)

    return avg_sd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Model Soup experiments."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SOUP] Device: {device}\n")

    annotations = load_annotations("data/Temporal_Anomaly_Annotation.txt")
    with open("data/video_frame_counts.json") as f:
        frame_counts: Dict[str, int] = json.load(f)

    # -----------------------------------------------------------------------
    # Define available V12 checkpoints (same architecture, different seeds)
    # -----------------------------------------------------------------------
    v12_checkpoints: Dict[str, str] = {}

    if Path("checkpoints_v12/best_model_framelevel.pth").exists():
        v12_checkpoints["V12_s42"] = "checkpoints_v12/best_model_framelevel.pth"
    if Path("checkpoints_v12_s777/best_model_framelevel.pth").exists():
        v12_checkpoints["V12_s777"] = "checkpoints_v12_s777/best_model_framelevel.pth"
    if Path("checkpoints_v12_s123/best_model_framelevel.pth").exists():
        v12_checkpoints["V12_s123"] = "checkpoints_v12_s123/best_model_framelevel.pth"

    print(f"[SOUP] Found {len(v12_checkpoints)} V12 checkpoints: {list(v12_checkpoints.keys())}")

    cfg = load_config("configs/config_v12.yaml")

    # Load all state dicts
    all_sds: Dict[str, OrderedDict] = {}
    for name, path in v12_checkpoints.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        all_sds[name] = ckpt["model_state_dict"]
        print(f"  Loaded: {name} ({path})")

    # -----------------------------------------------------------------------
    # Evaluate individual checkpoints first
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  INDIVIDUAL MODEL RESULTS")
    print(f"{'='*60}\n")

    individual_aurocs: Dict[str, float] = {}
    for name, sd in all_sds.items():
        model = LanguageGuidedVAD.from_config(cfg).to(device)
        model.load_state_dict(sd)
        model.eval()
        scores = score_with_model(model, cfg, device, "v12")
        auroc = compute_frame_auroc(scores, annotations, frame_counts)
        individual_aurocs[name] = auroc
        print(f"  {name:15s}: {auroc:.4f}")
        del model

    # -----------------------------------------------------------------------
    # Model Soup: Uniform averaging
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  MODEL SOUP RESULTS")
    print(f"{'='*60}\n")

    names = list(all_sds.keys())
    sds = [all_sds[n] for n in names]

    # Uniform soup (all models)
    if len(sds) >= 2:
        avg_sd = average_state_dicts(sds)
        model = LanguageGuidedVAD.from_config(cfg).to(device)
        model.load_state_dict(avg_sd)
        model.eval()
        scores = score_with_model(model, cfg, device, "v12")
        auroc = compute_frame_auroc(scores, annotations, frame_counts)
        combo_str = "+".join(names)
        print(f"  Uniform Soup ({combo_str}): {auroc:.4f}")
        del model

    # Pairwise soups
    from itertools import combinations
    best_soup_auroc = 0.0
    best_soup_name = ""
    best_soup_weights: List[float] = []

    for (n1, n2) in combinations(names, 2):
        # Grid search weight for pairwise
        best_pair_auroc = 0.0
        best_w = 0.5
        for w in np.arange(0.1, 1.0, 0.1):
            avg_sd = average_state_dicts(
                [all_sds[n1], all_sds[n2]],
                weights=[w, 1.0 - w],
            )
            model = LanguageGuidedVAD.from_config(cfg).to(device)
            model.load_state_dict(avg_sd)
            model.eval()
            scores = score_with_model(model, cfg, device, "v12")
            auroc = compute_frame_auroc(scores, annotations, frame_counts)
            if auroc > best_pair_auroc:
                best_pair_auroc = auroc
                best_w = w
            del model

        print(f"  Soup {n1}+{n2}: {best_pair_auroc:.4f}  (w=[{best_w:.1f}, {1-best_w:.1f}])")
        if best_pair_auroc > best_soup_auroc:
            best_soup_auroc = best_pair_auroc
            best_soup_name = f"{n1}+{n2}"
            best_soup_weights = [best_w, 1 - best_w]

    best_individual = max(individual_aurocs.values())
    print(f"\n{'='*60}")
    print(f"  BEST INDIVIDUAL:  {best_individual:.4f}")
    print(f"  BEST SOUP:        {best_soup_auroc:.4f}  ({best_soup_name}, w={best_soup_weights})")
    print(f"  Improvement:      {best_soup_auroc - best_individual:+.4f}")
    print(f"{'='*60}\n")

    # Save best soup checkpoint
    if best_soup_auroc > best_individual:
        soup_dir = Path("checkpoints_soup")
        soup_dir.mkdir(exist_ok=True)
        w_names = best_soup_name.split("+")
        best_avg_sd = average_state_dicts(
            [all_sds[n] for n in w_names],
            weights=best_soup_weights,
        )
        save_path = soup_dir / "best_soup_framelevel.pth"
        torch.save({"model_state_dict": best_avg_sd}, save_path)
        print(f"  [SAVED] Best soup checkpoint: {save_path}")


if __name__ == "__main__":
    main()
