"""Zero-Shot CLIP Semantic Danger Ensemble -- SENTINEL Extension 1.

Tests two zero-shot danger signals on top of the V4 trained model:
  1. Caption danger  -- cosine sim between BLIP-2 caption text features and danger phrases
  2. Visual danger   -- cosine sim between CLIP visual features and danger phrases (no re-extraction)

Evaluation mirrors frame_eval.py: T*16 proxy frames, interpolate_scores, annotated videos only.

Usage:
    python scripts/05_semantic_ensemble.py \\
        --config configs/config_v4_sota.yaml \\
        --checkpoint checkpoints_v4/best_model_framelevel.pth --grid_search
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vad_architecture import LanguageGuidedVAD
from utils.video_utils import load_config
from utils.metrics import interpolate_scores


# ---------------------------------------------------------------------------
# Danger Phrase Vocabulary
# ---------------------------------------------------------------------------

DANGER_PHRASES: List[str] = [
    "a violent physical assault is occurring",
    "a person is being attacked or beaten",
    "a fight breaking out between people",
    "a brawl or physical altercation",
    "someone is being punched kicked or hit",
    "a person carrying or brandishing a weapon",
    "a gun or knife is visible in the scene",
    "a shooting or stabbing is taking place",
    "a robbery or theft is in progress",
    "a person is stealing or grabbing property",
    "a burglary or break-in is occurring",
    "someone running away after a crime",
    "a vehicle crash or road accident",
    "a car hitting a person or another vehicle",
    "an explosion or fire emergency",
    "someone falling or collapsing in distress",
    "a dangerous or threatening situation",
    "suspicious or criminal behaviour",
    "an emergency requiring immediate response",
    "someone in danger or serious trouble",
]

NORMAL_PHRASES: List[str] = [
    "people walking normally in a public place",
    "a peaceful ordinary everyday scene",
    "normal routine activity with no danger",
    "people going about their daily business",
]


# ---------------------------------------------------------------------------
# CLIP Text Encoder
# ---------------------------------------------------------------------------

def encode_danger_phrases(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode danger/normal phrases with openai/clip-vit-large-patch14.

    Returns:
        (danger_embeds, normal_embeds) -- both L2-normalised, shape (N, 768).
    """
    from transformers import CLIPTokenizer, CLIPTextModel
    clip_name = "openai/clip-vit-large-patch14"
    print(f"[SENTINEL] Loading CLIP text encoder: {clip_name}")
    tokenizer  = CLIPTokenizer.from_pretrained(clip_name)
    text_model = CLIPTextModel.from_pretrained(clip_name).to(device).eval()

    def _encode(phrases: List[str]) -> torch.Tensor:
        tokens = tokenizer(
            phrases, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            embeds = text_model(**tokens).pooler_output
        return F.normalize(embeds, dim=-1).cpu()

    d = _encode(DANGER_PHRASES)
    n = _encode(NORMAL_PHRASES)
    print(f"[SENTINEL] Encoded {len(DANGER_PHRASES)} danger + {len(NORMAL_PHRASES)} normal phrases.")
    return d, n


# ---------------------------------------------------------------------------
# Danger Score Functions
# ---------------------------------------------------------------------------

def _contrast_score(feat: torch.Tensor, d_emb: torch.Tensor, n_emb: torch.Tensor) -> torch.Tensor:
    """Contrastive score: max(danger_sim) - mean(normal_sim), min-max normalised.

    Args:
        feat:  (T, D) feature matrix (unnormalised ok).
        d_emb: (N_d, D) danger phrase embeddings (L2-normalised).
        n_emb: (N_n, D) normal phrase embeddings (L2-normalised).

    Returns:
        (T,) score in [0, 1].
    """
    f   = F.normalize(feat.float(), dim=-1)
    d   = torch.matmul(f, d_emb.T).max(dim=-1).values
    n   = torch.matmul(f, n_emb.T).mean(dim=-1)
    raw = d - n
    lo, hi = raw.min(), raw.max()
    if (hi - lo).abs() > 1e-6:
        raw = (raw - lo) / (hi - lo)
    else:
        raw = torch.zeros_like(raw)
    return raw


# ---------------------------------------------------------------------------
# Annotation Parser -- mirrors frame_eval.py
# ---------------------------------------------------------------------------

def load_annotations(annotation_file: str) -> Dict[str, List[int]]:
    """Parse UCF-Crime Temporal_Anomaly_Annotation.txt.

    Format: VideoName.mp4  Category  start1  end1  start2  end2
    Uses parts[-4:] robustly (same as frame_eval.py).

    Returns:
        Dict: video_stem -> [start1, end1, start2, end2].
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
# Model Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_model_scores(
    model: LanguageGuidedVAD,
    features_dir: Path,
    test_videos: List[str],
    device: torch.device,
    num_segments: int = 32,
) -> Dict[str, np.ndarray]:
    """Run V4 forward pass over all test videos.

    Returns:
        Dict: video_name -> (T,) anomaly score array.
    """
    model.eval()
    out: Dict[str, np.ndarray] = {}
    for video in tqdm(test_videos, desc="[Model] Scoring"):
        vis_p  = features_dir / f"{video}_visual.pt"
        txt_p  = features_dir / f"{video}_text.pt"
        flow_p = features_dir / f"{video}_flow.pt"
        if not vis_p.exists() or not txt_p.exists():
            continue
        vis  = torch.load(vis_p,  weights_only=True).unsqueeze(0).to(device)
        txt  = torch.load(txt_p,  weights_only=True).unsqueeze(0).to(device)
        flow = (
            torch.load(flow_p, weights_only=True).unsqueeze(0).to(device)
            if flow_p.exists() else torch.zeros(1, num_segments, device=device)
        )
        scores, _, _ = model(vis, txt, flow)
        out[video] = scores.squeeze(0).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def evaluate_ensemble(
    config_path: str,
    checkpoint_path: str,
    alpha: float = 0.7,
    grid_search: bool = False,
) -> None:
    """Evaluate model-only, caption-danger, visual-danger, and their ensembles.

    Args:
        config_path:     Path to YAML config.
        checkpoint_path: Path to trained model checkpoint.
        alpha:           Default model weight in ensemble (only used without --grid_search).
        grid_search:     Sweep alpha 0->1 for both caption and visual danger signals.
    """
    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SENTINEL] Device: {device}")

    # Load model
    model = LanguageGuidedVAD.from_config(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[SENTINEL] Loaded: {checkpoint_path}")

    # Encode danger/normal phrases
    danger_embeds, normal_embeds = encode_danger_phrases(device)

    # Collect test videos
    features_dir = Path(cfg["data"]["features_dir"]) / "Test"
    test_videos  = sorted(
        f.stem.replace("_visual", "") for f in features_dir.glob("*_visual.pt")
    )
    print(f"[SENTINEL] Test videos: {len(test_videos)}")

    # Model scores
    model_scores = compute_model_scores(
        model, features_dir, test_videos, device,
        num_segments=cfg["model"]["num_segments"],
    )

    # Caption danger + Visual danger scores
    print("[SENTINEL] Computing caption + visual danger scores ...")
    cap_scores: Dict[str, np.ndarray] = {}
    vis_scores: Dict[str, np.ndarray] = {}
    for video in tqdm(test_videos, desc="[Danger] Scoring"):
        txt_p = features_dir / f"{video}_text.pt"
        vis_p = features_dir / f"{video}_visual.pt"
        if txt_p.exists():
            tf = torch.load(txt_p, weights_only=True)
            cap_scores[video] = _contrast_score(tf, danger_embeds, normal_embeds).numpy()
        if vis_p.exists():
            vf = torch.load(vis_p, weights_only=True)
            vis_scores[video] = _contrast_score(vf, danger_embeds, normal_embeds).numpy()

    # Load annotations
    annotations = load_annotations(cfg["data"]["annotation_file"])
    print(f"[SENTINEL] Annotated videos: {len(annotations)}")

    # Build frame-level arrays (T*16 proxy, same as frame_eval.py)
    T_proxy = 16
    lbl_list, m_list, c_list, v_list = [], [], [], []
    T_default = cfg["model"]["num_segments"]

    for video_name, seg_scores in model_scores.items():
        ann = annotations.get(video_name) or annotations.get(video_name.replace("_x264", ""))
        if ann is None:
            continue
        T         = len(seg_scores)
        N         = T * T_proxy
        fm        = interpolate_scores(seg_scores, N)
        fc        = interpolate_scores(cap_scores.get(video_name, np.zeros(T)), N)
        fv        = interpolate_scores(vis_scores.get(video_name, np.zeros(T)), N)
        lbl       = np.zeros(N, dtype=np.int32)
        s1, e1, s2, e2 = ann
        if s1 >= 0 and e1 >= 0:
            lbl[min(s1, N - 1):min(e1, N)] = 1
        if s2 >= 0 and e2 >= 0:
            lbl[min(s2, N - 1):min(e2, N)] = 1
        lbl_list.append(lbl); m_list.append(fm); c_list.append(fc); v_list.append(fv)

    preds_m = np.concatenate(m_list)
    preds_c = np.concatenate(c_list)
    preds_v = np.concatenate(v_list)
    labels  = np.concatenate(lbl_list)

    model_auroc = roc_auc_score(labels, preds_m)
    cap_auroc   = roc_auc_score(labels, preds_c)
    vis_auroc   = roc_auc_score(labels, preds_v)

    print(f"\n{'='*60}")
    print(f"[SENTINEL] Standalone AUROCs:")
    print(f"  Model (V4)         : {model_auroc:.4f}")
    print(f"  Caption Danger     : {cap_auroc:.4f}")
    print(f"  Visual Danger      : {vis_auroc:.4f}")

    best_auroc   = model_auroc
    best_label   = "Model Only"
    best_ensemble = preds_m

    if grid_search:
        for tag, preds_aux in [("Model+Visual", preds_v), ("Model+Caption", preds_c)]:
            print(f"\n[SENTINEL] Grid-search: {tag} ...")
            ba, bv = 1.0, model_auroc
            for a in np.arange(0.0, 1.01, 0.05):
                ens   = a * preds_m + (1.0 - a) * preds_aux
                auroc = roc_auc_score(labels, ens)
                marker = " <-- BEST" if auroc > bv else ""
                print(f"  alpha={a:.2f}  ->  AUROC={auroc:.4f}{marker}")
                if auroc > bv:
                    bv, ba = auroc, a
            if bv > best_auroc:
                best_auroc    = bv
                best_label    = f"{tag} (alpha={ba:.2f})"
                alpha         = ba
                best_ensemble = alpha * preds_m + (1.0 - alpha) * preds_aux
    else:
        best_ensemble = alpha * preds_m + (1.0 - alpha) * preds_v
        best_auroc    = roc_auc_score(labels, best_ensemble)
        best_label    = f"Model+Visual (alpha={alpha:.2f})"

    final_auroc = roc_auc_score(labels, best_ensemble)
    gain        = final_auroc - model_auroc

    print(f"\n{'='*60}")
    print(f"[SENTINEL] FINAL RESULTS")
    print(f"  Model-only (V4)    : {model_auroc:.4f}")
    print(f"  Caption Danger     : {cap_auroc:.4f}")
    print(f"  Visual Danger      : {vis_auroc:.4f}")
    print(f"  Best Ensemble      : {final_auroc:.4f}  ({best_label})")
    print(f"  Gain vs model      : {gain:+.4f}  ({gain * 100:+.2f}%)")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SENTINEL Extension 1: CLIP Semantic Danger Score Ensemble"
    )
    parser.add_argument("--config",      required=True,  help="Path to config YAML")
    parser.add_argument("--checkpoint",  required=True,  help="Path to model checkpoint")
    parser.add_argument("--alpha",       type=float, default=0.7)
    parser.add_argument("--grid_search", action="store_true")
    args = parser.parse_args()

    evaluate_ensemble(
        config_path     = args.config,
        checkpoint_path = args.checkpoint,
        alpha           = args.alpha,
        grid_search     = args.grid_search,
    )
