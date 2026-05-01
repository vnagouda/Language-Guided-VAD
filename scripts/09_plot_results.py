"""Generates IEEE-style Academic Plots for the MSc Thesis.

This script loads a trained model, runs inference on the test set, and generates:
1. Video-Level and Frame-Level ROC Curves.
2. Frame-Level Precision-Recall (PR) Curve.
3. Qualitative Temporal Score Plots: Overlays the model's predicted anomaly 
   score curve against the ground-truth anomaly window for selected videos.

Usage:
    python scripts/09_plot_results.py --config configs/config_v5.yaml --checkpoint checkpoints_v5/best_model_framelevel.pth
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vad_architecture import LanguageGuidedVAD
from utils.video_utils import load_config
from utils.metrics import interpolate_scores

def load_annotations(annotation_file: str) -> Dict[str, List[int]]:
    ann: Dict[str, List[int]] = {}
    with open(annotation_file, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 5:
                name = parts[0].replace(".mp4", "")
                ann[name] = [int(x) for x in parts[-4:]]
    return ann

@torch.no_grad()
def generate_plots(config_path: str, checkpoint_path: str, output_dir: str):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PLOTS] Device: {device}")

    # Load Model
    model = LanguageGuidedVAD.from_config(cfg).to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as e:
        print(f"[ERROR] Could not load checkpoint: {e}")
        return
    model.eval()

    T = cfg["model"]["num_segments"]
    T_proxy = 16
    features_dir = Path(cfg["data"]["features_dir"]) / "Test"
    annotations = load_annotations(cfg["data"]["annotation_file"])
    all_vis = sorted(features_dir.glob("*_visual.pt"))
    test_videos = [f.stem.replace("_visual", "") for f in all_vis]

    # Containers for global metrics
    preds_frame, labels_frame = [], []

    # Select a few specific videos for qualitative temporal plotting
    qualitative_videos = ["Abuse028_x264", "Arrest001_x264", "Explosion021_x264"]
    qualitative_data = {}

    print(f"[PLOTS] Running inference on {len(test_videos)} videos...")
    for video in tqdm(test_videos):
        vis_p = features_dir / f"{video}_visual.pt"
        txt_p = features_dir / f"{video}_text.pt"
        flow_p = features_dir / f"{video}_flow.pt"

        if not vis_p.exists() or not txt_p.exists():
            continue

        vis = torch.load(vis_p, weights_only=True).unsqueeze(0).to(device)
        txt = torch.load(txt_p, weights_only=True).unsqueeze(0).to(device)
        flow = (
            torch.load(flow_p, weights_only=True).unsqueeze(0).to(device)
            if flow_p.exists() else torch.zeros(1, T, device=device)
        )

        scores, _, _ = model(vis, txt, flow)
        scores_np = scores.squeeze(0).cpu().numpy()

        # Frame-level expansion
        N = T * T_proxy
        frame_pred = interpolate_scores(scores_np, N)
        lbl = np.zeros(N, dtype=np.int32)
        
        ann = annotations.get(video) or annotations.get(video.replace("_x264", ""))
        if ann is not None:
            s1, e1, s2, e2 = ann
            if s1 >= 0 and e1 >= 0:
                lbl[min(s1, N-1):min(e1, N)] = 1
            if s2 >= 0 and e2 >= 0:
                lbl[min(s2, N-1):min(e2, N)] = 1
        
        preds_frame.append(frame_pred)
        labels_frame.append(lbl)

        # Save data for qualitative plots if it's one of our targets
        if video in qualitative_videos:
            qualitative_data[video] = {"scores": frame_pred, "labels": lbl}

    p_frame = np.concatenate(preds_frame)
    l_frame = np.concatenate(labels_frame)

    # ---------------------------------------------------------
    # 1. ROC Curve
    # ---------------------------------------------------------
    print("[PLOTS] Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(l_frame, p_frame)
    roc_auc = roc_auc_score(l_frame, p_frame)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"V5 Model (AUROC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("Frame-Level ROC Curve on UCF-Crime", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "roc_curve.pdf")
    plt.savefig(out_path / "roc_curve.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. Precision-Recall Curve
    # ---------------------------------------------------------
    print("[PLOTS] Generating Precision-Recall Curve...")
    precision, recall, _ = precision_recall_curve(l_frame, p_frame)
    pr_auc = average_precision_score(l_frame, p_frame)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="purple", lw=2, label=f"V5 Model (AP = {pr_auc:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title("Frame-Level Precision-Recall Curve", fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "pr_curve.pdf")
    plt.savefig(out_path / "pr_curve.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. Qualitative Temporal Score Plots
    # ---------------------------------------------------------
    print("[PLOTS] Generating Qualitative Curves...")
    for video, data in qualitative_data.items():
        scores = data["scores"]
        labels = data["labels"]
        frames = np.arange(len(scores))

        plt.figure(figsize=(12, 4))
        
        # Plot score curve
        plt.plot(frames, scores, color="blue", lw=2.5, label="Predicted Anomaly Score")
        
        # Shade ground truth regions
        gt_active = False
        start_idx = 0
        for i, lbl in enumerate(labels):
            if lbl == 1 and not gt_active:
                gt_active = True
                start_idx = i
            elif lbl == 0 and gt_active:
                gt_active = False
                plt.axvspan(start_idx, i, color="red", alpha=0.3, label="Ground Truth Anomaly" if start_idx == 0 else "")
        # Handle case where anomaly goes to the very end
        if gt_active:
            plt.axvspan(start_idx, len(labels), color="red", alpha=0.3)

        plt.xlim([0, len(scores)])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Frame Index (Proxy)", fontsize=14)
        plt.ylabel("Anomaly Score", fontsize=14)
        plt.title(f"Qualitative Result: {video}", fontsize=16)
        
        handles, labels_leg = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels_leg, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=12)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path / f"qualitative_{video}.pdf")
        plt.savefig(out_path / f"qualitative_{video}.png", dpi=300)
        plt.close()

    print(f"\n[DONE] All plots saved to {out_path}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_v5.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results_v5/plots")
    args = parser.parse_args()

    generate_plots(args.config, args.checkpoint, args.output_dir)
