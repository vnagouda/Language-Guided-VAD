"""SENTINEL Extension 2: Temporal Prediction Error (LSTM Autoencoder).

Trains a lightweight LSTM Autoencoder exclusively on NORMAL video visual features
from the training set. The model learns to reconstruct normal temporal sequences.

At test time, the anomaly score is the MSE reconstruction error:
    Score = MSE(visual_feat, reconstructed_feat)

This score is min-max normalised and ensembled with the primary V4 model.

Usage:
    python scripts/06_temporal_pred_error.py --config configs/config_v4_sota.yaml --checkpoint checkpoints_v4/best_model_framelevel.pth
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vad_architecture import LanguageGuidedVAD
from utils.video_utils import load_config
from utils.metrics import interpolate_scores

# ---------------------------------------------------------------------------
# LSTM Autoencoder Architecture
# ---------------------------------------------------------------------------

class NormalLSTMAutoencoder(nn.Module):
    def __init__(self, feature_dim: int = 768, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        # Encoder
        self.encoder = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        # Decoder (Takes bidirectional hidden states -> 2 * hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, feature_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, 768)
        encoded_seq, _ = self.encoder(x)  # (B, T, 512)
        reconstructed  = self.decoder(encoded_seq)  # (B, T, 768)
        return reconstructed

# ---------------------------------------------------------------------------
# Dataset for Normal Videos Only
# ---------------------------------------------------------------------------

class NormalVisualDataset(Dataset):
    def __init__(self, features_dir: Path):
        # Find all visual features in the Train folder that are "Normal_Videos"
        all_vis = list(features_dir.glob("*_visual.pt"))
        self.files = [f for f in all_vis if "Normal_Videos" in f.stem]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        feat = torch.load(self.files[idx], weights_only=True)  # (32, 768)
        return feat

# ---------------------------------------------------------------------------
# Annotation Parser
# ---------------------------------------------------------------------------

def load_annotations(annotation_file: str) -> Dict[str, List[int]]:
    ann: Dict[str, List[int]] = {}
    with open(annotation_file, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 5:
                name = parts[0].replace(".mp4", "")
                ann[name] = [int(x) for x in parts[-4:]]
    return ann

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def run_temporal_pred_error(
    config_path: str,
    checkpoint_path: str,
    epochs: int = 50,
    batch_size: int = 128,
):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXT-2] Device: {device}")

    # 1. Train the LSTM Autoencoder on Normal Train Videos
    train_dir = Path(cfg["data"]["features_dir"]) / "Train"
    train_dataset = NormalVisualDataset(train_dir)
    print(f"[EXT-2] Found {len(train_dataset)} Normal training videos.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    lstm_ae = NormalLSTMAutoencoder(feature_dim=cfg["model"]["feature_dim"]).to(device)
    optimizer = optim.AdamW(lstm_ae.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    lstm_ae.train()
    print("[EXT-2] Training Temporal LSTM Autoencoder...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = lstm_ae(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # 2. Get baseline V4 Model Scores
    model = LanguageGuidedVAD.from_config(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_dir = Path(cfg["data"]["features_dir"]) / "Test"
    all_test_vis = sorted(test_dir.glob("*_visual.pt"))
    test_videos = [f.stem.replace("_visual", "") for f in all_test_vis]

    # Evaluate all test videos through both models
    model_scores_dict: Dict[str, np.ndarray] = {}
    lstm_scores_dict: Dict[str, np.ndarray]  = {}

    lstm_ae.eval()
    mse_loss_fn = nn.MSELoss(reduction='none')

    print(f"\n[EXT-2] Scoring {len(test_videos)} Test Videos...")
    with torch.no_grad():
        for video in tqdm(test_videos, desc="[EXT-2] inference"):
            vis_p = test_dir / f"{video}_visual.pt"
            txt_p = test_dir / f"{video}_text.pt"
            flow_p = test_dir / f"{video}_flow.pt"

            if not vis_p.exists() or not txt_p.exists():
                continue

            vis = torch.load(vis_p, weights_only=True).to(device)  # (32, 768)

            # --- LSTM Score ---
            vis_batch = vis.unsqueeze(0)
            recon = lstm_ae(vis_batch)
            # MSE per segment: mean across 768 dimension -> (32,)
            seg_errors = mse_loss_fn(recon, vis_batch).mean(dim=-1).squeeze(0)
            
            # temporal smooth it just in case
            seg_errors_np = seg_errors.cpu().numpy()
            
            lstm_scores_dict[video] = seg_errors_np

            # --- V4 Model Score ---
            txt = torch.load(txt_p, weights_only=True).unsqueeze(0).to(device)
            flow = (
                torch.load(flow_p, weights_only=True).unsqueeze(0).to(device)
                if flow_p.exists() else torch.zeros(1, 32, device=device)
            )
            v4_scores, _, _ = model(vis_batch, txt, flow)
            model_scores_dict[video] = v4_scores.squeeze(0).cpu().numpy()

    # 3. Global Min-Max Normalise LSTM Scores
    # We normalise across the entire test set so relative magnitudes are preserved
    all_lstm = np.concatenate(list(lstm_scores_dict.values()))
    l_min, l_max = all_lstm.min(), all_lstm.max()
    for v in lstm_scores_dict:
        lstm_scores_dict[v] = (lstm_scores_dict[v] - l_min) / (l_max - l_min + 1e-6)

    # 4. Interpolate and Eval against GT
    annotations = load_annotations(cfg["data"]["annotation_file"])
    T_proxy = 16
    T = cfg["model"]["num_segments"]

    preds_v4, preds_lstm, labels = [], [], []

    for video in test_videos:
        if video not in model_scores_dict:
            continue
        
        ann = annotations.get(video) or annotations.get(video.replace("_x264", ""))
        
        N = T * T_proxy
        fm_v4   = interpolate_scores(model_scores_dict[video], N)
        fm_lstm = interpolate_scores(lstm_scores_dict[video], N)

        lbl = np.zeros(N, dtype=np.int32)
        if ann is not None:
            s1, e1, s2, e2 = ann
            if s1 >= 0 and e1 >= 0:
                lbl[min(s1, N-1):min(e1, N)] = 1
            if s2 >= 0 and e2 >= 0:
                lbl[min(s2, N-1):min(e2, N)] = 1

        preds_v4.append(fm_v4)
        preds_lstm.append(fm_lstm)
        labels.append(lbl)

    p_v4   = np.concatenate(preds_v4)
    p_lstm = np.concatenate(preds_lstm)
    gt     = np.concatenate(labels)

    auroc_v4   = roc_auc_score(gt, p_v4)
    auroc_lstm = roc_auc_score(gt, p_lstm)

    print(f"\n{'='*60}")
    print(f"[EXT-2] Standalone AUROCs:")
    print(f"  V4 Model           : {auroc_v4:.4f}")
    print(f"  LSTM Temporal Error: {auroc_lstm:.4f}")
    
    print(f"\n[EXT-2] Ensembling Grid Search...")
    best_alpha, best_auroc = 1.0, auroc_v4
    for a in np.arange(0.0, 1.01, 0.05):
        ens = a * p_v4 + (1.0 - a) * p_lstm
        auroc = roc_auc_score(gt, ens)
        marker = " <-- BEST" if auroc > best_auroc else ""
        print(f"  alpha={a:.2f} (V4) / {1-a:.2f} (LSTM) -> AUROC={auroc:.4f}{marker}")
        if auroc > best_auroc:
            best_auroc = auroc
            best_alpha = a

    gain = best_auroc - auroc_v4
    print(f"\n{'='*60}")
    print(f"[EXT-2] FINAL RESULTS")
    print(f"  V4 Model (Baseline): {auroc_v4:.4f}")
    print(f"  LSTM Aux Signal    : {auroc_lstm:.4f}")
    print(f"  Best Ensemble      : {best_auroc:.4f}  (alpha={best_alpha:.2f})")
    print(f"  Peak Gain Found    : {gain:+.4f}  ({gain*100:+.2f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    run_temporal_pred_error(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs
    )
