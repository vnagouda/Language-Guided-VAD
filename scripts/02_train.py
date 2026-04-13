"""Training script for the Language-Guided WS-VAD model.

Implements the MIL training loop:
    1. Load pre-extracted .pt features via VADDataset / DataLoader
    2. Instantiate LanguageGuidedVAD model + MILRankingLoss + Adam optimizer
    3. Per epoch: forward pass, compute loss, backprop
    4. End-of-epoch: evaluate on test set, log AUROC
    5. Save best checkpoint by AUROC

Usage:
    python scripts/02_train.py
    python scripts/02_train.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vad_architecture import LanguageGuidedVAD
from utils.video_utils import load_config, set_seed
from utils.dataset import VADDataset, get_dataloaders
from utils.losses import MILRankingLoss
from utils.metrics import compute_auroc, interpolate_scores


def train(config_path: str) -> None:
    """Main training pipeline.

    Args:
        config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- DataLoaders ---
    train_loader, test_loader = get_dataloaders(config)
    print(f"[INFO] Train samples: {len(train_loader.dataset)}")
    print(f"[INFO] Test  samples: {len(test_loader.dataset)}")

    if len(train_loader.dataset) == 0:
        print("[ERROR] No training samples found.  Run 01_extract_features.py first.")
        return

    # --- Model, Loss, Optimizer ---
    model = LanguageGuidedVAD.from_config(config).to(device)
    criterion = MILRankingLoss.from_config(config)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["lr_scheduler"]["step_size"],
        gamma=config["training"]["lr_scheduler"]["gamma"],
    )

    # --- Training Loop ---
    epochs: int = config["training"]["epochs"]
    log_interval: int = config["training"]["log_interval"]
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_auroc: float = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch_idx, (visual, text, labels) in enumerate(pbar):
            visual = visual.to(device)  # (B, 32, 512)
            text = text.to(device)      # (B, 32, 512)
            labels = labels.to(device)  # (B,)

            # Forward pass
            scores = model(visual, text)  # (B, 32)

            # Split into abnormal and normal bags
            abn_mask = labels == 1
            nor_mask = labels == 0

            if abn_mask.sum() == 0 or nor_mask.sum() == 0:
                # Need both abnormal and normal samples for MIL loss
                continue

            scores_abn = scores[abn_mask]  # (B_abn, 32)
            scores_nor = scores[nor_mask]  # (B_nor, 32)

            # Compute loss
            loss_dict = criterion(scores_abn, scores_nor)
            total_loss = loss_dict["total_loss"]

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item())
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

            if (batch_idx + 1) % log_interval == 0:
                avg_loss = np.mean(epoch_losses[-log_interval:])
                print(
                    f"  [Epoch {epoch}] Batch {batch_idx+1}: "
                    f"Loss={avg_loss:.4f} "
                    f"(rank={loss_dict['ranking_loss'].item():.4f}, "
                    f"smooth={loss_dict['smoothness_loss'].item():.6f}, "
                    f"sparse={loss_dict['sparsity_loss'].item():.6f})"
                )

        scheduler.step()
        criterion.update_tau(epoch)

        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        print(
            f"Epoch {epoch}/{epochs} — Avg Loss: {avg_epoch_loss:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.2e}, "
            f"tau: {criterion.tau:.4f}"
        )

        # --- End-of-Epoch Evaluation ---
        if len(test_loader.dataset) > 0:
            auroc = evaluate_epoch(model, test_loader, device)
            print(f"  Test AUROC: {auroc:.4f}")

            if auroc > best_auroc:
                best_auroc = auroc
                ckpt_path = checkpoint_dir / "best_model.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "auroc": auroc,
                }, ckpt_path)
                print(f"  ★ New best AUROC: {best_auroc:.4f} — saved to {ckpt_path}")

    # Save final model
    final_path = checkpoint_dir / "final_model.pth"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_auroc": best_auroc,
    }, final_path)
    print(f"\n[DONE] Training complete.  Best AUROC: {best_auroc:.4f}")


@torch.no_grad()
def evaluate_epoch(
    model: LanguageGuidedVAD,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Run inference on test set and compute a simplified AUROC.

    For a quick epoch-level metric, we compute AUROC at the **segment level**
    (not frame level) by using the max segment score per video as the
    video-level anomaly score.

    Args:
        model: The trained LanguageGuidedVAD model.
        test_loader: DataLoader for the test split.
        device: Torch device.

    Returns:
        float: Segment-level AUROC over the test set.
    """
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []

    for visual, text, labels in test_loader:
        visual = visual.to(device)
        text = text.to(device)

        scores = model(visual, text)  # (B, 32)
        # Use max score per video as the video-level anomaly prediction
        max_scores = scores.max(dim=1).values  # (B,)

        all_scores.extend(max_scores.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    all_scores_np = np.array(all_scores)
    all_labels_np = np.array(all_labels)

    try:
        auroc = compute_auroc(all_scores_np, all_labels_np)
    except ValueError:
        auroc = 0.5  # Undefined if only one class present

    return auroc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Language-Guided WS-VAD model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    train(args.config)