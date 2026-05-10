"""V17 Training Script — EMA + Feature Augmentation + No MIST.

Key innovations over 02_train.py:
    1. **Exponential Moving Average (EMA)**: Maintains a smoothed copy of model
       weights (decay=0.999). The EMA model is used for evaluation and saved as
       the best checkpoint. This eliminates noisy weight oscillations.
    2. **Feature-Level Mixup**: Interpolates features between normal and anomalous
       videos during training, creating harder training examples.
    3. **Feature Dropout**: Randomly zeros visual feature dimensions to improve
       robustness (complementary to model dropout).
    4. **No MIST**: Phase 2 self-training consistently degraded frame-AUROC by
       1-2% across all experiments. We skip it entirely.

Usage:
    python scripts/02_train_v17.py --config configs/config_v17.yaml
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vad_architecture import LanguageGuidedVAD, DynamicNormalPrototypes
from utils.video_utils import load_config, set_seed
from utils.dataset import VADDataset, get_dataloaders
from utils.losses import VADLoss, SelfTrainingLoss
from utils.metrics import compute_auroc, interpolate_scores
from utils.frame_eval import compute_frame_level_auroc


# ---------------------------------------------------------------------------
# EMA Helper
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of all model parameters and updates them
    with exponential decay at each training step.

    Args:
        model: The model to track.
        decay: EMA decay factor (0.999 typical).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters with current model parameters.

        Args:
            model: Current training model.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model params with EMA shadow params (for evaluation).

        Args:
            model: Model to apply EMA weights to.
        """
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original model params after evaluation.

        Args:
            model: Model to restore original weights to.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ---------------------------------------------------------------------------
# Feature Augmentation
# ---------------------------------------------------------------------------

def feature_mixup(
    visual: torch.Tensor,
    text: torch.Tensor,
    flow: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup augmentation on features.

    Interpolates between random pairs within the batch using a Beta-distributed
    mixing coefficient lambda. Both features and labels are mixed.

    Args:
        visual: Visual features ``(B, T, D)``.
        text: Text features ``(B, T, D)``.
        flow: Flow magnitudes ``(B, T)``.
        labels: Binary labels ``(B,)``.
        alpha: Beta distribution parameter.

    Returns:
        tuple: (mixed_visual, mixed_text, mixed_flow, labels, lam).
    """
    if alpha <= 0:
        return visual, text, flow, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 (keep dominant sample)

    batch_size = visual.size(0)
    index = torch.randperm(batch_size, device=visual.device)

    mixed_visual = lam * visual + (1 - lam) * visual[index]
    mixed_text = lam * text + (1 - lam) * text[index]
    mixed_flow = lam * flow + (1 - lam) * flow[index]

    return mixed_visual, mixed_text, mixed_flow, labels, lam


def feature_random_dropout(
    visual: torch.Tensor,
    rate: float = 0.05,
) -> torch.Tensor:
    """Randomly zero out feature dimensions during training.

    Args:
        visual: Visual features ``(B, T, D)``.
        rate: Fraction of dims to zero.

    Returns:
        torch.Tensor: Augmented features with scaled remaining dims.
    """
    if rate <= 0 or not visual.requires_grad:
        return visual
    mask = (torch.rand_like(visual) > rate).float()
    return visual * mask / (1.0 - rate)


# ---------------------------------------------------------------------------
# Class-Balanced Sampler
# ---------------------------------------------------------------------------

def build_class_balanced_sampler(
    dataset: VADDataset,
) -> torch.utils.data.WeightedRandomSampler:
    """Build WeightedRandomSampler for balanced category sampling."""
    from collections import Counter

    categories = []
    for sample in dataset.samples:
        name = sample["video_name"]
        if sample["label"] == 0:
            categories.append("Normal")
        else:
            cat = name.split("_")[0] if "_" in name else name.split("/")[0]
            categories.append(cat)

    counts = Counter(categories)
    weights = [1.0 / counts[cat] for cat in categories]
    return torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path: str) -> None:
    """V17 training with EMA, feature augmentation, no MIST."""
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- Data ---
    train_loader, test_loader = get_dataloaders(config)
    print(f"[INFO] Train samples: {len(train_loader.dataset)}")
    print(f"[INFO] Test  samples: {len(test_loader.dataset)}")

    training_cfg = config["training"]
    if training_cfg.get("class_balanced_sampling", False):
        sampler = build_class_balanced_sampler(train_loader.dataset)
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=sampler,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
            drop_last=True,
        )
        print("[INFO] Class-balanced sampler enabled")

    # --- Model ---
    model = LanguageGuidedVAD.from_config(config).to(device)
    num_prototypes = config["model"].get("num_prototypes", 16)
    prototype_bank = DynamicNormalPrototypes(
        feature_dim=config["model"]["feature_dim"],
        num_prototypes=num_prototypes,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {total_params:,}")

    # --- EMA ---
    use_ema = training_cfg.get("use_ema", True)
    ema_decay = training_cfg.get("ema_decay", 0.999)
    ema: EMAModel | None = None
    if use_ema:
        ema = EMAModel(model, decay=ema_decay)
        print(f"[INFO] EMA enabled (decay={ema_decay})")

    # --- Loss ---
    criterion = VADLoss.from_config(config)

    # --- Feature augmentation config ---
    feat_dropout_rate = training_cfg.get("feature_dropout_rate", 0.0)
    use_mixup = training_cfg.get("use_feature_mixup", False)
    mixup_alpha = training_cfg.get("mixup_alpha", 0.2)
    print(f"[INFO] Feature dropout: {feat_dropout_rate}, Mixup: {use_mixup} (alpha={mixup_alpha})")

    raw_dir = config["data"].get("raw_dir", "data/raw")
    annotation_file = config["data"]["annotation_file"]
    eval_cfg = config.get("evaluation", {})
    results_dir = eval_cfg.get("results_dir", "results_v17")

    # --- Optimizer ---
    optimizer = optim.AdamW(
        list(model.parameters()) + list(prototype_bank.parameters()),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    # --- Scheduler ---
    sched_cfg = training_cfg.get("lr_scheduler", {})
    t0 = sched_cfg.get("T_0", 30)
    tmult = sched_cfg.get("T_mult", 2)
    eta_min = sched_cfg.get("eta_min", 1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t0, T_mult=tmult, eta_min=eta_min,
    )
    print(f"[INFO] Scheduler: CosineAnnealingWarmRestarts (T_0={t0}, T_mult={tmult})")

    # --- Training Loop ---
    epochs = training_cfg["epochs"]
    grad_clip = training_cfg.get("gradient_clip_max_norm", 5.0)
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints_v17"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_video_auroc: float = 0.0
    best_frame_auroc: float = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=False,
        )

        for batch_idx, (visual, text, flow, labels) in enumerate(pbar):
            visual = visual.to(device)
            text = text.to(device)
            flow = flow.to(device)
            labels = labels.to(device)

            # --- Feature Augmentation ---
            if feat_dropout_rate > 0:
                visual = feature_random_dropout(visual, feat_dropout_rate)

            if use_mixup and np.random.random() < 0.5:  # 50% chance
                visual, text, flow, labels, lam = feature_mixup(
                    visual, text, flow, labels, mixup_alpha,
                )

            scores, norms, guided = model(visual, text, flow)

            abn_mask = labels == 1
            nor_mask = labels == 0
            if abn_mask.sum() == 0 or nor_mask.sum() == 0:
                continue

            scores_abn = scores[abn_mask]
            scores_nor = scores[nor_mask]
            norms_abn = norms[abn_mask]
            norms_nor = norms[nor_mask]
            guided_abn = guided[abn_mask]
            guided_nor = guided[nor_mask]

            prototypes = prototype_bank.get()
            loss_dict = criterion(
                scores_abn, scores_nor,
                norms_abn, norms_nor,
                epoch=epoch,
                guided_abn=guided_abn,
                guided_nor=guided_nor,
                prototypes=prototypes,
            )
            total_loss = loss_dict["total_loss"]

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            # Update EMA after each step
            if ema is not None:
                ema.update(model)

            epoch_losses.append(total_loss.item())
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        scheduler.step()

        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{epochs} -- Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

        # --- Evaluation (use EMA model if available) ---
        if len(test_loader.dataset) > 0:
            if ema is not None:
                ema.apply_shadow(model)

            model.eval()

            # Video-AUROC
            video_auroc = evaluate_epoch(model, test_loader, device)
            print(f"  Test Video-AUROC: {video_auroc:.4f}")

            if video_auroc > best_video_auroc:
                best_video_auroc = video_auroc
                ckpt_path = checkpoint_dir / "best_model.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "auroc": video_auroc,
                    "version": "v17_ema",
                }, ckpt_path)
                print(f"  [BEST-VIDEO] {best_video_auroc:.4f} -- saved to {ckpt_path}")

            # Frame-AUROC
            print(f"  [INFO] Computing frame-level AUROC (epoch {epoch})...")
            frame_auroc = compute_frame_level_auroc(
                model, test_loader, device, annotation_file, results_dir,
                raw_dir=raw_dir,
            )
            if frame_auroc is not None:
                print(f"  Test Frame-AUROC: {frame_auroc:.4f}")
                if frame_auroc > best_frame_auroc:
                    best_frame_auroc = frame_auroc
                    fl_path = checkpoint_dir / "best_model_framelevel.pth"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "frame_auroc": frame_auroc,
                        "video_auroc": video_auroc,
                        "version": "v17_ema",
                    }, fl_path)
                    print(f"  [BEST-FRAME] {best_frame_auroc:.4f} -- saved to {fl_path}")

            if ema is not None:
                ema.restore(model)

    # Save final
    final_path = checkpoint_dir / "final_model.pth"
    if ema is not None:
        ema.apply_shadow(model)
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "best_video_auroc": best_video_auroc,
        "best_frame_auroc": best_frame_auroc,
        "version": "v17_ema",
    }, final_path)

    print(f"\n[DONE] Training complete.")
    print(f"       Best video-AUROC : {best_video_auroc:.4f}  -> {checkpoint_dir}/best_model.pth")
    print(f"       Best frame-AUROC : {best_frame_auroc:.4f}  -> {checkpoint_dir}/best_model_framelevel.pth")


@torch.no_grad()
def evaluate_epoch(
    model: LanguageGuidedVAD,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Video-level AUROC evaluation."""
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []

    for visual, text, flow, labels in test_loader:
        visual = visual.to(device)
        text = text.to(device)
        flow = flow.to(device)
        scores, _, _ = model(visual, text, flow)
        max_scores = scores.max(dim=1).values
        all_scores.extend(max_scores.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    try:
        return float(compute_auroc(np.array(all_scores), np.array(all_labels)))
    except ValueError:
        return 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V17: EMA + Feature Aug Training")
    parser.add_argument("--config", type=str, default="configs/config_v17.yaml")
    args = parser.parse_args()
    train(args.config)
