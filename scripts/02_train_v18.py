"""V18 Fine-Tuning Script — Focal Loss Refinement from V12_s777 Checkpoint.

Key design: Load a pre-trained checkpoint and refine with Focal MIL Loss
at very low LR. Focal loss focuses gradient on hard boundary segments
that the original training couldn't resolve, while preserving the strong
learned representations.

Usage:
    python scripts/02_train_v18.py --config configs/config_v18.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vad_architecture import LanguageGuidedVAD, DynamicNormalPrototypes
from utils.video_utils import load_config, set_seed
from utils.dataset import VADDataset, get_dataloaders
from utils.losses import FocalMILLoss
from utils.metrics import compute_auroc
from utils.frame_eval import compute_frame_level_auroc


def build_class_balanced_sampler(
    dataset: VADDataset,
) -> torch.utils.data.WeightedRandomSampler:
    """Build WeightedRandomSampler for balanced category sampling.

    Args:
        dataset: VADDataset with ``samples`` attribute.

    Returns:
        WeightedRandomSampler with per-sample weights.
    """
    from collections import Counter
    categories = []
    for s in dataset.samples:
        name = s["video_name"]
        if s["label"] == 0:
            categories.append("Normal")
        else:
            prefix = ''.join(c for c in name.split('_')[0] if not c.isdigit())
            categories.append(prefix if prefix else "Anomaly")

    counts = Counter(categories)
    n_total = len(categories)
    weights = [n_total / counts[c] for c in categories]
    return torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True,
    )


def train(config_path: str) -> None:
    """V18 fine-tuning with Focal Loss from pre-trained checkpoint.

    Args:
        config_path: Path to V18 YAML configuration.
    """
    config = load_config(config_path)
    set_seed(config["seed"])
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

    # --- Load Pre-Trained Checkpoint ---
    resume_path = training_cfg.get("resume_from", None)
    if resume_path and Path(resume_path).exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        pretrained_auroc = checkpoint.get("frame_auroc", checkpoint.get("auroc", "?"))
        pretrained_epoch = checkpoint.get("epoch", "?")
        print(f"[V18] Loaded checkpoint: {resume_path}")
        print(f"[V18] Pre-trained Frame-AUROC: {pretrained_auroc}, Epoch: {pretrained_epoch}")
        print(f"[V18] Fine-tuning with Focal Loss (γ={config['loss'].get('focal_gamma', 2.0)})")
    else:
        print(f"[WARN] No checkpoint found at {resume_path} — training from scratch!")

    # --- Focal Loss ---
    criterion = FocalMILLoss.from_config(config)
    gamma = config["loss"].get("focal_gamma", 2.0)
    print(f"[INFO] Loss: FocalMILLoss (γ={gamma}, λ_mag={criterion.lambda_magnitude}, "
          f"λ_ant={criterion.lambda_antagonistic})")

    raw_dir = config["data"].get("raw_dir", "data/raw")
    annotation_file = config["data"]["annotation_file"]
    eval_cfg = config.get("evaluation", {})
    results_dir = eval_cfg.get("results_dir", "results_v18")

    # --- Optimizer (lower LR for fine-tuning) ---
    optimizer = optim.AdamW(
        list(model.parameters()) + list(prototype_bank.parameters()),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    print(f"[INFO] Optimizer: AdamW (LR={training_cfg['learning_rate']:.2e})")

    # --- Scheduler ---
    sched_cfg = training_cfg.get("lr_scheduler", {})
    t0 = sched_cfg.get("T_0", 25)
    tmult = sched_cfg.get("T_mult", 2)
    eta_min = sched_cfg.get("eta_min", 1e-7)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t0, T_mult=tmult, eta_min=eta_min,
    )
    print(f"[INFO] Scheduler: CosineAnnealingWarmRestarts (T_0={t0}, T_mult={tmult})")

    # --- Training Loop ---
    epochs = training_cfg["epochs"]
    grad_clip = training_cfg.get("gradient_clip_max_norm", 1.0)
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints_v18"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_video_auroc: float = 0.0
    best_frame_auroc: float = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} [Focal Fine-Tune]",
            leave=False,
        )

        for batch_idx, (visual, text, flow, labels) in enumerate(pbar):
            visual = visual.to(device)
            text = text.to(device)
            flow = flow.to(device)
            labels = labels.to(device)

            scores, norms, guided = model(visual, text, flow)

            abn_mask = labels == 1
            nor_mask = labels == 0
            if abn_mask.sum() == 0 or nor_mask.sum() == 0:
                continue

            scores_abn = scores[abn_mask]
            scores_nor = scores[nor_mask]
            norms_abn = norms[abn_mask]
            norms_nor = norms[nor_mask]

            loss_dict = criterion(
                scores_abn, scores_nor,
                norms_abn, norms_nor,
                epoch=epoch,
            )
            total_loss = loss_dict["total_loss"]

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_losses.append(total_loss.item())
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        scheduler.step()

        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{epochs} [Focal Fine-Tune] -- "
              f"Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

        # --- Evaluation ---
        if len(test_loader.dataset) > 0:
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
                    "version": "v18_focal",
                }, ckpt_path)
                print(f"  [BEST-VIDEO] {best_video_auroc:.4f} -- saved to {ckpt_path}")

            # Frame-level AUROC every epoch
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
                        "version": "v18_focal",
                    }, fl_path)
                    print(f"  [BEST-FRAME] {best_frame_auroc:.4f} -- saved to {fl_path}")

    # Save final
    final_path = checkpoint_dir / "final_model.pth"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "best_video_auroc": best_video_auroc,
        "best_frame_auroc": best_frame_auroc,
        "version": "v18_focal",
    }, final_path)

    print(f"\n[DONE] V18 Fine-Tuning complete.")
    print(f"       Best video-AUROC : {best_video_auroc:.4f}")
    print(f"       Best frame-AUROC : {best_frame_auroc:.4f}")
    print(f"       Checkpoints: {checkpoint_dir}/")


@torch.no_grad()
def evaluate_epoch(
    model: LanguageGuidedVAD,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Video-level AUROC evaluation.

    Args:
        model: The trained model.
        test_loader: Test DataLoader.
        device: Torch device.

    Returns:
        float: Video-level AUROC.
    """
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
    parser = argparse.ArgumentParser(
        description="V18: Focal Loss Fine-Tuning from V12_s777"
    )
    parser.add_argument("--config", type=str, default="configs/config_v18.yaml")
    args = parser.parse_args()
    train(args.config)
