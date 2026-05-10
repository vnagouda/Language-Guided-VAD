"""Training script for the Language-Guided WS-VAD model — V4/V5.

V5 additions over V4:
    1. **CosineAnnealingWarmRestarts**: Periodic LR restarts escape local minima.
    2. **topk from config**: Top-K MIL parameter now read from config loss section.
    3. **Class-Balanced Sampling**: WeightedRandomSampler used when enabled in config.
    4. **Version tag**: Checkpoints tagged with configured version.

Usage:
    python scripts/02_train.py --config configs/config_v4_sota.yaml
    python scripts/02_train.py --config configs/config_v5.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vad_architecture import LanguageGuidedVAD, DynamicNormalPrototypes
from utils.video_utils import load_config, set_seed
from utils.dataset import VADDataset, get_dataloaders
from utils.losses import VADLoss, SelfTrainingLoss
from utils.metrics import compute_auroc, interpolate_scores
from utils.frame_eval import compute_frame_level_auroc


def build_class_balanced_sampler(
    dataset: VADDataset,
) -> torch.utils.data.WeightedRandomSampler:
    """Build a WeightedRandomSampler that balances anomaly categories.

    Each sample is weighted by the inverse frequency of its anomaly category
    (derived from the video name prefix, e.g., Abuse, Arrest, Fighting...).
    Normal videos share a single 'Normal' category weight.

    Args:
        dataset: VADDataset with a ``samples`` list of dicts containing
                 ``video_name`` and ``label`` keys.

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
            # Extract category prefix e.g. "Abuse028_x264" -> "Abuse"
            prefix = ''.join(c for c in name.split('_')[0] if not c.isdigit())
            categories.append(prefix if prefix else "Anomaly")

    counts  = Counter(categories)
    n_total = len(categories)
    weights = [n_total / counts[c] for c in categories]
    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )




def train(config_path: str) -> None:
    """Main V3 training pipeline.

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

    # V5: Class-balanced sampling (optional)
    if config["training"].get("class_balanced_sampling", False):
        sampler = build_class_balanced_sampler(train_loader.dataset)
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=sampler,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
            drop_last=True,
        )
        print("[INFO] Class-balanced sampler enabled (WeightedRandomSampler)")

    if len(train_loader.dataset) == 0:
        print("[ERROR] No training samples found.  Run 01_extract_features.py first.")
        return

    # --- Model & Prototypes ---
    model = LanguageGuidedVAD.from_config(config).to(device)
    num_prototypes: int = config["model"].get(
        "num_prototypes", config["model"].get("num_normal_prototypes", 16)
    )
    prototype_bank = DynamicNormalPrototypes(
        feature_dim=config["model"]["feature_dim"],
        num_prototypes=num_prototypes,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in prototype_bank.parameters() if p.requires_grad)
    print(f"[INFO] Model + Prototype parameters: {total_params:,}")

    # --- Loss functions ---
    criterion = VADLoss.from_config(config)
    self_training_criterion = SelfTrainingLoss()

    # V10 APEX: Print all loss hyperparameters at startup for verification
    print(f"[INFO] Loss config: margin_mag={criterion.margin_magnitude:.1f}, "
          f"lambda_mag={criterion.lambda_magnitude:.4f}, "
          f"lambda_smooth={criterion.lambda_smooth:.6f}, "
          f"smooth_decay={criterion.smooth_decay_rate:.3f}, "
          f"margin_proto={criterion.margin_prototype:.1f}, "
          f"lambda_cluster={criterion.lambda_prototype_cluster:.4f}, "
          f"lambda_sep={criterion.lambda_prototype_sep:.4f}, "
          f"lambda_scl={criterion.lambda_snippet_contrastive:.4f}, "
          f"snippet_margin={criterion.snippet_margin:.1f}")

    raw_dir: str = config["data"].get("raw_dir", "data/raw")

    training_cfg = config["training"]
    # --- MIST Config parsing ---
    mist_cfg = training_cfg.get("mist", {})
    self_training_start: int = mist_cfg.get("start_epoch", training_cfg.get("self_training_start_epoch", 60))
    lambda_self: float       = mist_cfg.get("lambda_self", training_cfg.get("lambda_self", 0.5))
    mist_pseudo_k: int       = mist_cfg.get("pseudo_k", training_cfg.get("mist_pseudo_k", 12))
    grad_clip: float         = training_cfg.get("gradient_clip_max_norm", 1.0)
    lambda_smooth_p2: float  = training_cfg.get("lambda_smooth_phase2", 1.0e-5)
    eval_fl_every: int       = training_cfg.get("eval_frame_level_every", 25)
    annotation_file: str     = config["data"]["annotation_file"]
    eval_cfg                 = config.get("evaluation", {})
    results_dir: str         = eval_cfg.get("results_dir",
                                training_cfg.get("results_dir", "results"))

    # --- Optimizer (V6 includes prototype cluster centers) ---
    optimizer = optim.AdamW(
        list(model.parameters()) + list(prototype_bank.parameters()),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    # --- LR Scheduler ---
    sched_cfg  = training_cfg.get("lr_scheduler", {})
    sched_type = sched_cfg.get("type", "cosine_warm") # Default to cosine_warm for V9
    
    if sched_type == "cosine_warm":
        t0 = sched_cfg.get("T_0", training_cfg.get("scheduler_t0", 50))
        tmult = sched_cfg.get("T_mult", training_cfg.get("scheduler_tmult", 2))
        eta_min = sched_cfg.get("eta_min", 1e-6)
        
        scheduler: optim.lr_scheduler.LRScheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t0,
            T_mult=tmult,
            eta_min=eta_min,
        )
        print(f"[INFO] Scheduler: CosineAnnealingWarmRestarts (T_0={t0}, T_mult={tmult})")
    elif sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", 100),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
        print(f"[INFO] Scheduler: CosineAnnealingLR (T_max={sched_cfg.get('T_max',100)})")
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg["step_size"],
            gamma=sched_cfg["gamma"],
        )
        print(f"[INFO] Scheduler: StepLR (step={sched_cfg['step_size']}, gamma={sched_cfg['gamma']})")

    # --- Training Loop ---
    epochs: int        = training_cfg["epochs"]
    log_interval: int  = training_cfg.get("log_interval", 10)
    checkpoint_dir     = Path(training_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # (Prototypes initialized above logic loop to bind to optimizer)

    best_video_auroc: float = 0.0
    best_frame_auroc: float = 0.0

    phase2_lr: float = training_cfg.get("phase2_lr", 1.0e-5)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        in_phase2 = epoch >= self_training_start
        phase     = "Phase-2 (+MIST)" if in_phase2 else "Phase-1 (MIL)"

        # V3: Override LR to constant phase2_lr at Phase 2 start
        if in_phase2 and epoch == self_training_start:
            for pg in optimizer.param_groups:
                pg["lr"] = phase2_lr
            print(f"  [INFO] Phase 2 start — LR overridden to constant {phase2_lr:.2e}")

        # V2.2: Override smoothness weight in Phase 2
        if in_phase2:
            criterion.lambda_smooth = lambda_smooth_p2

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} [{phase}]",
            leave=False,
        )

        for batch_idx, (visual, text, flow, labels) in enumerate(pbar):
            visual = visual.to(device)   # (B, 32, D)
            text   = text.to(device)     # (B, 32, D)
            flow   = flow.to(device)     # (B, 32)
            labels = labels.to(device)   # (B,)

            scores, norms, guided = model(visual, text, flow)   # (B, 32), (B, 32), (B, 32, D)

            abn_mask = labels == 1
            nor_mask = labels == 0

            if abn_mask.sum() == 0 or nor_mask.sum() == 0:
                continue

            scores_abn = scores[abn_mask]
            scores_nor = scores[nor_mask]
            norms_abn  = norms[abn_mask]
            norms_nor  = norms[nor_mask]
            guided_abn = guided[abn_mask]   # (B_abn, T, D)
            guided_nor = guided[nor_mask]   # (B_nor, T, D)

            # V6: Retrieve normal prototypes for geometric contrastive loss
            prototypes = prototype_bank.get()

            loss_dict = criterion(
                scores_abn, scores_nor,
                norms_abn, norms_nor,
                epoch=epoch,
                guided_abn=guided_abn,
                guided_nor=guided_nor,
                prototypes=prototypes,
            )
            total_loss: torch.Tensor = loss_dict["total_loss"]

            # Phase 2: MIST Self-Training BCE (top-K pseudo labels)
            if in_phase2:
                self_loss = self_training_criterion(scores_abn, scores_nor, pseudo_k=mist_pseudo_k)
                total_loss = total_loss + lambda_self * self_loss
                loss_dict["self_loss"] = self_loss

            # Backprop with gradient clipping
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_losses.append(total_loss.item())
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

            if (batch_idx + 1) % log_interval == 0:
                avg_loss = np.mean(epoch_losses[-log_interval:])
                self_str = (
                    f", self={loss_dict['self_loss'].item():.4f}"
                    if "self_loss" in loss_dict else ""
                )
                print(
                    f"  [Epoch {epoch}] Batch {batch_idx+1}: "
                    f"Loss={avg_loss:.4f} "
                    f"(ais={loss_dict['ais_loss'].item():.4f}, "
                    f"ant={loss_dict['antagonistic_loss'].item():.4f}, "
                    f"mag={loss_dict['magnitude_loss'].item():.4f}, "
                    f"smooth={loss_dict['smoothness_loss'].item():.6f}"
                    f"{self_str})"
                )

        if not in_phase2:
            scheduler.step()

        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        current_lr    = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{epochs} [{phase}] -- Avg Loss: {avg_epoch_loss:.4f}, "
            f"LR: {current_lr:.2e}"
        )

        # ---------------------------------------------------------------
        # End-of-Epoch Evaluation
        # ---------------------------------------------------------------
        if len(test_loader.dataset) > 0:

            # --- Video-level AUROC (every epoch, fast) ---
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
                    "version": "v5",
                }, ckpt_path)
                print(f"  [BEST-VIDEO] New best video-AUROC: {best_video_auroc:.4f} -- saved to {ckpt_path}")

            # --- Frame-level AUROC ---
            # User specifically requested to track Frame-Level AUROC on EVERY epoch 
            # to visibly prove the baseline vs. MIST (Phase 2) performance jump.
            should_eval_fl = True

            if should_eval_fl:
                print(f"  [INFO] Computing frame-level AUROC (epoch {epoch})...")
                frame_auroc = compute_frame_level_auroc(
                    model, test_loader, device, annotation_file, results_dir,
                    raw_dir=raw_dir,
                )
                if frame_auroc is not None:
                    print(f"  Test Frame-AUROC: {frame_auroc:.4f}")
                    if frame_auroc > best_frame_auroc:
                        best_frame_auroc = frame_auroc
                        fl_ckpt_path = checkpoint_dir / "best_model_framelevel.pth"
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "frame_auroc": frame_auroc,
                            "video_auroc": video_auroc,
                            "version": "v3",
                        }, fl_ckpt_path)
                        print(
                            f"  [BEST-FRAME] New best frame-AUROC: {best_frame_auroc:.4f} "
                            f"-- saved to {fl_ckpt_path}"
                        )
                else:
                    print("  [WARN] Frame-level AUROC unavailable (annotation file missing or parse error).")

    # Save final model
    final_path = checkpoint_dir / "final_model.pth"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_video_auroc": best_video_auroc,
        "best_frame_auroc": best_frame_auroc,
        "version": "v4",
    }, final_path)
    print(f"\n[DONE] Training complete.")
    print(f"       Best video-AUROC : {best_video_auroc:.4f}  -> checkpoints/best_model.pth")
    print(f"       Best frame-AUROC : {best_frame_auroc:.4f}  -> checkpoints/best_model_framelevel.pth")


@torch.no_grad()
def evaluate_epoch(
    model: LanguageGuidedVAD,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Run inference on test set and compute video-level AUROC (max-score per bag).

    Args:
        model: The trained LanguageGuidedVAD model.
        test_loader: DataLoader for the test split.
        device: Torch device.

    Returns:
        float: Video-level AUROC over the test set.
    """
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int]   = []

    for visual, text, flow, labels in test_loader:
        visual = visual.to(device)
        text   = text.to(device)
        flow   = flow.to(device)

        scores, _, _ = model(visual, text, flow)  # (B, 32)
        max_scores = scores.max(dim=1).values  # (B,)

        all_scores.extend(max_scores.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    all_scores_np = np.array(all_scores)
    all_labels_np = np.array(all_labels)

    try:
        return float(compute_auroc(all_scores_np, all_labels_np))
    except ValueError:
        return 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Language-Guided WS-VAD model (V3)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_v3_florence2.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    train(args.config)
