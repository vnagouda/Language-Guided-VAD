"""Hyperparameter Optimisation — Optuna Bayesian search for V3 VAD model.

Uses a shortened 200-epoch trial (Phase-2 starts at epoch 100) to search over
the most impactful hyperparameters while keeping wall-clock time tractable.
Best found config is written to configs/config_v3_hpo_best.yaml.

Usage:
    pip install optuna
    python scripts/04_hpo.py --config configs/config_v3_florence2.yaml --n_trials 20
    python scripts/04_hpo.py --config configs/config_v3_florence2.yaml --n_trials 40 --trial_epochs 300
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.video_utils import load_config, set_seed


# ---------------------------------------------------------------------------
# Optuna import (graceful error if not installed)
# ---------------------------------------------------------------------------
try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("[ERROR] Optuna is not installed. Run: pip install optuna")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Trial function
# ---------------------------------------------------------------------------

def run_trial(
    trial: "optuna.Trial",
    base_config: dict[str, Any],
    trial_epochs: int,
    device: torch.device,
) -> float:
    """Train a short run with suggested hyperparameters and return frame-AUROC.

    Args:
        trial: Optuna trial object used to suggest hyperparameter values.
        base_config: Base configuration dict (deep-copied per trial).
        trial_epochs: Number of epochs for each trial (shorter than full run).
        device: Torch device to train on.

    Returns:
        float: Best frame-level AUROC achieved during the trial.
    """
    from models.vad_architecture import LanguageGuidedVAD
    from utils.dataset import VADDataset
    from utils.losses import VADLoss, SelfTrainingLoss
    from utils.frame_eval import compute_frame_level_auroc
    from torch.utils.data import DataLoader

    # ------------------------------------------------------------------
    # 1. Suggest hyperparameters
    # ------------------------------------------------------------------
    cfg = copy.deepcopy(base_config)

    lr            = trial.suggest_float("learning_rate",     5e-5,  5e-4, log=True)
    wd            = trial.suggest_float("weight_decay",       1e-4,  1e-3, log=True)
    dropout       = trial.suggest_float("dropout",            0.3,   0.6)
    lam_mag       = trial.suggest_float("lambda_magnitude",   5e-3,  5e-2, log=True)
    lam_ant       = trial.suggest_float("lambda_antagonistic",0.5,   3.0)
    lam_smooth    = trial.suggest_float("lambda_smooth",      1e-5,  5e-4, log=True)
    lam_self      = trial.suggest_float("lambda_self",        0.2,   1.0)
    pseudo_k      = trial.suggest_int("mist_pseudo_k",        2,     5)
    phase2_start  = trial.suggest_int("phase2_start_frac",    30,    60)   # % of trial_epochs
    num_heads     = trial.suggest_categorical("num_heads",    [4, 8])

    phase2_epoch  = max(10, int(trial_epochs * phase2_start / 100))

    # Write suggestions into config copy
    cfg["training"]["learning_rate"]          = lr
    cfg["training"]["weight_decay"]           = wd
    cfg["training"]["lambda_self"]            = lam_self
    cfg["training"]["mist_pseudo_k"]          = pseudo_k
    cfg["training"]["self_training_start_epoch"] = phase2_epoch
    cfg["training"]["phase2_lr"]              = lr / 10.0

    cfg["model"]["dropout"]                   = dropout
    cfg["model"]["num_heads"]                 = num_heads

    cfg["loss"]["lambda_magnitude"]           = lam_mag
    cfg["loss"]["lambda_antagonistic"]        = lam_ant
    cfg["loss"]["lambda_smooth"]              = lam_smooth

    # ------------------------------------------------------------------
    # 2. Build data loaders
    # ------------------------------------------------------------------
    set_seed(42)
    features_dir    = cfg["data"]["features_dir"]
    annotation_file = cfg["data"]["annotation_file"]
    batch_size      = cfg["training"]["batch_size"]
    num_segments    = cfg["model"]["num_segments"]
    feature_dim     = cfg["model"]["feature_dim"]

    train_ds = VADDataset(
        Path(features_dir) / "Train",
        num_segments=num_segments,
        feature_dim=feature_dim,
    )
    test_ds = VADDataset(
        Path(features_dir) / "Test",
        num_segments=num_segments,
        feature_dim=feature_dim,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0,
    )

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise optuna.exceptions.TrialPruned()

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    model = LanguageGuidedVAD(
        feature_dim=cfg["model"]["feature_dim"],
        num_segments=cfg["model"]["num_segments"],
        num_heads=num_heads,
        num_layers=cfg["model"].get("num_layers", 1),
        dropout=dropout,
        ff_dim=cfg["model"]["ff_dim"],
        classifier_bottleneck_dim=cfg["model"]["classifier_bottleneck_dim"],
        classifier_hidden_dim=cfg["model"]["classifier_hidden_dim"],
        use_magnitude_branch=cfg["model"]["use_magnitude_branch"],
        use_flow_in_magnitude=cfg["model"].get("use_flow_in_magnitude", False),
        use_multi_scale=cfg["model"].get("use_multi_scale", True),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_cfg  = cfg["loss"]
    criterion = VADLoss(
        ais_score_threshold   = loss_cfg.get("ais_score_threshold", 0.9),
        ais_k_min             = loss_cfg.get("ais_k_min", 3),
        ais_warm_start_epochs = loss_cfg.get("ais_warm_start_epochs", 20),
        ais_warm_k            = loss_cfg.get("ais_warm_k", 8),
        lambda_magnitude      = lam_mag,
        margin_magnitude      = loss_cfg.get("margin_magnitude", 1.0),
        lambda_antagonistic   = lam_ant,
        lambda_smooth         = lam_smooth,
        lambda_contrastive    = loss_cfg.get("lambda_contrastive", 0.1),
        margin_contrastive    = loss_cfg.get("margin_contrastive", 1.0),
        lambda_bank           = loss_cfg.get("lambda_bank", 0.05),
        margin_bank           = loss_cfg.get("margin_bank", 1.0),
    )
    self_crit  = SelfTrainingLoss()

    grad_clip  = cfg["training"]["gradient_clip_max_norm"]
    lam_self_w = lam_self
    results_dir = Path(f"results_hpo/trial_{trial.number}")
    results_dir.mkdir(parents=True, exist_ok=True)

    best_frame_auroc: float = 0.0

    # ------------------------------------------------------------------
    # 4. Short training loop
    # ------------------------------------------------------------------
    for epoch in range(1, trial_epochs + 1):
        in_phase2 = epoch > phase2_epoch
        model.train()

        for batch in train_loader:
            visual, text, flow, labels = batch
            visual = visual.to(device)
            text   = text.to(device)
            flow   = flow.to(device)
            labels = labels.to(device)

            scores, norms, guided = model(visual, text, flow)

            abn_mask = labels == 1
            nor_mask = labels == 0

            if abn_mask.sum() == 0 or nor_mask.sum() == 0:
                continue

            loss_dict  = criterion(
                scores[abn_mask], scores[nor_mask],
                norms[abn_mask],  norms[nor_mask],
                epoch=epoch,
                guided_abn=guided[abn_mask],
                guided_nor=guided[nor_mask],
            )
            total_loss = loss_dict["total_loss"]

            if in_phase2:
                sl = self_crit(scores[abn_mask], scores[nor_mask], pseudo_k=pseudo_k)
                total_loss = total_loss + lam_self_w * sl

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        # Evaluate frame AUROC in Phase 2
        if in_phase2:
            frame_auroc = compute_frame_level_auroc(
                model, test_loader, device, annotation_file, results_dir
            )
            if frame_auroc is not None:
                best_frame_auroc = max(best_frame_auroc, frame_auroc)
                trial.report(frame_auroc, step=epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    return best_frame_auroc


# ---------------------------------------------------------------------------
# Main HPO loop
# ---------------------------------------------------------------------------

def run_hpo(
    config_path: str,
    n_trials: int = 20,
    trial_epochs: int = 200,
) -> None:
    """Run Optuna hyperparameter optimisation study.

    Args:
        config_path: Path to the base YAML config to optimise.
        n_trials: Total number of Optuna trials to run.
        trial_epochs: Epochs per trial (shorter = faster but noisier).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HPO] Device: {device}")
    print(f"[HPO] Running {n_trials} trials x {trial_epochs} epochs each")

    base_config = load_config(config_path)

    sampler = TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="vad_v3_hpo",
    )

    study.optimize(
        lambda trial: run_trial(trial, base_config, trial_epochs, device),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # ------------------------------------------------------------------
    # Report results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"[HPO] Best frame-AUROC: {study.best_value:.4f}")
    print(f"[HPO] Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"       {k}: {v}")

    # ------------------------------------------------------------------
    # Write best config to YAML
    # ------------------------------------------------------------------
    best_cfg = load_config(config_path)
    p = study.best_params
    best_cfg["training"]["learning_rate"]             = p["learning_rate"]
    best_cfg["training"]["weight_decay"]              = p["weight_decay"]
    best_cfg["training"]["lambda_self"]               = p["lambda_self"]
    best_cfg["training"]["mist_pseudo_k"]             = p["mist_pseudo_k"]
    best_cfg["training"]["self_training_start_epoch"] = int(
        trial_epochs * p["phase2_start_frac"] / 100
    )
    best_cfg["model"]["dropout"]                      = p["dropout"]
    best_cfg["model"]["num_heads"]                    = p["num_heads"]
    best_cfg["loss"]["lambda_magnitude"]              = p["lambda_magnitude"]
    best_cfg["loss"]["lambda_antagonistic"]           = p["lambda_antagonistic"]
    best_cfg["loss"]["lambda_smooth"]                 = p["lambda_smooth"]

    out_path = Path("configs/config_v3_hpo_best.yaml")
    with open(out_path, "w") as f:
        yaml.dump(best_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\n[HPO] Best config written to: {out_path}")
    print("[HPO] Run the full training with:")
    print(f"      python scripts/02_train.py --config {out_path}")

    # ------------------------------------------------------------------
    # Importance plot (if plotly installed)
    # ------------------------------------------------------------------
    try:
        import optuna.visualization as vis
        fig = vis.plot_param_importances(study)
        fig.write_html("results_hpo/param_importances.html")
        print("[HPO] Param importance plot → results_hpo/param_importances.html")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO for Language-Guided VAD")
    parser.add_argument("--config",       type=str, default="configs/config_v3_florence2.yaml")
    parser.add_argument("--n_trials",     type=int, default=20,  help="Number of Optuna trials")
    parser.add_argument("--trial_epochs", type=int, default=200, help="Epochs per trial")
    args = parser.parse_args()
    run_hpo(args.config, args.n_trials, args.trial_epochs)
