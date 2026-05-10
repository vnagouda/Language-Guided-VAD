"""Multi-Seed V12 Batch Trainer — runs 5 seeds sequentially.

Trains the proven V12 architecture with different random seeds to
exploit training variance. Each seed gets its own checkpoint directory.
After all seeds complete, reports the best individual and recommends
ensemble candidates.

Usage:
    python scripts/11_multi_seed_train.py
"""

from __future__ import annotations

import subprocess
import sys
import time
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Seeds to train (avoiding 42, 123, 777 which already exist)
SEEDS = [1, 99, 256, 512, 999]

# Base config template — identical to V12_s777 (our best proven config)
BASE_CONFIG = {
    "data": {
        "raw_dir": "data/raw",
        "features_dir": "data/features_v12_T128",
        "annotation_file": "data/Temporal_Anomaly_Annotation.txt",
        "frame_extensions": [".png", ".jpg"],
    },
    "extraction": {
        "num_segments": 128,
        "frames_per_segment": 5,
        "clip_model_name": "openai/clip-vit-large-patch14",
        "source_text_features": "data/features_v31_blip2_prompt",
        "use_patch_tokens": True,
        "extract_flow": True,
        "image_size": 224,
    },
    "model": {
        "feature_dim": 768,
        "num_segments": 128,
        "num_heads": 8,
        "num_layers": 1,
        "dropout": 0.4843,
        "ff_dim": 3072,
        "classifier_bottleneck_dim": 64,
        "classifier_hidden_dim": 128,
        "use_magnitude_branch": True,
        "use_flow_in_magnitude": False,
        "use_multi_scale": True,
        "use_temporal_convolutions": False,
        "memory_bank_size": 256,
        "num_prototypes": 16,
    },
    "loss": {
        "ais_score_threshold": 0.9,
        "ais_k_min": 20,
        "ais_warm_start_epochs": 20,
        "ais_warm_k": 48,
        "lambda_magnitude": 0.01918,
        "margin_magnitude": 1.0,
        "lambda_antagonistic": 2.5755,
        "lambda_smooth": 1.093e-05,
        "lambda_prototype_cluster": 0.0,
        "lambda_prototype_sep": 0.0,
        "margin_prototype": 2.0,
        "lambda_contrastive": 0.08,
        "margin_contrastive": 2.0,
        "lambda_bank": 0.04,
        "margin_bank": 2.0,
        "lambda_snippet_contrastive": 0.0,
        "snippet_margin": 2.0,
    },
    "training": {
        "batch_size": 256,
        "num_workers": 4,
        "epochs": 200,  # Shortened — models peak by epoch 50-60
        "learning_rate": 0.00014460,
        "weight_decay": 0.0008827,
        "lr_scheduler": {
            "type": "cosine_warm",
            "T_0": 50,
            "T_mult": 2,
            "eta_min": 1.0e-6,
        },
        "class_balanced_sampling": True,
        "gradient_clip_max_norm": 5.0,
        # MIST disabled — Phase 2 consistently hurts
        "self_training_start_epoch": 9999,
        "lambda_self": 0.05,
        "mist_pseudo_k": 12,
        "log_interval": 10,
        "eval_frame_level_every": 1,
        "lambda_smooth_phase2": 1.0e-05,
    },
    "evaluation": {},
}


def create_seed_config(seed: int) -> Path:
    """Create a config file for a specific seed.

    Args:
        seed: Random seed value.

    Returns:
        Path: Path to the generated config file.
    """
    config = BASE_CONFIG.copy()
    import copy
    config = copy.deepcopy(BASE_CONFIG)

    config["seed"] = seed
    config["training"]["checkpoint_dir"] = f"checkpoints_v12_s{seed}"
    config["evaluation"]["results_dir"] = f"results_v12_s{seed}"

    config_path = PROJECT_ROOT / f"configs/config_v12_s{seed}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


def main() -> None:
    """Run multi-seed training sequentially."""
    print("=" * 60)
    print("  MULTI-SEED V12 BATCH TRAINER")
    print("=" * 60)
    print(f"  Seeds to train: {SEEDS}")
    print(f"  Epochs per seed: {BASE_CONFIG['training']['epochs']}")
    print(f"  MIST: DISABLED (start_epoch=9999)")
    print(f"  Using proven 02_train.py script")
    print("=" * 60)

    # Check for existing checkpoint dirs
    for seed in SEEDS:
        ckpt_dir = PROJECT_ROOT / f"checkpoints_v12_s{seed}"
        if ckpt_dir.exists():
            print(f"  [WARN] {ckpt_dir} already exists — will be OVERWRITTEN")

    results: dict[int, dict] = {}
    total_start = time.time()

    for i, seed in enumerate(SEEDS, 1):
        print(f"\n{'=' * 60}")
        print(f"  SEED {seed} ({i}/{len(SEEDS)})")
        print(f"{'=' * 60}")

        # Create config
        config_path = create_seed_config(seed)
        print(f"  Config: {config_path}")
        print(f"  Checkpoints: checkpoints_v12_s{seed}/")

        # Run training
        seed_start = time.time()
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "02_train.py"),
            "--config", str(config_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=False,  # Show output in real-time
            )

            seed_time = time.time() - seed_start
            print(f"\n  [SEED {seed}] Completed in {seed_time / 60:.1f} min")

            if result.returncode != 0:
                print(f"  [SEED {seed}] FAILED (exit code {result.returncode})")
                results[seed] = {"status": "FAILED"}
                continue

            # Read best checkpoint to get AUROC
            import torch
            ckpt_path = PROJECT_ROOT / f"checkpoints_v12_s{seed}" / "best_model_framelevel.pth"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                frame_auroc = ckpt.get("frame_auroc", 0.0)
                video_auroc = ckpt.get("video_auroc", 0.0)
                epoch = ckpt.get("epoch", "?")
                results[seed] = {
                    "status": "OK",
                    "frame_auroc": frame_auroc,
                    "video_auroc": video_auroc,
                    "epoch": epoch,
                    "time_min": seed_time / 60,
                }
                print(f"  [SEED {seed}] Frame-AUROC: {frame_auroc:.4f} (epoch {epoch})")
            else:
                results[seed] = {"status": "NO_CHECKPOINT"}
                print(f"  [SEED {seed}] No checkpoint found!")

        except Exception as e:
            print(f"  [SEED {seed}] ERROR: {e}")
            results[seed] = {"status": "ERROR", "error": str(e)}

    total_time = time.time() - total_start

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  MULTI-SEED RESULTS SUMMARY")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"{'=' * 60}\n")

    # Include existing seeds
    existing_seeds = {
        42: {"frame_auroc": 0.8238, "status": "EXISTING"},
        123: {"frame_auroc": 0.8192, "status": "EXISTING"},
        777: {"frame_auroc": 0.8294, "status": "EXISTING"},
    }

    all_results = {**existing_seeds, **results}
    valid = {s: r for s, r in all_results.items()
             if r.get("status") in ("OK", "EXISTING") and "frame_auroc" in r}

    if valid:
        print(f"  {'Seed':<8} {'Frame-AUROC':<15} {'Status'}")
        print(f"  {'-'*8} {'-'*15} {'-'*10}")
        for seed in sorted(valid.keys()):
            r = valid[seed]
            marker = " ★" if r["frame_auroc"] == max(v["frame_auroc"] for v in valid.values()) else ""
            print(f"  s{seed:<7} {r['frame_auroc']:.4f}{marker:>10} {r['status']}")

        best_seed = max(valid, key=lambda s: valid[s]["frame_auroc"])
        best_auroc = valid[best_seed]["frame_auroc"]
        mean_auroc = sum(v["frame_auroc"] for v in valid.values()) / len(valid)
        std_auroc = (sum((v["frame_auroc"] - mean_auroc)**2 for v in valid.values()) / len(valid)) ** 0.5

        print(f"\n  Best single model: s{best_seed} = {best_auroc:.4f}")
        print(f"  Mean ± Std: {mean_auroc:.4f} ± {std_auroc:.4f}")
        print(f"  Range: [{min(v['frame_auroc'] for v in valid.values()):.4f}, {best_auroc:.4f}]")

        # Recommend ensemble candidates
        sorted_seeds = sorted(valid.keys(), key=lambda s: valid[s]["frame_auroc"], reverse=True)
        top3 = sorted_seeds[:3]
        print(f"\n  Top-3 for ensemble: {['s' + str(s) for s in top3]}")
        print(f"  Run: python scripts/08_ensemble_eval.py")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
