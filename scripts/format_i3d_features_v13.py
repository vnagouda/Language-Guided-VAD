"""Format 1024-dim I3D features (10-crop, category-subdirectory layout) → T=128 .pt tensors.

The raw I3D features are stored as:
    data/{Train,Test}/RGB/{Category}/{VideoName}.npy        (crop 0)
    data/{Train,Test}/RGB/{Category}/{VideoName}__1.npy     (crop 1)
    ...
    data/{Train,Test}/RGB/{Category}/{VideoName}__9.npy     (crop 9)

Each .npy file has shape (T_variable, 1024).

This script:
    1. Loads all 10 crop files per video.
    2. Averages across the 10 crops → (T_variable, 1024).
    3. Applies adaptive temporal pooling to T=128 → (128, 1024).
    4. Saves as {VideoName}_i3d.pt to data/features_v13_i3d/{Train,Test}/.

Usage:
    python scripts/format_i3d_features_v13.py
"""

from __future__ import annotations

import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


TARGET_T: int = 128
FEATURE_DIM: int = 1024


def discover_videos(rgb_dir: Path) -> dict[str, list[Path]]:
    """Discover all videos and their crop files from category subdirectories.

    Args:
        rgb_dir: Path to the RGB directory containing category subdirectories.

    Returns:
        dict: Mapping of video_name → list of crop file paths (up to 10).
    """
    videos: dict[str, list[Path]] = defaultdict(list)

    for cat_dir in sorted(rgb_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        for npy_file in sorted(cat_dir.glob("*.npy")):
            # Parse video name: "Abuse001_x264.npy" → "Abuse001_x264"
            # Parse crop index: "Abuse001_x264__3.npy" → base "Abuse001_x264"
            stem = npy_file.stem
            if "__" in stem:
                base_name = stem.split("__")[0]
            else:
                base_name = stem
            videos[base_name].append(npy_file)

    return dict(videos)


def process_video(crop_paths: list[Path]) -> torch.Tensor:
    """Average 10 crops and pool to T=128.

    Args:
        crop_paths: List of paths to the crop .npy files for one video.

    Returns:
        torch.Tensor: Pooled features of shape (128, 1024).
    """
    crops = [np.load(str(p)) for p in crop_paths]

    # All crops should have the same temporal length
    min_t = min(c.shape[0] for c in crops)
    crops_trimmed = [c[:min_t] for c in crops]

    # Average across crops: (T_var, 1024)
    averaged = np.mean(np.stack(crops_trimmed, axis=0), axis=0)

    # Convert to tensor for adaptive pooling
    feat = torch.from_numpy(averaged).float()  # (T_var, 1024)

    if feat.shape[0] == TARGET_T:
        return feat

    # Adaptive temporal pooling: (T_var, 1024) → (128, 1024)
    # F.adaptive_avg_pool1d expects (B, C, T)
    feat_perm = feat.T.unsqueeze(0)  # (1, 1024, T_var)
    pooled = F.adaptive_avg_pool1d(feat_perm, TARGET_T)  # (1, 1024, 128)
    return pooled.squeeze(0).T  # (128, 1024)


def format_split(
    rgb_dir: Path,
    output_dir: Path,
    split_name: str,
) -> None:
    """Format all videos in a split.

    Args:
        rgb_dir: Path to data/{split}/RGB/.
        output_dir: Path to save formatted .pt files.
        split_name: Name for progress bar.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(rgb_dir)
    print(f"\n--- Formatting {split_name} Features ---")
    print(f"Found {len(videos)} unique videos in {rgb_dir}")

    for video_name in tqdm(sorted(videos.keys()), desc=f"Converting {split_name}"):
        crop_paths = videos[video_name]
        out_path = output_dir / f"{video_name}_i3d.pt"

        if out_path.exists():
            continue

        try:
            feat = process_video(crop_paths)
            assert feat.shape == (TARGET_T, FEATURE_DIM), (
                f"{video_name}: expected ({TARGET_T}, {FEATURE_DIM}), "
                f"got {feat.shape}"
            )
            torch.save(feat, out_path)
        except Exception as exc:
            print(f"[WARN] Skipping {video_name}: {exc}")


def main() -> None:
    """Entry point: format Train and Test I3D features."""
    base = Path("data")
    output_base = base / "features_v13_i3d"

    # Train
    train_rgb = base / "Train" / "RGB"
    if train_rgb.exists():
        format_split(train_rgb, output_base / "Train", "Train")
    else:
        print(f"[WARN] Train RGB dir not found: {train_rgb}")

    # Test
    test_rgb = base / "Test" / "RGB"
    if test_rgb.exists():
        format_split(test_rgb, output_base / "Test", "Test")
    else:
        print(f"[WARN] Test RGB dir not found: {test_rgb}")

    print("\n[DONE] I3D formatting complete.")


if __name__ == "__main__":
    main()
