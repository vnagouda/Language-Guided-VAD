"""PyTorch Dataset and DataLoader utilities for loading pre-extracted .pt features.

After offline feature extraction (``scripts/01_extract_features.py``), each video
is represented by two ``.pt`` files:
    - ``{video_name}_visual.pt``  → Tensor[32, 512]
    - ``{video_name}_text.pt``    → Tensor[32, 512]

This module provides:
    - :class:`VADDataset` — loads those tensors and pairs them with video-level
      labels for training with the MIL ranking loss.
    - :func:`get_dataloaders` — convenience factory that builds train and test
      :class:`torch.utils.data.DataLoader` instances.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader


class VADDataset(Dataset):
    """Video Anomaly Detection dataset that loads pre-extracted .pt features.

    Each sample is a tuple ``(visual_features, text_features, label)`` where
    both feature tensors have shape ``(32, 512)`` and ``label`` is an int
    (0 = normal, 1 = anomaly).

    The dataset scans the features directory for matching pairs of
    ``*_visual.pt`` and ``*_text.pt`` files.  Video-level labels are inferred
    from the parent directory name passed during feature extraction.

    Args:
        features_dir: Path to the directory containing ``.pt`` feature files
            for a given split (e.g., ``data/features/Train``).
        num_segments: Expected number of temporal segments (T=32).
        feature_dim: Expected feature dimensionality (512 for CLIP).
    """

    def __init__(
        self,
        features_dir: str | Path,
        num_segments: int = 32,
        feature_dim: int = 512,
    ) -> None:
        super().__init__()
        self.features_dir = Path(features_dir)
        self.num_segments = num_segments
        self.feature_dim = feature_dim

        # Discover all visual feature files and build sample list
        self.samples: list[dict[str, Any]] = []
        self._scan_directory()

    def _scan_directory(self) -> None:
        """Scan the features directory for matching visual/text .pt pairs.

        Expected directory structure::

            features_dir/
                {video_name}_visual.pt
                {video_name}_text.pt
                {video_name}_label.pt   (scalar tensor: 0 or 1)

        Or alternatively, labels can be stored in a single manifest file.
        We support both patterns by checking for ``*_label.pt`` first and
        falling back to a ``labels.pt`` dict-manifest.
        """
        if not self.features_dir.exists():
            return

        # Find all visual feature files
        visual_files = sorted(self.features_dir.glob("*_visual.pt"))

        for vis_path in visual_files:
            # Derive the video name and companion paths
            video_name = vis_path.stem.replace("_visual", "")
            text_path = vis_path.parent / f"{video_name}_text.pt"
            label_path = vis_path.parent / f"{video_name}_label.pt"

            if not text_path.exists():
                continue  # Skip if text features are missing

            # Load label
            if label_path.exists():
                label = int(torch.load(label_path, weights_only=True).item())
            else:
                # Fallback: infer from filename convention
                # Videos with "Normal" in the name are label=0, else label=1
                label = 0 if "Normal" in video_name else 1

            self.samples.append({
                "video_name": video_name,
                "visual_path": vis_path,
                "text_path": text_path,
                "label": label,
            })

    def __len__(self) -> int:
        """Return the total number of video samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Load and return a single video sample.

        Args:
            index: Index into the dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor, int]:
                - ``visual_features``: Shape ``(32, 512)``.
                - ``text_features``: Shape ``(32, 512)``.
                - ``label``: ``0`` for normal, ``1`` for anomaly.
        """
        sample = self.samples[index]

        visual: torch.Tensor = torch.load(
            sample["visual_path"], map_location="cpu", weights_only=True
        )
        text: torch.Tensor = torch.load(
            sample["text_path"], map_location="cpu", weights_only=True
        )

        # Safety: ensure correct shape
        assert visual.shape == (self.num_segments, self.feature_dim), (
            f"Expected visual shape ({self.num_segments}, {self.feature_dim}), "
            f"got {visual.shape} for {sample['video_name']}"
        )
        assert text.shape == (self.num_segments, self.feature_dim), (
            f"Expected text shape ({self.num_segments}, {self.feature_dim}), "
            f"got {text.shape} for {sample['video_name']}"
        )

        return visual, text, sample["label"]


def get_dataloaders(
    config: dict,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders from the configuration.

    The DataLoaders handle batching and shuffling.  The collate function
    groups samples into separate abnormal and normal mini-batches, which is
    essential for the MIL ranking loss.

    Args:
        config: Full configuration dict loaded from ``config.yaml``.

    Returns:
        tuple[DataLoader, DataLoader]: ``(train_loader, test_loader)``.
    """
    features_dir = Path(config["data"]["features_dir"])
    num_segments = config["model"]["num_segments"]
    feature_dim = config["model"]["feature_dim"]
    batch_size = config["training"]["batch_size"]

    train_dataset = VADDataset(
        features_dir=features_dir / "Train",
        num_segments=num_segments,
        feature_dim=feature_dim,
    )
    test_dataset = VADDataset(
        features_dir=features_dir / "Test",
        num_segments=num_segments,
        feature_dim=feature_dim,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader
