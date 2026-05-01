"""PyTorch Dataset and DataLoader utilities for loading pre-extracted .pt features.

After offline feature extraction (``scripts/01_extract_features.py``), each video
is represented by ``.pt`` files in a uniquely-named feature directory:

    - ``{video_name}_visual.pt``  → Tensor[32, D]   (D=512 V2.x / D=768 V3)
    - ``{video_name}_text.pt``    → Tensor[32, D]
    - ``{video_name}_flow.pt``    → Tensor[32]       (optional, V3 only)
    - ``{video_name}_label.pt``   → scalar int       (0=normal, 1=anomaly)

This module provides:
    - :class:`VADDataset` — loads those tensors and pairs them with video-level
      labels for training with the MIL ranking loss.
    - :func:`get_dataloaders` — convenience factory that builds train and test
      :class:`torch.utils.data.DataLoader` instances.

Backward compatibility:
    If ``{video_name}_flow.pt`` does not exist (V2.x feature directories), a
    zero tensor of shape ``(num_segments,)`` is returned as the flow output.
    This allows the same training script to work with both V2.x and V3 features.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader


class VADDataset(Dataset):
    """Video Anomaly Detection dataset that loads pre-extracted .pt features.

    Each sample is a 4-tuple:
        ``(visual_features, text_features, flow_magnitudes, label)``

    Tensors:
        - ``visual_features``:  shape ``(T, D)`` — CLIP visual embeddings.
        - ``text_features``:    shape ``(T, D)`` — CLIP text embeddings.
        - ``flow_magnitudes``:  shape ``(T,)``  — optical flow scalars (zero
                                if ``_flow.pt`` not found — V2.x compatibility).
        - ``label``:            int, 0 = normal, 1 = anomaly.

    Args:
        features_dir: Path to directory containing ``.pt`` feature files
            for a given split (e.g., ``data/features_florence2_vitl14_5f_patch/Train``).
        num_segments: Expected number of temporal segments T (default 32).
        feature_dim: Expected feature dimensionality D (512 for V2.x, 768 for V3).
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

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Load and return a single video sample.

        Args:
            index: Index into the dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
                - ``visual_features``: Shape ``(T, D)``.
                - ``text_features``:   Shape ``(T, D)``.
                - ``flow_magnitudes``: Shape ``(T,)`` — zeros if no flow file.
                - ``label``:           ``0`` for normal, ``1`` for anomaly.
        """
        sample = self.samples[index]

        visual: torch.Tensor = torch.load(
            sample["visual_path"], map_location="cpu", weights_only=True
        )
        text: torch.Tensor = torch.load(
            sample["text_path"], map_location="cpu", weights_only=True
        )

        # Optional flow features — zero fallback for V2.x feature directories
        flow_path = sample["visual_path"].parent / f"{sample['video_name']}_flow.pt"
        if flow_path.exists():
            flow: torch.Tensor = torch.load(
                flow_path, map_location="cpu", weights_only=True
            )  # (T,)
        else:
            flow = torch.zeros(self.num_segments, dtype=torch.float32)

        # Shape assertions
        assert visual.shape == (self.num_segments, self.feature_dim), (
            f"Expected visual ({self.num_segments}, {self.feature_dim}), "
            f"got {visual.shape} for {sample['video_name']}"
        )
        assert text.shape == (self.num_segments, self.feature_dim), (
            f"Expected text ({self.num_segments}, {self.feature_dim}), "
            f"got {text.shape} for {sample['video_name']}"
        )

        return visual, text, flow, sample["label"]


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
