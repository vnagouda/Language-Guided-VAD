"""PyTorch Dataset and DataLoader utilities for loading pre-extracted .pt features for V13.

This is a standalone copy of dataset.py modified to load the Tri-Modal V13 features:
    - `{video_name}_visual.pt`  → Tensor[128, 768] (CLIP Visual)
    - `{video_name}_text.pt`    → Tensor[128, 768] (BLIP-2 Text)
    - `{video_name}_flow.pt`    → Tensor[128]      (Flow scalar)
    - `{video_name}_i3d.pt`     → Tensor[128, 1024] (I3D Motion)
    - `{video_name}_label.pt`   → scalar int       (0=normal, 1=anomaly)

Returns a 5-tuple: (visual, text, flow, i3d, label)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader


class TriModalVADDataset(Dataset):
    """Video Anomaly Detection dataset for Tri-Modal V13 architecture.

    Each sample is a 5-tuple:
        ``(visual_features, text_features, flow_magnitudes, i3d_features, label)``

    Args:
        features_dir: Path to directory containing ``.pt`` feature files (CLIP/BLIP-2/Flow).
        i3d_dir: Path to directory containing ``_i3d.pt`` feature files.
        num_segments: Expected number of temporal segments T (default 128).
        feature_dim: Expected feature dimensionality D (768).
        i3d_dim: Expected I3D dimensionality (1024).
        require_i3d: If True, skip videos without I3D features (Option A).
    """

    def __init__(
        self,
        features_dir: str | Path,
        i3d_dir: str | Path,
        num_segments: int = 128,
        feature_dim: int = 768,
        i3d_dim: int = 1024,
        require_i3d: bool = False,
    ) -> None:
        super().__init__()
        self.features_dir = Path(features_dir)
        self.i3d_dir = Path(i3d_dir)
        self.num_segments = num_segments
        self.feature_dim = feature_dim
        self.i3d_dim = i3d_dim
        self.require_i3d = require_i3d

        self.samples: list[dict[str, Any]] = []
        self._scan_directory()

    def _scan_directory(self) -> None:
        if not self.features_dir.exists():
            return

        visual_files = sorted(self.features_dir.glob("*_visual.pt"))

        for vis_path in visual_files:
            video_name = vis_path.stem.replace("_visual", "")
            text_path = vis_path.parent / f"{video_name}_text.pt"
            i3d_path = self.i3d_dir / f"{video_name}_i3d.pt"
            label_path = vis_path.parent / f"{video_name}_label.pt"

            if not text_path.exists():
                continue

            # Option A: skip videos without I3D features to avoid bias
            if self.require_i3d and not i3d_path.exists():
                continue

            if label_path.exists():
                label = int(torch.load(label_path, weights_only=True).item())
            else:
                label = 0 if "Normal" in video_name else 1

            self.samples.append({
                "video_name": video_name,
                "visual_path": vis_path,
                "text_path": text_path,
                "i3d_path": i3d_path,
                "label": label,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        sample = self.samples[index]

        visual: torch.Tensor = torch.load(
            sample["visual_path"], map_location="cpu", weights_only=True
        )
        text: torch.Tensor = torch.load(
            sample["text_path"], map_location="cpu", weights_only=True
        )

        flow_path = sample["visual_path"].parent / f"{sample['video_name']}_flow.pt"
        if flow_path.exists():
            flow: torch.Tensor = torch.load(
                flow_path, map_location="cpu", weights_only=True
            )
        else:
            flow = torch.zeros(self.num_segments, dtype=torch.float32)

        if sample["i3d_path"].exists():
            i3d: torch.Tensor = torch.load(
                sample["i3d_path"], map_location="cpu", weights_only=True
            )
        else:
            # Fallback for missing I3D features
            i3d = torch.zeros(self.num_segments, self.i3d_dim, dtype=torch.float32)

        # Basic shape enforcement
        if visual.shape[0] != self.num_segments:
            # For robustness, we interpolate if lengths mismatch
            visual = torch.nn.functional.interpolate(visual.unsqueeze(0).transpose(1,2), size=self.num_segments, mode='linear').transpose(1,2).squeeze(0)
            text = torch.nn.functional.interpolate(text.unsqueeze(0).transpose(1,2), size=self.num_segments, mode='linear').transpose(1,2).squeeze(0)
            flow = torch.nn.functional.interpolate(flow.unsqueeze(0).unsqueeze(0), size=self.num_segments, mode='linear').squeeze(0).squeeze(0)

        if i3d.shape[0] != self.num_segments:
            i3d = torch.nn.functional.interpolate(i3d.unsqueeze(0).transpose(1,2), size=self.num_segments, mode='linear').transpose(1,2).squeeze(0)

        return visual, text, flow, i3d, sample["label"]


def get_dataloaders_v13(
    config: dict,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders for V13 Tri-Modal architecture."""
    features_dir = Path(config["data"]["features_dir"])
    i3d_dir = Path(config["data"].get("i3d_dir", "data/features_v13_i3d"))
    num_segments = config["model"]["num_segments"]
    feature_dim = config["model"]["feature_dim"]
    i3d_dim = config["model"].get("i3d_dim", 1024)
    batch_size = config["training"]["batch_size"]
    require_i3d = config["data"].get("require_i3d", False)

    train_dataset = TriModalVADDataset(
        features_dir=features_dir / "Train",
        i3d_dir=i3d_dir / "Train",
        num_segments=num_segments,
        feature_dim=feature_dim,
        i3d_dim=i3d_dim,
        require_i3d=require_i3d,
    )
    test_dataset = TriModalVADDataset(
        features_dir=features_dir / "Test",
        i3d_dir=i3d_dir / "Test",
        num_segments=num_segments,
        feature_dim=feature_dim,
        i3d_dim=i3d_dim,
        require_i3d=False,  # Always include all test videos
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
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
