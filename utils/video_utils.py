"""Utility functions for video/frame loading and project-wide configuration.

This module provides:
- YAML config loading
- Deterministic seed setting for reproducibility
- Uniform T=32 segment sampling from directories of pre-extracted PNG frames

The UCF-Crime dataset stores frames as `{VideoName}_x264_{FrameNum}.png` in flat
class directories (e.g., `data/raw/Train/Abuse/Abuse001_x264_0.png`).  This module
groups frames by video name, sorts numerically, and samples exactly T=32 representative
frames via uniform stride.
"""

from __future__ import annotations

import os
import re
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load the central YAML configuration file.

    Args:
        config_path: Path (relative or absolute) to the YAML config file.

    Returns:
        dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Set deterministic seeds for torch, numpy, and random.

    Ensures reproducible behaviour across runs.  Should be called at the
    very start of every script.

    Args:
        seed: Integer seed value (loaded from config.yaml).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN for full reproducibility (may reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Frame filename parsing
# ---------------------------------------------------------------------------

# Regex to extract (video_name, frame_number) from filenames like
# "Abuse001_x264_1230.png".  The video name is everything before the last
# occurrence of "_<digits>.<ext>".
_FRAME_REGEX = re.compile(
    r"^(?P<video_name>.+)_(?P<frame_num>\d+)\.(?:png|jpg|jpeg)$",
    re.IGNORECASE,
)


def _parse_frame_filename(filename: str) -> tuple[str, int] | None:
    """Parse a frame filename into (video_name, frame_number).

    Args:
        filename: Base filename, e.g. ``"Abuse001_x264_1230.png"``.

    Returns:
        tuple[str, int] | None: ``("Abuse001_x264", 1230)`` or ``None``
        if the filename does not match the expected pattern.
    """
    match = _FRAME_REGEX.match(filename)
    if match is None:
        return None
    return match.group("video_name"), int(match.group("frame_num"))


# ---------------------------------------------------------------------------
# Directory scanning — discover videos from flat frame directories
# ---------------------------------------------------------------------------

def discover_videos_in_class_dir(
    class_dir: str | Path,
    frame_extensions: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Discover all videos (and their frames) inside a single class directory.

    The UCF-Crime dataset stores all frames from all videos of a given class
    in a single flat directory.  This function groups them by video name,
    for example::

        {
            "Abuse001_x264": [Path("Abuse001_x264_0.png"), ...],
            "Abuse002_x264": [Path("Abuse002_x264_0.png"), ...],
        }

    Frames within each video are sorted **numerically** by frame number so
    that chronological ordering is guaranteed even when the raw sort order
    is lexicographic (e.g., ``10`` before ``100``).

    Args:
        class_dir: Path to a class directory (e.g. ``data/raw/Train/Abuse``).
        frame_extensions: List of accepted image extensions including the
            leading dot, e.g. ``[".png", ".jpg"]``.  Defaults to
            ``[".png", ".jpg"]``.

    Returns:
        dict[str, list[Path]]: Mapping from video name to a sorted list of
        frame file paths.
    """
    if frame_extensions is None:
        frame_extensions = [".png", ".jpg"]

    class_dir = Path(class_dir)
    videos: dict[str, list[tuple[int, Path]]] = {}

    for entry in class_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in frame_extensions:
            continue

        parsed = _parse_frame_filename(entry.name)
        if parsed is None:
            continue

        video_name, frame_num = parsed
        videos.setdefault(video_name, []).append((frame_num, entry))

    # Sort each video's frames numerically by frame number
    sorted_videos: dict[str, list[Path]] = {}
    for video_name, frame_list in videos.items():
        frame_list.sort(key=lambda x: x[0])
        sorted_videos[video_name] = [path for _, path in frame_list]

    return sorted_videos


def discover_all_videos(
    split_dir: str | Path,
    frame_extensions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Discover every video across all class directories in a Train/ or Test/ split.

    Iterates over class sub-directories, groups frames by video, and assigns
    binary labels:  ``1`` = anomaly class, ``0`` = ``NormalVideos``.

    Args:
        split_dir: Path to a split directory (e.g. ``data/raw/Train``).
        frame_extensions: Accepted image file extensions.

    Returns:
        list[dict[str, Any]]: A list of dictionaries, each containing:
            - ``"video_name"`` (str): Unique video identifier, e.g.
              ``"Abuse001_x264"``.
            - ``"class_name"`` (str): Anomaly class name or
              ``"NormalVideos"``.
            - ``"label"`` (int): ``0`` for normal, ``1`` for anomaly.
            - ``"frames"`` (list[Path]): Chronologically sorted list of
              frame file paths.
    """
    split_dir = Path(split_dir)
    all_videos: list[dict[str, Any]] = []

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        label = 0 if class_name == "NormalVideos" else 1

        videos = discover_videos_in_class_dir(class_dir, frame_extensions)
        for video_name, frames in sorted(videos.items()):
            all_videos.append({
                "video_name": video_name,
                "class_name": class_name,
                "label": label,
                "frames": frames,
            })

    return all_videos


# ---------------------------------------------------------------------------
# Uniform T=32 segment sampling from image sequences
# ---------------------------------------------------------------------------

def sample_image_sequence_uniform(
    frame_paths: list[Path],
    num_segments: int = 32,
) -> list[np.ndarray]:
    """Sample exactly T=``num_segments`` frames uniformly from an image sequence.

    Mathematical approach:
        Given ``N`` total frames and ``T`` desired segments, the stride is
        ``stride = N / T``.  For segment ``i`` we pick the frame at index
        ``floor(i * stride + stride / 2)`` (center of each segment bin).

    Each frame is loaded with OpenCV (``cv2.imread``) and converted from
    BGR → RGB colour space.

    Args:
        frame_paths: Chronologically sorted list of absolute paths to PNG/JPG
            frame images for a single video.
        num_segments: Number of segments to divide the video into (T=32).

    Returns:
        list[np.ndarray]: Exactly ``num_segments`` RGB images, each of shape
        ``(H, W, 3)`` with dtype ``uint8``.

    Raises:
        ValueError: If the video has fewer frames than ``num_segments``.
        RuntimeError: If any frame file cannot be read.
    """
    total_frames = len(frame_paths)
    if total_frames < num_segments:
        raise ValueError(
            f"Video has {total_frames} frames but {num_segments} segments "
            f"are required.  Cannot sample."
        )

    stride = total_frames / num_segments
    sampled: list[np.ndarray] = []

    for i in range(num_segments):
        # Center of segment i
        idx = int(i * stride + stride / 2)
        idx = min(idx, total_frames - 1)  # safety clamp

        img = cv2.imread(str(frame_paths[idx]))
        if img is None:
            raise RuntimeError(
                f"Failed to read frame: {frame_paths[idx]}"
            )
        # Convert from BGR (OpenCV default) to RGB
        img_rgb: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sampled.append(img_rgb)

    return sampled


def sample_segment_clips(
    frame_paths: list[Path],
    num_segments: int = 32,
    frames_per_segment: int = 4,
) -> list[list[np.ndarray]]:
    """Sample multiple frames per segment for more robust feature extraction.

    For each of the ``T=num_segments`` temporal bins, this function samples
    ``frames_per_segment`` evenly-spaced frames.  This allows the feature
    extractor to average features across multiple frames per segment,
    producing more temporally-robust representations.

    Args:
        frame_paths: Chronologically sorted list of frame image paths.
        num_segments: Number of temporal segments (T=32).
        frames_per_segment: Number of frames to sample within each segment.

    Returns:
        list[list[np.ndarray]]: Outer list has ``num_segments`` elements.
        Each inner list has ``frames_per_segment`` RGB images of shape
        ``(H, W, 3)`` with dtype ``uint8``.

    Raises:
        ValueError: If the video has fewer frames than ``num_segments``.
    """
    total_frames = len(frame_paths)
    if total_frames < num_segments:
        raise ValueError(
            f"Video has {total_frames} frames but {num_segments} segments "
            f"are required."
        )

    stride = total_frames / num_segments
    clips: list[list[np.ndarray]] = []

    for i in range(num_segments):
        seg_start = int(i * stride)
        seg_end = int((i + 1) * stride)
        seg_end = min(seg_end, total_frames)

        seg_len = seg_end - seg_start
        if seg_len <= 0:
            seg_start = max(0, seg_end - 1)
            seg_len = 1

        # Evenly-spaced indices within this segment
        if frames_per_segment >= seg_len:
            indices = list(range(seg_start, seg_end))
            # Pad by repeating the last frame
            while len(indices) < frames_per_segment:
                indices.append(seg_end - 1)
        else:
            step = seg_len / frames_per_segment
            indices = [seg_start + int(j * step + step / 2)
                       for j in range(frames_per_segment)]
            indices = [min(idx, total_frames - 1) for idx in indices]

        segment_frames: list[np.ndarray] = []
        for idx in indices:
            img = cv2.imread(str(frame_paths[idx]))
            if img is None:
                raise RuntimeError(f"Failed to read frame: {frame_paths[idx]}")
            segment_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        clips.append(segment_frames)

    return clips
