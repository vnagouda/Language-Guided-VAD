"""Build the standard UCF-Crime ground truth labels matching SOTA evaluation.

The standard protocol requires knowing each video's ORIGINAL frame count.
Since we only have subsampled frames, we estimate the original count using:
- For anomalous videos: max(annotation_max_frame + 1, extracted_frames * ratio)
- For normal videos: extracted_frames * median_subsampling_ratio

The subsampling ratio is computed from anomalous videos where we know both
the extracted count AND the annotation frame indices (which reference the
original video frame space).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def build_gt_and_frame_counts(
    annotation_file: str = "data/Temporal_Anomaly_Annotation.txt",
    raw_dir: str = "data/raw",
) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    """Build video frame counts and parse annotations.

    Estimates original video frame counts by computing the subsampling
    ratio from anomalous videos (annotation_max / extracted_frames) and
    applying it to normal videos.

    Args:
        annotation_file: Path to the temporal annotation file.
        raw_dir: Path to the raw extracted frames directory.

    Returns:
        Tuple of (frame_counts dict, annotations dict).
    """
    raw_path = Path(raw_dir) / "Test"

    # Count extracted frames per video
    extracted_counts: Dict[str, int] = defaultdict(int)
    for cat_dir in sorted(raw_path.iterdir()):
        if not cat_dir.is_dir():
            continue
        for f in cat_dir.iterdir():
            if f.suffix in (".png", ".jpg"):
                parts = f.stem.rsplit("_", 1)
                if len(parts) == 2:
                    extracted_counts[parts[0]] += 1

    # Parse annotations
    annotations: Dict[str, List[int]] = {}
    with open(annotation_file, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 5:
                name = parts[0].replace(".mp4", "")
                annotations[name] = [int(x) for x in parts[-4:]]

    # Compute subsampling ratios from anomalous videos
    ratios: List[float] = []
    for name, ann in annotations.items():
        max_ann = max(v for v in ann if v > 0) if any(v > 0 for v in ann) else 0
        extracted = extracted_counts.get(name, 0)
        if max_ann > 0 and extracted > 0:
            ratios.append(max_ann / extracted)

    median_ratio = sorted(ratios)[len(ratios) // 2] if ratios else 6.0
    print(f"[GT] Median subsampling ratio: {median_ratio:.2f}x")
    print(f"[GT] Computed from {len(ratios)} anomalous videos")

    # Build frame counts
    frame_counts: Dict[str, int] = {}
    for name, ann in annotations.items():
        max_ann = max(v for v in ann if v > 0) if any(v > 0 for v in ann) else 0
        extracted = extracted_counts.get(name, 0)

        if max_ann > 0:
            # Anomalous video: use max annotation frame as minimum
            # Also estimate from extracted * ratio in case video continues after anomaly
            estimated = int(extracted * median_ratio) if extracted > 0 else 0
            frame_counts[name] = max(max_ann + 1, estimated)
        else:
            # Normal video: estimate from extracted frames * ratio
            if extracted > 0:
                frame_counts[name] = int(extracted * median_ratio)
            else:
                frame_counts[name] = 512  # fallback

    return frame_counts, annotations


if __name__ == "__main__":
    counts, anns = build_gt_and_frame_counts()

    total_frames = sum(counts.values())
    total_anomaly = 0
    for name, ann in anns.items():
        s1, e1, s2, e2 = ann
        if s1 >= 0 and e1 >= 0:
            total_anomaly += e1 - s1
        if s2 >= 0 and e2 >= 0:
            total_anomaly += e2 - s2

    total_normal = total_frames - total_anomaly
    print(f"\n[GT] Total frames: {total_frames:,}")
    print(f"[GT] Anomaly frames: {total_anomaly:,}")
    print(f"[GT] Normal frames: {total_normal:,}")
    print(f"[GT] Anomaly ratio: {total_anomaly/total_frames:.1%}")
    print(f"\n[GT] Compare to SOTA: ~5-7% anomaly ratio")

    # Save
    with open("data/video_frame_counts.json", "w") as f:
        json.dump(counts, f, indent=2)
    print(f"[GT] Saved {len(counts)} entries to data/video_frame_counts.json")
