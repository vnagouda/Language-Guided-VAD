"""Frame-level AUROC evaluation utility — shared by 02_train.py and 04_hpo.py.

V10 APEX critical fix:
    - Replaced hardcoded ``T * 16`` frame count with actual per-video frame
      counts derived from the raw data directory.
    - Removed destructive post-hoc Gaussian smoothing that was smearing
      the sharp temporal boundaries learned by PDC + MIST.

The standard UCF-Crime evaluation protocol requires interpolating segment
scores to the *actual* video frame count, not an arbitrary estimate. The V9
bug silently mislabeled 74.9% of anomaly frames as normal during AUROC
computation, capping achievable performance far below SOTA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.data


def _count_raw_frames(raw_dir: str, video_name: str) -> int:
    """Count actual extracted frames for a video from the raw data directory.

    Searches through all category subdirectories in the raw Test directory
    for PNG/JPG files matching the video name prefix.

    Args:
        raw_dir: Path to the raw data root (e.g., ``data/raw``).
        video_name: Video identifier (e.g., ``Abuse028_x264``).

    Returns:
        int: Number of frames found. Returns 0 if no frames found.
    """
    raw_path = Path(raw_dir)
    test_dir = raw_path / "Test"

    if not test_dir.exists():
        return 0

    count = 0
    for cat_dir in test_dir.iterdir():
        if not cat_dir.is_dir():
            continue
        # Frames are stored as {video_name}_{frame_idx}.png
        for ext in (".png", ".jpg"):
            count += len(list(cat_dir.glob(f"{video_name}_*{ext}")))
        if count > 0:
            break  # Found the category, no need to keep searching

    return count


def compute_frame_level_auroc(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    annotation_file: str,
    results_dir: str,
    raw_dir: str = "data/raw",
) -> Optional[float]:
    """Compute frame-level AUROC against temporal annotation file.

    V10 APEX: Uses actual per-video frame counts from the raw data directory
    instead of the hardcoded ``T * 16`` estimate. Gaussian smoothing has been
    removed to preserve the sharp temporal boundaries learned by PDC + MIST.

    Args:
        model: The trained LanguageGuidedVAD model (eval mode set internally).
        test_loader: DataLoader for the test split (batch_size=1).
        device: Torch device for inference.
        annotation_file: Path to ``Temporal_Anomaly_Annotation.txt``.
        results_dir: Directory to save intermediate score curves (created if
            absent).
        raw_dir: Path to the raw data root for actual frame counting.

    Returns:
        Optional[float]: Frame-level AUROC in [0, 1], or ``None`` if the
        annotation file is missing or fewer than 2 label classes are present.
    """
    from utils.metrics import compute_auroc, interpolate_scores  # local import avoids circular dep
    import json as _json

    ann_path = Path(annotation_file)
    if not ann_path.exists():
        return None

    model.eval()
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # V11: Load pre-computed frame counts (estimates original video lengths)
    _frame_count_cache: dict[str, int] | None = None
    fc_path = Path("data/video_frame_counts.json")
    if fc_path.exists():
        with open(fc_path) as _fc_fh:
            _frame_count_cache = _json.load(_fc_fh)
        print(f"  [EVAL] Loaded {len(_frame_count_cache)} frame counts from {fc_path}")

    # ------------------------------------------------------------------
    # Parse annotation file
    # Format: VideoName  ClassName  Start1  End1  Start2  End2
    # Use parts[-4:] to robustly handle optional class-name column.
    # ------------------------------------------------------------------
    annotations: dict[str, list[int]] = {}
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    name = parts[0].replace(".mp4", "")
                    vals = [int(x) for x in parts[-4:]]
                    annotations[name] = vals
    except Exception as exc:
        print(f"  [WARN] Annotation file parse error: {exc}")
        return None

    # ------------------------------------------------------------------
    # Collect per-video score curves from the test set
    # ------------------------------------------------------------------
    video_scores: dict[str, np.ndarray] = {}
    dataset = test_loader.dataset

    for idx in range(len(dataset)):  # type: ignore[arg-type]
        item = dataset[idx]          # (visual, text, flow, label) or (visual, text, label)
        visual_item = item[0]
        text_item   = item[1]
        # flow is item[2] if 4-tuple, else zeros
        flow_item   = item[2] if len(item) == 4 else torch.zeros(visual_item.shape[0], 1)

        visual_t = visual_item.unsqueeze(0).to(device)
        text_t   = text_item.unsqueeze(0).to(device)
        flow_t   = flow_item.unsqueeze(0).to(device)

        with torch.no_grad():
            scores, _, _ = model(visual_t, text_t, flow_t)

        score_curve = scores.squeeze(0).cpu().numpy()  # (T,)

        # V10 APEX: NO post-hoc Gaussian smoothing.
        # The PDC module and AIS loss already enforce temporal coherence
        # during training. Post-hoc smoothing destroys the sharp boundaries
        # that MIST learns, directly reducing Frame-AUROC.

        video_name  = dataset.samples[idx]["video_name"]
        video_scores[video_name] = score_curve

    # ------------------------------------------------------------------
    # Build frame-level predictions and binary labels
    # ------------------------------------------------------------------
    all_frame_preds:  list[np.ndarray] = []
    all_frame_labels: list[np.ndarray] = []

    frame_count_source_stats = {"raw": 0, "fallback": 0}

    for video_name, seg_scores in video_scores.items():
        ann = annotations.get(video_name)
        if ann is None:
            # Try without codec suffix
            clean = video_name.replace("_x264", "")
            ann   = annotations.get(clean)
        if ann is None:
            continue

        # V11: Use pre-computed ORIGINAL video frame counts.
        # Normal videos were severely under-counted (extracted frames only,
        # missing the 6.3x subsampling factor), inflating anomaly ratio
        # from ~11% (SOTA-comparable) to 24.6% (artificially hard).
        start1, end1, start2, end2 = ann

        # Try JSON lookup first (built by scripts/build_gt.py)
        if _frame_count_cache is not None:
            lookup_name = video_name
            if lookup_name not in _frame_count_cache:
                lookup_name = video_name.replace("_x264", "")
            if lookup_name in _frame_count_cache:
                num_frames = _frame_count_cache[lookup_name]
                frame_count_source_stats["raw"] += 1
            else:
                # Fallback for videos not in JSON
                actual_frames = _count_raw_frames(raw_dir, video_name)
                max_ann_frame = max(v for v in ann if v > 0) if any(v > 0 for v in ann) else 0
                num_frames = max(max_ann_frame + 1, actual_frames) if max_ann_frame > 0 else max(actual_frames, len(seg_scores) * 16)
                frame_count_source_stats["fallback"] += 1
        else:
            # No JSON file — fall back to raw directory counting
            actual_frames = _count_raw_frames(raw_dir, video_name)
            max_ann_frame = max(v for v in ann if v > 0) if any(v > 0 for v in ann) else 0
            if actual_frames > 0:
                if max_ann_frame > 0:
                    num_frames = max(max_ann_frame + 1, actual_frames)
                else:
                    num_frames = actual_frames
                frame_count_source_stats["raw"] += 1
            else:
                num_frames = max_ann_frame + 1 if max_ann_frame > 0 else len(seg_scores) * 16
                frame_count_source_stats["fallback"] += 1

        frame_scores = interpolate_scores(seg_scores, num_frames)

        frame_labels = np.zeros(num_frames, dtype=np.int32)
        if start1 >= 0 and end1 >= 0:
            frame_labels[min(start1, num_frames - 1):min(end1, num_frames)] = 1
        if start2 >= 0 and end2 >= 0:
            frame_labels[min(start2, num_frames - 1):min(end2, num_frames)] = 1

        all_frame_preds.append(frame_scores)
        all_frame_labels.append(frame_labels)

    if not all_frame_preds:
        return None

    preds  = np.concatenate(all_frame_preds)
    labels = np.concatenate(all_frame_labels)

    if len(set(labels.tolist())) < 2:
        return None

    try:
        auroc = float(compute_auroc(preds, labels))
        # Log frame count source statistics
        print(f"  [EVAL] Frame counts: {frame_count_source_stats['raw']} from raw data, "
              f"{frame_count_source_stats['fallback']} fallback")
        print(f"  [EVAL] Total frames evaluated: {len(preds)} "
              f"(anomaly: {labels.sum()}, normal: {(labels == 0).sum()})")
        return auroc
    except Exception as exc:
        print(f"  [WARN] AUROC compute error: {exc}")
        return None
