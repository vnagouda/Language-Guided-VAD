"""Frame-level evaluation metrics for Video Anomaly Detection.

Provides the primary evaluation metric — frame-level AUROC — and utilities
to interpolate T=32 segment-level scores to per-frame granularity.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def interpolate_scores(
    segment_scores: np.ndarray,
    num_frames: int,
) -> np.ndarray:
    """Upsample segment-level anomaly scores to frame-level via linear interpolation.

    Given T=32 segment scores, this produces ``num_frames`` frame-level scores
    by linearly interpolating between segment centres.

    Args:
        segment_scores: 1-D array of shape ``(T,)`` with T segment scores.
        num_frames: Total number of frames in the original video.

    Returns:
        np.ndarray: 1-D array of shape ``(num_frames,)`` with per-frame scores.
    """
    num_segments = len(segment_scores)
    # x-coordinates of segment centres (normalised to [0, 1])
    seg_x = np.linspace(0.0, 1.0, num_segments)
    # x-coordinates of each frame (normalised to [0, 1])
    frame_x = np.linspace(0.0, 1.0, num_frames)

    frame_scores: np.ndarray = np.interp(frame_x, seg_x, segment_scores)
    return frame_scores


def compute_auroc(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """Compute the frame-level Area Under the ROC Curve (AUROC).

    This is the primary success metric for WS-VAD.  Both inputs must be 1-D
    arrays of the same length (total number of test frames across all videos).

    Args:
        predictions: Frame-level anomaly scores, shape ``(N_frames,)``.
        ground_truth: Frame-level binary labels (0/1), shape ``(N_frames,)``.

    Returns:
        float: AUROC value in [0, 1].  Higher is better.

    Raises:
        ValueError: If ground_truth contains only one class (AUROC undefined).
    """
    unique_labels = np.unique(ground_truth)
    if len(unique_labels) < 2:
        raise ValueError(
            f"AUROC is undefined when ground_truth contains only class "
            f"{unique_labels}.  Need both 0 and 1."
        )

    auroc: float = float(roc_auc_score(ground_truth, predictions))
    return auroc
