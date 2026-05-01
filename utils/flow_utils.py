"""Optical flow magnitude extraction utilities.

Uses OpenCV Farneback dense optical flow to compute a per-segment motion
signal during offline feature extraction. High optical flow magnitude
correlates with rapid pixel movement, which is strongly characteristic of
anomalous events such as fights, vehicle crashes, and chasing behaviour.

Mathematical formulation:
    Given consecutive segment centre-frames F_t and F_{t+1}, dense optical
    flow produces a 2D displacement field (u_{x,y}, v_{x,y}) for every pixel.
    Per-segment magnitude scalar:

        m_t = (1 / HW) * Σ_{x,y} sqrt(u_{x,y}^2 + v_{x,y}^2)

    This yields a shape (32,) vector of motion signals, one per temporal segment,
    saved as ``{video_name}_flow.pt``.

Usage:
    from utils.flow_utils import extract_video_flow
    flow_arr = extract_video_flow(center_frames, num_segments=32)  # np.ndarray (32,)
    torch.save(torch.tensor(flow_arr), flow_out_path)
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def _check_cv2() -> None:
    """Raise a clear error if opencv-python is not installed."""
    if not _CV2_AVAILABLE:
        raise ImportError(
            "opencv-python is required for optical flow extraction. "
            "Install it with: pip install opencv-python"
        )


def extract_flow_magnitude(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> float:
    """Compute mean Farneback dense optical flow magnitude between two frames.

    Converts RGB frames to greyscale, runs the Farneback algorithm, and
    returns the mean pixel displacement magnitude across the entire frame.

    Mathematical result:
        m = mean(sqrt(u^2 + v^2))  over all pixel positions (x, y)

    Args:
        frame_a: First frame as uint8 numpy array of shape (H, W, 3) in RGB
                 order, or (H, W) for greyscale.
        frame_b: Second frame, same shape and dtype as frame_a.

    Returns:
        float: Mean optical flow magnitude (pixels displaced per frame).
               Larger values indicate faster movement / more motion.

    Raises:
        ImportError: If opencv-python is not installed.
        ValueError: If frame shapes do not match.
    """
    _check_cv2()

    if frame_a.shape != frame_b.shape:
        raise ValueError(
            f"Frame shapes must match: {frame_a.shape} != {frame_b.shape}"
        )

    # Convert RGB → greyscale for Farneback (expects single-channel input)
    if frame_a.ndim == 3 and frame_a.shape[2] == 3:
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
    else:
        gray_a = frame_a.astype(np.uint8)
        gray_b = frame_b.astype(np.uint8)

    # Farneback dense optical flow
    # Parameters follow the OpenCV recommended defaults for general video
    flow: np.ndarray = cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=0.5,   # image pyramid scale (0.5 = each layer is ½ the previous)
        levels=3,        # number of pyramid layers
        winsize=15,      # averaging window size
        iterations=3,    # iterations per pyramid level
        poly_n=5,        # neighbourhood pixel size for polynomial expansion
        poly_sigma=1.2,  # Gaussian std for polynomial expansion
        flags=0,
    )                    # flow.shape == (H, W, 2): [..., 0]=u, [..., 1]=v

    # Compute per-pixel magnitude, then take the spatial mean
    magnitude: np.ndarray = np.sqrt(
        flow[..., 0] ** 2 + flow[..., 1] ** 2
    )                    # (H, W)
    return float(magnitude.mean())


def extract_video_flow(
    center_frames: list[np.ndarray],
    num_segments: int = 32,
) -> np.ndarray:
    """Compute per-segment optical flow magnitude for a full video.

    Uses the centre-frame of each temporal segment to compute the inter-segment
    motion signal. Segment t=0 has no prior frame, so it is assigned the same
    value as t=1.

    Args:
        center_frames: List of numpy arrays (uint8, RGB), one per segment. 
                       Length must equal num_segments.
        num_segments:  Expected number of temporal segments T (default 32).

    Returns:
        np.ndarray: Shape (num_segments,), dtype float32.
                    flow[t] = mean optical flow magnitude between segments t-1 and t.
                    flow[0] = flow[1] (replicated, no prior frame for segment 0).

    Raises:
        AssertionError: If len(center_frames) != num_segments.
        ImportError:    If opencv-python is not installed.
    """
    assert len(center_frames) == num_segments, (
        f"Expected {num_segments} center frames, got {len(center_frames)}"
    )

    flow_mags: np.ndarray = np.zeros(num_segments, dtype=np.float32)

    for t in range(1, num_segments):
        flow_mags[t] = extract_flow_magnitude(
            center_frames[t - 1], center_frames[t]
        )

    # Segment 0 has no preceding frame — replicate segment 1's value
    flow_mags[0] = flow_mags[1]

    return flow_mags
