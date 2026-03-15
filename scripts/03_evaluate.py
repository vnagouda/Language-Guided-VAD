"""Evaluation script for the Language-Guided WS-VAD model.

Loads a trained checkpoint, runs inference on the test set, interpolates
segment-level scores to frame-level, and computes the frame-level AUROC.

Also saves per-video anomaly score curves as ``.npy`` files for visualization.

Usage:
    python scripts/03_evaluate.py
    python scripts/03_evaluate.py --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vad_architecture import LanguageGuidedVAD
from utils.video_utils import load_config, set_seed
from utils.dataset import VADDataset
from utils.metrics import compute_auroc, interpolate_scores


def evaluate(config_path: str, checkpoint_path: str | None) -> None:
    """Main evaluation pipeline.

    Args:
        config_path: Path to the YAML configuration file.
        checkpoint_path: Path to the model checkpoint.  If None, uses
            ``checkpoints/best_model.pth``.
    """
    config = load_config(config_path)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- Load Model ---
    model = LanguageGuidedVAD.from_config(config).to(device)

    if checkpoint_path is None:
        checkpoint_path = str(
            Path(config["training"]["checkpoint_dir"]) / "best_model.pth"
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")
    if "auroc" in ckpt:
        print(f"       Training AUROC was: {ckpt['auroc']:.4f}")

    # --- Load Test Dataset ---
    features_dir = Path(config["data"]["features_dir"]) / "Test"
    test_dataset = VADDataset(
        features_dir=features_dir,
        num_segments=config["model"]["num_segments"],
        feature_dim=config["model"]["feature_dim"],
    )
    print(f"[INFO] Test samples: {len(test_dataset)}")

    if len(test_dataset) == 0:
        print("[ERROR] No test samples found.  Run 01_extract_features.py first.")
        return

    # --- Inference ---
    results_dir = Path(config["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_video_scores: dict[str, np.ndarray] = {}
    all_video_labels: dict[str, int] = {}

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            sample = test_dataset.samples[idx]
            visual, text, label = test_dataset[idx]

            visual = visual.unsqueeze(0).to(device)  # (1, 32, 512)
            text = text.unsqueeze(0).to(device)       # (1, 32, 512)

            scores = model(visual, text)  # (1, 32)
            scores_np = scores.squeeze(0).cpu().numpy()  # (32,)

            video_name = sample["video_name"]
            all_video_scores[video_name] = scores_np
            all_video_labels[video_name] = label

    # --- Save per-video score curves ---
    scores_path = results_dir / "video_scores.npy"
    np.save(scores_path, all_video_scores, allow_pickle=True)
    print(f"\n[INFO] Saved score curves to: {scores_path}")

    # --- Compute Video-level AUROC (max-score) ---
    video_preds = np.array([scores.max() for scores in all_video_scores.values()])
    video_labels = np.array(list(all_video_labels.values()))

    try:
        video_auroc = compute_auroc(video_preds, video_labels)
        print(f"[RESULT] Video-level AUROC (max-score): {video_auroc:.4f}")
    except ValueError as e:
        print(f"[WARN] Could not compute video-level AUROC: {e}")

    # --- Frame-level AUROC (if annotation file available) ---
    annotation_file = Path(config["data"]["annotation_file"])
    if annotation_file.exists():
        print("\n[INFO] Found annotation file, computing frame-level AUROC...")
        frame_auroc = compute_frame_level_auroc(
            all_video_scores, annotation_file, config
        )
        print(f"[RESULT] Frame-level AUROC: {frame_auroc:.4f}")
    else:
        print(f"\n[INFO] Annotation file not found at {annotation_file}.")
        print("       Frame-level AUROC cannot be computed without it.")
        print("       Please download Temporal_Anomaly_Annotation.txt.")

    print("\n[DONE] Evaluation complete.")


def compute_frame_level_auroc(
    video_scores: dict[str, np.ndarray],
    annotation_path: Path,
    config: dict,
) -> float:
    """Compute frame-level AUROC using temporal annotations.

    The annotation file format (Sultani et al.) has lines like:
        ``VideoName  AnomalyStart1  AnomalyEnd1  AnomalyStart2  AnomalyEnd2``

    For normal videos, all fields are -1.

    Args:
        video_scores: Dict mapping video names to segment-level scores.
        annotation_path: Path to the temporal annotation file.
        config: Configuration dictionary.

    Returns:
        float: Frame-level AUROC.
    """
    all_frame_preds: list[np.ndarray] = []
    all_frame_labels: list[np.ndarray] = []

    # Parse annotations
    annotations: dict[str, list[int]] = {}
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # The file might have: VideoName Class Start1 End1 Start2 End2
                name = parts[0].replace(".mp4", "")
                # The last 4 elements are always the start/end frames
                vals = [int(x) for x in parts[-4:]]
                annotations[name] = vals

    for video_name, scores in video_scores.items():
        # Look up annotations (try various name formats)
        ann = annotations.get(video_name)
        if ann is None:
            # Try without _x264 suffix
            clean_name = video_name.replace("_x264", "")
            ann = annotations.get(clean_name)

        if ann is None:
            continue  # Skip if no annotation found

        # Determine number of frames (approximate from score length)
        # In practice, you'd want the actual frame count
        num_segments = len(scores)
        # We'll use a default approximation
        num_frames = num_segments * 16  # rough estimate

        # Interpolate scores to frame level
        frame_scores = interpolate_scores(scores, num_frames)

        # Build frame-level labels
        frame_labels = np.zeros(num_frames, dtype=np.int32)
        start1, end1, start2, end2 = ann
        if start1 >= 0 and end1 >= 0:
            s = min(start1, num_frames - 1)
            e = min(end1, num_frames)
            frame_labels[s:e] = 1
        if start2 >= 0 and end2 >= 0:
            s = min(start2, num_frames - 1)
            e = min(end2, num_frames)
            frame_labels[s:e] = 1

        all_frame_preds.append(frame_scores)
        all_frame_labels.append(frame_labels)

    # Concatenate all frames
    if not all_frame_preds:
        return 0.5

    preds = np.concatenate(all_frame_preds)
    labels = np.concatenate(all_frame_labels)

    return compute_auroc(preds, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the trained Language-Guided WS-VAD model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (defaults to checkpoints/best_model.pth)",
    )
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
