"""V12 Feature Extraction: CLIP-only visual at T=128 + interpolated BLIP-2 text.

This script:
  1. Extracts CLIP ViT-L/14 visual features at T=128 segments (~7 hours GPU)
  2. Interpolates existing T=32 BLIP-2 text features to T=128 (instant)
  3. Copies flow features with interpolation to T=128

Does NOT run BLIP-2 captioning — reuses existing T=32 text features.

Usage:
    python scripts/extract_v12_features.py --split Train
    python scripts/extract_v12_features.py --split Test
    python scripts/extract_v12_features.py --split Test --max_videos 5  # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.flow_utils import extract_video_flow
from utils.video_utils import (
    discover_all_videos,
    load_config,
    set_seed,
)


# ---------------------------------------------------------------------------
# Visual feature extraction (same as 01_extract_features.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_visual_features(
    frames_5: list[Image.Image],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    """Extract CLIP ViT-L/14 visual features via patch token mean (5-frame avg).

    Args:
        frames_5: List of 5 PIL Images — evenly-spaced frames within one segment.
        clip_model: Loaded CLIP ViT-L/14 model (eval mode, on device).
        clip_processor: Corresponding CLIPProcessor.
        device: Torch device.

    Returns:
        torch.Tensor: Shape (768,) on CPU — 5-frame averaged patch feature.
    """
    inputs = clip_processor(
        images=frames_5, return_tensors="pt", padding=True
    ).to(device)
    pixel_values: torch.Tensor = inputs["pixel_values"]  # (5, 3, 224, 224)

    hidden: torch.Tensor = clip_model.vision_model(
        pixel_values=pixel_values
    ).last_hidden_state  # (5, 257, 1024)

    patch_tokens: torch.Tensor = hidden[:, 1:, :]  # (5, 256, 1024)
    pooled: torch.Tensor = patch_tokens.mean(dim=1)  # (5, 1024)
    projected: torch.Tensor = clip_model.visual_projection(pooled)  # (5, 768)

    return projected.mean(dim=0).cpu()


# ---------------------------------------------------------------------------
# Interpolate existing T=32 text features to T=128
# ---------------------------------------------------------------------------

def interpolate_text_features(
    text_t32: torch.Tensor,
    target_t: int = 128,
) -> torch.Tensor:
    """Linearly interpolate T=32 text features to target_t segments.

    Text descriptions change slowly across a video, so linear interpolation
    of text embeddings is a valid approximation.

    Args:
        text_t32: Text features of shape (32, D).
        target_t: Target number of segments (default: 128).

    Returns:
        torch.Tensor: Interpolated text features of shape (target_t, D).
    """
    # F.interpolate expects (N, C, L) — add batch dim, transpose to (1, D, 32)
    t = text_t32.unsqueeze(0).permute(0, 2, 1)  # (1, D, 32)
    t_interp = F.interpolate(t, size=target_t, mode="linear", align_corners=True)
    return t_interp.permute(0, 2, 1).squeeze(0)  # (target_t, D)


def interpolate_flow_features(
    flow_t32: torch.Tensor,
    target_t: int = 128,
) -> torch.Tensor:
    """Linearly interpolate T=32 flow magnitudes to target_t segments.

    Args:
        flow_t32: Flow magnitudes of shape (32,).
        target_t: Target number of segments (default: 128).

    Returns:
        torch.Tensor: Interpolated flow of shape (target_t,).
    """
    f = flow_t32.unsqueeze(0).unsqueeze(0)  # (1, 1, 32)
    f_interp = F.interpolate(f, size=target_t, mode="linear", align_corners=True)
    return f_interp.squeeze(0).squeeze(0)  # (target_t,)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="V12 CLIP-only T=128 extraction")
    parser.add_argument("--config", type=str, default="configs/config_v12.yaml")
    parser.add_argument("--split", type=str, default=None, help="Train or Test")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    ext_cfg = config["extraction"]
    num_segments: int = ext_cfg["num_segments"]  # 128
    frames_per_seg: int = ext_cfg["frames_per_segment"]  # 5
    clip_model_name: str = ext_cfg["clip_model_name"]
    frame_extensions: list[str] = config["data"].get("frame_extensions", [".png", ".jpg"])

    # Source T=32 text features
    source_text_dir: str = ext_cfg.get("source_text_features", "data/features_v31_blip2_prompt")

    print(f"[INFO] Target segments: T={num_segments}")
    print(f"[INFO] Frames per segment: {frames_per_seg}")
    print(f"[INFO] Source text features: {source_text_dir}")

    # Load CLIP only (NO BLIP-2!)
    print(f"[INFO] Loading CLIP: {clip_model_name}")
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    feature_dim: int = config["model"]["feature_dim"]
    print(f"[INFO] Feature dim: {feature_dim}")

    raw_dir = Path(config["data"]["raw_dir"])
    features_dir = Path(config["data"]["features_dir"])
    splits = [args.split] if args.split else ["Train", "Test"]

    for split_name in splits:
        split_raw = raw_dir / split_name
        out_dir = features_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        source_text_split = Path(source_text_dir) / split_name

        print(f"\n{'='*60}")
        print(f" Processing split: {split_name}")
        print(f"{'='*60}")

        videos = discover_all_videos(split_raw, frame_extensions)
        if args.max_videos is not None:
            videos = videos[:args.max_videos]
            print(f"[INFO] Smoke-test mode: capped at {args.max_videos} videos")

        print(f"[INFO] Discovered {len(videos)} videos")
        skipped = 0
        text_interp_count = 0
        text_missing_count = 0

        for video_info in tqdm(videos, desc=f"Extracting [{split_name}]"):
            video_name: str = video_info["video_name"]
            label: int = video_info["label"]
            frame_paths: list[Path] = video_info["frames"]

            vis_out = out_dir / f"{video_name}_visual.pt"
            txt_out = out_dir / f"{video_name}_text.pt"
            flow_out = out_dir / f"{video_name}_flow.pt"
            lbl_out = out_dir / f"{video_name}_label.pt"

            # Resume: skip if visual already exists
            if args.resume and vis_out.exists() and txt_out.exists():
                skipped += 1
                continue

            # Handle short videos by allowing frame reuse (no skipping!)
            total_frames = len(frame_paths)
            if total_frames < 2:
                print(f"[SKIP] {video_name}: only {total_frames} frame(s)")
                continue

            # ---------------------------------------------------------------
            # Step 1: Sample frames for CLIP visual (T=128 segments)
            # Use np.linspace to spread segments evenly across available frames.
            # For short videos (< 128 frames), segments will share frames.
            # ---------------------------------------------------------------
            multi_indices: list[list[int]] = []
            center_indices: list[int] = []

            # Evenly-spaced segment centers across the video
            seg_centers = np.linspace(0, total_frames - 1, num_segments, dtype=float)

            for t in range(num_segments):
                center = seg_centers[t]
                # Segment boundaries (half-segment radius around center)
                half_seg = max(1, (total_frames / num_segments) / 2)
                seg_start = max(0, int(center - half_seg))
                seg_end = min(total_frames, int(center + half_seg) + 1)
                if seg_end <= seg_start:
                    seg_end = seg_start + 1

                positions = np.linspace(seg_start, seg_end - 1, frames_per_seg, dtype=int)
                positions = np.clip(positions, 0, total_frames - 1).tolist()
                multi_indices.append(positions)
                center_indices.append(int(np.clip(round(center), 0, total_frames - 1)))

            try:
                # Load segment frames as PIL for CLIP
                segment_pil_frames: list[list[Image.Image]] = []
                for t in range(num_segments):
                    five_pils = [
                        Image.open(frame_paths[idx]).convert("RGB")
                        for idx in multi_indices[t]
                    ]
                    segment_pil_frames.append(five_pils)

            except Exception as e:
                print(f"[ERROR] {video_name} frame load failed: {e}")
                continue

            # ---------------------------------------------------------------
            # Step 2: Extract CLIP visual features (T=128)
            # ---------------------------------------------------------------
            visual_feats: list[torch.Tensor] = []
            for t in range(num_segments):
                try:
                    feat = extract_visual_features(
                        segment_pil_frames[t], clip_model, clip_processor, device
                    )
                    visual_feats.append(feat)
                except Exception as e:
                    print(f"[ERROR] {video_name} seg {t}: {e}")
                    visual_feats.append(torch.zeros(feature_dim))

            visual_tensor = torch.stack(visual_feats)  # (128, 768)

            # ---------------------------------------------------------------
            # Step 3: Interpolate T=32 text features to T=128
            # ---------------------------------------------------------------
            source_txt = source_text_split / f"{video_name}_text.pt"
            if source_txt.exists():
                text_t32 = torch.load(source_txt, weights_only=True)  # (32, 768)
                text_tensor = interpolate_text_features(text_t32, num_segments)
                text_interp_count += 1
            else:
                # No source text — create zero tensor
                print(f"[WARN] No source text for {video_name}")
                text_tensor = torch.zeros(num_segments, feature_dim)
                text_missing_count += 1

            # ---------------------------------------------------------------
            # Step 4: Flow — interpolate from source or extract fresh
            # ---------------------------------------------------------------
            source_flow = source_text_split / f"{video_name}_flow.pt"
            if source_flow.exists():
                flow_t32 = torch.load(source_flow, weights_only=True)  # (32,)
                flow_tensor = interpolate_flow_features(flow_t32, num_segments)
            else:
                # Extract flow from center frames
                try:
                    center_frames_np = [
                        np.array(Image.open(frame_paths[ci]).convert("RGB"))
                        for ci in center_indices
                    ]
                    flow_tensor = extract_video_flow(center_frames_np)
                except Exception:
                    flow_tensor = torch.zeros(num_segments)

            # ---------------------------------------------------------------
            # Step 5: Save
            # ---------------------------------------------------------------
            torch.save(visual_tensor, vis_out)
            torch.save(text_tensor, txt_out)
            torch.save(flow_tensor, flow_out)
            torch.save(torch.tensor(label), lbl_out)

        print(f"\n[DONE] {split_name}: skipped {skipped}, "
              f"text interpolated {text_interp_count}, text missing {text_missing_count}")


if __name__ == "__main__":
    main()
