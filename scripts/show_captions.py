"""Caption Visualiser — shows Florence-2 descriptions for one video.

Picks the first N segments of a chosen video, runs Florence-2 captioning,
and prints the generated text alongside the segment index.  Useful for
qualitatively verifying that the captioner produces meaningful descriptions.

Usage:
    python scripts/show_captions.py
    python scripts/show_captions.py --video Abuse028_x264 --segments 8
    python scripts/show_captions.py --video Normal_Videos001_x264 --segments 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.video_utils import load_config


def sample_frames(video_dir: Path, num_frames: int = 5) -> list[Image.Image]:
    """Sample up to num_frames evenly-spaced frames from a video directory."""
    exts = {".jpg", ".jpeg", ".png"}
    frames = sorted(
        [f for f in video_dir.iterdir() if f.suffix.lower() in exts],
        key=lambda p: p.name,
    )
    if not frames:
        return []
    idxs = [int(i * (len(frames) - 1) / max(num_frames - 1, 1))
            for i in range(num_frames)]
    return [Image.open(frames[i]).convert("RGB") for i in idxs]


def show_captions(video_name: str, num_segments: int = 8, start_segment: int = 0) -> None:
    """Load Florence-2 and print captions for a range of segments of a video.

    Args:
        video_name: Name of the video directory (without extension).
        num_segments: How many consecutive segments to caption.
        start_segment: Which segment index (0-based) to start from.
    """
    config = load_config("configs/config_v3_florence2.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # --- Find video directory ---
    raw_dir = Path(config["data"]["raw_dir"])
    candidates = list(raw_dir.rglob(video_name))
    if not candidates:
        # Try with common suffixes
        candidates = list(raw_dir.rglob(f"{video_name}*"))
    if not candidates:
        print(f"[ERROR] Could not find video directory for: {video_name}")
        print(f"        Searched in: {raw_dir}")
        return

    video_dir = candidates[0]
    if video_dir.is_file():
        video_dir = video_dir.parent
    print(f"[INFO] Video dir: {video_dir}")

    # --- Load Florence-2 ---
    from transformers import AutoProcessor, AutoModelForCausalLM

    florence_name = config["extraction"]["florence2_model_name"]
    print(f"[INFO] Loading Florence-2: {florence_name} (may take 30s)...")

    processor = AutoProcessor.from_pretrained(
        florence_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        florence_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    print("[INFO] Florence-2 loaded.\n")

    # --- Discover frames belonging to THIS video only, sorted NUMERICALLY ---
    # UCF-Crime frames are named {video_name}_{frame_number}.png.
    # Lexicographic sort (default) is WRONG: "1010" < "110" as a string.
    # Extract the integer frame number and sort by numeric value.
    exts = {".jpg", ".jpeg", ".png"}

    def frame_number(p: Path) -> int:
        """Extract integer frame number from filename stem suffix."""
        stem = p.stem  # e.g. "Abuse028_x264_1010"
        # The frame number is everything after the last underscore
        try:
            return int(stem.rsplit("_", 1)[-1])
        except ValueError:
            return 0

    all_frames = sorted(
        [
            f for f in video_dir.iterdir()
            if f.suffix.lower() in exts and f.stem.startswith(video_name)
        ],
        key=frame_number,   # ← numeric sort, not lexicographic
    )
    total = len(all_frames)

    # Show what original frame numbers we have
    if all_frames:
        first_fn = frame_number(all_frames[0])
        last_fn  = frame_number(all_frames[-1])
        print(f"[INFO] Extracted frames: {total}  (frame {first_fn} → {last_fn})")

    # Annotation hint: which segments map to frame range [ann_start, ann_end]?
    # Read from annotation file if available
    try:
        ann_file = Path("data/Temporal_Anomaly_Annotation.txt")
        if ann_file.exists():
            for line in ann_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    vname = parts[0].replace(".mp4", "")
                    if vname == video_name:
                        vals = [int(x) for x in parts[-4:]]
                        s1, e1, s2, e2 = vals
                        if s1 >= 0 and total > 0:
                            # Map annotation frames to segment indices
                            all_fnums = [frame_number(f) for f in all_frames]
                            anom_idxs = [i for i, fn in enumerate(all_fnums) if s1 <= fn <= e1]
                            if anom_idxs:
                                seg_s = int(anom_idxs[0] * 32 / total)
                                seg_e = int(anom_idxs[-1] * 32 / total)
                                print(f"[INFO] Annotation: anomaly frames {s1}–{e1} "
                                      f"→ approx segments {seg_s+1}–{seg_e+1} of 32")
    except Exception:
        pass
    print(f"[INFO] Total frames in video: {total}")
    print(f"[INFO] Showing segments {start_segment+1}–{start_segment+num_segments} of 32\n")
    print("=" * 70)

    T = 32
    frames_per_seg = config["extraction"]["frames_per_segment"]

    for seg_idx in range(start_segment, min(start_segment + num_segments, T)):
        # Map segment index → frame indices (same logic as extraction script)
        seg_start = int(seg_idx * total / T)
        seg_end   = int((seg_idx + 1) * total / T)
        seg_frames_all = all_frames[seg_start:seg_end]

        if not seg_frames_all:
            print(f"Segment {seg_idx+1:02d}: [NO FRAMES]")
            continue

        # Sample up to frames_per_seg from this segment
        n = min(frames_per_seg, len(seg_frames_all))
        idxs = [int(i * (len(seg_frames_all) - 1) / max(n - 1, 1))
                for i in range(n)]
        sampled = [Image.open(seg_frames_all[i]).convert("RGB")
                   for i in idxs]

        # Use middle frame as representative for display
        mid_frame_path = seg_frames_all[len(seg_frames_all) // 2]

        # --- Florence-2 caption ---
        # Use the middle frame for captioning (same as extraction)
        pil_img = sampled[len(sampled) // 2]
        prompt  = "<MORE_DETAILED_CAPTION>"

        inputs = processor(text=prompt, images=pil_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
            )

        raw_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = processor.post_process_generation(
            raw_text,
            task=prompt,
            image_size=pil_img.size,
        )
        caption = parsed.get(prompt, raw_text).strip()

        print(f"Segment {seg_idx+1:02d} | Frame: {mid_frame_path.name}")
        print(f"  Caption: {caption}")
        print()

    print("=" * 70)
    print("[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show Florence-2 captions for one video")
    parser.add_argument(
        "--video",
        type=str,
        default="Abuse028_x264",
        help="Video directory name (default: Abuse028_x264)",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=8,
        help="Number of segments to caption (default: 8)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start from this segment index (0-based, default: 0)",
    )
    args = parser.parse_args()
    show_captions(args.video, args.segments, args.start)
