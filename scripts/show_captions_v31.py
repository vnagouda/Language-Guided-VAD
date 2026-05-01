"""Caption Visualiser — shows BLIP-2 descriptions for one video.

Picks the first N segments of a chosen video, runs BLIP-2 captioning with an Anomaly prompt,
and prints the generated text alongside the segment index. Useful for
qualitatively verifying that the captioner produces meaningful anomaly descriptions.

Usage:
    python scripts/show_captions_v31.py
    python scripts/show_captions_v31.py --video Abuse028_x264 --segments 8
    python scripts/show_captions_v31.py --video Normal_Videos001_x264 --segments 8
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


def show_captions(video_name: str, num_segments: int = 8, start_segment: int = 0) -> None:
    """Load BLIP-2 and print captions for a range of segments of a video."""
    config = load_config("configs/config_v3.1_blip2_prompt.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # --- Find video directory ---
    raw_dir = Path(config["data"]["raw_dir"])
    candidates = list(raw_dir.rglob(video_name))
    if not candidates:
        candidates = list(raw_dir.rglob(f"{video_name}*"))
    if not candidates:
        print(f"[ERROR] Could not find video directory for: {video_name}")
        print(f"        Searched in: {raw_dir}")
        return

    video_dir = candidates[0]
    if video_dir.is_file():
        video_dir = video_dir.parent
    print(f"[INFO] Video dir: {video_dir}")

    # --- Load BLIP-2 ---
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    blip2_name = config["extraction"]["blip2_model_name"]
    print(f"[INFO] Loading BLIP-2: {blip2_name} in FP16 (may take a moment)...")

    processor = Blip2Processor.from_pretrained(blip2_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        blip2_name,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    print("[INFO] BLIP-2 loaded.\n")

    exts = {".jpg", ".jpeg", ".png"}

    def frame_number(p: Path) -> int:
        stem = p.stem
        try:
            return int(stem.rsplit("_", 1)[-1])
        except ValueError:
            return 0

    all_frames = sorted(
        [
            f for f in video_dir.iterdir()
            if f.suffix.lower() in exts and f.stem.startswith(video_name)
        ],
        key=frame_number,
    )
    total = len(all_frames)

    if all_frames:
        first_fn = frame_number(all_frames[0])
        last_fn  = frame_number(all_frames[-1])
        print(f"[INFO] Extracted frames: {total}  (frame {first_fn} → {last_fn})")

    # Annotation hint
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
    prompt = config["extraction"]["blip2_prompt"]

    for seg_idx in range(start_segment, min(start_segment + num_segments, T)):
        seg_start = int(seg_idx * total / T)
        seg_end   = int((seg_idx + 1) * total / T)
        seg_frames_all = all_frames[seg_start:seg_end]

        if not seg_frames_all:
            print(f"Segment {seg_idx+1:02d}: [NO FRAMES]")
            continue

        n = min(frames_per_seg, len(seg_frames_all))
        idxs = [int(i * (len(seg_frames_all) - 1) / max(n - 1, 1))
                for i in range(n)]
        sampled = [Image.open(seg_frames_all[i]).convert("RGB")
                   for i in idxs]

        mid_frame_path = seg_frames_all[len(seg_frames_all) // 2]
        pil_img = sampled[len(sampled) // 2]

        inputs = processor(images=pil_img, text=prompt, return_tensors="pt").to(device, torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=40)

        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        print(f"Segment {seg_idx+1:02d} | Frame: {mid_frame_path.name}")
        print(f"  Caption: {caption}")
        print()

    print("=" * 70)
    print("[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show BLIP-2 captions for one video")
    parser.add_argument(
        "--video",
        type=str,
        default="Abuse028_x264",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    show_captions(args.video, args.segments, args.start)
