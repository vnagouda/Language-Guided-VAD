"""Offline Feature Extraction: CLIP Visual + BLIP-2/CLIP Text features.

This script processes the UCF-Crime dataset (stored as directories of PNG frames)
and produces per-video ``.pt`` tensors:
    - ``{video_name}_visual.pt``  →  Tensor[32, 512]
    - ``{video_name}_text.pt``    →  Tensor[32, 512]
    - ``{video_name}_label.pt``   →  Tensor scalar (0 or 1)

Key design decisions:
    - All inference runs in ``torch.no_grad()`` + ``.eval()`` mode.
    - Tensors are moved to CPU before saving to prevent GPU OOM.
    - Supports ``--resume`` to skip already-extracted videos.
    - Supports ``--split`` to process Train or Test individually.

Usage:
    python scripts/01_extract_features.py
    python scripts/01_extract_features.py --resume --split Train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

# Add project root to sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.video_utils import (
    load_config,
    set_seed,
    discover_all_videos,
    sample_image_sequence_uniform,
)


def extract_features(config_path: str, resume: bool, split: str | None) -> None:
    """Main feature extraction pipeline.

    Args:
        config_path: Path to the YAML configuration file.
        resume: If True, skip videos whose features already exist.
        split: Process only ``"Train"`` or ``"Test"``.  If None, process both.
    """
    config = load_config(config_path)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- Load CLIP (full model for both visual and text features) ---
    clip_model_name: str = config["extraction"]["clip_model_name"]
    print(f"[INFO] Loading CLIP model: {clip_model_name}")
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    clip_model.eval()

    # --- Optionally load BLIP-2 for captioning ---
    blip2_model_name: str = config["extraction"]["blip2_model_name"]
    try:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        print(f"[INFO] Loading BLIP-2 model: {blip2_model_name}")
        blip2_processor = Blip2Processor.from_pretrained(blip2_model_name)
        blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            blip2_model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        blip2_model.eval()
        use_blip2 = True
        print("[INFO] BLIP-2 loaded successfully — will generate captions.")
    except Exception as e:
        print(f"[WARN] Could not load BLIP-2 ({e}). "
              f"Falling back to class-name-based text prompts.")
        use_blip2 = False

    # --- Determine splits to process ---
    raw_dir = Path(config["data"]["raw_dir"])
    features_dir = Path(config["data"]["features_dir"])
    num_segments: int = config["extraction"]["num_segments"]
    frame_extensions: list[str] = config["data"]["frame_extensions"]

    splits = [split] if split else ["Train", "Test"]

    for split_name in splits:
        split_dir = raw_dir / split_name
        out_dir = features_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing split: {split_name}")
        print(f"{'='*60}")

        videos = discover_all_videos(split_dir, frame_extensions)
        print(f"[INFO] Discovered {len(videos)} videos")

        for video_info in tqdm(videos, desc=f"Extracting [{split_name}]"):
            video_name: str = video_info["video_name"]
            label: int = video_info["label"]
            class_name: str = video_info["class_name"]
            frame_paths: list[Path] = video_info["frames"]

            # Check for resume
            vis_out = out_dir / f"{video_name}_visual.pt"
            txt_out = out_dir / f"{video_name}_text.pt"
            lbl_out = out_dir / f"{video_name}_label.pt"

            if resume and vis_out.exists() and txt_out.exists():
                continue

            # --- Skip videos with too few frames ---
            if len(frame_paths) < num_segments:
                print(f"[SKIP] {video_name}: only {len(frame_paths)} frames "
                      f"(need {num_segments})")
                continue

            # --- Sample T=32 representative frames ---
            try:
                sampled_frames_np = sample_image_sequence_uniform(
                    frame_paths, num_segments
                )
            except RuntimeError as e:
                print(f"[ERROR] {video_name}: {e}")
                continue

            # Convert to PIL images for HuggingFace processors
            pil_images = [Image.fromarray(f) for f in sampled_frames_np]

            # --- Extract Visual Features ---
            with torch.no_grad():
                vis_inputs = clip_processor(
                    images=pil_images, return_tensors="pt", padding=True
                )
                pixel_values = vis_inputs["pixel_values"].to(device)
                visual_outputs = clip_model.get_image_features(
                    pixel_values=pixel_values
                )
                if isinstance(visual_outputs, torch.Tensor):
                    visual_features = visual_outputs
                elif hasattr(visual_outputs, "image_embeds") and visual_outputs.image_embeds is not None:
                    visual_features = visual_outputs.image_embeds
                elif hasattr(visual_outputs, "pooler_output") and visual_outputs.pooler_output is not None:
                    visual_features = visual_outputs.pooler_output
                else:
                    visual_features = visual_outputs[0]
                    
                if visual_features.shape[-1] != 512 and hasattr(clip_model, "visual_projection"):
                    visual_features = clip_model.visual_projection(visual_features)
                
                visual_features = visual_features.cpu()

            # --- Generate Captions & Extract Text Features ---
            if use_blip2:
                # Generate captions with BLIP-2 (one per segment)
                captions: list[str] = []
                for pil_img in pil_images:
                    with torch.no_grad():
                        b2_inputs = blip2_processor(
                            images=pil_img, return_tensors="pt"
                        ).to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
                        gen_ids = blip2_model.generate(**b2_inputs, max_new_tokens=50)
                        cap = blip2_processor.batch_decode(
                            gen_ids, skip_special_tokens=True
                        )[0].strip()
                    captions.append(cap)
            else:
                # Fallback: use class-name-based prompts
                prompt = (
                    f"A surveillance video showing {class_name.lower()} activity."
                    if label == 1
                    else "A surveillance video showing normal activity."
                )
                captions = [prompt] * num_segments

            # Encode captions with CLIP text encoder
            with torch.no_grad():
                text_inputs = clip_tokenizer(
                    captions,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(device)
                text_outputs = clip_model.get_text_features(**text_inputs)
                if isinstance(text_outputs, torch.Tensor):
                    text_features = text_outputs
                elif hasattr(text_outputs, "text_embeds") and text_outputs.text_embeds is not None:
                    text_features = text_outputs.text_embeds
                elif hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
                    text_features = text_outputs.pooler_output
                else:
                    text_features = text_outputs[0]
                    
                if text_features.shape[-1] != 512 and hasattr(clip_model, "text_projection"):
                    text_features = clip_model.text_projection(text_features)
                    
                text_features = text_features.cpu()

            # --- Save tensors ---
            torch.save(visual_features, vis_out)
            torch.save(text_features, txt_out)
            torch.save(torch.tensor(label), lbl_out)

    print("\n[DONE] Feature extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline feature extraction for Language-Guided WS-VAD"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos whose features already exist",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["Train", "Test"],
        help="Process only a specific split",
    )
    args = parser.parse_args()
    extract_features(args.config, args.resume, args.split)
