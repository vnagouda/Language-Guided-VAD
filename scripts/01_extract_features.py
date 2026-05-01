"""Offline Feature Extraction V3: Florence-2 + CLIP ViT-L/14 + 5-frame + Patch + Flow.

Produces per-video ``.pt`` tensors in a uniquely-named directory:
    ``data/features_florence2_vitl14_5f_patch/``

Saved tensors per video:
    ``{video_name}_visual.pt``  → Tensor[32, 768]  (5-frame averaged patch features)
    ``{video_name}_text.pt``    → Tensor[32, 768]  (Florence-2 → CLIP text encode)
    ``{video_name}_flow.pt``    → Tensor[32]       (Farneback flow magnitude)
    ``{video_name}_label.pt``   → Tensor scalar    (0 = normal, 1 = anomalous)

Key design decisions (V3 vs baseline):
    - **Captioner**: Florence-2-large (``microsoft/florence-2-large``) replaces
      BLIP-2-OPT-2.7B. Florence-2's ``<MORE_DETAILED_CAPTION>`` task produces
      spatially-grounded, action-focused descriptions superior for VAD.
    - **CLIP backbone**: ViT-L/14 (768-dim) replaces ViT-B/16 (512-dim) for
      richer feature geometry and learnable magnitude discrimination.
    - **5-frame averaging**: 5 evenly-spaced frames per segment are encoded and
      averaged, capturing intra-segment motion rather than a single frozen snapshot.
    - **Patch tokens**: Mean of patch token hidden states (excludes CLS) gives
      spatially-aware features; local anomalous regions are not diluted.
    - **Optical flow**: Farneback inter-segment flow magnitude provides a direct
      motion signal that appearance-only features cannot capture.

VRAM budget (RTX 4060 8GB):
    Florence-2-large (fp16): ~1.5 GB
    CLIP ViT-L/14     (fp16): ~2.5 GB
    Activations + overhead:  ~1.5 GB
    Total:                   ~5.5 GB ✓ (safe headroom on 8 GB)

Usage:
    python scripts/01_extract_features.py --config configs/config_v3_florence2.yaml --split Train
    python scripts/01_extract_features.py --config configs/config_v3_florence2.yaml --split Test
    python scripts/01_extract_features.py --config configs/config_v3_florence2.yaml --split Test --max_videos 5  # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.flow_utils import extract_video_flow
from utils.video_utils import (
    discover_all_videos,
    load_config,
    sample_image_sequence_uniform,
    set_seed,
)


# ---------------------------------------------------------------------------
# Visual feature extraction (ViT-L/14 patch tokens, 5-frame average)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_visual_features(
    frames_5: list[Image.Image],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    """Extract CLIP ViT-L/14 visual features via patch token mean (5-frame average).

    For each of the 5 input frames:
        1. Run ViT vision encoder → last_hidden_state shape (1, 257, 1024)
           where dim-0 of the sequence is the CLS token.
        2. Exclude CLS token → patch tokens (1, 256, 1024).
        3. Mean-pool patches → (1, 1024).
        4. Apply CLIP visual projection → (1, 768).
    Then average the 5 projected vectors → (768,).

    Using patch tokens instead of the CLS token preserves local spatial
    information. Anomalous events (weapon, aggressive posture) occupy a
    fraction of the frame, and their contribution survives patch-mean better
    than the single global CLS summary vector.

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

    # ViT vision encoder — last_hidden_state: (5, seq_len, hidden_dim)
    # seq_len = 1 (CLS) + num_patches = 1 + 256 = 257 for ViT-L/14
    hidden: torch.Tensor = clip_model.vision_model(
        pixel_values=pixel_values
    ).last_hidden_state  # (5, 257, 1024)

    # Exclude CLS token (index 0) → patch tokens only
    patch_tokens: torch.Tensor = hidden[:, 1:, :]  # (5, 256, 1024)

    # Spatial mean-pool → (5, 1024)
    pooled: torch.Tensor = patch_tokens.mean(dim=1)

    # CLIP visual projection: 1024 → 768
    projected: torch.Tensor = clip_model.visual_projection(pooled)  # (5, 768)

    # Temporal average over 5 frames → (768,)
    return projected.mean(dim=0).cpu()


# ---------------------------------------------------------------------------
# Text feature extraction (Florence-2 caption → CLIP text encode)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_florence2_text(
    center_frame: Image.Image,
    florence_model: AutoModelForCausalLM,
    florence_processor: AutoProcessor,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    clip_tokenizer: CLIPTokenizer,
    task: str,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    """Generate Florence-2 caption for a frame, then CLIP-encode the caption.

    Pipeline:
        Image → Florence-2 (<MORE_DETAILED_CAPTION>) → natural language caption
              → CLIP text tokenizer → CLIP text encoder → (768,) embedding

    The ``<MORE_DETAILED_CAPTION>`` task produces spatially-aware, action-focused
    descriptions trained on the FLD-5B dataset (5.4B dense annotations). This
    generates descriptions like "a person aggressively grabbing another person
    by the collar near a vehicle" rather than generic object lists.

    Args:
        center_frame: Center PIL Image of the temporal segment.
        florence_model: Loaded Florence-2-large model (fp16, eval, on device).
        florence_processor: Corresponding AutoProcessor.
        clip_model: Loaded CLIP ViT-L/14 model for text encoding.
        clip_processor: Corresponding CLIPProcessor.
        clip_tokenizer: Corresponding CLIPTokenizer.
        task: Florence-2 task string (``"<MORE_DETAILED_CAPTION>"``).
        device: Torch device.

    Returns:
        tuple[torch.Tensor, str]:
            - text_feat: Shape (768,) on CPU — CLIP text embedding.
            - caption: The raw generated caption string (for logging/debug).
    """
    # --- Florence-2 caption generation ---
    f2_inputs = florence_processor(
        text=task,
        images=center_frame,
        return_tensors="pt",
    ).to(device, dtype=torch.float16)

    gen_ids: torch.Tensor = florence_model.generate(
        **f2_inputs,
        max_new_tokens=100,
        num_beams=3,
        do_sample=False,
    )
    raw_output: str = florence_processor.batch_decode(
        gen_ids, skip_special_tokens=False
    )[0]
    parsed: dict = florence_processor.post_process_generation(
        raw_output,
        task=task,
        image_size=(center_frame.width, center_frame.height),
    )
    caption: str = parsed[task]

    # --- CLIP text encode ---
    tok = clip_tokenizer(
        [caption],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    text_feat: torch.Tensor = clip_model.get_text_features(**tok).squeeze(0).cpu()
    # (768,)

    return text_feat, caption


@torch.no_grad()
def extract_blip2_text(
    center_frame: Image.Image,
    blip2_model: Blip2ForConditionalGeneration,
    blip2_processor: Blip2Processor,
    clip_model: CLIPModel,
    clip_tokenizer: CLIPTokenizer,
    prompt: str,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    """Generate BLIP-2 caption via a specific prompt, then CLIP-encode it.

    Args:
        center_frame: Center PIL Image of the temporal segment.
        blip2_model: Loaded BLIP-2 model.
        blip2_processor: Corresponding Blip2Processor.
        clip_model: Loaded CLIP ViT-L/14 model.
        clip_tokenizer: Corresponding CLIPTokenizer.
        prompt: Custom anomaly-seeking prompt string.
        device: Torch device.

    Returns:
        tuple[torch.Tensor, str]: (text_feat, caption)
    """
    inputs = blip2_processor(images=center_frame, text=prompt, return_tensors="pt").to(device, torch.float16)
    
    # Generate response
    generated_ids = blip2_model.generate(**inputs, max_new_tokens=40)
    caption = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # If the response is extremely short/empty due to "no anomaly", default string might be needed,
    # but theoretically clip won't crash on empty string, just gives zero-ish text features, 
    # though it's better to provide a default empty description padding.
    if not caption:
        caption = "Normal scene, no anomaly detected."

    # Encode with CLIP
    tok = clip_tokenizer(
        [caption],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    text_feat: torch.Tensor = clip_model.get_text_features(**tok).squeeze(0).cpu()

    return text_feat, caption

# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_features(
    config_path: str,
    resume: bool,
    split: str | None,
    max_videos: int | None = None,
) -> None:
    """Main V3 feature extraction pipeline.

    Saves per-video tensors to:
        ``{config.data.features_dir}/{split}/{video_name}_visual.pt``  (32, 768)
        ``{config.data.features_dir}/{split}/{video_name}_text.pt``    (32, 768)
        ``{config.data.features_dir}/{split}/{video_name}_flow.pt``    (32,)
        ``{config.data.features_dir}/{split}/{video_name}_label.pt``   scalar

    Args:
        config_path: Path to YAML config (e.g. ``configs/config_v3_florence2.yaml``).
        resume: If True, skip videos whose .pt files already exist.
        split: ``"Train"`` or ``"Test"``. If None, processes both.
        max_videos: Optional cap on number of videos (for smoke testing).
    """
    config = load_config(config_path)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    ext_cfg = config["extraction"]
    clip_model_name: str = ext_cfg["clip_model_name"]
    captioner: str = ext_cfg.get("captioner", "florence2")  # "florence2" or "blip2"
    
    num_segments: int = ext_cfg["num_segments"]
    frames_per_seg: int = ext_cfg["frames_per_segment"]
    use_patch_tokens: bool = ext_cfg.get("use_patch_tokens", True)
    extract_flow: bool = ext_cfg.get("extract_flow", True)
    frame_extensions: list[str] = config["data"]["frame_extensions"]

    florence_model = None
    florence_processor = None
    blip2_model = None
    blip2_processor = None
    task_or_prompt = ""

    if captioner == "florence2":
        florence_model_name: str = ext_cfg["florence2_model_name"]
        task_or_prompt = ext_cfg["florence2_task"]
        print(f"[INFO] Loading Florence-2: {florence_model_name}")
        florence_processor = AutoProcessor.from_pretrained(
            florence_model_name, trust_remote_code=True
        )
        florence_model = AutoModelForCausalLM.from_pretrained(
            florence_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device).eval()

    elif captioner == "blip2":
        blip2_model_name: str = ext_cfg["blip2_model_name"]
        task_or_prompt = ext_cfg["blip2_prompt"]
        print(f"[INFO] Loading BLIP-2: {blip2_model_name} in FP16")
        blip2_processor = Blip2Processor.from_pretrained(blip2_model_name)
        blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            blip2_model_name,
            torch_dtype=torch.float16,
        ).to(device).eval()
    else:
        raise ValueError(f"Unknown captioner: {captioner}")

    print(f"[INFO] Loading CLIP: {clip_model_name}")
    clip_model: CLIPModel = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor: CLIPProcessor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

    # -----------------------------------------------------------------------
    # Determine feature_dim from model
    # -----------------------------------------------------------------------
    feature_dim: int = config["model"]["feature_dim"]   # 768 for ViT-L/14
    print(f"[INFO] Feature dim: {feature_dim}")
    print(f"[INFO] Frames per segment: {frames_per_seg} (temporal averaging)")
    print(f"[INFO] Patch tokens: {use_patch_tokens}")
    print(f"[INFO] Optical flow: {extract_flow}")
    print(f"[INFO] Output dir: {config['data']['features_dir']}")

    raw_dir = Path(config["data"]["raw_dir"])
    features_dir = Path(config["data"]["features_dir"])
    splits = [split] if split else ["Train", "Test"]

    # -----------------------------------------------------------------------
    # Per-split extraction loop
    # -----------------------------------------------------------------------
    for split_name in splits:
        split_raw = raw_dir / split_name
        out_dir = features_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f" Processing split: {split_name}")
        print(f"{'='*60}")

        videos = discover_all_videos(split_raw, frame_extensions)
        if max_videos is not None:
            videos = videos[:max_videos]
            print(f"[INFO] Smoke-test mode: capped at {max_videos} videos")

        print(f"[INFO] Discovered {len(videos)} videos")
        skipped = 0

        for video_info in tqdm(videos, desc=f"Extracting [{split_name}]"):
            video_name: str = video_info["video_name"]
            label: int = video_info["label"]
            frame_paths: list[Path] = video_info["frames"]

            vis_out  = out_dir / f"{video_name}_visual.pt"
            txt_out  = out_dir / f"{video_name}_text.pt"
            flow_out = out_dir / f"{video_name}_flow.pt"
            lbl_out  = out_dir / f"{video_name}_label.pt"

            # --- Resume: skip if all outputs already exist ---
            if resume and vis_out.exists() and txt_out.exists() and flow_out.exists():
                skipped += 1
                continue

            # --- Skip videos too short for segmentation ---
            if len(frame_paths) < num_segments:
                print(f"[SKIP] {video_name}: {len(frame_paths)} frames < {num_segments}")
                continue

            # ---------------------------------------------------------------
            # Step 1: Sample frames
            # ---------------------------------------------------------------
            # For each of T=32 segments, sample frames_per_seg (5) frames
            # evenly within the segment + 1 centre frame for captioning & flow.
            total_frames = len(frame_paths)
            seg_size = total_frames // num_segments

            # Centre frame index for each segment (for captions & flow)
            center_indices: list[int] = [
                (t * seg_size) + seg_size // 2
                for t in range(num_segments)
            ]
            center_indices = [min(i, total_frames - 1) for i in center_indices]

            # Sample frames_per_seg evenly within each segment
            multi_indices: list[list[int]] = []
            for t in range(num_segments):
                seg_start = t * seg_size
                seg_end = min(seg_start + seg_size, total_frames)
                if seg_end <= seg_start:
                    seg_end = seg_start + 1
                positions = np.linspace(seg_start, seg_end - 1, frames_per_seg, dtype=int)
                positions = np.clip(positions, 0, total_frames - 1).tolist()
                multi_indices.append(positions)

            try:
                # Load centre frames as numpy arrays (for flow)
                center_frames_np: list[np.ndarray] = []
                for ci in center_indices:
                    img = Image.open(frame_paths[ci]).convert("RGB")
                    center_frames_np.append(np.array(img))

                # Load segment frames as PIL (for CLIP visual)
                segment_pil_frames: list[list[Image.Image]] = []
                for t in range(num_segments):
                    five_pils = [
                        Image.open(frame_paths[idx]).convert("RGB")
                        for idx in multi_indices[t]
                    ]
                    segment_pil_frames.append(five_pils)

                # Centre frames as PIL (for Florence-2 captioning)
                center_pil_frames: list[Image.Image] = [
                    Image.open(frame_paths[ci]).convert("RGB")
                    for ci in center_indices
                ]

            except Exception as e:
                print(f"[ERROR] {video_name} frame load failed: {e}")
                continue

            # ---------------------------------------------------------------
            # Step 2: Extract visual features (patch tokens, 5-frame avg)
            # ---------------------------------------------------------------
            visual_feats: list[torch.Tensor] = []
            for t in range(num_segments):
                try:
                    feat = extract_visual_features(
                        segment_pil_frames[t], clip_model, clip_processor, device
                    )
                    visual_feats.append(feat)
                except Exception as e:
                    print(f"[ERROR] {video_name} visual seg {t}: {e}")
                    visual_feats.append(torch.zeros(feature_dim))

            visual_tensor = torch.stack(visual_feats, dim=0)  # (32, 768)

            # ---------------------------------------------------------------
            # Step 3: Generate captions & extract text features (1 per segment)
            # ---------------------------------------------------------------
            text_feats: list[torch.Tensor] = []
            for t in range(num_segments):
                try:
                    if captioner == "florence2":
                        tfeat, _ = extract_florence2_text(
                            center_pil_frames[t],
                            florence_model, florence_processor,
                            clip_model, clip_processor, clip_tokenizer,
                            task_or_prompt, device,
                        )
                    else:
                        tfeat, _ = extract_blip2_text(
                            center_pil_frames[t],
                            blip2_model, blip2_processor,
                            clip_model, clip_tokenizer,
                            task_or_prompt, device,
                        )
                    text_feats.append(tfeat)
                except Exception as e:
                    print(f"[ERROR] {video_name} text seg {t}: {e}")
                    text_feats.append(torch.zeros(feature_dim))

            text_tensor = torch.stack(text_feats, dim=0)  # (32, 768)

            # ---------------------------------------------------------------
            # Step 4: Extract optical flow magnitude
            # ---------------------------------------------------------------
            if extract_flow:
                try:
                    flow_arr = extract_video_flow(center_frames_np, num_segments)
                except Exception as e:
                    print(f"[WARN] {video_name} flow failed: {e}")
                    flow_arr = np.zeros(num_segments, dtype=np.float32)
            else:
                flow_arr = np.zeros(num_segments, dtype=np.float32)

            flow_tensor = torch.tensor(flow_arr, dtype=torch.float32)  # (32,)

            # ---------------------------------------------------------------
            # Step 5: Save to disk (CPU tensors)
            # ---------------------------------------------------------------
            torch.save(visual_tensor, vis_out)
            torch.save(text_tensor, txt_out)
            torch.save(flow_tensor, flow_out)
            torch.save(torch.tensor(label, dtype=torch.long), lbl_out)

        if skipped > 0:
            print(f"[INFO] Skipped {skipped} already-extracted videos (--resume mode)")

    print("\n[DONE] V3 feature extraction complete.")
    print(f"[INFO] Features saved to: {features_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3 Offline Feature Extraction — Florence-2 + CLIP ViT-L/14",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_v3_florence2.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos whose features already exist (safe restart)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["Train", "Test"],
        help="Process only Train or Test. Default: both.",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Cap on videos to process (for smoke testing). E.g. --max_videos 5",
    )
    args = parser.parse_args()
    extract_features(args.config, args.resume, args.split, args.max_videos)
