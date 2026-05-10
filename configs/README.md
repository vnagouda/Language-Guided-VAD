# Configuration Guide

This document explains every key in the YAML config files used for training, evaluation, and feature extraction.

---

## Top-level

| Key | Description |
|-----|-------------|
| `seed` | Random seed for reproducibility. Set in all scripts via `set_seed()`. Default: `42` |

---

## `data`

| Key | Description |
|-----|-------------|
| `raw_dir` | Path to raw extracted video frames (e.g. `data/raw`) |
| `features_dir` | Path to pre-extracted `.pt` feature files for Train/Test splits |
| `annotation_file` | Path to `Temporal_Anomaly_Annotation.txt` (UCF-Crime temporal labels) |
| `frame_extensions` | List of accepted image formats for frame extraction (e.g. `.png`, `.jpg`) |

---

## `extraction`

Controls how raw frames are turned into feature vectors.

| Key | Description |
|-----|-------------|
| `num_segments` | Number of temporal segments per video. Always `32` |
| `frames_per_segment` | Frames sampled per segment before averaging features |
| `clip_model_name` | HuggingFace CLIP model for visual features (e.g. `openai/clip-vit-large-patch14`) |
| `captioner` | Caption model to use — `blip2` or `florence2` |
| `blip2_model_name` | BLIP-2 model identifier on HuggingFace |
| `blip2_prompt` | Prompt passed to BLIP-2 during VQA-style captioning |
| `use_patch_tokens` | If `true`, uses patch-level CLIP tokens instead of CLS token |
| `extract_flow` | If `true`, computes optical flow magnitudes per segment |
| `image_size` | Input resolution for CLIP (224 for ViT models) |
| `extraction_batch_size` | Batch size during feature extraction (reduce if GPU OOM) |

---

## `model`

| Key | Description |
|-----|-------------|
| `feature_dim` | CLIP embedding dimension — `512` for ViT-B/16, `768` for ViT-L/14 |
| `num_segments` | Must match `extraction.num_segments`. Always `32` |
| `num_heads` | Number of attention heads in the cross-attention block |
| `num_layers` | Number of stacked cross-attention layers (single-scale mode only) |
| `dropout` | Dropout probability in attention and classifier layers |
| `ff_dim` | Hidden dimension of the feed-forward network (typically `4 × feature_dim`) |
| `classifier_bottleneck_dim` | Compression dimension in the hourglass classifier |
| `classifier_hidden_dim` | Expansion dimension in the hourglass classifier |
| `use_magnitude_branch` | Enables the L2-norm magnitude scoring branch |
| `use_flow_in_magnitude` | If `true`, magnitude branch takes both visual norm and optical flow |
| `use_multi_scale` | Enables multi-scale cross-attention at T=32, T=16, T=8 (V4 default) |
| `memory_bank_size` | Size of the FIFO normal feature memory bank (V4) |
| `num_prototypes` | Number of learnable normal prototype cluster centres (V6) |

---

## `loss`

### AIS (Adaptive Instance Selection)

| Key | Description |
|-----|-------------|
| `ais_score_threshold` | Score threshold `r` for counting high-confidence anomaly segments |
| `ais_k_min` | Minimum K floor after warm-start ends |
| `ais_warm_start_epochs` | Epochs to use fixed `ais_warm_k` before adaptive K kicks in |
| `ais_warm_k` | Fixed K used during the warm-start phase |

### Loss weights

| Key | Description |
|-----|-------------|
| `lambda_magnitude` | Weight for the magnitude ranking loss (RTFM/MGFN) |
| `margin_magnitude` | Hinge margin for magnitude ranking |
| `lambda_antagonistic` | Weight for the antagonistic loss (Light-WVAD) |
| `lambda_smooth` | Weight for temporal smoothness penalty |
| `lambda_contrastive` | Weight for feature contrastive loss (V4) |
| `margin_contrastive` | Hinge margin for feature contrastive loss |
| `lambda_bank` | Weight for memory bank contrastive loss (V4) |
| `margin_bank` | Hinge margin for memory bank contrastive loss |
| `lambda_prototype_cluster` | Weight for normal prototype clustering loss (V6) |
| `lambda_prototype_sep` | Weight for anomaly-prototype separation loss (V6) |
| `margin_prototype` | Hinge margin for prototype separation |
| `lambda_snippet_contrastive` | Weight for snippet contrastive learning — V10 novel contribution |
| `snippet_margin` | Margin for snippet contrastive loss |
| `smooth_decay_rate` | Exponential decay rate applied to `lambda_smooth` after MIST starts |

---

## `training`

| Key | Description |
|-----|-------------|
| `batch_size` | Number of videos per batch |
| `epochs` | Total training epochs |
| `learning_rate` | Initial learning rate for AdamW |
| `weight_decay` | L2 regularisation coefficient |
| `num_workers` | DataLoader worker processes (0 = main thread, increase for faster loading) |
| `gradient_clip_max_norm` | Max norm for gradient clipping |
| `checkpoint_dir` | Directory to save model checkpoints |
| `log_interval` | Print loss every N batches |
| `eval_frame_level_every` | Evaluate frame-level AUROC every N epochs |
| `phase2_lr` | Constant learning rate used once MIST (Phase 2) starts |
| `lambda_smooth_phase2` | Smoothness weight override during Phase 2 |
| `class_balanced_sampling` | If `true`, uses WeightedRandomSampler to balance anomaly categories |

### `training.lr_scheduler`

| Key | Description |
|-----|-------------|
| `type` | Scheduler type: `cosine_warm`, `cosine`, or `step` |
| `T_0` | Restart period for `cosine_warm` |
| `T_mult` | Period multiplier after each restart for `cosine_warm` |
| `T_max` | Max epochs for `cosine` scheduler |
| `eta_min` | Minimum learning rate |
| `step_size` | Step size for `step` scheduler |
| `gamma` | Decay factor for `step` scheduler |

### `training.mist`

| Key | Description |
|-----|-------------|
| `start_epoch` | Epoch at which Phase 2 (MIST self-training) begins |
| `lambda_self` | Weight for the MIST self-training BCE loss |
| `pseudo_k` | Number of top-scoring segments marked as pseudo-positive per video |

---

## `evaluation`

| Key | Description |
|-----|-------------|
| `results_dir` | Directory to save score curves and evaluation outputs |
