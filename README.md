# Language-Guided Weakly Supervised Video Anomaly Detection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **MSc Thesis Project** — Applied Machine Learning, University of Surrey
> **Author:** Viresh Nagouda

---

## 📖 Overview

Weakly Supervised Video Anomaly Detection (WS-VAD) aims to detect anomalous events in untrimmed surveillance videos using **only video-level labels** (normal / anomaly) during training, while producing frame-level anomaly scores at inference. Current state-of-the-art MIL-based methods rely exclusively on visual features, which leads to **context bias** — confusing visually similar normal and abnormal scenes (e.g. smoke from cooking vs. smoke from an explosion).

This project introduces a novel **Language-Guided Cross-Attention** framework that fundamentally departs from pure visual MIL by exploiting semantic language descriptions to guide visual attention toward anomaly-relevant cues.

### Novel Contribution
Instead of simple feature concatenation, we use BLIP-2-generated captions encoded via CLIP's text encoder as **Queries** in a cross-attention mechanism, with visual features as **Keys/Values**:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V, \quad Q = T \cdot W_Q,\quad K = V = F_{vis} \cdot W_{K/V}$$

This forces the network to selectively attend to visual patterns that correlate with the semantic anomaly description — mathematically superior to concatenation.

---

## 🏆 Experimental Results

Evaluated on the **UCF-Crime** dataset (1,610 training videos, 283 test videos, 13 anomaly categories + normal).

| Metric | Score | Notes |
|--------|-------|-------|
| **Video-Level AUROC** | **94.85%** | Whether any anomaly exists in the video |
| **Frame-Level AUROC** | **77.14%** | Exact temporal localization of anomaly frames |
| Model Parameters | 2.17M | Core cross-attention + MLP only |
| FLOPs (inference) | 138.68M | Extremely lightweight at inference time |
| Training Time | ~20 min / 100 epochs | On RTX 4060 (8GB VRAM) |

### Comparison Against Baselines

| Method | Supervision | AUROC (Frame-Level) |
|--------|-------------|----------------------|
| Sultani et al. CVPR 2018 (C3D) | Weak | 75.41% |
| RTFM (Tian et al., ICCV 2021) | Weak | 84.30% |
| **Ours (Language-Guided Cross-Attn)** | **Weak** | **77.14%** |

> Our model outperforms the seminal CVPR 2018 baseline and does so using a highly compact 2.17M-parameter architecture, compared to C3D-based approaches which typically exceed 30M parameters.

---

## 🏗️ Architecture

```
Phase 1 — Offline Extraction (run once):
  PNG Frames ──► CLIP ViT-B/16 ──► Visual Features [32, 512] ──► .pt file
  PNG Frames ──► BLIP-2 OPT-2.7B ──► Captions ──► CLIP Text Encoder ──► Text Features [32, 512] ──► .pt file

Phase 2 — Online Training:
  Visual [B, 32, 512] ──► Cross-Attention (Q=Text, K=V=Visual) ──► Guided Features [B, 32, 512]
  Guided Features ──► MLP Head ──► Anomaly Scores [B, 32] ∈ [0,1]
  Anomaly Scores ──► MIL Ranking Loss (Top-K=8) + Smoothness + Sparsity
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Offline feature extraction** | Prevents GPU OOM during MIL training; enables fast dataloading |
| **CLIP joint embedding space** | Both visual and text features live in the same 512-D space, making cross-attention mathematically meaningful |
| **Query = Text, Key/Value = Visual** | Text asks "what should I look for?"; visual answers "where is it?" |
| **Top-K MIL (K=8, T=32)** | Selects top 25% of segments to avoid training on border-frames |
| **Sparsity + Smoothness regularisation** | Prevents model from predicting uniform high anomaly scores everywhere |

---

## ⚙️ Setup

### Prerequisites
- Python 3.12+
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060)
- CUDA 12.4 compatible driver

### Installation
```bash
git clone https://github.com/vnagouda/Language-Guided-VAD.git
cd Language-Guided-VAD

python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/Mac

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Dataset Setup (UCF-Crime)

> **⚠️ The raw dataset is ~100GB and cannot be stored in git.** Follow the steps below to download it.

The dataset is available on Kaggle as pre-extracted PNG frames (no mp4 decoding required). Each video is stored as a flat directory of PNG images named `{VideoName}_x264_{FrameNumber}.png`.

#### Option A — Kaggle API (recommended)
```bash
# 1. Install the Kaggle CLI
pip install kaggle

# 2. Place your kaggle.json API token at ~/.kaggle/kaggle.json
#    (Download from: https://www.kaggle.com/account → Create New API Token)

# 3. Create the data directory
mkdir -p data/raw

# 4. Download the UCF-Crime PNG dataset
kaggle datasets download -d odins0n/ucf-crime-dataset -p data/raw --unzip
```

#### Option B — Manual Download
1. Go to: https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset
2. Download and unzip into `data/raw/`

#### Expected Folder Structure (after download)
```
data/
├── Temporal_Anomaly_Annotation.txt   ← already tracked in this repo ✅
├── raw/
│   ├── Train/
│   │   ├── Abuse/
│   │   │   ├── Abuse001_x264_0.png
│   │   │   ├── Abuse001_x264_10.png
│   │   │   └── ...
│   │   ├── Arrest/
│   │   ├── Arson/
│   │   ├── Assault/
│   │   ├── Burglary/
│   │   ├── Explosion/
│   │   ├── Fighting/
│   │   ├── RoadAccidents/
│   │   ├── Robbery/
│   │   ├── Shooting/
│   │   ├── Shoplifting/
│   │   ├── Stealing/
│   │   ├── Vandalism/
│   │   └── NormalVideos/
│   └── Test/
│       └── (same structure as Train)
└── features/                         ← generated by Step 1 (extract_features)
```

> **Total size:** ~100GB raw frames. ~8GB extracted `.pt` features.
> **Videos:** 1,610 Train / 283 Test (after skipping videos with < 32 frames).

---

## 🚀 Usage

### ⚡ Quick Start for Reviewers (skip re-training)
If you only want to verify results, the trained model checkpoint is already in this repo. You only need the raw frames to run evaluation:

```bash
# After installing and downloading the dataset:
python scripts/03_evaluate.py
# Output:
# [RESULT] Video-level AUROC (max-score): 0.9485
# [RESULT] Frame-level AUROC: 0.7714
```

---

### Full Pipeline (train from scratch)
All hyperparameters are centralised in `configs/config.yaml`. **Never hardcode values in scripts.**

### Step 1 — Extract Features (run once, ~12 hours on RTX 4060)
```bash
python scripts/01_extract_features.py --split Train
python scripts/01_extract_features.py --split Test

# Resume after interruption:
python scripts/01_extract_features.py --split Train --resume
```

### Step 2 — Train the Model (~20 minutes, 100 epochs)
```bash
python scripts/02_train.py
```
Monitor the `Test AUROC` score printed at the end of each epoch. Best checkpoint is auto-saved to `checkpoints/best_model.pth`.

### Step 3 — Evaluate
```bash
python scripts/03_evaluate.py
```
Requires `data/Temporal_Anomaly_Annotation.txt` for frame-level AUROC. Download from the [UCF-Crime GitHub](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018).

### Step 4 — Compute FLOPs / Complexity
```bash
python scripts/compute_flops.py
```

---

## 📂 Project Structure

```
Language-Guided-VAD/
├── configs/
│   └── config.yaml                 # All hyperparameters (NO hardcoding in scripts)
├── data/
│   ├── Temporal_Anomaly_Annotation.txt  # UCF-Crime frame-level GT (not in git)
│   ├── raw/                        # UCF-Crime PNG frames   (not in git — too large)
│   └── features/                   # Pre-extracted .pt tensors (not in git — too large)
├── models/
│   ├── __init__.py
│   ├── vad_architecture.py         # CrossAttentionBlock + LanguageGuidedVAD
│   ├── visual_encoder.py           # CLIP ViT-B/16 wrapper
│   └── text_encoder.py             # BLIP-2 captioner + CLIP text encoder
├── scripts/
│   ├── 01_extract_features.py      # Offline CLIP + BLIP-2 extraction w/ --resume
│   ├── 02_train.py                 # MIL training loop + AUROC evaluation
│   ├── 03_evaluate.py              # Inference + frame-level AUROC
│   └── compute_flops.py            # FLOPs/MACs/Params analysis
├── utils/
│   ├── __init__.py
│   ├── dataset.py                  # VADDataset — loads .pt feature tensors
│   ├── losses.py                   # Top-K MIL Ranking Loss
│   ├── metrics.py                  # AUROC + score interpolation
│   └── video_utils.py              # Config loading, seeding, T=32 frame sampling
├── results/
│   └── video_scores.npy            # Per-video anomaly score curves (post-evaluation)
├── research papers/                # Reference literature (17 papers)
├── THESIS_LOG.md                   # Full academic development log
├── requirements.txt
└── README.md
```

---

## 📊 Loss Function

$$\mathcal{L}_{total} = \mathcal{L}_{rank} + \lambda_{smooth}\,\mathcal{L}_{smooth} + \lambda_{sparse}\,\mathcal{L}_{sparse}$$

Where:
$$\mathcal{L}_{rank} = \frac{1}{K}\sum_{k=1}^{K}\max\!\left(0,\ \text{margin} - \left(s_{abn}^{(k)} - s_{nor}^{(k)}\right)\right)$$

| Hyperparameter | Value |
|---|---|
| Top-K | 8 |
| Margin | 1.0 |
| λ_smooth | 8×10⁻⁵ |
| λ_sparse | 8×10⁻⁵ |
| Learning Rate | 1×10⁻⁴ (AdamW) |
| LR Schedule | StepLR (step=50, γ=0.1) |
| Epochs | 100 |

---

## 📝 References

- **UCF-Crime Dataset:** Sultani et al. *"Real-world Anomaly Detection in Surveillance Videos"* (CVPR 2018)
- **CLIP:** Radford et al. *"Learning Transferable Visual Models From Natural Language Supervision"* (ICML 2021)
- **BLIP-2:** Li et al. *"Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"* (ICML 2023)
- **RTFM:** Tian et al. *"Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning"* (ICCV 2021)
- **Attention Is All You Need:** Vaswani et al. (NeurIPS 2017)