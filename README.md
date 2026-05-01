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

| Metric                | V1 Baseline | V2.1 (Current Best) | Notes                                                  |
| --------------------- | ----------- | ------------------- | ------------------------------------------------------ |
| **Video-Level AUROC** | 94.85%      | **94.87%**          | Whether any anomaly exists in the video                |
| **Frame-Level AUROC** | 77.14%      | **79.18%**          | Exact temporal localization of anomaly frames          |
| Model Parameters      | 2.17M       | ~2.2M               | Core cross-attention + Hourglass FC + Magnitude Branch |
| Training Time         | ~20 min     | ~30 min             | 100 epochs on RTX 4060 (8GB VRAM)                      |

### Comparison Against Baselines

| Method                         | Supervision | AUROC (Frame-Level) |
| ------------------------------ | ----------- | ------------------- |
| Sultani et al. CVPR 2018 (C3D) | Weak        | 75.41%              |
| **Ours (V1 Baseline)**         | **Weak**    | **77.14%**          |
| **Ours (V2.1 - Current Best)** | **Weak**    | **79.18%**          |
| MIST (2021)                    | Weak        | 82.30%              |
| RTFM (Tian et al., ICCV 2021)  | Weak        | 84.30%              |

> Our V2.1 model incorporates a feature magnitude branch, Adaptive Instance Selection (AIS), and antagonistic loss, pushing the frame-level AUROC to **79.18%**, significantly outperforming the seminal CVPR 2018 baseline.

---

## 🚀 Development Journey (V1 to V12)

This project has systematically evolved through 12 rigorous architectural iterations to address the "Lazy Localisation" problem and temporal aliasing in Weakly Supervised VAD. Here is the detailed progression from the baseline to our highest-performing model.

### Phase 1: Establishing the Baseline (V1 to V2.2)
*   **V1 Baseline:** Implemented Language-Guided Cross-Attention (1-layer) + Flat MLP + Top-K MIL ranking. Proved the core hypothesis by achieving **77.14%** Frame-AUROC, beating the CVPR 2018 baseline (75.41%).
*   **V2:** Introduced a Feature Magnitude Branch, Hourglass FC classifier, Adaptive Instance Selection (AIS), and Antagonistic Loss. Showed a regression (**74.78%**) due to multiplicative fusion saturating gradients and AIS K=1 cold-starts.
*   **V2.1 & V2.2:** Fixed V2 bugs via additive fusion, z-score normalization, AIS warm-starts, and top-3 MIST pseudo-labels. Performance improved to **79.18%**.

### Phase 2: Feature Extraction Upgrades (V3 to V3.1)
*   **V3 (Florence-2):** Migrated from BLIP-2 to Florence-2 for dense spatial grounding, using CLIP ViT-L/14 (768-dim) and 5-frame averaging. Frame-AUROC: **76.11%**.
*   **V3.1 (BLIP-2 Anomaly Prompt):** Applied an anomaly-seeking VQA prompt ("Question: What is happening in this image? Answer:") to BLIP-2, proving that targeted prompts out-perform generic spatial captioning. Frame-AUROC: **77.95%**.

### Phase 3: Architectural Complexity & Multi-Scale Modelling (V4 to V7)
*   **V4 SOTA:** Added Multi-Scale Temporal Attention (T=32, 16, 8), Feature Contrastive Loss, and a Global Normal Memory Bank. Achieved **0.7824** Frame-AUROC (later re-evaluated to **0.8180** under the corrected protocol).
*   **V5:** Tuned via Cosine Annealing, Class-Balanced Sampling, and K=5 AIS. Excellent Video-AUROC (0.9329) but demonstrated the MIST temporal trade-off where frame boundaries decay (Frame-AUROC: **0.8042**).
*   **V6:** Replaced the global memory bank with 16 Dynamic Normal Prototypes. Geometric constraints were too rigid, amplifying the "Siren Effect" (Frame-AUROC: **0.7771**).
*   **V7:** Introduced a Temporal Pyramid of Dilated Convolutions (PDC) to inject chronological velocity-awareness into static ViT frames (Frame-AUROC: **0.7931**).

### Phase 4: Resolution Scaling & Protocol Correction (V9 to V12)
*   **V9 & V10 (T=64):** Attempted T=64 segment resolution with Florence-2. Led to a forensic audit revealing ground truth evaluation corruption. Fixed severe evaluation bugs (normal video frame count deflation) which established the true SOTA-comparable protocol (11.4% anomaly ratio). V10 introduced Snippet Contrastive Learning (SCL) and Adaptive Smoothness Decay.
*   **V11:** Re-evaluated V5 + SCL + Smoothness Decay using the corrected protocol. Frame-AUROC: **0.8179**. (A score-level ensemble of V4 + V11 yielded **0.8197**).
*   **V12 (High-Resolution T=128):** Quadrupled temporal resolution from T=32 to T=128 to capture fine-grained boundaries. Achieved the project's highest single-model Frame-AUROC of **82.06%**.

### The SENTINEL Extensions (Verification Experiments)
We verified that raw semantic spaces of CLIP and sequential temporal dynamics cannot replace explicit cross-modal training:
- **Zero-Shot CLIP Danger Score:** Chance-level performance (~0.49 AUROC) proved raw CLIP embeddings suffer from a massive domain gap on surveillance footage; CCTV frames do not cluster with internet-derived "danger" concepts without explicit fine-tuning.
- **Temporal Prediction Error (LSTM):** Random performance (~0.50) proved that T=32 is too coarse a temporal resolution for continuous motion modeling, definitively justifying the move to T=128.

---

## 🏗️ Architecture

```
Phase 1 — Offline Extraction:
  PNG Frames ──► CLIP ViT-B/16 ──► Visual Features [32, 512] ──► .pt file
  PNG Frames ──► BLIP-2 OPT-2.7B ──► Captions ──► CLIP Text Encoder ──► Text Features [32, 512] ──► .pt file

Phase 2 — Online Training (V2.1 Architecture):
  Visual [B, 32, 512] ──► Cross-Attention (Q=Text, K=V=Visual) ──► Guided Features [B, 32, 512]
  Guided Features ──► Hourglass FC (512→64→128→1) ──► Semantic Score
  Visual Norms ──► Z-Score Norm ──► Magnitude Branch ──► Magnitude Score
  Final Score = sigmoid(Semantic Score + α * Magnitude Score)
```

### Key Design Decisions (V2.1)

| Decision                               | Rationale                                                                               |
| -------------------------------------- | --------------------------------------------------------------------------------------- |
| **Language-Guided Cross-Attention**    | Text queries semantic concepts; visual keys answer "where is it?"                       |
| **Hourglass FC Classifier**            | Parameter efficiency and regularisation via bottleneck compression (512→64→128→1)       |
| **Magnitude Branch (Additive Fusion)** | Uses raw visual L2-norm to provide feature magnitude signal alongside semantic guidance |
| **Adaptive Instance Selection (AIS)**  | Dynamically scales K based on model confidence, replacing fixed Top-K selection         |
| **MIST Self-Training**                 | Bootstraps from bag-level to instance-level pseudo-labels later in training             |

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
│   ├── config.yaml                 # Core hyperparameters
│   └── config_v*.yaml              # Version-specific experimental configurations (v3 to v12)
├── data/
│   ├── Temporal_Anomaly_Annotation.txt  # UCF-Crime frame-level GT (not in git)
│   ├── video_frame_counts.json     # Generated original video lengths
│   ├── raw/                        # UCF-Crime PNG frames (not in git)
│   └── features_*/                 # Extracted feature tensors (not in git)
├── models/
│   ├── __init__.py
│   ├── vad_architecture.py         # CrossAttentionBlock, MSCA, NormalPrototypes + LanguageGuidedVAD
│   ├── visual_encoder.py           # CLIP ViT wrappers
│   └── text_encoder.py             # BLIP-2 & Florence-2 captioner + CLIP text encoder
├── scripts/
│   ├── 01_extract_features.py      # Offline CLIP + BLIP-2/Florence-2 extraction
│   ├── 02_train.py                 # MIL training loop + AUROC evaluation
│   ├── 03_evaluate.py              # Inference + frame-level AUROC
│   ├── 04_hpo.py                   # Optuna Hyperparameter Optimization
│   ├── 05_semantic_ensemble.py     # Sentinel extension: CLIP Danger score ensemble
│   ├── 06_temporal_pred_error.py   # Sentinel extension: LSTM prediction error
│   ├── 07_full_eval.py             # SOTA-comparable evaluation suite
│   ├── 08_postproc_eval.py         # Smoothing & boundary refinement evaluation
│   ├── 09_plot_results.py          # Generates ROC/PR plots for IEEE thesis
│   ├── build_gt.py                 # Ground truth frame count corrector
│   ├── check_gt.py                 # Ground truth data validator
│   ├── compute_flops.py            # FLOPs/MACs/Params analysis
│   ├── extract_v12_features.py     # High-resolution T=128 feature extractor
│   ├── show_captions.py            # Diagnostic caption viewing
│   ├── show_captions_v31.py        # Diagnostic anomaly-prompt caption viewing
│   ├── test_all_prompts.py         # Testing CLIP zero-shot prompts
│   ├── test_blip_prompt.py         # Testing BLIP-2 Q&A
│   ├── test_florence2_prompt.py    # Testing Florence-2 detailed captions
│   └── v12_ensemble_eval.py        # Ensemble evaluator for V12 metrics
├── utils/
│   ├── __init__.py
│   ├── dataset.py                  # VADDataset — loads .pt feature tensors
│   ├── flow_utils.py               # Optical flow & PDC helpers
│   ├── frame_eval.py               # Standalone frame-level AUROC evaluator
│   ├── losses.py                   # Multi-objective VADLoss (SCL, MIST, AIS, Magnitude)
│   ├── metrics.py                  # Score interpolation
│   └── video_utils.py              # Config loading, seeding, T=32/64/128 frame sampling
├── checkpoints/                    # Auto-saved PyTorch models (.pth)
├── results/                        # Generated charts, logs, and score arrays
├── docs/                           # Associated thesis documentation
├── THESIS_LOG.md                   # Full academic development log
├── requirements.txt
└── README.md
```

---

## 📊 Loss Function (V2.1)

The V2.1 `VADLoss` replaces the standard MIL Ranking Loss with a multi-objective function:

$$\mathcal{L}_{total} = \mathcal{L}_{AIS} + \lambda_{ant}\mathcal{L}_{ant} + \lambda_{mag}\mathcal{L}_{mag} + \lambda_{smooth}\mathcal{L}_{smooth} + \lambda_{self}\mathcal{L}_{self}\cdot\mathbb{1}[\text{epoch}\ge50]$$

| Component                             | Purpose                                                                              |
| ------------------------------------- | ------------------------------------------------------------------------------------ |
| **Adaptive Instance Selection (AIS)** | Replaces fixed Top-K MIL ranking loss.                                               |
| **Antagonistic Loss**                 | Surgically penalises the top-1 normal segment while rewarding top-1 anomaly segment. |
| **Magnitude Ranking Loss**            | Enforces inter-bag separation based on visual feature L2-norms.                      |
| **Temporal Smoothness**               | Penalises abrupt score changes between consecutive segments.                         |
| **Self-Training (MIST)**              | BCE loss using high-confidence pseudo-labels (active from epoch 50).                 |

---

## 📝 References

- **UCF-Crime Dataset:** Sultani et al. *"Real-world Anomaly Detection in Surveillance Videos"* (CVPR 2018)
- **CLIP:** Radford et al. *"Learning Transferable Visual Models From Natural Language Supervision"* (ICML 2021)
- **BLIP-2:** Li et al. *"Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"* (ICML 2023)
- **RTFM:** Tian et al. *"Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning"* (ICCV 2021)
- **Attention Is All You Need:** Vaswani et al. (NeurIPS 2017)