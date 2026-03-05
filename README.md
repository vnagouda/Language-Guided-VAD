# Language-Guided Weakly Supervised Video Anomaly Detection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **MSc Thesis Project** - Applied Machine Learning, University of Surrey  
> **Author:** Viresh Nagouda

## 📖 Overview
Weakly Supervised Video Anomaly Detection (WS-VAD) aims to detect anomalous events in untrimmed surveillance videos using only video-level labels (normal/anomaly) during training. Current state-of-the-art methods rely solely on visual features, which often leads to **context bias** — confusing visually similar normal and abnormal scenes.

This project introduces a **Language-Guided Cross-Attention** framework. By leveraging vision-language models (BLIP-2 and CLIP), we explicitly guide the visual representation toward anomaly-relevant cues using semantic language descriptions.

## ✨ Key Features
- **Language Guidance:** Uses BLIP-2 captions and CLIP text embeddings to explicitly guide visual attention.
- **Cross-Attention Mechanism:** Text features act as Queries, while visual features act as Keys/Values, forcing the model to attend to semantically relevant visual cues.
- **Top-K MIL Ranking Loss:** Incorporates robust multiple-instance learning with temporal smoothness and sparsity regularizations.
- **Highly Modular:** Clean, deeply documented, configuration-driven PyTorch codebase.
- **Pre-extraction Pipeline:** Offline extraction of heavy visual and text features to `[32, 512]` point tensors, preventing GPU memory bottlenecks during MIL training.

---

## 🏗️ Architecture

### 1. Offline Feature Extraction
- **Visual:** UCF-Crime PNG frames $\rightarrow$ CLIP ViT-B/16 $\rightarrow$ Visual Features $\in \mathbb{R}^{32 \times 512}$
- **Text:** PNG frames $\rightarrow$ BLIP-2 Captioner $\rightarrow$ CLIP Text Encoder $\rightarrow$ Text Features $\in \mathbb{R}^{32 \times 512}$

### 2. Language-Guided VAD Model
$$
\begin{aligned}
Q &= T \cdot W_Q \quad \text{(Text Query)} \\
K &= V \cdot W_K \quad \text{(Visual Key)} \\
V &= V \cdot W_V \quad \text{(Visual Value)} \\
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
\end{aligned}
$$
The cross-attended features are then passed through an MLP to generate per-segment anomaly scores $\in [0, 1]$.

---

## ⚙️ Installation

### Prerequisites
- Python 3.12+
- NVIDIA GPU (Tested on RTX 4060 with 8GB+ VRAM)
- CUDA 12.4 compatible driver

### Setup Environment
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Language-Guided-VAD.git
cd Language-Guided-VAD

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# 3. Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Configuration
All hyperparameters (paths, model dims, learning rates) are centralized in `configs/config.yaml`. Modify this file before running any scripts.

### 2. Feature Extraction
Extract visual and text features offline to avoid memory bottlenecks during training. Supports resuming from interruptions.
```bash
python scripts/01_extract_features.py --split Train
python scripts/01_extract_features.py --split Test
```

### 3. Training
Train the Cross-Attention MIL model using the pre-extracted `.pt` features.
```bash
python scripts/02_train.py --config configs/config.yaml
```

### 4. Evaluation
Evaluate the model on the test set and calculate the frame-level AUROC.
*(Note: Requires the `Temporal_Anomaly_Annotation.txt` file in `data/`)*
```bash
python scripts/03_evaluate.py --checkpoint checkpoints/best_model.pth
```

---

## 📂 Project Structure

```text
Language-Guided-VAD/
├── configs/
│   └── config.yaml              # Centralized hyperparameters
├── data/
│   ├── raw/                     # UCF-Crime PNG frames
│   └── features/                # Pre-extracted .pt feature tensors
├── models/
│   ├── vad_architecture.py      # Cross-Attention + MLP classifier
│   ├── visual_encoder.py        # CLIP ViT wrapper
│   └── text_encoder.py          # BLIP-2 + CLIP text encoder
├── scripts/
│   ├── 01_extract_features.py   # Offline feature extraction pipeline
│   ├── 02_train.py              # MIL training loop
│   └── 03_evaluate.py           # Evaluation and AUROC computation
├── utils/
│   ├── dataset.py               # PyTorch Dataset for .pt features
│   ├── losses.py                # Top-K MIL Ranking Loss
│   ├── metrics.py               # AUROC and Score Interpolation
│   └── video_utils.py           # T=32 sampling & directory parsing
├── THESIS_LOG.md                # Comprehensive developmental log
├── requirements.txt             # Pip dependencies
└── README.md                    # Project documentation
```

---

## 📝 Acknowledgements
- **UCF-Crime Dataset:** Sultani et al. "Real-world Anomaly Detection in Surveillance Videos" (CVPR 2018)
- **CLIP:** Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021)
- **BLIP-2:** Li et al. "Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" (ICML 2023)