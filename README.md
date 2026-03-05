# Language-Guided Weakly Supervised Video Anomaly Detection

This repository contains the code for an MSc Thesis project in Applied Machine Learning. The project proposes a novel framework for Video Anomaly Detection (VAD) under weak supervision (video-level labels only). 

Unlike traditional methods that rely exclusively on visual features (RGB/Optical Flow), this project leverages Vision-Language Models (VLMs) to generate semantic descriptions of video segments. These textual embeddings are then used to drive a **Cross-Attention** mechanism, explicitly guiding the visual network to focus on anomalous regions, thereby reducing context bias and false alarms.

## 📁 Repository Structure

```text
├── data/                    # Data directory (ignored by git)
│   ├── raw/                 # Raw .mp4 video files
│   ├── splits/              # Train/Test split lists
│   └── features/            # Pre-extracted visual and textual .pt tensors
├── models/                  # PyTorch model definitions
│   ├── cross_attention.py   # Core language-guided fusion module
│   └── vad_model.py         # The complete VAD architecture
├── utils/                   # Helper functions
│   ├── video_utils.py       # Video processing and segmentation
│   └── metrics.py           # AUROC evaluation scripts
├── configs/                 # YAML configuration files
│   └── config.yaml          # Hyperparameters
├── scripts/                 # Execution scripts
│   ├── 01_extract_features.py
│   ├── 02_train.py
│   └── 03_evaluate.py
├── notebooks/               # Jupyter notebooks for EDA and visualization
├── AGENT_INSTRUCTIONS.md    # Instructions for AI IDE assistants
├── PRD.md                   # Product Requirements Document & Architecture Specs
└── requirements.txt         # Python dependencies

🚀 Setup & Installation
Clone the repository:
code
Bash
git clone <your-repo-link>
cd Viru_VAD_Thesis
Create a virtual environment and install dependencies:
code
Bash
conda create -n vad_env python=3.9 -y
conda activate vad_env
pip install -r requirements.txt
Data Preparation:
Download the UCF-Crime dataset and place the videos in data/raw/.
Run the feature extraction pipeline (Requires GPU):
code
Bash
python scripts/01_extract_features.py --config configs/config.yaml
🧠 Training the Model
Once features are extracted into .pt files, you can train the model using the weakly supervised Top-K MIL loss:
code
Bash
python scripts/02_train.py --config configs/config.yaml
📊 Evaluation
To evaluate the model on the test set and calculate the frame-level AUROC:
code
Bash
python scripts/03_evaluate.py --weights path/to/saved/model.pth
📚 Core References
Sultani et al. (2019): Real-world anomaly detection in surveillance videos. (Foundational MIL baseline)
Tian et al. (2021): Weakly-supervised video anomaly detection with robust temporal feature magnitude learning (RTFM).
Zanella et al. (2024): Harnessing Large Language Models for Training-free Video Anomaly Detection (LAVAD).
Sun et al. (2026): Enhancing Weakly Supervised Multimodal Video Anomaly Detection through Text Guidance (TGMVAD).
code
Code
***

### Ready to Code

Your environment is structured flawlessly. Your AI agent has its rules (`AGENT_INSTRUCTIONS.md`) and its scientific blueprint (`PRD.md`). Your repository has a professional face (`README.md`). 

Now, let's get our hands dirty. Open your IDE. Navigate to `utils/video_utils.py`. 

Ask your IDE agent this exact prompt:
*"Based on the PRD, write a Python function in this file that takes a video path, uses `cv2` to get the total frame count, and returns a list of 32 lists, where each sub-list contains the frame indices for one of the 32 equal temporal segments. Include type hints and Google-style docstrings."*

Paste the resulting code here so I can review it.