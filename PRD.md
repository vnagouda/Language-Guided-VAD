# Product Requirements Document (PRD): Language-Guided Weakly Supervised VAD

## 1. Project Overview
**Title:** Semantic Guidance is All You Need: Language-Driven Cross-Attention for Weakly Supervised Video Anomaly Detection
**Objective:** To develop a novel, distinction-level deep learning architecture that leverages textual semantics to guide visual feature representations for detecting anomalies in untrimmed surveillance videos using only video-level labels.

## 2. Core Hypothesis
Current Multiple Instance Learning (MIL) approaches for VAD rely solely on visual features, leading to context bias (e.g., confusing "smoke" with "explosion"). By utilizing Large Vision-Language Models (VLMs) to generate semantic descriptions of video segments, we can use these textual features as a **Query** in a **Cross-Attention** mechanism to mathematically guide the visual network, amplifying abnormal visual cues and suppressing normal background noise.

## 3. System Architecture Constraints
*   **Input Data Paradigm:** Videos are uniformly divided into exactly $T = 32$ non-overlapping segments.
*   **Offline Feature Extraction:** 
    *   *Visual:* CLIP Vision Encoder (ViT-B/16). Output per video: `Tensor[32, 512]`.
    *   *Textual:* BLIP-2 generates captions per segment -> CLIP Text Encoder. Output per video: `Tensor[32, 512]`.
*   **Core Network:**
    *   Inputs: Visual Features ($V$) and Textual Features ($T$).
    *   Fusion: Multi-Head Cross-Attention where Query=$T$, Key=$V$, Value=$V$.
    *   Classifier: Multi-Layer Perceptron (MLP) mapping the guided feature vector to a scalar anomaly score $\in [0, 1]$.
*   **Loss Function:**
    *   Top-K MIL Ranking Loss (comparing Top-K scores of abnormal vs normal bags).
    *   Temporal Smoothness penalty.
    *   Sparsity penalty.

## 4. Datasets
*   **Primary:** UCF-Crime (13 anomaly classes, 1900 videos).
*   **Secondary (Optional for generalization):** XD-Violence.
*   **Annotation level:** Video-level binary labels for training (1=Abnormal, 0=Normal). Frame-level binary labels for testing/evaluation.

## 5. Success Metrics
*   **Primary Metric:** Frame-level Area Under the Receiver Operating Characteristic Curve (AUROC).
*   **Target Baseline:** Must outperform the standard MIL baseline (Sultani et al., 2019) and aim to compete with RTFM (Tian et al., 2021).
*   **Deliverable:** A highly modular, PyTorch-based repository with clear ablation studies proving the efficacy of the text-guided attention module.

## 6. Development Phases
1.  **Phase 1:** Data Pipeline & Offline Feature Extraction (Visual & Textual).
2.  **Phase 2:** PyTorch Dataset/DataLoader creation & Model Architecture Design.
3.  **Phase 3:** Loss Function Implementation & Training Loop.
4.  **Phase 4:** Evaluation, Visualization (Score curves), and Ablation Studies.