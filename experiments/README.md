# Experiments: Language-Guided Weakly Supervised Video Anomaly Detection

**Author:** Madhu Bagroy  
**Programme:** MSc Artificial Intelligence  
**Institution:** University of Surrey  
**Branch:** experiment/ais-instance-selection  

---

## Overview

This document summarises experiments conducted to understand what drives performance in weakly supervised video anomaly detection (VAD).

The primary focus is on improving frame-level localisation, where only video-level labels are available during training.

## Problem Setting

### Available supervision
- Video-level labels only:
  - 1 → anomalous video  
  - 0 → normal video  
- No frame-level labels during training

### Objective
Learn to assign anomaly scores to temporal segments of a video using only weak supervision.

---

## Data Representation

Each video is converted into a fixed-length representation.

### Temporal segmentation
- Each video is divided into 32 non-overlapping temporal segments
- From each segment, one representative frame (centre frame) is sampled

### Visual features
- CLIP image encoder applied to each sampled frame  
- Output: [32, 512]

### Text features
- BLIP generates one caption per sampled frame  
- CLIP text encoder converts captions to embeddings  
- Output: [32, 512]

### Final representation
Each video is represented as:
- Visual features: [32, 512]
- Text features: [32, 512]
- Label: video-level (0 or 1)

### Important assumption
A single frame is used as a proxy for each segment.  
This is efficient but may miss anomalies occurring between sampled frames.


## Main-Branch Baseline

The baseline model (main branch) is a semantic-guided MIL model:

- Visual features (CLIP)
- Text features (BLIP + CLIP)
- Cross-attention between text and visual features
- Trained with Top-K Multiple Instance Learning (MIL) ranking loss
- Trained for 100 epochs

### Limitation
Top-K MIL uses hard selection of a few segments, making it:
- sensitive to noisy predictions  
- unstable early in training  
- limited in capturing full temporal context  


## Experiment Groups

### 1. Visual-only Ablation

**Change:**  
- Removed text guidance and cross-attention  
- Used visual features only with Top-K MIL  

**Purpose:**  
Evaluate the contribution of semantic guidance to anomaly localisation.

---

### 2. Magnitude-Based Feature Variants

**Purpose:**  
Evaluate whether feature magnitude provides useful signal for anomaly detection.

#### E3: Multiplicative Fusion

**Formulation:**  
score_final = score_semantic × score_magnitude  

**Interpretation:**  
The semantic score is scaled by magnitude, so higher magnitude amplifies the score while lower magnitude suppresses it.

---

#### E4: Weighted Fusion

**Formulation:**  
score_final = 0.8 × score_semantic + 0.2 × score_magnitude  

**Interpretation:**  
Magnitude contributes as a secondary signal alongside the semantic score.

---

### 3. AIS-Style Soft Instance Selection (Proposed)

**Change:**  
Replaces Top-K hard selection with soft weighting over all segments.

**Formulation:**  
Segment scores are converted into weights using a temperature-controlled softmax.  
Higher-scoring segments receive larger weights, and the final video score is computed as a weighted sum of all segment scores.

**Purpose:**  
Improve instance selection under weak supervision by reducing sensitivity to noisy high-scoring segments.

**Temperature behaviour:**  
- High τ → broad weighting (early training)  
- Low τ → focused selection (later training)

**Key idea:**  
Move from coarse (exploration) to focused (selection) learning.


## Experimental Results

| Exp ID | Model Variant | Type | Epochs | Video AUROC | Frame AUROC |
|--------|--------------|------|--------|-------------|-------------|
| E1 | Visual-only | Top-K MIL | 5 | 0.9388 | 0.7339 |
| E2 | Semantic-guided | Top-K MIL | 5 | 0.9432 | 0.7707 |
| E3 | Semantic + Magnitude | Multiplicative | 5 | 0.9469 | 0.7717 |
| E4 | Semantic + Magnitude | Weighted | 5 | 0.9455 | 0.7700 |
| E5 | Semantic + AIS | Soft selection | 5 | 0.9462 | 0.7770 |
| E6 | Semantic + AIS | Soft selection | 100 | 0.9477 | 0.7898 |


## Additional Analysis: Feature Magnitude

| Metric | Train (Anomaly) | Train (Normal) | Test (Anomaly) | Test (Normal) |
|--------|----------------|----------------|----------------|----------------|
| Mean Norm | 10.41 | 10.58 | 10.41 | 10.55 |
| Std Dev | 0.36 | 0.39 | 0.37 | 0.31 |

Observation:  
Normal segments have slightly higher magnitude than anomalous ones.

Conclusion:  
Feature magnitude is not a reliable anomaly signal.

## Key Findings

- Semantic guidance significantly improves localisation  
- Feature magnitude alone is unreliable  
- Magnitude fusion provides only marginal gains  
- Instance selection is the main bottleneck  
- AIS-style soft selection provides the strongest improvement  
- Frame-level AUROC reveals improvements better than video-level metrics  

---

## Contribution

This work identifies instance selection as the primary limitation in weakly supervised VAD under this segment-based representation.

Key insight:

Improving how segments are selected and weighted is more impactful than modifying feature representations.

The proposed AIS-style soft weighting:
- addresses instability in Top-K MIL  
- improves frame-level localisation  
- provides a simple and effective alternative to hard selection  

## Files

- README.md — overview of experimental design, setup, and key findings  
- results_summary.md — detailed results  
- analyse_visual_norms.py — magnitude analysis  
- experiment_log.md — experiment notes  

---

## Notes

- All experiments use the same feature extraction pipeline  
- Comparisons are controlled and consistent  
- Frame-level AUROC is the primary evaluation metric  

## References

- Sultani et al., *Real-World Anomaly Detection in Surveillance Videos*, CVPR 2018  
- Tian et al., *RTFM: Weakly-Supervised Video Anomaly Detection With Robust Temporal Feature Magnitude Learning*, ICCV 2021  
- Wu et al., *A Lightweight Video Anomaly Detection Model with Weak Supervision and Adaptive Instance Selection* — motivates improved instance selection beyond hard Top-K MIL