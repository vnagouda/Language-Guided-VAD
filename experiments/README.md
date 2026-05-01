# Experiments: Language-Guided Weakly Supervised Video Anomaly Detection

**Author:** Madhu Bagroy  
**Programme:** MSc Artificial Intelligence  
**Institution:** University of Surrey  
**Branch:** experiment/ais-instance-selection  

This experiment package documents Madhu Bagroy’s investigation into improving weakly supervised video anomaly detection through controlled comparisons of semantic guidance, feature magnitude, and AIS-style soft instance selection.

---

## Overview

This work investigates which design choices most influence performance in weakly supervised video anomaly detection (VAD), with particular emphasis on frame-level localisation.

The objective is not only to distinguish anomalous from normal videos, but to improve the model’s ability to identify which temporal segments correspond to anomalous events, despite having only video-level supervision during training.

A key hypothesis explored is that instance selection — how segment-level predictions are aggregated into a video-level signal — significantly influences performance in this setting.

To examine this, a set of controlled experiments is conducted:
- removing semantic guidance (visual-only ablation)
- introducing feature magnitude as an auxiliary signal
- replacing Top-K selection with AIS-style soft instance selection (proposed)

All experiments are evaluated under a consistent pipeline to isolate the impact of each design choice.

---

## Problem Setting

### Available supervision
Training uses video-level labels only:
- 1 → anomalous video  
- 0 → normal video  

No frame-level labels are available during training.

### Objective
The model must learn to assign anomaly scores to temporal segments such that:
- anomalous videos receive higher scores than normal videos  
- anomalous segments are more accurately localised  

### Challenge
Each video is represented by multiple temporal segments, but supervision is only provided at the video level. The model must therefore infer which segments are responsible for anomaly without direct temporal guidance.

This makes instance selection a critical factor in learning.

---

## Data Representation

Each video is converted into a fixed-length segment-level representation.

### Temporal segmentation
- Each video is divided into 32 non-overlapping temporal segments  
- From each segment, a single representative frame (centre frame) is sampled  

### Visual features
- Each frame is encoded using the CLIP image encoder  
- Output per video: [32, 512]  

### Text features
- BLIP generates one caption per sampled frame  
- Captions are encoded using the CLIP text encoder  
- Output per video: [32, 512]  

### Final representation
Each video is represented as:
- Visual features: [32, 512]  
- Text features: [32, 512]  
- Label: video-level (0 or 1)  

### Important implication
Using a single frame per segment is efficient but may miss anomalies occurring between sampled frames. This highlights the importance of effective instance selection when learning from weak supervision.

---

## Main-Branch Baseline

The baseline system is a semantic-guided MIL model.

### Components
- CLIP visual features  
- BLIP + CLIP text features  
- Cross-attention (text-guided visual representation)  
- Top-K Multiple Instance Learning (MIL) ranking loss with K = 8 (8 segments selected out of 32)  
- Training for 100 epochs  

### Limitation
Top-K MIL uses hard selection of a small subset of segments:

- Only 8 out of 32 segments contribute directly to the loss  
- Most temporal information is ignored  
- Early in training, high scores may be assigned to incorrect segments  
- These incorrect segments can be repeatedly selected and reinforced  

As a result, supervision is limited and can become unstable, particularly during early training.

This motivates exploring alternative instance selection strategies.

---

## What Is Instance Selection?

Instance selection refers to how segment-level scores are aggregated into a video-level signal during training.

### Baseline: Top-K MIL
- Assign scores to all segments  
- Select top-K segments  
- Compute video score from selected segments  

Example:

Segment scores:
[0.10, 0.20, 0.05, 0.90, 0.80, 0.15]

Top-2 selection:
[0.90, 0.80]

Video score:
0.85

### Limitation
- ignores most segments  
- sensitive to noisy peaks  
- uses hard selection  
- provides sparse and potentially unstable supervision  

---

## AIS-Style Soft Instance Selection

The proposed approach replaces hard selection with soft weighting over all segments.

### Method
- Convert segment scores into weights using temperature-controlled softmax  
- Higher-scoring segments receive larger weights  
- Compute video score as a weighted combination of all segments  

### Key difference

| Aspect | Top-K MIL | AIS-style |
|--------|----------|----------|
| Selection | Hard | Soft |
| Segments used | Top-K | All |
| Stability | Lower | Higher |
| Learning signal | Sparse | Distributed |

### Intuition
Top-K selection commits early to a few segments.  
AIS-style selection allows the model to consider all segments while gradually focusing on the most relevant ones.

---

## Baseline Reference (Main Branch)

The main branch baseline (semantic-guided cross-attention with Top-K MIL, K = 8, trained for 100 epochs) reports:

- Video-level AUROC: ~94.85%  
- Frame-level AUROC: ~77.14%  

These values serve as the reference point for evaluating all modifications.

---

## Experiment Design and Analysis

All experiments are designed to isolate the effect of specific components while keeping the rest of the pipeline unchanged. This ensures that observed differences can be attributed to the modification being tested.

The experiments focus on three aspects:
- semantic guidance  
- feature magnitude  
- instance selection  

---

### E1: Visual-only Ablation

**Change**  
- Removed text guidance and cross-attention  
- Used visual features only with Top-K MIL (K = 8)

**Motivation**  
To evaluate whether semantic guidance contributes meaningfully to anomaly localisation.

**Expectation**  
Removing semantic guidance should reduce performance.

**Result**  
- Frame AUROC: ~0.7339  

**Interpretation**  
Semantic guidance provides useful contextual information that improves localisation.

---

### E2: Semantic-Guided Baseline (Reference)

**Setup**  
- Visual + text features  
- Cross-attention  
- Top-K MIL (K = 8)

**Purpose**  
Serves as the reference baseline.

**Result**  
- Frame AUROC: ~0.7707  

**Insight**  
Cross-modal interaction improves localisation compared to visual-only scoring.

---

### E3: Magnitude-Based Multiplicative Fusion

**Change**  
score_final = score_semantic × score_magnitude  

**Motivation**  
Evaluate whether feature magnitude amplifies anomaly signal.

**Expectation**  
Magnitude may strengthen anomaly discrimination.

**Result**  
- Frame AUROC: ~0.7717  

**Interpretation**  
Magnitude provides only a weak signal.

---

### E4: Magnitude-Based Weighted Fusion

**Change**  
score_final = 0.8 × score_semantic + 0.2 × score_magnitude  

**Motivation**  
Test magnitude as a secondary signal.

**Expectation**  
A softer combination may stabilise performance.

**Result**  
- Frame AUROC: ~0.7700  

**Interpretation**  
Magnitude has limited impact.

---

## Feature Magnitude Analysis

To better understand the behaviour observed in E3 and E4, feature magnitude (L2 norm of segment embeddings) was analysed across anomalous and normal segments.

| Metric | Train (Anomaly) | Train (Normal) | Test (Anomaly) | Test (Normal) |
|--------|----------------|----------------|----------------|----------------|
| Mean Norm | 10.41 | 10.58 | 10.41 | 10.55 |
| Std Dev | 0.36 | 0.39 | 0.37 | 0.31 |

### Observations

- Normal segments exhibit slightly higher average feature magnitude than anomalous segments  
- The difference in magnitude between anomalous and normal segments is small  
- The distributions significantly overlap  

### Interpretation

These results suggest that feature magnitude does not provide a strong or consistent signal for distinguishing anomalous segments in this setup.

This helps explain the results observed in:
- E3 (multiplicative fusion) → only marginal improvement  
- E4 (weighted fusion) → negligible change  

Since magnitude does not clearly separate anomalous from normal segments, incorporating it into the scoring function provides limited benefit.

### Implication

Improving anomaly detection performance in this setup is less about augmenting feature representations (e.g., magnitude) and more about improving how segment-level information is aggregated — as explored in the AIS-style instance selection experiments.

---

### E5: AIS-Style Soft Instance Selection (Short Training)

**Change**  
- Replaced Top-K MIL with soft weighting over all segments  
- Video score computed as a weighted sum of segment scores  
- Weights derived using a temperature-controlled softmax  

**Motivation**  
Top-K MIL provides supervision from only a small subset of segments and is sensitive to noisy high-scoring segments, especially early in training.

This experiment tests whether using all segments with soft weighting provides a more stable learning signal.

---

**Training behaviour**

At the start of training:
- Temperature (τ) is high (~1.0)  
- Weight distribution is relatively uniform  
- Many segments contribute to the video score  

As training progresses:
- Temperature decreases  
- Weight distribution becomes sharper  
- Higher-scoring segments receive more emphasis  

This creates a transition from broad exploration to more focused selection.

---

**Result (5 epochs)**  
- Frame AUROC: ~0.7770  

**Interpretation**  
Soft weighting improves early-stage stability and leads to better localisation even with limited training.

---

### E6: AIS-Style Soft Instance Selection (Full Training)

**Setup**  
Same as E5, trained for 100 epochs.

---

**Training dynamics**

The temperature schedule leads to:

- Early stage: broad weighting across segments  
- Mid stage: gradual differentiation between segments  
- Late stage: focused weighting on high-scoring segments  

---

**Key difference from Top-K MIL**

- Top-K makes hard selections early  
- AIS delays commitment and allows refinement over time  

---

**Result (100 epochs)**  
- Frame AUROC: ~0.7898  

---

**Interpretation**

The improvement over E5 indicates that:

- benefits of soft selection accumulate over training  
- the model converges to more reliable segment-level importance  
- localisation improves as selection becomes more refined  

---

**Overall implication**

AIS-style soft instance selection improves:

- training stability in early stages  
- selection precision in later stages  

making it a more robust alternative to hard Top-K selection under weak supervision.

---

## Comparison to Baseline

- Baseline (Top-K MIL): ~0.7714 frame AUROC  
- AIS-style soft selection: ~0.7898  frame AUROC

This indicates improved localisation without modifying feature extraction or architecture.

---

## Key Findings

- Semantic guidance improves localisation  
- Feature magnitude provides limited benefit  
- Instance selection is a key limitation in this setup  
- AIS-style soft selection improves frame-level AUROC  
- Frame-level AUROC is a more informative metric for localisation  

---

## Contribution

This work provides:

1. Evidence that instance selection significantly influences performance in this setup  
2. Controlled comparison of alternative strategies  
3. AIS-style soft selection improving localisation without additional model complexity  
4. Insight that loss design can be more impactful than feature augmentation  

### Key insight
In this setup, improving how segments are selected and weighted has a stronger impact on localisation than modifying feature representations.

---

## Files

- README.md — overview  
- results_summary.md — detailed results  
- analyse_visual_norms.py — magnitude analysis  
- experiment_log.md — experiment notes  

---

## Notes

- Pipeline kept consistent across experiments  
- Frame-level AUROC used as primary evaluation metric  
- Designed to be interpretable without code inspection  

---

## References

- Sultani et al., *Real-World Anomaly Detection in Surveillance Videos*, CVPR 2018  
- Tian et al., *RTFM: Weakly-Supervised Video Anomaly Detection With Robust Temporal Feature Magnitude Learning*, ICCV 2021  
- Wu et al., *A Lightweight Video Anomaly Detection Model with Weak Supervision and Adaptive Instance Selection*  