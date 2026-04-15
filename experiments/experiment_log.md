# Experiment Log

## Overview

This log captures the progression of experiments conducted to improve weakly supervised video anomaly detection.

---

## Baseline

- Semantic-guided model (cross-attention)
- Top-K MIL loss
- 100 epochs

Observation:
- Strong video-level performance
- Frame-level localisation can be improved

---

## Experiment 1: Visual-only Ablation

Change:
- Removed text guidance

Result:
- Significant drop in frame-level AUROC

Insight:
- Semantic guidance is important for localisation

---

## Experiment 2: Magnitude-based Fusion

Variants:
- Multiplicative fusion
- Weighted fusion

Result:
- Small improvements
- Inconsistent across variants

Insight:
- Feature magnitude is not a reliable anomaly signal

---

## Experiment 3: AIS-style Soft Instance Selection

Change:
- Replaced Top-K with soft weighting
- Introduced temperature decay

Result:
- Consistent improvement in frame-level AUROC
- Best result achieved (0.7898)

Insight:
- Instance selection is the main bottleneck in weakly supervised VAD

---

## Final Takeaway

Improving instance selection is more impactful than modifying feature representations in this setup.