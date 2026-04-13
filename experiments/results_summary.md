## 📊 Experimental Results Summary

| Exp ID | Model Variant                         | Fusion Type     | Epochs | Video AUROC | Frame AUROC | Δ Video AUROC | Δ Frame AUROC | Key Insight                                  |
|--------|--------------------------------------|-----------------|--------|-------------|-------------|----------------|----------------|----------------------------------------------|
| E1     | Visual-only                          | —               | 5      | 0.9388      | 0.7339      | —              | —              | Strong baseline; limited localisation         |
| E2     | Semantic-guided (Cross-Attention)    | —               | 5      | 0.9432      | 0.7707      | +0.0044        | +0.0368        | Major improvement in frame-level localisation |
| E3     | Semantic + Magnitude (v1)            | Multiplicative  | 5      | 0.9469      | 0.7717      | +0.0037        | +0.0010        | Modest gain; magnitude acts as complement     |
| E4     | Semantic + Magnitude (v2)           | Weighted (0.8/0.2) | 5      | 0.9455      | 0.7700      | -0.0014        | -0.0017        | Slight drop vs v1; magnitude contribution sensitive to fusion |
| E5 | Semantic + AIS-style soft ranking | Soft instance selection | 5 | 0.9462 | 0.7770 | +0.0030 | +0.0063 | Best frame-level result so far; improves segment selection under weak supervision |
| E6 | Semantic + AIS-style soft ranking | Soft instance selection | 100 | 0.9477 | 0.7898 | — | — | Best overall result; AIS outperformed all previous variants . Stronger instance selection improves localization |

---

## 🔍 Additional Analysis: Visual Feature Magnitude

| Metric    | Train (Anomaly) | Train (Normal) | Test (Anomaly) | Test (Normal) |
|-----------|------------------|----------------|----------------|----------------|
| Mean Norm | 10.41           | 10.58          | 10.41          | 10.55          |
| Std Dev   | 0.36            | 0.39           | 0.37           | 0.31           |

**Observation:** Normal segments exhibit slightly higher feature magnitude than anomaly segments.

---

## 🧠 Key Findings


- Semantic guidance is the primary driver of localisation performance  
  Introducing cross-attention improves frame-level AUROC over visual-only scoring.

- Feature magnitude alone is not a reliable indicator of anomaly  
  Normal segments often have slightly higher norms than anomalous ones.

- Magnitude provides only marginal gains when fused with semantic features  
  Improvements are small and mainly at video level.

- The main bottleneck lies in instance selection under weak supervision  
  Top-K MIL ignores most segments and is sensitive to noisy high scores.

- AIS-style soft instance selection provides the most effective improvement  
  Frame-level AUROC improves to 0.7898, the best result achieved.

- Training dynamics play an important role: a gradual transition from broad to selective segment weighting improves learning stability.
The temperature decay schedule enables the model to initially leverage information from a wide range of segments before progressively focusing on the most informative ones.

- Loss design is more impactful than feature augmentation  
  Instance selection improvements outperform magnitude-based fusion.

- Improvements at frame level reveal limitations of video-level metrics  
  Frame-level AUROC captures gains that video-level AUROC misses.