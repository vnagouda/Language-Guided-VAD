## Experimental Results Summary

| Exp ID | Model Variant                         | Fusion / Selection Type       | Epochs | Video AUROC | Frame AUROC | Key Insight |
|--------|--------------------------------------|-------------------------------|--------|-------------|-------------|-------------|
| E1     | Visual-only                          | Top-K MIL                     | 5      | 0.9388      | 0.7339      | Removing semantic guidance substantially reduces localisation performance |
| E2     | Semantic-guided (Cross-Attention)    | Top-K MIL                     | 5      | 0.9432      | 0.7707      | Semantic guidance significantly improves frame-level localisation |
| E3     | Semantic + Magnitude (v1)            | Multiplicative fusion         | 5      | 0.9469      | 0.7717      | Magnitude acts as a weak complementary signal |
| E4     | Semantic + Magnitude (v2)            | Weighted fusion (0.8 / 0.2)   | 5      | 0.9455      | 0.7700      | Performance is sensitive to fusion strategy; weighted fusion does not outperform v1 |
| E5     | Semantic + AIS-style soft ranking    | Soft instance selection       | 5      | 0.9462      | 0.7770      | Improved segment selection under weak supervision; best frame-level result among 5-epoch runs |
| E6     | Semantic + AIS-style soft ranking    | Soft instance selection       | 100    | 0.9477      | 0.7898      | Best overall result; soft instance selection consistently improves localisation |

---

## Additional Analysis: Visual Feature Magnitude

| Metric    | Train (Anomaly) | Train (Normal) | Test (Anomaly) | Test (Normal) |
|-----------|------------------|----------------|----------------|----------------|
| Mean Norm | 10.41            | 10.58          | 10.41          | 10.55          |
| Std Dev   | 0.36             | 0.39           | 0.37           | 0.31           |

**Observation:**  
Normal segments exhibit slightly higher feature magnitude than anomalous segments.

**Interpretation:**  
Feature magnitude alone is not a reliable indicator of anomaly in this setting.

---

## Key Findings

- Semantic guidance is a strong driver of localisation performance  
  Cross-attention improves frame-level AUROC over visual-only models.

- Feature magnitude is not a reliable anomaly signal  
  Normal segments often exhibit slightly higher norms than anomalous ones.

- Magnitude-based fusion provides only marginal gains  
  Improvements are limited and sensitive to how magnitude is incorporated.

- Instance selection is the primary bottleneck under weak supervision  
  Top-K MIL concentrates supervision on a small subset of segments and is sensitive to noisy high scores.

- AIS-style soft instance selection provides the most effective improvement  
  Frame-level AUROC improves to **0.7898**, the best result achieved.

- Training dynamics play an important role  
  The temperature decay schedule enables a transition from broad to selective segment weighting, supporting more robust learning.

- Loss design is more impactful than feature augmentation  
  Improvements in instance selection outperform magnitude-based fusion strategies.

- Frame-level evaluation reveals improvements not captured by video-level metrics  
  Localisation gains are more clearly reflected in frame-level AUROC.