## 📊 Experimental Results Summary

| Exp ID | Model Variant                         | Fusion Type     | Epochs | Video AUROC | Frame AUROC | Δ Video AUROC | Δ Frame AUROC | Key Insight                                  |
|--------|--------------------------------------|-----------------|--------|-------------|-------------|----------------|----------------|----------------------------------------------|
| E1     | Visual-only                          | —               | 5      | 0.9388      | 0.7339      | —              | —              | Strong baseline; limited localisation         |
| E2     | Semantic-guided (Cross-Attention)    | —               | 5      | 0.9432      | 0.7707      | +0.0044        | +0.0368        | Major improvement in frame-level localisation |
| E3     | Semantic + Magnitude (v1)            | Multiplicative  | 5      | 0.9469      | 0.7717      | +0.0037        | +0.0010        | Modest gain; magnitude acts as complement     |

---

## 🔍 Additional Analysis: Visual Feature Magnitude

| Metric    | Train (Anomaly) | Train (Normal) | Test (Anomaly) | Test (Normal) |
|-----------|------------------|----------------|----------------|----------------|
| Mean Norm | 10.41           | 10.58          | 10.41          | 10.55          |
| Std Dev   | 0.36            | 0.39           | 0.37           | 0.31           |

**Observation:** Normal segments exhibit slightly higher feature magnitude than anomaly segments.

---

## 🧠 Key Findings

- Semantic guidance significantly improves frame-level AUROC (+3.7%)
- Magnitude is not directly aligned with anomaly (normal > anomaly in norm)
- Magnitude provides a small but consistent improvement at video level
- Multiplicative fusion may be too restrictive  

