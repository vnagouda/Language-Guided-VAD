## 📊 Experimental Results Summary

| Exp ID | Model Variant | Fusion Type | Epochs | Video AUROC | Frame AUROC | Δ Video AUROC | Δ Frame AUROC | Key Insight |
|--------|--------------|-------------|--------|-------------|-------------|----------------|----------------|-------------|
| E1 | Visual-only | — | 5 | 0.9388 | 0.7339 | — | — | Strong baseline; limited localisation |
| E2 | Semantic-guided (Cross-Attention) | — | 5 | 0.9432 | 0.7707 | +0.0044 | +0.0368 | Major improvement in frame-level localisation |
| E3 | Semantic + Magnitude (v1) | Multiplicative | 5 | 0.9469 | 0.7717 | +0.0037 | +0.0010 | Modest gain; magnitude acts as complementary signal |

---

## 🔍 Additional Analysis: Visual Feature Magnitude

| Metric | Train (Anomaly) | Train (Normal) | Test (Anomaly) | Test (Normal) |
|--------|----------------|----------------|----------------|----------------|
| Mean Norm | 10.41 | 10.58 | 10.41 | 10.55 |
| Std Dev | 0.36 | 0.39 | 0.37 | 0.31 |
| Observation | \multicolumn{4}{c}{Normal segments exhibit slightly higher feature magnitude than anomaly segments} |

---

## 🧠 Key Findings

1. **Semantic guidance is the dominant contributor**
   - Significant improvement in frame-level AUROC (+3.7%)
   - Enables better localisation of anomalous segments

2. **Magnitude is not directly aligned with anomaly**
   - Anomaly segments do **not** exhibit higher feature norms
   - Raw magnitude is not a reliable standalone anomaly signal

3. **Magnitude provides complementary signal**
   - Small but consistent improvement in video-level AUROC
   - Minimal positive impact on frame-level AUROC

4. **Multiplicative fusion is likely suboptimal**
   - Assumes alignment between semantic and magnitude signals
   - May suppress useful anomaly signals when magnitude is low

---

## 🎯 Interpretation

These results suggest that while semantic alignment captures *what* constitutes an anomaly, feature magnitude encodes a different aspect of the representation space. A learned transformation of magnitude can provide complementary information, but its effectiveness depends critically on how it is integrated with semantic signals.