"""Unit tests for utils/metrics.py — interpolate_scores and compute_auroc."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import interpolate_scores, compute_auroc


# ---------------------------------------------------------------------------
# interpolate_scores
# ---------------------------------------------------------------------------

class TestInterpolateScores:

    def test_output_length_matches_num_frames(self):
        """Output should have exactly num_frames elements."""
        scores = np.array([0.1, 0.5, 0.9, 0.3])
        result = interpolate_scores(scores, num_frames=100)
        assert len(result) == 100

    def test_single_segment_broadcast(self):
        """A single segment score should broadcast to all frames."""
        scores = np.array([0.7])
        result = interpolate_scores(scores, num_frames=50)
        assert len(result) == 50
        assert np.allclose(result, 0.7)

    def test_same_length_is_identity(self):
        """When num_frames == num_segments, output equals input."""
        scores = np.array([0.1, 0.4, 0.7, 0.9])
        result = interpolate_scores(scores, num_frames=4)
        assert np.allclose(result, scores)

    def test_values_within_score_range(self):
        """Interpolated values should stay within [min, max] of input."""
        scores = np.array([0.2, 0.8, 0.5, 0.1])
        result = interpolate_scores(scores, num_frames=64)
        assert result.min() >= scores.min() - 1e-6
        assert result.max() <= scores.max() + 1e-6

    def test_monotone_segment_produces_monotone_frames(self):
        """Strictly increasing segments should produce increasing frame scores."""
        scores = np.linspace(0.0, 1.0, 8)
        result = interpolate_scores(scores, num_frames=32)
        assert np.all(np.diff(result) >= 0)

    def test_output_dtype_is_float(self):
        """Output should be a float array."""
        scores = np.array([0.3, 0.6, 0.9])
        result = interpolate_scores(scores, num_frames=16)
        assert result.dtype in (np.float32, np.float64)


# ---------------------------------------------------------------------------
# compute_auroc
# ---------------------------------------------------------------------------

class TestComputeAUROC:

    def test_perfect_classifier_returns_1(self):
        """A perfect classifier should return AUROC = 1.0."""
        preds  = np.array([0.9, 0.8, 0.1, 0.05])
        labels = np.array([1,   1,   0,   0])
        assert compute_auroc(preds, labels) == pytest.approx(1.0)

    def test_random_classifier_returns_approx_05(self):
        """A random classifier should return AUROC ≈ 0.5."""
        rng = np.random.default_rng(42)
        preds  = rng.random(1000)
        labels = rng.integers(0, 2, size=1000)
        auroc = compute_auroc(preds, labels)
        assert 0.45 < auroc < 0.55

    def test_worst_classifier_returns_0(self):
        """Inverted predictions should return AUROC = 0.0."""
        preds  = np.array([0.1, 0.05, 0.9, 0.8])
        labels = np.array([1,   1,    0,   0])
        assert compute_auroc(preds, labels) == pytest.approx(0.0)

    def test_single_class_raises_value_error(self):
        """AUROC is undefined when only one class is present."""
        preds  = np.array([0.3, 0.5, 0.7])
        labels = np.array([0,   0,   0])
        with pytest.raises(ValueError):
            compute_auroc(preds, labels)

    def test_return_type_is_float(self):
        """Return type should be a Python float."""
        preds  = np.array([0.9, 0.1])
        labels = np.array([1, 0])
        result = compute_auroc(preds, labels)
        assert isinstance(result, float)
