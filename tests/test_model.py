"""Unit tests for models/vad_architecture.py — LanguageGuidedVAD forward pass."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vad_architecture import LanguageGuidedVAD, DynamicNormalPrototypes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(**kwargs) -> LanguageGuidedVAD:
    defaults = dict(
        feature_dim=64,   # small dim for fast tests
        num_segments=32,
        num_heads=4,
        ff_dim=128,
        classifier_bottleneck_dim=16,
        classifier_hidden_dim=32,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return LanguageGuidedVAD(**defaults)


def _make_inputs(B: int = 2, T: int = 32, D: int = 64):
    visual = torch.randn(B, T, D)
    text   = torch.randn(B, T, D)
    flow   = torch.zeros(B, T)
    return visual, text, flow


# ---------------------------------------------------------------------------
# LanguageGuidedVAD
# ---------------------------------------------------------------------------

class TestLanguageGuidedVAD:

    def test_output_shapes(self):
        """forward() should return (scores, norms, guided) with correct shapes."""
        B, T, D = 2, 32, 64
        model = _make_model(feature_dim=D, num_segments=T)
        visual, text, flow = _make_inputs(B, T, D)

        scores, norms, guided = model(visual, text, flow)

        assert scores.shape  == (B, T)
        assert norms.shape   == (B, T)
        assert guided.shape  == (B, T, D)

    def test_scores_in_0_1(self):
        """Anomaly scores should be in [0, 1] after sigmoid."""
        model = _make_model()
        visual, text, flow = _make_inputs()
        scores, _, _ = model(visual, text, flow)
        assert scores.min().item() >= 0.0
        assert scores.max().item() <= 1.0

    def test_single_scale_mode(self):
        """Single-scale mode (use_multi_scale=False) should still run correctly."""
        model = _make_model(use_multi_scale=False)
        visual, text, flow = _make_inputs()
        scores, norms, guided = model(visual, text, flow)
        assert scores.shape == (2, 32)

    def test_no_magnitude_branch(self):
        """Model without magnitude branch should still produce correct output."""
        model = _make_model(use_magnitude_branch=False)
        visual, text, flow = _make_inputs()
        scores, norms, guided = model(visual, text, flow)
        assert scores.shape == (2, 32)

    def test_gradients_flow(self):
        """Gradients should flow back through the model."""
        model = _make_model()
        visual, text, flow = _make_inputs()
        visual.requires_grad_(True)
        scores, _, _ = model(visual, text, flow)
        scores.sum().backward()
        assert visual.grad is not None

    def test_repr(self):
        """__repr__ should return a non-empty descriptive string."""
        model = _make_model()
        r = repr(model)
        assert "LanguageGuidedVAD" in r
        assert "dim=" in r

    def test_batch_size_1(self):
        """Model should handle batch size of 1."""
        model = _make_model()
        visual, text, flow = _make_inputs(B=1)
        scores, norms, guided = model(visual, text, flow)
        assert scores.shape == (1, 32)


# ---------------------------------------------------------------------------
# DynamicNormalPrototypes
# ---------------------------------------------------------------------------

class TestDynamicNormalPrototypes:

    def test_get_returns_normalised_prototypes(self):
        """get() should return L2-normalised prototypes."""
        bank = DynamicNormalPrototypes(feature_dim=64, num_prototypes=8)
        protos = bank.get()
        norms = protos.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_prototype_shape(self):
        """Prototypes should have shape (num_prototypes, feature_dim)."""
        bank = DynamicNormalPrototypes(feature_dim=64, num_prototypes=16)
        assert bank.get().shape == (16, 64)
