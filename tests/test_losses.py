"""Unit tests for utils/losses.py — VADLoss and SelfTrainingLoss."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.losses import VADLoss, SelfTrainingLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scores(batch: int, T: int = 32, high: bool = True) -> torch.Tensor:
    """Create synthetic anomaly scores."""
    base = 0.7 if high else 0.2
    return torch.clamp(torch.randn(batch, T) * 0.1 + base, 0.0, 1.0)

def _make_norms(batch: int, T: int = 32, high: bool = True) -> torch.Tensor:
    """Create synthetic L2-norm values."""
    base = 3.0 if high else 1.0
    return torch.clamp(torch.randn(batch, T) * 0.5 + base, 0.1, 10.0)


# ---------------------------------------------------------------------------
# VADLoss
# ---------------------------------------------------------------------------

class TestVADLoss:

    def setup_method(self):
        self.loss_fn = VADLoss()
        self.B = 4
        self.T = 32

    def test_forward_returns_dict(self):
        """forward() should return a dict of loss tensors."""
        scores_abn = _make_scores(self.B, self.T, high=True)
        scores_nor = _make_scores(self.B, self.T, high=False)
        norms_abn  = _make_norms(self.B, self.T, high=True)
        norms_nor  = _make_norms(self.B, self.T, high=False)

        result = self.loss_fn(scores_abn, scores_nor, norms_abn, norms_nor, epoch=1)
        assert isinstance(result, dict)
        assert "total_loss" in result

    def test_total_loss_is_scalar(self):
        """total_loss should be a scalar tensor."""
        scores_abn = _make_scores(self.B, self.T, high=True)
        scores_nor = _make_scores(self.B, self.T, high=False)
        norms_abn  = _make_norms(self.B, self.T, high=True)
        norms_nor  = _make_norms(self.B, self.T, high=False)

        result = self.loss_fn(scores_abn, scores_nor, norms_abn, norms_nor, epoch=1)
        assert result["total_loss"].shape == torch.Size([])

    def test_total_loss_is_non_negative(self):
        """total_loss should always be >= 0."""
        scores_abn = _make_scores(self.B, self.T, high=True)
        scores_nor = _make_scores(self.B, self.T, high=False)
        norms_abn  = _make_norms(self.B, self.T, high=True)
        norms_nor  = _make_norms(self.B, self.T, high=False)

        result = self.loss_fn(scores_abn, scores_nor, norms_abn, norms_nor, epoch=1)
        assert result["total_loss"].item() >= 0.0

    def test_all_loss_components_present(self):
        """All expected loss keys should be in the output dict."""
        scores_abn = _make_scores(self.B, self.T, high=True)
        scores_nor = _make_scores(self.B, self.T, high=False)
        norms_abn  = _make_norms(self.B, self.T, high=True)
        norms_nor  = _make_norms(self.B, self.T, high=False)

        result = self.loss_fn(scores_abn, scores_nor, norms_abn, norms_nor, epoch=1)
        for key in ["ais_loss", "antagonistic_loss", "magnitude_loss", "smoothness_loss"]:
            assert key in result, f"Missing key: {key}"

    def test_loss_is_differentiable(self):
        """total_loss should support backpropagation."""
        scores_abn = _make_scores(self.B, self.T, high=True).requires_grad_(True)
        scores_nor = _make_scores(self.B, self.T, high=False).requires_grad_(True)
        norms_abn  = _make_norms(self.B, self.T, high=True)
        norms_nor  = _make_norms(self.B, self.T, high=False)

        result = self.loss_fn(scores_abn, scores_nor, norms_abn, norms_nor, epoch=1)
        result["total_loss"].backward()
        assert scores_abn.grad is not None

    def test_repr(self):
        """__repr__ should return a non-empty string."""
        r = repr(self.loss_fn)
        assert isinstance(r, str) and len(r) > 0


# ---------------------------------------------------------------------------
# SelfTrainingLoss
# ---------------------------------------------------------------------------

class TestSelfTrainingLoss:

    def setup_method(self):
        self.loss_fn = SelfTrainingLoss()

    def test_output_is_scalar(self):
        """Output should be a scalar tensor."""
        scores_abn = _make_scores(4, 32, high=True)
        scores_nor = _make_scores(4, 32, high=False)
        result = self.loss_fn(scores_abn, scores_nor)
        assert result.shape == torch.Size([])

    def test_output_is_non_negative(self):
        """BCE loss should always be >= 0."""
        scores_abn = _make_scores(4, 32, high=True)
        scores_nor = _make_scores(4, 32, high=False)
        result = self.loss_fn(scores_abn, scores_nor)
        assert result.item() >= 0.0

    def test_pseudo_k_respected(self):
        """Different pseudo_k values should produce different losses."""
        scores_abn = _make_scores(4, 32, high=True)
        scores_nor = _make_scores(4, 32, high=False)
        loss_k3 = self.loss_fn(scores_abn, scores_nor, pseudo_k=3).item()
        loss_k8 = self.loss_fn(scores_abn, scores_nor, pseudo_k=8).item()
        assert loss_k3 != loss_k8
