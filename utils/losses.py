"""Top-K MIL Ranking Loss with Temporal Smoothness and Sparsity Penalties.

This module implements the loss function for Weakly Supervised VAD, following
the MIL paradigm of Sultani et al. (2018) with extensions from RTFM (2021).

Mathematical Formulation:
    Given a batch of paired (abnormal, normal) bags, each bag is a video of
    T=32 segment-level anomaly scores:

    1. **Top-K Ranking Loss**:
       Select the Top-K highest scores from the abnormal bag and the Top-K
       highest from the normal bag, then apply hinge loss:
           L_rank = (1/K) * Σ max(0, margin - (s_abn^k - s_nor^k))

    2. **Temporal Smoothness Penalty**:
           L_smooth = Σ_{t=1}^{T-1} (s[t+1] - s[t])^2

    3. **Sparsity Penalty (L1)**:
           L_sparse = Σ_{t=1}^{T} |s[t]|

    4. **Total Loss**:
           L = L_rank + λ_smooth * L_smooth + λ_sparse * L_sparse
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MILRankingLoss(nn.Module):
    """Top-K Multiple Instance Learning Ranking Loss.

    This loss treats each video as a "bag" of T=32 segment instances.  It
    compares the top-K segment scores from abnormal bags against the top-K
    from normal bags, enforcing a margin between them.  Additional temporal
    smoothness and sparsity constraints regularise the predictions.

    Args:
        top_k: Number of top-scoring segments to select per bag.
        margin: Margin for the hinge ranking loss.
        lambda_smooth: Weight for the temporal smoothness penalty.
        lambda_sparse: Weight for the L1 sparsity penalty.
    """

    def __init__(
        self,
        top_k: int = 8,
        margin: float = 1.0,
        lambda_smooth: float = 8e-5,
        lambda_sparse: float = 8e-5,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.margin = margin
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse

    def _ranking_loss(
        self,
        scores_abn: torch.Tensor,
        scores_nor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Top-K hinge ranking loss.

        For each pair: select top-K scores from abnormal bag, top-K from
        normal bag.  Both are sorted descending and paired element-wise.

        Args:
            scores_abn: Anomaly scores for abnormal videos, shape ``(B_abn, 32)``.
            scores_nor: Anomaly scores for normal videos, shape ``(B_nor, 32)``.

        Returns:
            torch.Tensor: Scalar ranking loss.
        """
        # Top-K from each bag, sorted descending
        topk_abn, _ = torch.topk(scores_abn, self.top_k, dim=1)  # (B_abn, K)
        topk_nor, _ = torch.topk(scores_nor, self.top_k, dim=1)  # (B_nor, K)

        # If batch sizes differ, use the minimum
        min_batch = min(topk_abn.size(0), topk_nor.size(0))
        topk_abn = topk_abn[:min_batch]
        topk_nor = topk_nor[:min_batch]

        # Hinge loss: max(0, margin - (score_abn - score_nor))
        loss: torch.Tensor = torch.clamp(
            self.margin - (topk_abn - topk_nor), min=0.0
        ).mean()

        return loss

    def _temporal_smoothness(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute the temporal smoothness penalty.

        Penalises abrupt changes between consecutive segment scores:
            L_smooth = Σ (s[t+1] - s[t])² / (T-1)

        Args:
            scores: Anomaly scores of shape ``(Batch, 32)``.

        Returns:
            torch.Tensor: Scalar smoothness loss.
        """
        diff = scores[:, 1:] - scores[:, :-1]  # (B, 31)
        loss: torch.Tensor = (diff ** 2).mean()
        return loss

    def _sparsity(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute the L1 sparsity penalty on anomaly scores.

        Encourages the model to predict anomalies in only a few segments
        rather than uniformly:
            L_sparse = mean(|s|)

        Args:
            scores: Anomaly scores of shape ``(Batch, 32)``.

        Returns:
            torch.Tensor: Scalar sparsity loss.
        """
        loss: torch.Tensor = scores.abs().mean()
        return loss

    def forward(
        self,
        scores_abn: torch.Tensor,
        scores_nor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the combined MIL ranking loss.

        Args:
            scores_abn: Predicted anomaly scores for abnormal videos,
                shape ``(B_abn, 32)``.
            scores_nor: Predicted anomaly scores for normal videos,
                shape ``(B_nor, 32)``.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys:
                - ``"total_loss"``: Combined scalar loss for backpropagation.
                - ``"ranking_loss"``: The Top-K ranking component.
                - ``"smoothness_loss"``: The temporal smoothness component.
                - ``"sparsity_loss"``: The L1 sparsity component.
        """
        ranking = self._ranking_loss(scores_abn, scores_nor)
        smoothness = self._temporal_smoothness(scores_abn)
        sparsity = self._sparsity(scores_abn)

        total = ranking + self.lambda_smooth * smoothness + self.lambda_sparse * sparsity

        return {
            "total_loss": total,
            "ranking_loss": ranking,
            "smoothness_loss": smoothness,
            "sparsity_loss": sparsity,
        }

    @classmethod
    def from_config(cls, config: dict) -> "MILRankingLoss":
        """Construct the loss from a configuration dictionary.

        Args:
            config: Full configuration dict (loaded from ``config.yaml``).

        Returns:
            MILRankingLoss: Instantiated loss with config-driven parameters.
        """
        loss_cfg = config["loss"]
        return cls(
            top_k=loss_cfg["top_k"],
            margin=loss_cfg["margin"],
            lambda_smooth=loss_cfg["lambda_smooth"],
            lambda_sparse=loss_cfg["lambda_sparse"],
        )
