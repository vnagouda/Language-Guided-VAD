"""AIS-Style MIL Ranking Loss with Temporal Smoothness and Sparsity Penalties.

This module implements a weakly supervised loss for Video Anomaly Detection.
Compared with hard Top-K MIL, this version uses a temperature-controlled
softmax weighting over all segments. Early in training, the weighting is broad;
later, it becomes sharper and focuses more on the most suspicious segments.

Mathematical Formulation:
    Given a batch of abnormal and normal bags, each bag is a video of
    T=32 segment-level anomaly scores:

    1. **AIS-Style Soft Ranking Loss**:
       Compute soft weights over all segments using a temperature ``tau``:
           p_t = softmax(s_t / tau)

       Then compute a weighted pooled score per video:
           S = Σ p_t * s_t

       Finally apply hinge ranking loss:
           L_rank = mean(max(0, margin - (S_abn - S_nor)))

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
    """AIS-style Multiple Instance Learning Ranking Loss.

    This loss treats each video as a "bag" of T=32 segment instances. Instead
    of selecting a hard Top-K set of segments, it computes temperature-controlled
    softmax weights over all segments, producing a weighted pooled anomaly score
    per video. This reduces the risk of learning from noisy high-scoring segments
    early in training.

    Args:
        top_k: Kept for backward compatibility with the config, but not used in
            the AIS-style ranking loss.
        margin: Margin for the hinge ranking loss.
        lambda_smooth: Weight for the temporal smoothness penalty.
        lambda_sparse: Weight for the L1 sparsity penalty.
        tau_initial: Initial temperature for softmax weighting.
        tau_final: Final temperature after decay.
        tau_decay_epochs: Number of epochs over which to decay tau.
    """

    def __init__(
        self,
        top_k: int = 8,
        margin: float = 1.0,
        lambda_smooth: float = 8e-5,
        lambda_sparse: float = 8e-5,
        tau_initial: float = 1.0,
        tau_final: float = 0.07,
        tau_decay_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.margin = margin
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse

        self.tau_initial = tau_initial
        self.tau_final = tau_final
        self.tau_decay_epochs = tau_decay_epochs
        self.tau = tau_initial

    def update_tau(self, epoch: int) -> None:
        """Update the softmax temperature using exponential decay.

        Higher tau -> broader weighting across segments.
        Lower tau -> sharper focus on the highest-scoring segments.

        Args:
            epoch: Current training epoch (1-indexed).
        """
        if epoch >= self.tau_decay_epochs:
            self.tau = self.tau_final
            return

        ratio = epoch / max(1, self.tau_decay_epochs)
        self.tau = self.tau_initial * (
            (self.tau_final / self.tau_initial) ** ratio
        )

    def _ranking_loss(
        self,
        scores_abn: torch.Tensor,
        scores_nor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the AIS-style soft ranking loss.

        Uses temperature-controlled softmax weights over all 32 segments, then
        computes a weighted pooled score per video before applying hinge loss.

        Args:
            scores_abn: Anomaly scores for abnormal videos, shape ``(B_abn, 32)``.
            scores_nor: Anomaly scores for normal videos, shape ``(B_nor, 32)``.

        Returns:
            torch.Tensor: Scalar ranking loss.
        """
        # Soft weights over all segments
        weights_abn = torch.softmax(scores_abn / self.tau, dim=1)  # (B_abn, 32)
        weights_nor = torch.softmax(scores_nor / self.tau, dim=1)  # (B_nor, 32)

        # Weighted pooled score per video
        pooled_abn = (weights_abn * scores_abn).sum(dim=1)  # (B_abn,)
        pooled_nor = (weights_nor * scores_nor).sum(dim=1)  # (B_nor,)

        # If batch sizes differ, use the minimum
        min_batch = min(pooled_abn.size(0), pooled_nor.size(0))
        pooled_abn = pooled_abn[:min_batch]
        pooled_nor = pooled_nor[:min_batch]

        # Hinge loss: max(0, margin - (score_abn - score_nor))
        loss: torch.Tensor = torch.clamp(
            self.margin - (pooled_abn - pooled_nor), min=0.0
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
                - ``"ranking_loss"``: The AIS-style ranking component.
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
            tau_initial=loss_cfg["tau_initial"],
            tau_final=loss_cfg["tau_final"],
            tau_decay_epochs=loss_cfg["tau_decay_epochs"],
        )