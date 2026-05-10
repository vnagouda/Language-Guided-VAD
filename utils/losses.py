"""V4 SOTA Loss Functions for Weakly Supervised Video Anomaly Detection.

This module provides two core loss classes:

1. **VADLoss** *(primary, V4)* — Combines six components from the literature:

   a. **AIS-BCE Ranking Loss** (Light-WVAD, 2023) — Adaptive top-K BCE.
   b. **Antagonistic Loss** (Light-WVAD, 2023) — Targets hardest normal/anomaly.
   c. **Magnitude Ranking Loss** (RTFM ICCV 2021; MGFN AAAI 2023) — Hinge on L2-norm.
   d. **Temporal Smoothness** (Sultani et al. CVPR 2018) — Penalises abrupt transitions.
   e. **Feature Contrastive Loss** (V4, MGFN-inspired) — Maximises embedding-space
      separation between top-K anomalous guided features and all normal guided features.
   f. **Memory Bank Contrastive Loss** (V4, Cross-Batch ICME 2023) — Contrasts top-K
      anomalous features against a global FIFO bank of normal representations.

2. **SelfTrainingLoss** *(Phase-2 MIST)* — BCE on pseudo-labels with V4 temporal
   smoothing applied before top-K selection, eliminating flickering pseudo-labels.

Mathematical summary:
    L_total = L_AIS
            + λ_ant   · L_ant
            + λ_mag   · L_mag
            + λ_sm    · L_smooth
            + λ_ctr   · L_contrastive           (guided feature space)
            + λ_bank  · L_bank_contrastive      (memory bank)
            + λ_self  · L_self · 𝟙[Phase-2]   (applied externally)

Where:
    L_AIS    = −(1/K)Σ log(1−S_top-k^N) − (1/K)Σ log(S_top-k^P)
    L_ant    = S_top-1^N + (1 − S_top-1^P)
    L_mag    = max(0, Δ_mag − (mean‖f_abn‖_K − mean‖f_nor‖_K))
    L_smooth = Σ (s_{t+1} − s_t)² / (T−1)
    L_ctr    = max(0, Δ_ctr − ‖μ_abn_guided − μ_nor_guided‖₂)
    L_bank   = max(0, Δ_bank − ‖μ_abn_guided − μ_bank‖₂)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: Adaptive Instance Selection
# ---------------------------------------------------------------------------

def _compute_ais_k(
    scores_abn: torch.Tensor,
    scores_nor: torch.Tensor,
    threshold: float = 0.9,
    k_min: int = 3,
    warm_k: int = 8,
    epoch: int = 0,
    warm_start_epochs: int = 20,
) -> int:
    """Compute the adaptive K for a batch using the AIS confidence measure.

    V2.1 additions:
    - **Warm-start**: returns ``warm_k`` (fixed, like V1) for the first
      ``warm_start_epochs`` epochs to avoid the K=1 cold-start problem where
      only a single segment contributes gradient per video per step.
    - **K_min floor**: after warm-start, K is clamped to at least ``k_min``
      regardless of the confidence score ω, ensuring a minimum gradient signal.

    Mathematical formulation (Light-WVAD, 2023):
        ω = 1
            − (1/T) Σ S_i^N
            − (1/(2T−2)) Σ (|S_{i+1}^N − S_i^N| + |S_{i+1}^P − S_i^P|)
        K = max(k_min, floor(ω · #{i : S_i^P ≥ threshold}))

    Args:
        scores_abn: Anomaly scores for abnormal bags, shape ``(B_abn, T)``.
        scores_nor: Anomaly scores for normal bags, shape ``(B_nor, T)``.
        threshold: Score threshold ``r`` (default 0.9).
        k_min: Minimum K floor after warm-start (default 3).
        warm_k: Fixed K used during warm-start phase (default 8).
        epoch: Current training epoch (1-indexed).
        warm_start_epochs: Number of epochs to use fixed warm_k (default 20).

    Returns:
        int: Adaptive K clamped to [k_min, T].
    """
    T = scores_abn.size(1)

    # Warm-start: use fixed K for first N epochs (addresses cold-start)
    if epoch <= warm_start_epochs:
        return min(warm_k, T)

    # Mean normal score (lower is better)
    mean_nor: torch.Tensor = scores_nor.mean()
    # Temporal roughness of both bags
    diff_nor = (scores_nor[:, 1:] - scores_nor[:, :-1]).abs().mean()
    diff_abn = (scores_abn[:, 1:] - scores_abn[:, :-1]).abs().mean()

    omega = 1.0 - mean_nor.item() - 0.5 * (diff_nor.item() + diff_abn.item())
    omega = max(0.0, min(1.0, omega))

    high_conf_count = (scores_abn >= threshold).float().sum(dim=1).mean().item()
    k = max(k_min, int(omega * high_conf_count))   # floor at k_min
    k = min(k, T)
    return k


# ---------------------------------------------------------------------------
# V2.1 Primary Loss: VADLoss
# ---------------------------------------------------------------------------

class VADLoss(nn.Module):
    """V4 SOTA Combined VAD Loss.

    V4 adds over V2.1:
    - ``_feature_contrastive_loss``: Hinge loss in guided embedding space.
      Pulls anomalous cluster mean AWAY from normal cluster mean.
    - ``_memory_bank_contrastive_loss``: Same hinge but against the global
      memory bank of normal features (Cross-Batch style).
    - ``forward()`` accepts ``guided_abn``, ``guided_nor``, and optional
      ``bank_features`` tensors.

    Components:
    1. AIS-BCE ranking loss (Light-WVAD)     — adaptive Top-K selection
    2. Antagonistic loss (Light-WVAD)        — replaces L1 sparsity
    3. Magnitude ranking loss (RTFM/MGFN)   — supervises magnitude branch
    4. Temporal smoothness (Sultani et al.)  — temporal coherence
    5. Feature Contrastive Loss (MGFN-style) — embedding-space separation
    6. Memory Bank Contrastive Loss          — global normal negatives

    Args:
        ais_score_threshold: Score threshold ``r`` for AIS (default 0.9).
        ais_k_min: Minimum K floor after warm-start (default 3).
        ais_warm_start_epochs: Epochs to use fixed warm_k (default 20).
        ais_warm_k: Fixed K during warm-start (default 8).
        lambda_magnitude: Weight for magnitude ranking component.
        margin_magnitude: Hinge margin for magnitude ranking.
        lambda_antagonistic: Weight for antagonistic component.
        lambda_smooth: Weight for temporal smoothness penalty.
        lambda_prototype_cluster: Weight for prototype clustering loss (V6).
        lambda_prototype_sep: Weight for prototype separation loss (V6).
        margin_prototype: Hinge margin for pushing anomalies from prototypes (V6).
    """

    def __init__(
        self,
        ais_score_threshold: float = 0.9,
        ais_k_min: int = 3,
        ais_warm_start_epochs: int = 20,
        ais_warm_k: int = 8,
        lambda_magnitude: float = 1.0e-3,
        margin_magnitude: float = 1.0,
        lambda_antagonistic: float = 1.0,
        lambda_smooth: float = 8.0e-5,
        lambda_prototype_cluster: float = 0.05,
        lambda_prototype_sep: float = 0.05,
        margin_prototype: float = 1.0,
        lambda_snippet_contrastive: float = 0.0,
        snippet_margin: float = 2.0,
        smooth_decay_rate: float = 1.0,
        mist_start_epoch: int = 60,
    ) -> None:
        super().__init__()
        self.ais_score_threshold = ais_score_threshold
        self.ais_k_min = ais_k_min
        self.ais_warm_start_epochs = ais_warm_start_epochs
        self.ais_warm_k = ais_warm_k
        self.lambda_magnitude = lambda_magnitude
        self.margin_magnitude = margin_magnitude
        self.lambda_antagonistic = lambda_antagonistic
        self.lambda_smooth = lambda_smooth
        self.lambda_smooth_base = lambda_smooth  # Store original for decay
        self.lambda_prototype_cluster = lambda_prototype_cluster
        self.lambda_prototype_sep = lambda_prototype_sep
        self.margin_prototype = margin_prototype
        self.lambda_snippet_contrastive = lambda_snippet_contrastive
        self.snippet_margin = snippet_margin
        self.smooth_decay_rate = smooth_decay_rate
        self.mist_start_epoch = mist_start_epoch

    # ------------------------------------------------------------------
    # Component 1 — AIS-BCE Ranking Loss
    # ------------------------------------------------------------------

    def _ais_ranking_loss(
        self,
        scores_abn: torch.Tensor,
        scores_nor: torch.Tensor,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, int]:
        """Adaptive Instance Selection BCE ranking loss (V2.1: warm-start + K_min).

        Args:
            scores_abn: Anomaly scores, shape ``(B_abn, T)``.
            scores_nor: Anomaly scores, shape ``(B_nor, T)``.
            epoch: Current epoch (used for warm-start gate).

        Returns:
            tuple[torch.Tensor, int]: Scalar AIS loss and the K used.
        """
        k = _compute_ais_k(
            scores_abn, scores_nor,
            threshold=self.ais_score_threshold,
            k_min=self.ais_k_min,
            warm_k=self.ais_warm_k,
            epoch=epoch,
            warm_start_epochs=self.ais_warm_start_epochs,
        )

        # Top-K anomaly scores from pos/neg bags
        topk_abn, _ = torch.topk(scores_abn, k, dim=1)   # (B_abn, K)
        topk_nor, _ = torch.topk(scores_nor, k, dim=1)   # (B_nor, K)

        # BCE: positive bag → log(score), negative bag → log(1 − score)
        # Clamp for numerical stability
        eps = 1e-7
        topk_abn_c = topk_abn.clamp(eps, 1.0 - eps)
        topk_nor_c = topk_nor.clamp(eps, 1.0 - eps)

        loss_pos = -torch.log(topk_abn_c).mean()
        loss_neg = -torch.log(1.0 - topk_nor_c).mean()

        return loss_pos + loss_neg, k

    # ------------------------------------------------------------------
    # Component 2 — Antagonistic Loss
    # ------------------------------------------------------------------

    def _antagonistic_loss(
        self,
        scores_abn: torch.Tensor,
        scores_nor: torch.Tensor,
    ) -> torch.Tensor:
        """Antagonistic loss (Light-WVAD, 2023).

        Pushes the single most anomalous-looking normal segment toward 0
        and the single most confident anomaly segment toward 1:
            L_ant = S_top-1^N + (1 − S_top-1^P)

        This replaces the V1 L1 sparsity penalty.  The sparsity assumption
        fails when anomaly clips occupy ~20% of a T=32 video (as is common
        in UCF-Crime).  The antagonistic loss is targeted rather than
        indiscriminate.

        Args:
            scores_abn: Anomaly scores, shape ``(B_abn, T)``.
            scores_nor: Anomaly scores, shape ``(B_nor, T)``.

        Returns:
            torch.Tensor: Scalar antagonistic loss.
        """
        top1_nor = scores_nor.max(dim=1).values.mean()   # highest normal score
        top1_abn = scores_abn.max(dim=1).values.mean()   # highest anomaly score

        loss: torch.Tensor = top1_nor + (1.0 - top1_abn)
        return loss

    # ------------------------------------------------------------------
    # Component 3 — Magnitude Ranking Loss
    # ------------------------------------------------------------------

    def _magnitude_ranking_loss(
        self,
        norms_abn: torch.Tensor,
        norms_nor: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Feature magnitude hinge ranking loss (RTFM ICCV 2021; MGFN 2023).

        Enforces that the mean L2-norm of the top-K abnormal segments
        exceeds that of the top-K normal segments by at least ``margin``:
            L_mag = max(0, Δ − (mean‖f_abn‖_K − mean‖f_nor‖_K))

        Uses the same K computed by AIS for consistency.

        Args:
            norms_abn: Visual L2-norms for abnormal bags, shape ``(B_abn, T)``.
            norms_nor: Visual L2-norms for normal bags, shape ``(B_nor, T)``.
            k: Number of top-norm segments to select (from AIS).

        Returns:
            torch.Tensor: Scalar magnitude ranking loss.
        """
        k_clamped = max(1, min(k, norms_abn.size(1)))

        topk_norms_abn, _ = torch.topk(norms_abn, k_clamped, dim=1)  # (B_abn, K)
        topk_norms_nor, _ = torch.topk(norms_nor, k_clamped, dim=1)  # (B_nor, K)

        mean_abn_norm = topk_norms_abn.mean()
        mean_nor_norm = topk_norms_nor.mean()

        loss: torch.Tensor = torch.clamp(
            self.margin_magnitude - (mean_abn_norm - mean_nor_norm), min=0.0
        )
        return loss

    # ------------------------------------------------------------------
    # Component 4 — Temporal Smoothness (unchanged from V1)
    # ------------------------------------------------------------------

    def _temporal_smoothness(self, scores: torch.Tensor) -> torch.Tensor:
        """Temporal smoothness penalty.

        Penalises abrupt transitions between consecutive segment scores:
            L_smooth = (1/(T−1)) Σ (s_{t+1} − s_t)²

        Args:
            scores: Anomaly scores of shape ``(Batch, T)``.

        Returns:
            torch.Tensor: Scalar smoothness loss.
        """
        diff = scores[:, 1:] - scores[:, :-1]   # (B, T−1)
        loss: torch.Tensor = (diff ** 2).mean()
        return loss

    # ------------------------------------------------------------------
    # Component 5 & 6 — Dynamic Prototype Contrastive Loss (V6)
    # ------------------------------------------------------------------

    def _prototype_contrastive_loss(
        self,
        guided_abn: torch.Tensor,
        guided_nor: torch.Tensor,
        prototypes: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """V6 Dynamic Prototype Contrastive loss.

        Replaces both V4 Feature Contrastive and Memory Bank Contrastive losses.
        Evaluates the geometric separation of guided features against $M$ learnable
        normal prototypes.

        1. Normal Clustering: Pulls normal features towards their CLOSEST prototype.
           (Ensures the prototypes map out the true multimodal bounds of 'normal').
        2. Abnormal Separation: Pushes top-k anomalous features away from their
           CLOSEST prototype by `margin_prototype`. (Ensures strict anomaly boundaries).

        Args:
            guided_abn: Guided features for anomalous videos ``(B_abn, T, D)``.
            guided_nor: Guided features for normal videos ``(B_nor, T, D)``.
            prototypes: L2-normalised Learnable Prototypes ``(M, D)``.
            k: Top-K anomalous segments to select.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (Cluster Loss, Separation Loss)
        """
        # 1. Normal Clustering (Pull)
        # We cluster all frames of normal videos since they are entirely normal.
        B_n, T, D = guided_nor.shape
        gn_flat = guided_nor.view(B_n * T, D)  # (N, D)

        # Distance to all prototypes
        dists_nor = torch.cdist(gn_flat, prototypes, p=2) 
        min_dists_nor, _ = dists_nor.min(dim=1)  # (N,)
        cluster_loss = (min_dists_nor ** 2).mean()

        # 2. Abnormal Separation (Push)
        # Select top-K anomalous segments
        if guided_abn.size(0) > 0:
            k_c = max(1, min(k, guided_abn.size(1)))
            norms_abn = guided_abn.norm(dim=-1)            # (B_abn, T)
            _, topk_idx = torch.topk(norms_abn, k_c, dim=1)  # (B_abn, K)
            topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
            top_guided = guided_abn.gather(1, topk_idx_exp)    # (B_abn, K, D)

            ga_flat = top_guided.view(-1, D) # (B_abn * K, D)
            dists_abn = torch.cdist(ga_flat, prototypes, p=2)
            min_dists_abn, _ = dists_abn.min(dim=1) # (B_abn * K,)

            sep_loss = torch.clamp(self.margin_prototype - min_dists_abn, min=0.0).mean()
        else:
            sep_loss = torch.tensor(0.0, device=guided_nor.device)

        return cluster_loss, sep_loss

    # ------------------------------------------------------------------
    # Component 7 — Snippet Contrastive Learning (V10 APEX, Novel)
    # ------------------------------------------------------------------

    def _snippet_contrastive_loss(
        self,
        scores_abn: torch.Tensor,
        guided_abn: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """V10 APEX Snippet Contrastive Learning (SCL) — Novel Contribution.

        Enforces intra-video temporal discrimination by pushing the guided
        embeddings of the top-K highest-scored segments AWAY from the
        bottom-K lowest-scored segments within each anomalous video.

        This directly targets frame-level precision: the model must learn
        to produce geometrically distinct embeddings for anomalous vs.
        normal temporal regions within the same scene.

        Mathematical formulation:
            L_SCL = max(0, δ_scl - ‖μ_topK^guided - μ_botK^guided‖₂)

        Args:
            scores_abn: Anomaly scores for abnormal bags ``(B_abn, T)``.
            guided_abn: Guided features for abnormal bags ``(B_abn, T, D)``.
            k: Number of top/bottom segments to contrast.

        Returns:
            torch.Tensor: Scalar snippet contrastive loss.
        """
        if guided_abn.size(0) == 0:
            return torch.tensor(0.0, device=guided_abn.device)

        B, T, D = guided_abn.shape
        k_c = max(1, min(k, T // 2))  # Ensure top-K and bottom-K don't overlap

        # Top-K (most anomalous) and Bottom-K (most normal) segments per video
        _, topk_idx = torch.topk(scores_abn, k_c, dim=1)       # (B, K)
        _, botk_idx = torch.topk(scores_abn, k_c, dim=1, largest=False)  # (B, K)

        topk_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)     # (B, K, D)
        botk_exp = botk_idx.unsqueeze(-1).expand(-1, -1, D)     # (B, K, D)

        top_guided = guided_abn.gather(1, topk_exp)              # (B, K, D)
        bot_guided = guided_abn.gather(1, botk_exp)              # (B, K, D)

        # Mean embeddings per video, then average across batch
        mu_top = top_guided.mean(dim=1)  # (B, D)
        mu_bot = bot_guided.mean(dim=1)  # (B, D)

        dist = (mu_top - mu_bot).norm(dim=-1)  # (B,)
        loss: torch.Tensor = torch.clamp(self.snippet_margin - dist, min=0.0).mean()
        return loss

    # ------------------------------------------------------------------
    # Combined Forward (V10 APEX)
    # ------------------------------------------------------------------

    def forward(
        self,
        scores_abn: torch.Tensor,
        scores_nor: torch.Tensor,
        norms_abn: torch.Tensor,
        norms_nor: torch.Tensor,
        epoch: int = 0,
        guided_abn: torch.Tensor | None = None,
        guided_nor: torch.Tensor | None = None,
        prototypes: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the combined V10 APEX VAD loss.

        V10 additions over V6:
        - Snippet Contrastive Loss (intra-video temporal contrast)
        - Adaptive Smoothness Decay (epoch-dependent lambda_smooth)

        Args:
            scores_abn: Predicted anomaly scores for abnormal videos ``(B_abn, T)``.
            scores_nor: Predicted anomaly scores for normal videos ``(B_nor, T)``.
            norms_abn: Visual L2-norms for abnormal videos ``(B_abn, T)``.
            norms_nor: Visual L2-norms for normal videos ``(B_nor, T)``.
            epoch: Current training epoch (1-indexed). Used for AIS warm-start
                   and smoothness decay.
            guided_abn: Guided embeddings for abnormal bags ``(B_abn, T, D)``.
            guided_nor: Guided embeddings for normal bags ``(B_nor, T, D)``.
            prototypes: L2-normalised Learnable Prototypes ``(M, D)``.

        Returns:
            dict[str, torch.Tensor]: Loss components keyed by name.
        """
        ais_loss, k = self._ais_ranking_loss(scores_abn, scores_nor, epoch=epoch)
        antagonistic = self._antagonistic_loss(scores_abn, scores_nor)
        magnitude = self._magnitude_ranking_loss(norms_abn, norms_nor, k)
        smoothness = self._temporal_smoothness(scores_abn)

        # V10 APEX: Adaptive Smoothness Decay
        # Full strength during Phase 1, exponentially decays after MIST starts
        if self.smooth_decay_rate < 1.0 and epoch > self.mist_start_epoch:
            decay_exp = epoch - self.mist_start_epoch
            effective_smooth = self.lambda_smooth_base * (self.smooth_decay_rate ** decay_exp)
        else:
            effective_smooth = self.lambda_smooth

        total = (
            ais_loss
            + self.lambda_antagonistic * antagonistic
            + self.lambda_magnitude * magnitude
            + effective_smooth * smoothness
        )

        loss_dict: dict[str, torch.Tensor] = {
            "total_loss": total,
            "ais_loss": ais_loss,
            "antagonistic_loss": antagonistic,
            "magnitude_loss": magnitude,
            "smoothness_loss": smoothness,
        }

        # V6: Dynamic Prototype Contrastive Loss
        if guided_nor is not None and prototypes is not None:
            cluster_loss, sep_loss = self._prototype_contrastive_loss(
                guided_abn, guided_nor, prototypes, k
            )
            total = total + (self.lambda_prototype_cluster * cluster_loss) + (self.lambda_prototype_sep * sep_loss)
            loss_dict["prototype_cluster_loss"] = cluster_loss
            loss_dict["prototype_sep_loss"] = sep_loss

        # V10 APEX: Snippet Contrastive Learning (Novel)
        if self.lambda_snippet_contrastive > 0 and guided_abn is not None:
            scl_loss = self._snippet_contrastive_loss(scores_abn, guided_abn, k)
            total = total + self.lambda_snippet_contrastive * scl_loss
            loss_dict["snippet_contrastive_loss"] = scl_loss

        loss_dict["total_loss"] = total
        return loss_dict

    def __repr__(self) -> str:
        return (
            f"VADLoss(λ_ant={self.lambda_antagonistic}, λ_mag={self.lambda_magnitude}, "
            f"λ_smooth={self.lambda_smooth}, λ_proto_cluster={self.lambda_prototype_cluster}, "
            f"λ_proto_sep={self.lambda_prototype_sep}, λ_scl={self.lambda_snippet_contrastive})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "VADLoss":
        """Construct the V10 APEX loss from a configuration dictionary.

        Supports both V9-style YAML keys (``margin_mag``, ``lambda_mag``,
        ``lambda_contrastive``, ``margin_contrastive``) and canonical keys
        (``margin_magnitude``, ``lambda_magnitude``, etc.).

        Args:
            config: Full configuration dict (loaded from ``config.yaml``).

        Returns:
            VADLoss: Instantiated loss with config-driven parameters.
        """
        loss_cfg = config["loss"]

        # Helper to read with alias fallback
        def _get(primary: str, alias: str, default: float) -> float:
            return loss_cfg.get(primary, loss_cfg.get(alias, default))

        # MIST start epoch (needed for smoothness decay)
        mist_cfg = config.get("training", {}).get("mist", {})
        mist_start = mist_cfg.get("start_epoch",
                     config.get("training", {}).get("self_training_start_epoch", 60))

        return cls(
            ais_score_threshold=loss_cfg.get("ais_score_threshold", 0.9),
            ais_k_min=loss_cfg.get("ais_k_min", 3),
            ais_warm_start_epochs=loss_cfg.get("ais_warm_start_epochs", 20),
            ais_warm_k=loss_cfg.get("ais_warm_k", 8),
            lambda_magnitude=_get("lambda_magnitude", "lambda_mag", 1.0e-3),
            margin_magnitude=_get("margin_magnitude", "margin_mag", 1.0),
            lambda_antagonistic=loss_cfg.get("lambda_antagonistic", 1.0),
            lambda_smooth=loss_cfg.get("lambda_smooth", 8.0e-5),
            lambda_prototype_cluster=_get("lambda_prototype_cluster", "lambda_contrastive", 0.05),
            lambda_prototype_sep=_get("lambda_prototype_sep", "lambda_contrastive", 0.05),
            margin_prototype=_get("margin_prototype", "margin_contrastive", 1.0),
            lambda_snippet_contrastive=loss_cfg.get("lambda_snippet_contrastive", 0.0),
            snippet_margin=loss_cfg.get("snippet_margin", 2.0),
            smooth_decay_rate=loss_cfg.get("smooth_decay_rate", 1.0),
            mist_start_epoch=int(mist_start),
        )


# ---------------------------------------------------------------------------
# MIST Self-Training Helper (Phase 2, G4)
# ---------------------------------------------------------------------------

class SelfTrainingLoss(nn.Module):
    """MIST-style pseudo-label self-training BCE loss.

    V2.1 update: Uses **top-K pseudo labels** instead of top-1 argmax.
    Marking the top-3 highest-scoring segments as pseudo-positive provides
    richer gradient signal (3× more supervised instances) while remaining
    conservative enough to avoid noisy labelling.

    Mathematical formulation:
        ỹ_t = 1  if  t ∈ top-K(s_abn)   (K=mist_pseudo_k, default 3)
        ỹ_t = 0  otherwise
        L_self = BCE(s_abn, ỹ_abn) + BCE(s_nor, 0)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        scores_abn: torch.Tensor,
        scores_nor: torch.Tensor,
        pseudo_k: int = 3,
        smooth_window: int = 3,
    ) -> torch.Tensor:
        """Compute the MIST self-training BCE loss with top-K pseudo labels.

        V4 addition: **Temporal Score Smoothing** (MIST/Unbiased-MIL inspired).
        Before generating pseudo-labels, a 1D uniform moving-average kernel of
        size ``smooth_window`` is applied to the anomaly scores along the time
        dimension. This ensures pseudo-labels form temporally contiguous blobs
        rather than flickering on-off across adjacent segments.

        Mathematical formulation:
            s̃_t = (1/W) Σ_{j=t−W//2}^{t+W//2} s_j
            ỹ_t = 1  if  t ∈ top-K(s̃_abn)
            ỹ_t = 0  otherwise
            L_self = BCE(s_abn, ỹ_abn) + BCE(s_nor, 0)

        Note: Smoothing is applied to score selection only; gradients flow
        through the original ``scores_abn`` for proper back-propagation.

        Args:
            scores_abn: Predicted anomaly scores for abnormal videos ``(B_abn, T)``.
            scores_nor: Predicted anomaly scores for normal videos ``(B_nor, T)``.
            pseudo_k: Number of top-scoring segments to mark as pseudo-positive
                per anomaly bag (default 3).
            smooth_window: Temporal smoothing kernel size (default 3). Set to 1
                to disable smoothing (matches V3 behaviour).

        Returns:
            torch.Tensor: Scalar self-training BCE loss.
        """
        T = scores_abn.size(1)
        k = min(pseudo_k, T)

        # Apply temporal smoothing to guide top-K selection (V4)
        if smooth_window > 1:
            # Use 1D avg-pool as a uniform moving-average kernel
            # scores_abn: (B, T) → (B, 1, T) for conv1d
            s = scores_abn.unsqueeze(1)                      # (B, 1, T)
            pad = smooth_window // 2
            s_smooth = F.avg_pool1d(s, kernel_size=smooth_window, stride=1, padding=pad)  # (B, 1, T)
            s_smooth = s_smooth.squeeze(1)                   # (B, T)
        else:
            s_smooth = scores_abn

        # --- Anomaly bags: pseudo label = 1 at top-K smoothed positions, 0 elsewhere ---
        pseudo_abn = torch.zeros_like(scores_abn)
        _, topk_idx = torch.topk(s_smooth, k, dim=1)         # (B_abn, K)
        pseudo_abn.scatter_(1, topk_idx, 1.0)

        # --- Normal bags: all-zero pseudo labels ---
        pseudo_nor = torch.zeros_like(scores_nor)

        # BCE uses ORIGINAL (unsmoothed) scores for proper gradients
        loss_abn = F.binary_cross_entropy(scores_abn, pseudo_abn)
        loss_nor = F.binary_cross_entropy(scores_nor, pseudo_nor)
        return loss_abn + loss_nor


# ---------------------------------------------------------------------------
# V1 Legacy: MILRankingLoss (kept for backward compatibility)
# ---------------------------------------------------------------------------

class MILRankingLoss(nn.Module):
    """[V1 LEGACY] Top-K Multiple Instance Learning Ranking Loss.

    Retained for backward compatibility and ablation studies only.
    Not used in V2 training — use ``VADLoss`` instead.

    For the V2 training pipeline, see :class:`VADLoss`.

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
        lambda_smooth: float = 8.0e-5,
        lambda_sparse: float = 8.0e-5,
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
        """Compute the V1 Top-K hinge ranking loss.

        Args:
            scores_abn: Anomaly scores for abnormal videos, shape ``(B_abn, T)``.
            scores_nor: Anomaly scores for normal videos, shape ``(B_nor, T)``.

        Returns:
            torch.Tensor: Scalar ranking loss.
        """
        topk_abn, _ = torch.topk(scores_abn, self.top_k, dim=1)
        topk_nor, _ = torch.topk(scores_nor, self.top_k, dim=1)

        min_batch = min(topk_abn.size(0), topk_nor.size(0))
        topk_abn = topk_abn[:min_batch]
        topk_nor = topk_nor[:min_batch]

        loss: torch.Tensor = torch.clamp(
            self.margin - (topk_abn - topk_nor), min=0.0
        ).mean()
        return loss

    def _temporal_smoothness(self, scores: torch.Tensor) -> torch.Tensor:
        """Temporal smoothness penalty (V1).

        Args:
            scores: Anomaly scores of shape ``(Batch, T)``.

        Returns:
            torch.Tensor: Scalar smoothness loss.
        """
        diff = scores[:, 1:] - scores[:, :-1]
        loss: torch.Tensor = (diff ** 2).mean()
        return loss

    def _sparsity(self, scores: torch.Tensor) -> torch.Tensor:
        """L1 sparsity penalty (V1).

        Args:
            scores: Anomaly scores of shape ``(Batch, T)``.

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
        """Compute the V1 combined MIL ranking loss.

        Args:
            scores_abn: Predicted anomaly scores for abnormal videos,
                shape ``(B_abn, T)``.
            scores_nor: Predicted anomaly scores for normal videos,
                shape ``(B_nor, T)``.

        Returns:
            dict[str, torch.Tensor]: Dictionary with loss components.
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
        """Construct V1 loss from a configuration dictionary.

        Args:
            config: Full configuration dict (loaded from ``config.yaml``).

        Returns:
            MILRankingLoss: Instantiated V1 loss.
        """
        loss_cfg = config["loss"]
        return cls(
            top_k=loss_cfg.get("top_k", 8),
            margin=loss_cfg.get("margin", 1.0),
            lambda_smooth=loss_cfg.get("lambda_smooth", 8.0e-5),
            lambda_sparse=loss_cfg.get("lambda_sparse", 8.0e-5),
        )
