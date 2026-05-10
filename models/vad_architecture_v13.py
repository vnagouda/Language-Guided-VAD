"""Tri-Modal VAD Network — V13 Revised (Late Fusion + Stacked Optimizations).

V13 Revised improvements over V12:

1. **Late Fusion with Attention Gate**: I3D features are injected AFTER
   cross-attention via a per-segment learned gate, preserving the proven
   CLIP→CrossAttn pipeline identical to V12.

2. **LayerNorm on I3D Projection**: Aligns I3D feature distribution to
   CLIP scale, preventing gradient domination.

3. **Enhanced 4-Channel Magnitude Branch**: Feeds CLIP norms + flow +
   I3D norms + temporal difference norms for comprehensive motion scoring.

4. **Temporal Difference Features**: Computes Δi3d for anomaly onset detection.

Architecture overview (V13 Revised)::

    CLIP Visual (B,T,768) ──→ MultiScaleCrossAttn ──→ guided (B,T,768)
                                ↑ text Query                 │
    Text (B,T,768) ─────────────┘                             │
                                                              ↓
    I3D (B,T,1024) ──→ Proj+LN ──→ i3d_proj (B,T,768)       │
                                         │                    │
                              ModalityAttentionGate ──→ fused (B,T,768)
                                                              │
                                               HourglassClassifier → sem_logit
                                               EnhancedMagBranch  → mag_logit
                                               sigmoid(sem + gate*mag) → scores

    Returns: tuple[final_scores (B,T), visual_norms (B,T), guided (B,T,D)]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-module 1: Cross-Attention Block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Multi-Head Cross-Attention block with residual connection and FFN.

    Query = text features, Key = Value = visual features.

    Args:
        feature_dim: Input/output feature dimensionality.
        num_heads: Number of attention heads.
        ff_dim: Hidden dimension of the FFN.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        ff_dim: int = 3072,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        assert feature_dim % num_heads == 0
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, feature_dim), nn.Dropout(dropout),
        )

    def forward(
        self, text_features: torch.Tensor, visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention forward pass.

        Args:
            text_features:   ``(B, T, D)`` — Query.
            visual_features: ``(B, T, D)`` — Key and Value.

        Returns:
            torch.Tensor: Language-guided features ``(B, T, D)``.
        """
        attn_out, _ = self.cross_attn(
            query=text_features, key=visual_features, value=visual_features,
        )
        x = self.norm1(text_features + attn_out)
        return self.norm2(x + self.ffn(x))


# ---------------------------------------------------------------------------
# Sub-module 2: Multi-Scale Cross-Attention
# ---------------------------------------------------------------------------

class MultiScaleCrossAttention(nn.Module):
    """Multi-Scale Cross-Attention at 3 temporal scales [1, 2, 4].

    Args:
        feature_dim: CLIP feature dimensionality.
        num_heads: Attention heads.
        ff_dim: FFN hidden dim.
        dropout: Dropout probability.
        scales: Temporal pooling factors.
    """

    def __init__(
        self, feature_dim: int = 768, num_heads: int = 8,
        ff_dim: int = 3072, dropout: float = 0.5,
        scales: list[int] | None = None,
    ) -> None:
        super().__init__()
        if scales is None:
            scales = [1, 2, 4]
        self.scales = scales
        self.attn_blocks = nn.ModuleList([
            CrossAttentionBlock(feature_dim, num_heads, ff_dim, dropout)
            for _ in scales
        ])
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))

    def forward(
        self, text_features: torch.Tensor, visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-scale cross-attention forward.

        Args:
            text_features:   ``(B, T, D)``
            visual_features: ``(B, T, D)``

        Returns:
            torch.Tensor: Fused guided features ``(B, T, D)``.
        """
        B, T, D = visual_features.shape
        scale_outputs: list[torch.Tensor] = []
        fusion_weights = F.softmax(self.scale_weights, dim=0)

        for i, (scale, block) in enumerate(zip(self.scales, self.attn_blocks)):
            if scale == 1:
                v_scaled, t_scaled = visual_features, text_features
            else:
                v_perm = visual_features.permute(0, 2, 1)
                t_perm = text_features.permute(0, 2, 1)
                v_pool = F.avg_pool1d(v_perm, kernel_size=scale, stride=scale)
                t_pool = F.avg_pool1d(t_perm, kernel_size=scale, stride=scale)
                v_scaled = v_pool.permute(0, 2, 1)
                t_scaled = t_pool.permute(0, 2, 1)

            attended = block(text_features=t_scaled, visual_features=v_scaled)

            if scale != 1:
                attended_perm = attended.permute(0, 2, 1)
                attended_up = F.interpolate(
                    attended_perm, size=T, mode="linear", align_corners=False
                )
                attended = attended_up.permute(0, 2, 1)

            scale_outputs.append(fusion_weights[i] * attended)

        return torch.stack(scale_outputs, dim=0).sum(dim=0)


# ---------------------------------------------------------------------------
# Sub-module 3: Hourglass FC Classifier
# ---------------------------------------------------------------------------

class HourglassClassifier(nn.Module):
    """Hourglass-shaped FC classifier returning unnormalised logit.

    Args:
        feature_dim: Input dimensionality.
        bottleneck_dim: Compressed dimension.
        hidden_dim: Expansion dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self, feature_dim: int = 768, bottleneck_dim: int = 64,
        hidden_dim: int = 128, dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-segment semantic logits.

        Args:
            x: Guided features ``(B, T, D)``.

        Returns:
            torch.Tensor: Raw logits ``(B, T)``.
        """
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Sub-module 4: Enhanced Magnitude Branch (V13 — 4-channel)
# ---------------------------------------------------------------------------

class EnhancedMagnitudeBranch(nn.Module):
    """V13 Enhanced magnitude branch with up to 4 input channels.

    Channels:
        1. CLIP visual L2-norms (appearance magnitude)
        2. Optical flow magnitudes (pixel-level motion)
        3. I3D projected L2-norms (action-level motion magnitude)
        4. I3D temporal difference norms (motion transition / onset)

    All inputs are Z-score normalised per-video before the linear layers.

    Args:
        num_channels: Number of input channels (1–4).
        dropout: Dropout probability.
    """

    def __init__(self, num_channels: int = 4, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _zscore(self, x: torch.Tensor) -> torch.Tensor:
        """Per-video Z-score normalisation.

        Args:
            x: Shape ``(B, T)``.

        Returns:
            torch.Tensor: Normalised ``(B, T)``.
        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        return (x - mean) / std

    def forward(
        self,
        clip_norms: torch.Tensor,
        flow_magnitudes: torch.Tensor | None = None,
        i3d_norms: torch.Tensor | None = None,
        i3d_delta_norms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score multi-channel magnitudes → raw logit.

        Args:
            clip_norms:      ``(B, T)`` CLIP visual L2-norms.
            flow_magnitudes: ``(B, T)`` optical flow scalars.
            i3d_norms:       ``(B, T)`` I3D projected L2-norms.
            i3d_delta_norms: ``(B, T)`` I3D temporal difference norms.

        Returns:
            torch.Tensor: Raw magnitude logits ``(B, T)``.
        """
        channels = [self._zscore(clip_norms)]

        if flow_magnitudes is not None:
            channels.append(self._zscore(flow_magnitudes))
        else:
            channels.append(torch.zeros_like(clip_norms))

        if i3d_norms is not None:
            channels.append(self._zscore(i3d_norms))
        else:
            channels.append(torch.zeros_like(clip_norms))

        if i3d_delta_norms is not None:
            channels.append(self._zscore(i3d_delta_norms))
        else:
            channels.append(torch.zeros_like(clip_norms))

        # Stack to (B, T, num_channels), take only what we need
        x = torch.stack(channels[:self.num_channels], dim=-1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Sub-module 5: Modality Attention Gate (V13 — Late Fusion)
# ---------------------------------------------------------------------------

class ModalityAttentionGate(nn.Module):
    """Per-segment learned gate for blending guided features with I3D.

    gate_t = σ(Linear([guided_t || i3d_proj_t]))
    fused_t = guided_t + gate_t * i3d_proj_t

    When I3D is zero (missing data), gate→~0.5*bias, effectively ignored.

    Args:
        feature_dim: Dimensionality of guided and I3D projected features.
    """

    def __init__(self, feature_dim: int = 768) -> None:
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

    def forward(
        self, guided: torch.Tensor, i3d_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-gated fusion.

        Args:
            guided:   ``(B, T, D)`` language-guided features.
            i3d_proj: ``(B, T, D)`` projected I3D features.

        Returns:
            torch.Tensor: Fused features ``(B, T, D)``.
        """
        concat = torch.cat([guided, i3d_proj], dim=-1)  # (B, T, 2D)
        gate = torch.sigmoid(self.gate_fc(concat))       # (B, T, 1)
        return guided + gate * i3d_proj


# ---------------------------------------------------------------------------
# Sub-module 6: Learnable Normal Prototypes (V6)
# ---------------------------------------------------------------------------

class DynamicNormalPrototypes(nn.Module):
    """V6 Learnable Normal Prototypes.

    Args:
        feature_dim: Dimension of guided feature vectors.
        num_prototypes: Number of cluster centers.
    """

    def __init__(self, feature_dim: int = 768, num_prototypes: int = 16) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        nn.init.normal_(self.prototypes, std=0.02)

    def get(self) -> torch.Tensor:
        """Retrieve L2-normalised prototypes."""
        return F.normalize(self.prototypes, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Main model: TriModalVAD V13 Revised
# ---------------------------------------------------------------------------

class TriModalVAD(nn.Module):
    """Tri-Modal Fusion VAD Network — V13 Revised (Late Fusion).

    Core design: The CLIP → MultiScaleCrossAttention pipeline is kept
    IDENTICAL to V12. I3D is injected AFTER cross-attention via an
    attention-gated residual connection.

    Args:
        feature_dim: CLIP feature dimensionality (768).
        num_segments: T temporal segments per video.
        num_heads: Attention heads.
        num_layers: Stacked cross-attention blocks.
        ff_dim: FFN hidden dim.
        classifier_bottleneck_dim: Hourglass compress dim.
        classifier_hidden_dim: Hourglass expand dim.
        dropout: Dropout probability.
        use_magnitude_branch: Toggle magnitude branch.
        use_multi_scale: Use MultiScaleCrossAttention.
        use_i3d_fusion: Enable I3D fusion.
        i3d_dim: Raw I3D feature dimensionality (1024).
        i3d_fusion_type: 'late_attention_gate' or 'early_blend'.
        use_i3d_in_magnitude: Feed I3D norms to magnitude branch.
        use_i3d_temporal_diff: Compute and use Δi3d features.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_segments: int = 128,
        num_heads: int = 8,
        num_layers: int = 1,
        ff_dim: int = 3072,
        classifier_bottleneck_dim: int = 64,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.5,
        use_magnitude_branch: bool = True,
        use_multi_scale: bool = True,
        use_i3d_fusion: bool = True,
        i3d_dim: int = 1024,
        i3d_fusion_type: str = "late_attention_gate",
        use_i3d_in_magnitude: bool = True,
        use_i3d_temporal_diff: bool = True,
    ) -> None:
        super().__init__()

        self.num_segments = num_segments
        self.feature_dim = feature_dim
        self.use_magnitude_branch = use_magnitude_branch
        self.use_multi_scale = use_multi_scale
        self.use_i3d_fusion = use_i3d_fusion
        self.i3d_fusion_type = i3d_fusion_type
        self.use_i3d_in_magnitude = use_i3d_in_magnitude
        self.use_i3d_temporal_diff = use_i3d_temporal_diff

        # --- (V13) I3D Projection with LayerNorm ---
        if self.use_i3d_fusion:
            self.i3d_projection = nn.Sequential(
                nn.Linear(i3d_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # Only create attention gate for late fusion
            if i3d_fusion_type == "late_attention_gate":
                self.modality_gate = ModalityAttentionGate(feature_dim=feature_dim)
            else:
                # Early blend: learnable scalar alpha
                self.blend_alpha = nn.Parameter(torch.tensor(0.3))

        # --- Cross-attention stack ---
        if use_multi_scale:
            self.attention_module: nn.Module = MultiScaleCrossAttention(
                feature_dim=feature_dim, num_heads=num_heads,
                ff_dim=ff_dim, dropout=dropout, scales=[1, 2, 4],
            )
        else:
            self.attention_module = nn.Sequential(*[
                CrossAttentionBlock(feature_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ])

        # --- Hourglass classifier ---
        self.classifier = HourglassClassifier(
            feature_dim=feature_dim,
            bottleneck_dim=classifier_bottleneck_dim,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

        # --- Enhanced Magnitude Branch (Optimization 3) ---
        if use_magnitude_branch:
            mag_channels = 4 if (use_i3d_fusion and use_i3d_in_magnitude) else 2
            self.magnitude_branch: nn.Module | None = EnhancedMagnitudeBranch(
                num_channels=mag_channels, dropout=dropout,
            )
            self.fusion_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.magnitude_branch = None
            self.fusion_gate = None  # type: ignore[assignment]

    def _compute_temporal_diff_norms(
        self, i3d_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal difference norms: ||i3d_proj_{t+1} - i3d_proj_t||₂.

        Args:
            i3d_proj: ``(B, T, D)`` projected I3D features.

        Returns:
            torch.Tensor: ``(B, T)`` temporal difference norms (0-padded at end).
        """
        # Δ = i3d_proj[:, 1:, :] - i3d_proj[:, :-1, :]  → (B, T-1, D)
        delta = i3d_proj[:, 1:, :] - i3d_proj[:, :-1, :]
        delta_norms = delta.norm(dim=-1)  # (B, T-1)
        # Pad last position with 0
        pad = torch.zeros(delta_norms.shape[0], 1, device=delta_norms.device)
        return torch.cat([delta_norms, pad], dim=1)  # (B, T)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        flow_magnitudes: torch.Tensor | None = None,
        i3d_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """V13 Revised forward: CLIP→CrossAttn→Late I3D Fusion→Classifier.

        Args:
            visual_features:  ``(B, T, 768)`` CLIP visual features.
            text_features:    ``(B, T, 768)`` BLIP-2/CLIP text features.
            flow_magnitudes:  ``(B, T)`` optical flow scalars.
            i3d_features:     ``(B, T, i3d_dim)`` raw I3D features.

        Returns:
            tuple: (final_scores, visual_norms, guided) each ``(B, T, ...)``.
        """
        # --- Visual L2-norms (computed on PURE CLIP, before any fusion) ---
        visual_norms: torch.Tensor = visual_features.norm(dim=-1)  # (B, T)

        # --- (V16) Early Fusion: blend I3D into visual BEFORE cross-attention ---
        if self.use_i3d_fusion and self.i3d_fusion_type == "early_blend" and i3d_features is not None:
            i3d_proj = self.i3d_projection(i3d_features)  # (B, T, 768)
            alpha = torch.sigmoid(self.blend_alpha)
            visual_features = (1 - alpha) * visual_features + alpha * i3d_proj

        # --- Cross-Attention ---
        if self.use_multi_scale:
            guided: torch.Tensor = self.attention_module(
                text_features=text_features,
                visual_features=visual_features,
            )
        else:
            guided = text_features
            for layer in self.attention_module:
                guided = layer(text_features=guided, visual_features=visual_features)

        # --- (V14/V15) Late Fusion: I3D injected AFTER cross-attention ---
        i3d_proj_late: torch.Tensor | None = None
        i3d_norms: torch.Tensor | None = None
        i3d_delta_norms: torch.Tensor | None = None

        if self.use_i3d_fusion and self.i3d_fusion_type == "late_attention_gate" and i3d_features is not None:
            i3d_proj_late = self.i3d_projection(i3d_features)  # (B, T, 768)
            guided = self.modality_gate(guided, i3d_proj_late)  # Attention-gated fusion

            if self.use_i3d_in_magnitude:
                feat = i3d_proj_late if i3d_proj_late is not None else i3d_proj
                if feat is not None:
                    i3d_norms = feat.norm(dim=-1)

            if self.use_i3d_temporal_diff:
                feat = i3d_proj_late if i3d_proj_late is not None else i3d_proj
                if feat is not None:
                    i3d_delta_norms = self._compute_temporal_diff_norms(feat)

        # --- Semantic logit ---
        semantic_logit: torch.Tensor = self.classifier(guided)  # (B, T)

        # --- Magnitude branch ---
        if self.magnitude_branch is not None and self.fusion_gate is not None:
            mag_logit: torch.Tensor = self.magnitude_branch(
                clip_norms=visual_norms,
                flow_magnitudes=flow_magnitudes,
                i3d_norms=i3d_norms,
                i3d_delta_norms=i3d_delta_norms,
            )
            fused_logit = semantic_logit + self.fusion_gate * mag_logit
        else:
            fused_logit = semantic_logit

        final_scores: torch.Tensor = torch.sigmoid(fused_logit)  # (B, T)

        return final_scores, visual_norms, guided

    @classmethod
    def from_config(cls, config: dict) -> "TriModalVAD":
        """Construct the Tri-Modal model from a configuration dictionary.

        Args:
            config: Full config dict loaded from a YAML file.

        Returns:
            TriModalVAD: Instantiated model.
        """
        m = config["model"]
        return cls(
            feature_dim=m["feature_dim"],
            num_segments=m["num_segments"],
            num_heads=m["num_heads"],
            num_layers=m.get("num_layers", 1),
            ff_dim=m["ff_dim"],
            classifier_bottleneck_dim=m["classifier_bottleneck_dim"],
            classifier_hidden_dim=m["classifier_hidden_dim"],
            dropout=m["dropout"],
            use_magnitude_branch=m.get("use_magnitude_branch", True),
            use_multi_scale=m.get("use_multi_scale", True),
            use_i3d_fusion=m.get("use_i3d_fusion", True),
            i3d_dim=m.get("i3d_dim", 1024),
            i3d_fusion_type=m.get("i3d_fusion_type", "late_attention_gate"),
            use_i3d_in_magnitude=m.get("use_i3d_in_magnitude", True),
            use_i3d_temporal_diff=m.get("use_i3d_temporal_diff", True),
        )
