"""Language-Guided Cross-Attention VAD Network — V4 SOTA.

V4 improvements over V3:

1. **Multi-Scale Cross-Attention (MSBT, 'Enhancing WS-VAD via Text Guidance' ArXiv 2024):**
   Computes Language-Guided Cross-Attention at 3 temporal resolutions (T=32, T=16, T=8)
   using adaptive average pooling. Outputs are fused via learnable softmax-normalised
   weights, giving the model simultaneous fine-grained and coarse temporal reasoning.
   Expected gain: +2–3% Frame-AUROC.

2. **Normal Feature Memory Bank (Cross-Batch Clustering, ICME 2023):**
   An EMA-updated FIFO queue `(bank_size, 768)` stores guided representations from
   all normal videos seen during training. This provides a *global* negative set for
   contrastive supervision, preventing "hard-normal" (e.g., camera-shake) clips from
   being misclassified. Expected gain: +1–2% Frame-AUROC.

3. **forward() returns 3 values:**
   ``(final_scores, visual_norms, guided_features)`` — the guided embeddings are now
   exposed to the loss function for the V4 Feature Contrastive Loss.

Architecture overview (V4)::

    Text Features    (B, 32, 768) ──────────────────────────────────┐
                                                                     │ (MultiScaleCrossAttention)
    Visual Features  (B, 32, 768) ──── pool×[1,2,4] → CrossAttn ───┘
                                                        fuse (learnable)
                                                             │
                                                  guided (B, 32, 768)
                                                             │
                                              HourglassClassifier
                                            768 → 64 → 128 → logit (B, 32)
                                                      │ (semantic_logit)
                                                      ├── + gate × mag_logit
                                                      │             ↑
                                     [visual.norm(−1)] → MagnitudeBranch
                                                      │
                                             sigmoid(fused_logit)
                                                      │
                                              final_scores (B, 32) ∈ [0, 1]

    Returns: tuple[final_scores (B,T),  visual_norms (B,T),  guided (B,T,D)]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-module 1: Cross-Attention Block (unchanged from V3)
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """A single Multi-Head Cross-Attention block with residual connection and FFN.

    Query = text features (semantic guidance),  Key = Value = visual features.

    The block implements::

        attn_out = MultiHeadAttention(Q=text, K=visual, V=visual)
        x        = LayerNorm(text + attn_out)
        output   = LayerNorm(x + FFN(x))

    Args:
        feature_dim: Input/output feature dimensionality.
                     512 for ViT-B/16, 768 for ViT-L/14.
        num_heads: Number of attention heads. Must divide feature_dim evenly.
                   Default 8 → head_dim = 96 for 768-dim.
        ff_dim: Hidden dimension of the position-wise FFN. Standard: 4 × feature_dim.
        dropout: Dropout probability applied in FFN and attention.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        ff_dim: int = 3072,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        assert feature_dim % num_heads == 0, (
            f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads}). "
            f"head_dim = {feature_dim}/{num_heads} = {feature_dim/num_heads}"
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention forward pass.

        Args:
            text_features:   ``(Batch, T, feature_dim)`` — Query (language guide).
            visual_features: ``(Batch, T, feature_dim)`` — Key and Value.

        Returns:
            torch.Tensor: Language-guided features ``(Batch, T, feature_dim)``.
        """
        attn_out, _ = self.cross_attn(
            query=text_features,
            key=visual_features,
            value=visual_features,
        )
        x = self.norm1(text_features + attn_out)
        return self.norm2(x + self.ffn(x))


# ---------------------------------------------------------------------------
# Sub-module 2: Multi-Scale Cross-Attention (V4 — MSBT inspired)
# ---------------------------------------------------------------------------

class MultiScaleCrossAttention(nn.Module):
    """Multi-Scale Cross-Attention block (MSBT-lite).

    Computes Language-Guided Cross-Attention at 3 temporal scales:
        - Scale 1: T=32 (original, fine-grained)
        - Scale 2: T=16 (2× pooled, medium)
        - Scale 4: T=8  (4× pooled, coarse)

    The three attended feature sequences are upsampled back to T=32, then fused
    via softmax-normalised learnable scalar weights::

        fused = Σ_s  w_s * scale_s_output          (w_s = softmax(raw_weights))

    Mathematical justification:
        Attending at scale-2 allows the model to correlate an anomalous action
        (e.g., "person falling") with the surrounding context (e.g., "crowd
        forming"). The coarse scale captures long-range temporal dependencies
        that single-scale attention misses. This directly mirrors the Multi-Scale
        Behavioral Transformer in the 87.46% AUC paper.

    Args:
        feature_dim: CLIP feature dimensionality. Default 768 (ViT-L/14).
        num_heads: Attention heads (must divide feature_dim). Default 8.
        ff_dim: FFN hidden dim. Standard: 4 × feature_dim.
        dropout: Dropout probability.
        scales: Temporal pooling factors. Default [1, 2, 4].
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        ff_dim: int = 3072,
        dropout: float = 0.5,
        scales: list[int] | None = None,
    ) -> None:
        super().__init__()
        if scales is None:
            scales = [1, 2, 4]
        self.scales = scales

        self.attn_blocks = nn.ModuleList([
            CrossAttentionBlock(
                feature_dim=feature_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in scales
        ])

        # Learnable per-scale fusion weights (before softmax)
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))

    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-scale cross-attention forward pass.

        Args:
            text_features:   ``(Batch, T, feature_dim)``
            visual_features: ``(Batch, T, feature_dim)``

        Returns:
            torch.Tensor: Fused language-guided features ``(Batch, T, feature_dim)``.
        """
        B, T, D = visual_features.shape
        scale_outputs: list[torch.Tensor] = []

        fusion_weights = F.softmax(self.scale_weights, dim=0)  # (num_scales,)

        for i, (scale, block) in enumerate(zip(self.scales, self.attn_blocks)):
            if scale == 1:
                v_scaled = visual_features      # (B, T, D)
                t_scaled = text_features        # (B, T, D)
            else:
                # Pool along temporal dim: (B, T, D) → (B, T//scale, D)
                # Permute to (B, D, T) for F.avg_pool1d, then back
                v_perm = visual_features.permute(0, 2, 1)   # (B, D, T)
                t_perm = text_features.permute(0, 2, 1)     # (B, D, T)
                v_pool = F.avg_pool1d(v_perm, kernel_size=scale, stride=scale)  # (B, D, T//scale)
                t_pool = F.avg_pool1d(t_perm, kernel_size=scale, stride=scale)  # (B, D, T//scale)
                v_scaled = v_pool.permute(0, 2, 1)  # (B, T//scale, D)
                t_scaled = t_pool.permute(0, 2, 1)  # (B, T//scale, D)

            attended = block(text_features=t_scaled, visual_features=v_scaled)  # (B, T//scale, D)

            if scale != 1:
                # Upsample back to T: (B, T//scale, D) → (B, T, D)
                attended_perm = attended.permute(0, 2, 1)                          # (B, D, T//scale)
                attended_up   = F.interpolate(attended_perm, size=T, mode="linear", align_corners=False)
                attended = attended_up.permute(0, 2, 1)                            # (B, T, D)

            scale_outputs.append(fusion_weights[i] * attended)  # weighted

        fused: torch.Tensor = torch.stack(scale_outputs, dim=0).sum(dim=0)  # (B, T, D)
        return fused


# ---------------------------------------------------------------------------
# Sub-module 3: Hourglass FC Classifier (unchanged from V3)
# ---------------------------------------------------------------------------

class HourglassClassifier(nn.Module):
    """Hourglass-shaped FC classifier — returns unnormalised logit.

    Architecture::

        feature_dim →(compress)→ bottleneck_dim →(expand)→ hidden_dim → 1 (logit)
              768   →                64          →            128       → 1

    No Sigmoid here — applied externally after logit-level fusion with
    the magnitude branch (pre-sigmoid additive fusion, V2.1 design).

    Args:
        feature_dim: Input dimensionality (768 for ViT-L/14).
        bottleneck_dim: Compressed bottleneck dimension (default 64).
        hidden_dim: Expansion dimension (default 128).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        bottleneck_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            # NO Sigmoid — fused externally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-segment semantic logits.

        Args:
            x: Guided features ``(Batch, T, feature_dim)``.

        Returns:
            torch.Tensor: Raw logits ``(Batch, T)`` (unbounded).
        """
        return self.net(x).squeeze(-1)  # (B, T, 1) → (B, T)


# ---------------------------------------------------------------------------
# Sub-module 4: Enhanced Magnitude Branch — L2-norm + optical flow (V3)
# ---------------------------------------------------------------------------

class MagnitudeBranch(nn.Module):
    """Enhanced magnitude scoring branch — accepts L2-norm + flow magnitude.

    V3 change over V2.1:
        The branch now takes a 2-channel input: the visual feature L2-norm AND
        the optical flow magnitude. This provides both an appearance-magnitude
        signal (anomalous segments often have larger/different-magnitude features
        in ViT-L/14) and a direct motion signal (fights, crashes have high flow).

    Architecture::

        [norm_hat, flow_hat] → Linear(2→32) → ReLU → Linear(32→1) → logit
                                                                        ↑
                                                    NO Sigmoid (fused externally)

    Z-score normalisation (per-batch) is applied to both inputs independently
    to bring them to approximately N(0,1) before the linear layer.

    Mathematical formulation:
        norm_hat  = (‖f_t‖₂ − μ_norm) / (σ_norm + ε)
        flow_hat  = (m_t − μ_flow) / (σ_flow + ε)
        x_t       = [norm_hat_t, flow_hat_t]  ∈ ℝ²
        mag_logit = Linear_2(ReLU(Linear_1(x_t)))

    Backward-compatible mode (use_flow=False):
        Accepts only visual_norms → z-score → Linear(1→1), matching V2.1 exactly.

    Args:
        dropout: Dropout probability applied before the first Linear.
        use_flow: If True, accepts (norms, flow) 2-channel input (V3 default).
                  If False, accepts only norms (backward-compatible with V2.1).
    """

    def __init__(
        self,
        dropout: float = 0.5,
        use_flow: bool = True,
    ) -> None:
        super().__init__()
        self.use_flow = use_flow
        self.dropout = nn.Dropout(dropout)

        if use_flow:
            self.fc = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(1, 1),
            )

    def _zscore(self, x: torch.Tensor) -> torch.Tensor:
        """Per-batch Z-score normalisation along time dimension.

        Args:
            x: Shape ``(Batch, T)``.

        Returns:
            torch.Tensor: Normalised ``(Batch, T)`` ≈ N(0, 1) per video.
        """
        mean = x.mean(dim=1, keepdim=True)    # (B, 1)
        std  = x.std(dim=1, keepdim=True) + 1e-6
        return (x - mean) / std

    def forward(
        self,
        visual_norms: torch.Tensor,
        flow_magnitudes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score normalised magnitudes → raw magnitude logit.

        Args:
            visual_norms: L2-norms of visual features, shape ``(Batch, T)``.
            flow_magnitudes: Optional optical flow scalars, shape ``(Batch, T)``.
                             Required when ``use_flow=True``.

        Returns:
            torch.Tensor: Raw magnitude logits ``(Batch, T)`` (no sigmoid).
        """
        norm_hat: torch.Tensor = self._zscore(visual_norms)  # (B, T)

        if self.use_flow and flow_magnitudes is not None:
            flow_hat: torch.Tensor = self._zscore(flow_magnitudes)  # (B, T)
            x = torch.stack([norm_hat, flow_hat], dim=-1)   # (B, T, 2)
        else:
            x = norm_hat.unsqueeze(-1)                       # (B, T, 1)

        x = self.dropout(x)
        return self.fc(x).squeeze(-1)   # (B, T)


# ---------------------------------------------------------------------------
# Sub-module 5: Learnable Normal Prototypes (V6)
# ---------------------------------------------------------------------------

class DynamicNormalPrototypes(nn.Module):
    """V6 Learnable Normal Prototypes.

    Replaces the V4/V5 Global FIFO Memory Bank.
    Instead of maintaining a generic history of normal frames (which blends
    multimodal safe concepts into a blurry mean), this module learns $M$ distinct
    mathematical cluster centers.

    This ensures sharp geometric boundaries mapping 'walking vs driving vs standing'.

    Args:
        feature_dim: Dimension of guided feature vectors. Default 768.
        num_prototypes: Number of cluster centers. Default 16.
    """

    def __init__(self, feature_dim: int = 768, num_prototypes: int = 16) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        nn.init.normal_(self.prototypes, std=0.02)

    def get(self) -> torch.Tensor:
        """Retrieve L2-normalised prototypes."""
        return F.normalize(self.prototypes, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Sub-module 6: Pyramid of Dilated Convolutions (V7 Temporal Dynamics)
# ---------------------------------------------------------------------------

class PyramidDilatedConv(nn.Module):
    """V7 Pyramid of Dilated Convolutions (RTFM Style).

    Reconstructs chronological motion-gradients across static appearance frames
    by applying 1D Convolutions across the sequence dimension (T) with multiple
    dilation rates.

    Args:
        feature_dim: Dimensionality of the input visual features.
    """

    def __init__(self, feature_dim: int = 768) -> None:
        super().__init__()
        # PyTorch Conv1D expects (Batch, Channels, Length)
        # Dilations: 1 (adjacent), 2 (skip 1), 4 (skip 3)
        self.conv1 = nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=4, dilation=4)
        
        self.fusion = nn.Linear(feature_dim * 3, feature_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Visual features (Batch, T, feature_dim)
        Returns:
            torch.Tensor: Motion-infused features (Batch, T, feature_dim)
        """
        # Permute to (Batch, Channels, T)
        x_p = x.permute(0, 2, 1)
        
        out1 = self.relu(self.conv1(x_p))
        out2 = self.relu(self.conv2(x_p))
        out3 = self.relu(self.conv3(x_p))
        
        # Concatenate along channel dim: (B, Channels*3, T)
        cat_out = torch.cat([out1, out2, out3], dim=1)
        # Permute back: (B, T, Channels*3)
        cat_out = cat_out.permute(0, 2, 1)
        
        fused = self.dropout(self.relu(self.fusion(cat_out)))
        # Residual connection
        return fused + x


# ---------------------------------------------------------------------------
# Main model: LanguageGuidedVAD V4
# ---------------------------------------------------------------------------

class LanguageGuidedVAD(nn.Module):
    """Language-Guided Video Anomaly Detection Network — V4 SOTA.

    V4 changes over V3:
    - ``MultiScaleCrossAttention`` replaces the single ``CrossAttentionBlock``
      stack when ``use_multi_scale=True`` (default). Attends at T=32, T=16, T=8.
    - ``forward()`` now returns a 3-tuple:
      ``(final_scores, visual_norms, guided_features)``
      so the loss function can compute the Feature Contrastive Loss in embedding
      space (Component 1 of V4 loss).
    - ``NormalMemoryBank`` is maintained externally in the training loop and
      passed to the loss function. The architecture itself does not hold the bank
      (to allow the bank to persist across mini-batches without being part of
      back-propagation).
    - All V3 sub-modules remain unchanged for backward compatibility.

    Args:
        feature_dim: CLIP feature dimensionality. 768 for ViT-L/14 (V3/V4),
                     512 for ViT-B/16 (V2.x).
        num_segments: T = 32 temporal segments per video.
        num_heads: Attention heads (8 for both 512 and 768 dim).
        num_layers: Stacked cross-attention blocks.
        ff_dim: FFN hidden dim. Standard: 4 × feature_dim.
        classifier_bottleneck_dim: Hourglass compress dim.
        classifier_hidden_dim: Hourglass expand dim.
        dropout: Dropout probability.
        use_magnitude_branch: Toggle the entire magnitude branch.
        use_flow_in_magnitude: If True, MagnitudeBranch uses 2-channel input.
        use_multi_scale: If True, use ``MultiScaleCrossAttention`` (V4 default).
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_segments: int = 32,
        num_heads: int = 8,
        num_layers: int = 1,
        ff_dim: int = 3072,
        classifier_bottleneck_dim: int = 64,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.5,
        use_magnitude_branch: bool = True,
        use_flow_in_magnitude: bool = False,
        use_multi_scale: bool = True,
        use_temporal_convolutions: bool = False,
    ) -> None:
        super().__init__()

        self.num_segments = num_segments
        self.feature_dim = feature_dim
        self.use_magnitude_branch = use_magnitude_branch
        self.use_flow_in_magnitude = use_flow_in_magnitude
        self.use_multi_scale = use_multi_scale
        self.use_temporal_convolutions = use_temporal_convolutions

        # (V7) Temporal Motion Extractor
        if self.use_temporal_convolutions:
            self.temporal_extractor = PyramidDilatedConv(feature_dim=feature_dim)
        else:
            self.temporal_extractor = None

        # Cross-attention stack (single-scale or multi-scale)
        if use_multi_scale:
            self.attention_module: nn.Module = MultiScaleCrossAttention(
                feature_dim=feature_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                scales=[1, 2, 4],
            )
        else:
            self.attention_module = nn.Sequential(*[
                CrossAttentionBlock(
                    feature_dim=feature_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])

        # Hourglass FC classifier — outputs logit (no Sigmoid)
        self.classifier = HourglassClassifier(
            feature_dim=feature_dim,
            bottleneck_dim=classifier_bottleneck_dim,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

        # Magnitude branch
        if use_magnitude_branch:
            self.magnitude_branch: nn.Module | None = MagnitudeBranch(
                dropout=dropout,
                use_flow=use_flow_in_magnitude,
            )
            self.fusion_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.magnitude_branch = None
            self.fusion_gate = None  # type: ignore[assignment]

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        flow_magnitudes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """V4 forward pass: multi-scale cross-attention → additive logit fusion → sigmoid.

        Tensor flow::

            visual_features:   (B, T, 768)
            text_features:     (B, T, 768)
            guided (post-attn):(B, T, 768)   ← returned as 3rd element
            semantic_logit:    (B, T)
            visual_norms:      (B, T)
            mag_logit:         (B, T)
            fused_logit:       (B, T)
            final_scores:      (B, T) ∈ [0,1]

        Args:
            visual_features:  ``(Batch, T, feature_dim)`` raw visual features.
            text_features:    ``(Batch, T, feature_dim)`` CLIP text features.
            flow_magnitudes:  ``(Batch, T)`` per-segment optical flow scalars.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - **final_scores**: ``(Batch, T)`` anomaly scores ∈ [0, 1].
                - **visual_norms**: ``(Batch, T)`` raw L2-norms (for mag loss).
                - **guided**: ``(Batch, T, feature_dim)`` guided embeddings (for contrastive loss).
        """
        # --- (V7) Temporal Motion Extraction ---
        if self.use_temporal_convolutions and self.temporal_extractor is not None:
            visual_features = self.temporal_extractor(visual_features)

        # --- (Multi-Scale) Cross-Attention ---
        if self.use_multi_scale:
            guided: torch.Tensor = self.attention_module(
                text_features=text_features,
                visual_features=visual_features,
            )
        else:
            guided = text_features
            for layer in self.attention_module:
                guided = layer(text_features=guided, visual_features=visual_features)

        # --- Semantic logit from Hourglass FC ---
        semantic_logit: torch.Tensor = self.classifier(guided)  # (B, T)

        # --- Visual L2-norms (for magnitude loss and magnitude branch) ---
        visual_norms: torch.Tensor = visual_features.norm(dim=-1)  # (B, T)

        # --- Magnitude branch ---
        if self.magnitude_branch is not None and self.fusion_gate is not None:
            mag_logit: torch.Tensor = self.magnitude_branch(
                visual_norms, flow_magnitudes
            )  # (B, T)
            fused_logit: torch.Tensor = (
                semantic_logit + self.fusion_gate * mag_logit
            )
        else:
            fused_logit = semantic_logit

        final_scores: torch.Tensor = torch.sigmoid(fused_logit)  # (B, T)

        return final_scores, visual_norms, guided

    @classmethod
    def from_config(cls, config: dict) -> "LanguageGuidedVAD":
        """Construct the model from a configuration dictionary.

        Backward-compatible: V3 configs load cleanly (``use_multi_scale``
        defaults to True for V4; False behaviour matches V3 exactly).

        Args:
            config: Full config dict loaded from a YAML file.

        Returns:
            LanguageGuidedVAD: Instantiated model.
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
            use_flow_in_magnitude=m.get("use_flow_in_magnitude", False),
            use_multi_scale=m.get("use_multi_scale", True),
            use_temporal_convolutions=m.get("use_temporal_convolutions", False),
        )
