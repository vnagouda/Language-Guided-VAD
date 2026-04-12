"""Language-Guided Cross-Attention VAD Network.

This module implements the core trainable architecture for Weakly Supervised
Video Anomaly Detection.  The key innovation is the **Cross-Attention** fusion
mechanism where textual features serve as Queries to guide the visual
Keys/Values, departing from naive concatenation.

Architecture overview::

    Text Features  ──► Q ──┐
                           ├── Multi-Head Cross-Attention ──► LayerNorm + Residual
    Visual Features ──► K,V┘                                        │
                                                                    ▼
                                                            Feed-Forward Network
                                                                    │
                                                                    ▼
                                                        (Repeat × num_layers)
                                                                    │
                                                                    ▼
                                                        MLP Classifier Head
                                                                    │
                                                                    ▼
                                                        Anomaly Scores ∈ [0,1]
                                                        Shape: (Batch, 32)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """A single Multi-Head Cross-Attention block with residual connection and FFN.

    Implements the transformer-style cross-attention where:
        - **Query** = text features (semantic guidance signal)
        - **Key, Value** = visual features (scene information)

    This forces the network to attend to visual regions that are semantically
    relevant to the textual description, amplifying abnormal visual cues.

    Args:
        feature_dim: Dimensionality of input features (default 512 for CLIP).
        num_heads: Number of attention heads.
        ff_dim: Hidden dimension of the feed-forward sub-layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # Multi-Head Cross-Attention: Q=text, K=V=visual
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # (Batch, Seq, Dim)
        )

        # Layer norms (pre-norm is more stable, but we use post-norm
        # to match the standard Transformer formulation)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # Position-wise Feed-Forward Network
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
        """Forward pass of the cross-attention block.

        Mathematical formulation:
            attn_out = MultiHead(Q=text, K=visual, V=visual)
            x = LayerNorm(text + attn_out)          # Residual + Norm
            output = LayerNorm(x + FFN(x))           # Residual + Norm

        Args:
            text_features: Text embeddings of shape ``(Batch, 32, feature_dim)``.
                These serve as the Query — the semantic guidance signal.
            visual_features: Visual embeddings of shape ``(Batch, 32, feature_dim)``.
                These serve as both Key and Value.

        Returns:
            torch.Tensor: Guided features of shape ``(Batch, 32, feature_dim)``.
        """
        # Cross-Attention: Q=text, K=V=visual
        attn_output, _ = self.cross_attn(
            query=text_features,
            key=visual_features,
            value=visual_features,
        )

        # Residual connection + LayerNorm (add to text, not visual)
        x = self.norm1(text_features + attn_output)

        # Feed-Forward with Residual + LayerNorm
        ffn_output = self.ffn(x)
        output: torch.Tensor = self.norm2(x + ffn_output)

        return output


class LanguageGuidedVAD(nn.Module):
    """Language-Guided Video Anomaly Detection Network.

    Stacks ``num_layers`` :class:`CrossAttentionBlock` modules, then maps the
    guided features through an MLP classifier to produce per-segment anomaly
    scores in [0, 1].

    Args:
        feature_dim: Dimensionality of CLIP feature vectors (512).
        num_segments: Number of temporal segments per video (T=32).
        num_heads: Number of attention heads per cross-attention block.
        num_layers: Number of stacked cross-attention blocks.
        ff_dim: Hidden dimension of the feed-forward sub-layers.
        classifier_hidden_dim: Hidden dimension of the MLP classifier head.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_segments: int = 32,
        num_heads: int = 8,
        num_layers: int = 1,
        ff_dim: int = 2048,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_segments = num_segments
        self.feature_dim = feature_dim

        # Stack of Cross-Attention Blocks
        self.attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                feature_dim=feature_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # MLP Classifier Head: maps each segment's guided feature → scalar score
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: cross-attention fusion followed by per-segment scoring.

        Args:
            visual_features: Visual feature tensor of shape ``(Batch, 32, 512)``.
            text_features: Text feature tensor of shape ``(Batch, 32, 512)``.

        Returns:
            torch.Tensor: Anomaly scores of shape ``(Batch, 32)``, each value
            in the range [0, 1].
        """
        # Pass through stacked cross-attention blocks
        guided = text_features
        for layer in self.attention_layers:
            guided = layer(text_features=guided, visual_features=visual_features)

        # Classify: (B, 32, 512) → (B, 32, 1) → (B, 32)
        
        # Original semantic scoring (cross-attention guided)
        # scores: torch.Tensor = self.classifier(guided).squeeze(-1)

        # Visual-only baseline (no text guidance)
        scores: torch.Tensor = self.classifier(visual_features).squeeze(-1)

        return scores

    @classmethod
    def from_config(cls, config: dict) -> "LanguageGuidedVAD":
        """Construct the model from a configuration dictionary.

        Args:
            config: Full configuration dict (loaded from ``config.yaml``).

        Returns:
            LanguageGuidedVAD: Instantiated model with config-driven parameters.
        """
        model_cfg = config["model"]
        return cls(
            feature_dim=model_cfg["feature_dim"],
            num_segments=model_cfg["num_segments"],
            num_heads=model_cfg["num_heads"],
            num_layers=model_cfg["num_layers"],
            ff_dim=model_cfg["ff_dim"],
            classifier_hidden_dim=model_cfg["classifier_hidden_dim"],
            dropout=model_cfg["dropout"],
        )
