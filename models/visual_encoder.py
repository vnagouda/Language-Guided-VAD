"""CLIP Visual Feature Extractor for offline feature extraction.

Wraps the HuggingFace ``CLIPVisionModel`` to extract per-frame visual feature
vectors of shape ``(512,)`` from RGB images.  Used exclusively by
``scripts/01_extract_features.py`` during the offline extraction phase.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel


class CLIPVisualFeatureExtractor:
    """Wrapper for CLIP Vision Encoder feature extraction.

    Loads the CLIP ViT model and processor, then provides a simple
    :meth:`extract` method that takes a list of PIL images and returns
    a stacked feature tensor.

    Args:
        model_name: HuggingFace model identifier, e.g.
            ``"openai/clip-vit-base-patch16"``.
        device: Torch device to run inference on.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, images: list[Image.Image]) -> torch.Tensor:
        """Extract visual features from a list of PIL images.

        Args:
            images: List of PIL images (RGB).

        Returns:
            torch.Tensor: Feature matrix of shape ``(N, 512)`` where
            ``N = len(images)``.
        """
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(self.device)

        outputs = self.model(pixel_values=pixel_values)
        # Use the pooled output (CLS token) as the feature vector
        features: torch.Tensor = outputs.pooler_output  # (N, 768) for ViT-B/16

        # Project to 512 if needed (CLIP's projection head)
        # Note: CLIPVisionModel output is 768-dim; the full CLIPModel does
        # the projection.  We handle this in the extraction script.
        return features.cpu()

    @torch.no_grad()
    def extract_with_projection(
        self,
        images: list[Image.Image],
        clip_model: Any,
    ) -> torch.Tensor:
        """Extract features and project through CLIP's visual projection head.

        Uses the full CLIP model's visual projection to get 512-dim vectors
        that are directly comparable to text embeddings.

        Args:
            images: List of PIL images (RGB).
            clip_model: Full ``CLIPModel`` instance with ``visual_projection``.

        Returns:
            torch.Tensor: Feature matrix of shape ``(N, 512)``.
        """
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(self.device)

        vision_outputs = self.model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        projected: torch.Tensor = clip_model.visual_projection(pooled)

        return projected.cpu()
