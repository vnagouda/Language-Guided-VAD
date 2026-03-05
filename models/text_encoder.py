"""Text Feature Extraction: BLIP-2 Captioning + CLIP Text Encoding.

This module provides two classes used in the offline feature extraction pipeline:

1. :class:`BLIP2Captioner` — generates natural language captions for each video
   segment using BLIP-2 (Salesforce/blip2-opt-2.7b).
2. :class:`CLIPTextFeatureExtractor` — encodes text captions into 512-dim CLIP
   embedding vectors.

Together, the pipeline is: Frame → BLIP-2 → Caption → CLIP Text Encoder → [512].
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    CLIPModel,
    CLIPTokenizer,
)


class BLIP2Captioner:
    """Generate natural language captions from images using BLIP-2.

    Args:
        model_name: HuggingFace model identifier, e.g.
            ``"Salesforce/blip2-opt-2.7b"``.
        device: Torch device to run inference on.
        max_new_tokens: Maximum number of tokens to generate.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: torch.device | None = None,
        max_new_tokens: int = 50,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def caption(self, images: list[Image.Image]) -> list[str]:
        """Generate captions for a list of PIL images.

        Args:
            images: List of RGB PIL images to generate captions for.

        Returns:
            list[str]: One caption string per input image.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(
            self.device, dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )

        captions: list[str] = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [c.strip() for c in captions]


class CLIPTextFeatureExtractor:
    """Encode text strings into CLIP text embedding vectors.

    Uses the full ``CLIPModel`` to produce 512-dim text features that lie in
    the same embedding space as CLIP visual features.

    Args:
        model_name: HuggingFace CLIP model identifier.
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

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def extract(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of text strings into CLIP feature vectors.

        Args:
            texts: List of caption strings.

        Returns:
            torch.Tensor: Text feature matrix of shape ``(N, 512)``
            where ``N = len(texts)``.
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_features: torch.Tensor = self.model.get_text_features(**inputs)
        return text_features.cpu()
