"""Model package for Language-Guided WS-VAD.

Exposes the main trainable model::

    from models import LanguageGuidedVAD
"""

from models.vad_architecture import LanguageGuidedVAD, CrossAttentionBlock

__all__ = [
    "LanguageGuidedVAD",
    "CrossAttentionBlock",
]
