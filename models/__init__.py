"""Model package for Language-Guided WS-VAD — V2.

Exposes the main trainable model and all sub-modules::

    from models import LanguageGuidedVAD
    from models import MagnitudeBranch, HourglassClassifier

"""

from models.vad_architecture import (
    LanguageGuidedVAD,
    CrossAttentionBlock,
    HourglassClassifier,
    MagnitudeBranch,
)

__all__ = [
    "LanguageGuidedVAD",
    "CrossAttentionBlock",
    "HourglassClassifier",
    "MagnitudeBranch",
]
