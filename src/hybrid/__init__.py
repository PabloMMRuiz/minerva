"""
Hybrid matrix fusion methods.
"""

from .base import HybridFuser
from .fusion import (
    WeightedAverageFuser,
    ElementWiseMaxFuser,
    ElementWiseMinFuser,
    RankAverageFuser,
    SNFDiffusionFuser
)
from .learned import AttentionFuser

__all__ = [
    'HybridFuser',
    'WeightedAverageFuser',
    'ElementWiseMaxFuser',
    'ElementWiseMinFuser',
    'RankAverageFuser',
    'SNFDiffusionFuser',
    'AttentionFuser'
]
