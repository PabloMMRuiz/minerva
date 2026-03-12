"""
Matrix generation methods for creating adjacency matrices from time series.
"""

from .base import MatrixGenerator
from .correlation import (
    PearsonCorrelationGenerator,
    PartialCorrelationGenerator
)
from .dtw import DTWGenerator
from .causality import GrangerCausalityGenerator
from .information import MutualInformationGenerator
from .embedding import EmbeddingCosineGenerator

__all__ = [
    'MatrixGenerator',
    'PearsonCorrelationGenerator',
    'PartialCorrelationGenerator',
    'DTWGenerator',
    'GrangerCausalityGenerator',
    'MutualInformationGenerator',
    'EmbeddingCosineGenerator'
]
