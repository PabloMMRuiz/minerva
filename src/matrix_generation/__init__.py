"""
Matrix generation methods for creating adjacency matrices from time series.
"""

from .base import MatrixGenerator
from .correlation import (
    PearsonCorrelationGenerator,
)

__all__ = [
    'MatrixGenerator',
    'PearsonCorrelationGenerator',
]
