"""
Sparsification methods for adjacency matrices.
"""

from .base import Sparsifier, Normalizer
from .builder import (
    MatrixConstructionPipeline,
    make_adjacency_matrix,
    make_adjacency_from_generator,
    SPARSIFIER_REGISTRY,
    NORMALIZER_REGISTRY
)
from .masks import (
    TopKRowSparsifier,
    GlobalTopESparsifier,
    DegreeRegularizedGreedySparsifier,
    ThresholdWithBudgetSparsifier,
    KNNWithGlobalBudgetSparsifier,
    SpectralProxySparsifier,
    ResistanceSpectralSparsifier,
    DHondtSparsifier,
    MutualKNNSparsifier,
    MSTSparsifier,
    PMFGSparsifier
)
from .normalization import (
    RowL1Normalizer,
    RowSoftmaxNormalizer,
    GlobalSoftmaxNormalizer,
    RowMinMaxNormalizer,
    GlobalMinMaxNormalizer,
    BinaryNormalizer
)

__all__ = [
    'Sparsifier',
    'Normalizer',
    'MatrixConstructionPipeline',
    'make_adjacency_matrix',
    'make_adjacency_from_generator',
    'SPARSIFIER_REGISTRY',
    'NORMALIZER_REGISTRY',
    'TopKRowSparsifier',
    'GlobalTopESparsifier',
    'DegreeRegularizedGreedySparsifier',
    'ThresholdWithBudgetSparsifier',
    'KNNWithGlobalBudgetSparsifier',
    'SpectralProxySparsifier',
    'ResistanceSpectralSparsifier',
    'DHondtSparsifier',
    'MutualKNNSparsifier',
    'MSTSparsifier',
    'PMFGSparsifier',
    'RowL1Normalizer',
    'RowSoftmaxNormalizer',
    'GlobalSoftmaxNormalizer',
    'RowMinMaxNormalizer',
    'GlobalMinMaxNormalizer',
    'BinaryNormalizer'
]