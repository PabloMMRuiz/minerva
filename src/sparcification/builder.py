"""
Unified pipeline for adjacency matrix construction.
"""

import numpy as np
from typing import List, Optional, Union, Dict, Any
from .base import Sparsifier, Normalizer
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


class MatrixConstructionPipeline:
    """
    Orchestrates the process of generating, sparsifying, and normalizing adjacency matrices.
    """

    def __init__(
        self,
        sparsifiers: Optional[Union[Sparsifier, List[Sparsifier]]] = None,
        normalizers: Optional[Union[Normalizer, List[Normalizer]]] = None,
        fill_diag: bool = False
    ):
        self.sparsifiers = self._ensure_list(sparsifiers)
        self.normalizers = self._ensure_list(normalizers)
        self.fill_diag = fill_diag

    @staticmethod
    def _ensure_list(item):
        if item is None:
            return []
        return [item] if not isinstance(item, (list, tuple)) else list(item)

    def run(self, matrix: np.ndarray, k: int) -> np.ndarray:
        """
        Execute the pipeline: Sparsification -> Diagonal Fill -> Normalization.
        """
        N = matrix.shape[0]
        effective_k = max(0, k - N) if self.fill_diag else k
        
        # 1. Sparsification (cumulative if multiple)
        adj = np.abs(matrix)
        if self.sparsifiers:
            mask = np.ones_like(adj, dtype=bool)
            for sparsifier in self.sparsifiers:
                mask = mask & sparsifier.get_mask(matrix, effective_k)
            adj = np.where(mask, adj, 0)

        # 2. Diagonal fill
        if self.fill_diag:
            np.fill_diagonal(adj, 1.0)

        # 3. Normalization
        for normalizer in self.normalizers:
            adj = normalizer.normalize(adj)

        return adj.astype(np.float32)


# Registry for easy string-based access
SPARSIFIER_REGISTRY = {
    'top-k-row': TopKRowSparsifier,
    'top-k-global': GlobalTopESparsifier,
    'greedy-degree-regularize': DegreeRegularizedGreedySparsifier,
    'threshold-mask': ThresholdWithBudgetSparsifier,
    'top-k-row-global-limit': KNNWithGlobalBudgetSparsifier,
    'spectral-sparce': SpectralProxySparsifier,
    'strict-spectral-sparce': ResistanceSpectralSparsifier,
    'dhondt': DHondtSparsifier,
    'mutual-knn': MutualKNNSparsifier,
    'mst': MSTSparsifier,
    'pmfg': PMFGSparsifier
}

NORMALIZER_REGISTRY = {
    'row-l1': RowL1Normalizer,
    'row-softmax': RowSoftmaxNormalizer,
    'softmax': GlobalSoftmaxNormalizer,
    'row-minmax': RowMinMaxNormalizer,
    'minmax': GlobalMinMaxNormalizer,
    'make-1': BinaryNormalizer
}


def make_adjacency_matrix(
    corr_matrix: np.ndarray,
    k: int,
    mask_method: Optional[str] = None,
    norm_method: Optional[str] = None,
    norm_strength: float = 1.0,
    mask_params: Optional[Dict[str, Any]] = None,
    fill_diag: bool = False
) -> np.ndarray:
    """
    Legacy-compatible wrapper for the pipeline.
    """
    sparsifiers = []
    if mask_method:
        cls = SPARSIFIER_REGISTRY.get(mask_method)
        if not cls:
            raise ValueError(f"Unknown mask method: {mask_method}")
        sparsifiers.append(cls(**(mask_params or {})))

    normalizers = []
    if norm_method:
        cls = NORMALIZER_REGISTRY.get(norm_method)
        if not cls:
            raise ValueError(f"Unknown normalization method: {norm_method}")
        normalizers.append(cls(norm_strength=norm_strength))

    pipeline = MatrixConstructionPipeline(sparsifiers, normalizers, fill_diag)
    return pipeline.run(corr_matrix, k)


def make_adjacency_from_generator(
    generator: Any,
    time_series: np.ndarray,
    k: int,
    **kwargs
) -> np.ndarray:
    """
    Generate and sparsify in one call.
    """
    similarity_matrix = generator.generate(time_series)
    return make_adjacency_matrix(similarity_matrix, k, **kwargs)


# --------------- Hybrid Fusion Registry ---------------

try:
    from hybrid.fusion import (
        WeightedAverageFuser,
        ElementWiseMaxFuser,
        ElementWiseMinFuser,
        RankAverageFuser,
        SNFDiffusionFuser
    )

    FUSER_REGISTRY = {
        'weighted-average': WeightedAverageFuser,
        'max': ElementWiseMaxFuser,
        'min': ElementWiseMinFuser,
        'rank-average': RankAverageFuser,
        'snf': SNFDiffusionFuser
    }
except ImportError:
    FUSER_REGISTRY = {}


def make_hybrid_adjacency(
    matrices: list,
    k: int,
    fuse_method: str = 'weighted-average',
    fuse_params: Optional[Dict[str, Any]] = None,
    mask_method: Optional[str] = None,
    norm_method: Optional[str] = None,
    norm_strength: float = 1.0,
    mask_params: Optional[Dict[str, Any]] = None,
    fill_diag: bool = False
) -> np.ndarray:
    """
    Fuse multiple matrices and then apply sparsification + normalization.
    """
    cls = FUSER_REGISTRY.get(fuse_method)
    if not cls:
        raise ValueError(f"Unknown fuse method: {fuse_method}. Available: {list(FUSER_REGISTRY.keys())}")
    fuser = cls(**(fuse_params or {}))
    fused = fuser.fuse(matrices)
    return make_adjacency_matrix(fused, k, mask_method, norm_method, norm_strength, mask_params, fill_diag)
