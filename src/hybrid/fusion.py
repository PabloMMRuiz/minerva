"""
Unsupervised fusion algorithms for combining multiple similarity/adjacency matrices.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from .base import HybridFuser


class WeightedAverageFuser(HybridFuser):
    """
    Weighted average: A_hybrid = Σ wᵢ · Aᵢ.
    Weights are normalized to sum to 1.
    """

    def __init__(self, weights: Optional[List[float]] = None, params: Optional[Dict[str, Any]] = None):
        super().__init__("weighted_average", params)
        self.weights = weights

    def fuse(self, matrices: List[np.ndarray]) -> np.ndarray:
        self._validate_inputs(matrices)
        K = len(matrices)

        if self.weights is None:
            w = np.ones(K) / K
        else:
            if len(self.weights) != K:
                raise ValueError(f"Expected {K} weights, got {len(self.weights)}")
            w = np.array(self.weights, dtype=float)
            w /= w.sum()

        result = np.zeros_like(matrices[0], dtype=np.float64)
        for wi, mat in zip(w, matrices):
            result += wi * mat

        return result.astype(np.float32)


class ElementWiseMaxFuser(HybridFuser):
    """
    Element-wise maximum (union): keeps the strongest signal from any view.
    A_hybrid[i,j] = max(A₁[i,j], A₂[i,j], ...).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("element_wise_max", params)

    def fuse(self, matrices: List[np.ndarray]) -> np.ndarray:
        self._validate_inputs(matrices)
        stacked = np.stack(matrices, axis=0)
        return np.max(stacked, axis=0).astype(np.float32)


class ElementWiseMinFuser(HybridFuser):
    """
    Element-wise minimum (intersection): keeps only edges all views agree on.
    A_hybrid[i,j] = min(A₁[i,j], A₂[i,j], ...).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("element_wise_min", params)

    def fuse(self, matrices: List[np.ndarray]) -> np.ndarray:
        self._validate_inputs(matrices)
        stacked = np.stack(matrices, axis=0)
        return np.min(stacked, axis=0).astype(np.float32)


class RankAverageFuser(HybridFuser):
    """
    Rank-based fusion: rank all edges within each matrix, then average ranks.
    Scale-invariant — handles matrices with incompatible value ranges
    (e.g., Pearson ∈ [-1, 1] vs MI ∈ [0, ∞)).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("rank_average", params)

    def _rank_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Rank all elements; higher value → higher rank."""
        flat = matrix.flatten()
        order = flat.argsort().argsort()  # double argsort gives ranks
        return order.reshape(matrix.shape).astype(np.float64)

    def fuse(self, matrices: List[np.ndarray]) -> np.ndarray:
        self._validate_inputs(matrices)
        rank_sum = np.zeros_like(matrices[0], dtype=np.float64)

        for mat in matrices:
            rank_sum += self._rank_matrix(np.abs(mat))

        # Normalize to [0, 1]
        rmin, rmax = rank_sum.min(), rank_sum.max()
        if rmax - rmin > 1e-9:
            rank_sum = (rank_sum - rmin) / (rmax - rmin)

        return rank_sum.astype(np.float32)


class SNFDiffusionFuser(HybridFuser):
    """
    Similarity Network Fusion (SNF)-inspired diffusion fusion.

    Each matrix is treated as a similarity network. The algorithm iteratively
    diffuses information across views: each matrix is updated by propagating
    through a KNN-based kernel of every *other* matrix. After several
    iterations the matrices converge to a single fused view.

    Reference: Wang et al., "Similarity network fusion for aggregating data
    types on a genomic scale", Nature Methods, 2014.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        n_iterations: int = 20,
        mu: float = 0.5,
        params: Optional[Dict[str, Any]] = None
    ):
        super().__init__("snf_diffusion", params)
        self.k_neighbors = k_neighbors
        self.n_iterations = n_iterations
        self.mu = mu

    def _normalize_rows(self, W: np.ndarray) -> np.ndarray:
        """Row-normalize a similarity matrix (make each row sum to 1)."""
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return W / row_sums

    def _knn_kernel(self, W: np.ndarray, k: int) -> np.ndarray:
        """
        Build a sparse KNN kernel: for each node, keep only the top-k neighbors.
        """
        N = W.shape[0]
        S = np.zeros_like(W)
        for i in range(N):
            row = W[i, :].copy()
            row[i] = 0  # exclude self
            top_k_idx = np.argsort(row)[::-1][:k]
            S[i, top_k_idx] = row[top_k_idx]
        # Symmetrize
        S = (S + S.T) / 2.0
        return self._normalize_rows(S)

    def fuse(self, matrices: List[np.ndarray]) -> np.ndarray:
        self._validate_inputs(matrices)
        K = len(matrices)
        N = matrices[0].shape[0]

        # Step 1: Ensure non-negative similarities and normalize
        P_list = []
        S_list = []
        for mat in matrices:
            W = np.abs(mat).astype(np.float64)
            np.fill_diagonal(W, 0)
            P_list.append(self._normalize_rows(W))
            S_list.append(self._knn_kernel(W, self.k_neighbors))

        # Step 2: Iterative diffusion
        for _ in range(self.n_iterations):
            P_new = []
            for i in range(K):
                # Average of all OTHER normalized matrices
                others_avg = np.zeros((N, N), dtype=np.float64)
                for j in range(K):
                    if j != i:
                        others_avg += P_list[j]
                others_avg /= (K - 1)

                # Diffuse: P_i = S_i @ others_avg @ S_i.T
                updated = S_list[i] @ others_avg @ S_list[i].T
                # Re-normalize
                updated = self._normalize_rows(updated)
                P_new.append(updated)
            P_list = P_new

        # Step 3: Final fused matrix = average of all converged P
        fused = np.zeros((N, N), dtype=np.float64)
        for P in P_list:
            fused += P
        fused /= K

        # Symmetrize
        fused = (fused + fused.T) / 2.0
        return fused.astype(np.float32)
