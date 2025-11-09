"""
Main adjacency matrix builder that orchestrates masking and normalization.
"""

import numpy as np
from typing import Optional
from .masks import (
    top_k_row_mask,
    global_top_e_mask,
    degree_regularized_greedy_mask,
    threshold_with_budget_mask,
    spectral_sparsification_proxy_mask,
    resistance_spectral_sparsify,
    knn_with_global_budget_mask,
    dhondt_proportional_allocation_sum_correlation,
    dhondt_top_edge_allocation_mask,
)
from .normalization import normalize_adjacency


def make_adjacency_matrix(
    corr_matrix: np.ndarray,
    k: int,
    mask: Optional[str] = None,
    normalize: Optional[str] = None,
    norm_strength: float = 1.0,
    mask_params: Optional[dict] = None
) -> np.ndarray:
    """
    Creates a sparse adjacency matrix by selecting edges and normalizing.

    Pipeline: correlation matrix → masking → normalization → sparse adjacency

    Args:
        corr_matrix: The N x N correlation/similarity matrix
        k: Number of edges to keep (interpretation depends on mask method)
        mask: Sparsification method. Options:
              - 'top-k-row': Top-k per node (local budget)
              - 'top-k-global': Top-k globally
              - 'greedy-degree-regularize': Degree-regularized greedy
              - 'threshold-mask': Threshold-based with budget
              - 'spectral-sparce': Approximate spectral sparsification
              - 'strict-spectral-sparce': Exact spectral sparsification
              - 'top-k-row-global-limit': K-NN with global budget
              - 'dhont-corr-sum': D'Hondt allocation by correlation sum
              - 'dhont-top-edge': D'Hondt allocation by top edge
        normalize: Normalization method. Options:
                   - 'row-l1': Row sums to norm_strength
                   - 'row-softmax': Row-wise softmax
                   - 'softmax': Global softmax
                   - 'row-minmax': Row-wise min-max normalization
                   - 'minmax': Global min-max normalization
                   - 'make-1': Set non-zero edges to 1
                   - None: No normalization
        norm_strength: Scaling factor for normalization (default: 1.0)
        mask_params: Additional parameters for mask functions (e.g., penalty_factor)

    Returns:
        Sparse adjacency matrix (N x N)

    Raises:
        ValueError: If mask or normalize method is unknown
    """
    N = corr_matrix.shape[0]
    mask_params = mask_params or {}

    # Ensure matrix is positive (take absolute value)
    adj = np.abs(corr_matrix)

    # Apply masking
    if mask is None:
        # No sparsification - keep all edges
        mask_matrix = np.ones_like(adj, dtype=bool)

    elif mask == 'top-k-row':
        edges_per_node = int(k / N)
        mask_matrix = top_k_row_mask(corr_matrix, edges_per_node)

    elif mask == 'top-k-global':
        mask_matrix = global_top_e_mask(corr_matrix, k, directed=True)

    elif mask == 'greedy-degree-regularize':
        penalty_factor = mask_params.get('penalty_factor', 0.1)
        mask_matrix = degree_regularized_greedy_mask(
            corr_matrix, k, penalty_factor, directed=True
        )

    elif mask == 'threshold-mask':
        mask_matrix = threshold_with_budget_mask(corr_matrix, k)

    elif mask == 'spectral-sparce':
        mask_matrix = spectral_sparsification_proxy_mask(corr_matrix, k)

    elif mask == 'strict-spectral-sparce':
        mask_matrix, _ = resistance_spectral_sparsify(corr_matrix, k)

    elif mask == 'top-k-row-global-limit':
        k_per_node = mask_params.get('k_per_node', k // N * 3)
        mask_matrix = knn_with_global_budget_mask(corr_matrix, k, k_per_node)

    elif mask == 'dhont-corr-sum':
        mask_matrix = dhondt_proportional_allocation_sum_correlation(
            corr_matrix, k)

    elif mask == 'dhont-top-edge':
        mask_matrix = dhondt_top_edge_allocation_mask(corr_matrix, k)

    else:
        raise ValueError(f"Unknown mask method: {mask}")

    # Apply the mask
    sparse_adj = np.where(mask_matrix, adj, 0)

    # Apply normalization
    if normalize is not None:
        sparse_adj = normalize_adjacency(sparse_adj, normalize, norm_strength)

    return sparse_adj.astype(np.float32)


def make_adjacency_from_generator(
    generator,
    time_series: np.ndarray,
    k: int,
    mask: Optional[str] = None,
    normalize: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """
    Convenience function to generate and sparsify in one call.

    Args:
        generator: MatrixGenerator instance
        time_series: Time series data
        k: Number of edges to keep
        mask: Sparsification method
        normalize: Normalization method
        **kwargs: Additional arguments for make_adjacency_matrix

    Returns:
        Sparse adjacency matrix
    """
    # Generate similarity matrix
    similarity_matrix = generator.generate(time_series)

    # Sparsify
    sparse_adj = make_adjacency_matrix(
        similarity_matrix,
        k=k,
        mask=mask,
        normalize=normalize,
        **kwargs
    )

    return sparse_adj
