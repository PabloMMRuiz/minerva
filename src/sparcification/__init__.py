"""
Sparsification methods for adjacency matrices.
"""

from .builder import make_adjacency_matrix, make_adjacency_from_generator
from .masks import (
    top_k_row_mask,
    global_top_e_mask,
    degree_regularized_greedy_mask,
    threshold_with_budget_mask,
    spectral_sparsification_proxy_mask,
    knn_with_global_budget_mask,
    resistance_spectral_sparsify,
    dhondt_top_edge_allocation_mask,
    dhondt_proportional_allocation_sum_correlation,

)
from .normalization import normalize_adjacency

__all__ = [
    # matrix making
    'make_adjacency_matrix',
    'make_adjacency_from_generator',
    # masks
    'top_k_row_mask',
    'global_top_e_mask',
    'degree_regularized_greedy_mask',
    'threshold_with_budget_mask',
    'knn_with_global_budget_mask',
    'spectral_sparsification_proxy_mask',
    'resistance_spectral_sparsify',
    'dhondt_top_edge_allocation_mask',
    'dhondt_proportional_allocation_sum_correlation',

    # normalization
    'normalize_adjacency'
]