"""
Normalization methods for adjacency matrices.
"""

import numpy as np


def normalize_adjacency(
    sparse_adj: np.ndarray,
    method: str,
    norm_strength: float = 1.0
) -> np.ndarray:
    """
    Normalize sparse adjacency matrix.
    
    Args:
        sparse_adj: Sparse adjacency matrix (with zeros for removed edges)
        method: Normalization method
                - 'row-l1': Row sums to norm_strength
                - 'row-softmax': Row-wise softmax
                - 'softmax': Global softmax
                - 'row-minmax': Row-wise min-max
                - 'minmax': Global min-max
                - 'make-1': Set non-zero edges to 1
        norm_strength: Scaling factor for normalization
    
    Returns:
        Normalized adjacency matrix
    """
    N = sparse_adj.shape[0]
    zero_mask = sparse_adj < 1e-6
    
    if method == 'row-l1':
        return _normalize_row_l1(sparse_adj, norm_strength, zero_mask)
    
    elif method == 'row-softmax':
        return _normalize_row_softmax(sparse_adj, norm_strength, zero_mask)
    
    elif method == 'softmax':
        return _normalize_softmax(sparse_adj, norm_strength, zero_mask)
    
    elif method == 'row-minmax':
        return _normalize_row_minmax(sparse_adj, N, zero_mask)
    
    elif method == 'minmax':
        return _normalize_minmax(sparse_adj, zero_mask)
    
    elif method == 'make-1':
        return (~zero_mask).astype(np.float32) # ~ is numpy negation
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def _normalize_row_l1(
    sparse_adj: np.ndarray,
    norm_strength: float,
    zero_mask: np.ndarray
) -> np.ndarray:
    """Row-wise L1 normalization (sum of each row = norm_strength)."""
    row_sums = sparse_adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized_adj = norm_strength * sparse_adj / row_sums
    return normalized_adj.astype(np.float32)


def _normalize_row_softmax(
    sparse_adj: np.ndarray,
    norm_strength: float,
    zero_mask: np.ndarray
) -> np.ndarray:
    """Row-wise Softmax normalization."""
    exp_adj = np.exp(sparse_adj)
    exp_adj[zero_mask] = 0  # Reset original zeros to zero
    row_sums = exp_adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized_adj = norm_strength * exp_adj / row_sums
    return normalized_adj.astype(np.float32)


def _normalize_softmax(
    sparse_adj: np.ndarray,
    norm_strength: float,
    zero_mask: np.ndarray
) -> np.ndarray:
    """Global Softmax normalization."""
    exp_adj = np.exp(sparse_adj)
    exp_adj[zero_mask] = 0
    global_sum = exp_adj.sum()
    normalized_adj = norm_strength * exp_adj / global_sum
    return normalized_adj.astype(np.float32)


def _normalize_row_minmax(
    sparse_adj: np.ndarray,
    N: int,
    zero_mask: np.ndarray
) -> np.ndarray:
    """Row-wise Min-Max normalization."""
    normalized_adj = np.copy(sparse_adj)
    
    for i in range(N):
        row = sparse_adj[i, :]
        row_min = row[row > 0].min() if np.any(row > 0) else 0
        row_max = row.max()
        
        denominator = row_max - row_min
        if denominator > 1e-6:
            normalized_adj[i, :][~zero_mask[i, :]] = (
                row[~zero_mask[i, :]] - row_min
            ) / denominator
        # else: keep as is if all values are the same
    
    return normalized_adj.astype(np.float32)


def _normalize_minmax(
    sparse_adj: np.ndarray,
    zero_mask: np.ndarray
) -> np.ndarray:
    """Global Min-Max normalization."""
    global_min = sparse_adj[sparse_adj > 0].min() if np.any(sparse_adj > 0) else 0
    global_max = sparse_adj.max()
    denominator = global_max - global_min
    
    normalized_adj = np.copy(sparse_adj)
    if denominator > 1e-6:
        # Apply normalization only to non-zero elements
        normalized_adj[~zero_mask] = (
            sparse_adj[~zero_mask] - global_min
        ) / denominator
    
    return normalized_adj.astype(np.float32)