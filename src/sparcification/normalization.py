"""
Normalization methods implemented as Normalizer classes.
"""

import numpy as np
from .base import Normalizer


class RowL1Normalizer(Normalizer):
    """Row-wise L1 normalization (sum of each row = norm_strength)."""
    
    def __init__(self, norm_strength: float = 1.0):
        super().__init__("row_l1", norm_strength)
    
    def normalize(self, matrix: np.ndarray) -> np.ndarray:
        row_sums = np.abs(matrix).sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return (self.norm_strength * matrix / row_sums).astype(np.float32)


class RowSoftmaxNormalizer(Normalizer):
    """Row-wise Softmax normalization."""
    
    def __init__(self, norm_strength: float = 1.0):
        super().__init__("row_softmax", norm_strength)
    
    def normalize(self, matrix: np.ndarray) -> np.ndarray:
        zero_mask = np.abs(matrix) < 1e-6
        exp_adj = np.exp(matrix)
        exp_adj[zero_mask] = 0
        row_sums = exp_adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return (self.norm_strength * exp_adj / row_sums).astype(np.float32)


class GlobalSoftmaxNormalizer(Normalizer):
    """Global Softmax normalization."""
    
    def __init__(self, norm_strength: float = 1.0):
        super().__init__("global_softmax", norm_strength)
    
    def normalize(self, matrix: np.ndarray) -> np.ndarray:
        zero_mask = np.abs(matrix) < 1e-6
        exp_adj = np.exp(matrix)
        exp_adj[zero_mask] = 0
        global_sum = exp_adj.sum()
        if global_sum == 0:
            global_sum = 1
        return (self.norm_strength * exp_adj / global_sum).astype(np.float32)


class RowMinMaxNormalizer(Normalizer):
    """Row-wise Min-Max normalization."""
    
    def __init__(self, norm_strength: float = 1.0):
        super().__init__("row_minmax", norm_strength)
    
    def normalize(self, matrix: np.ndarray) -> np.ndarray:
        N = matrix.shape[0]
        normalized = np.copy(matrix)
        for i in range(N):
            row = matrix[i, :]
            nz_row = row[np.abs(row) > 1e-6]
            if nz_row.size > 0:
                r_min, r_max = nz_row.min(), nz_row.max()
                denom = r_max - r_min
                if denom > 1e-6:
                    mask = np.abs(row) > 1e-6
                    normalized[i, mask] = (row[mask] - r_min) / denom
        return normalized.astype(np.float32)


class GlobalMinMaxNormalizer(Normalizer):
    """Global Min-Max normalization."""
    
    def __init__(self, norm_strength: float = 1.0):
        super().__init__("global_minmax", norm_strength)
    
    def normalize(self, matrix: np.ndarray) -> np.ndarray:
        nz_vals = matrix[np.abs(matrix) > 1e-6]
        if nz_vals.size == 0:
            return matrix.astype(np.float32)
        
        g_min, g_max = nz_vals.min(), nz_vals.max()
        denom = g_max - g_min
        if denom < 1e-6:
            return matrix.astype(np.float32)
        
        normalized = np.copy(matrix)
        mask = np.abs(matrix) > 1e-6
        normalized[mask] = (matrix[mask] - g_min) / denom
        return normalized.astype(np.float32)


class BinaryNormalizer(Normalizer):
    """Set non-zero edges to 1."""
    
    def __init__(self, norm_strength: float = 1.0):
        super().__init__("binary", norm_strength)
    
    def normalize(self, matrix: np.ndarray) -> np.ndarray:
        return (np.abs(matrix) > 1e-6).astype(np.float32)