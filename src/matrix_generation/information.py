"""
Mutual Information matrix generator (non-linear dependencies).
"""

import numpy as np
from typing import Dict, Optional, Any
from sklearn.feature_selection import mutual_info_regression
from .base import MatrixGenerator
from data.preprocessing import reshape_time_series_2_d


class MutualInformationGenerator(MatrixGenerator):
    """
    Calculates the Mutual Information matrix.
    Captures non-linear dependencies between time series.
    """

    def __init__(self, n_neighbors: int = 3, feature_index: int = 0, params: Optional[Dict[str, Any]] = None):
        super().__init__("mutual_information", params)
        self.n_neighbors = n_neighbors
        self.feature_index = feature_index

    def generate(self, data: np.ndarray) -> np.ndarray:
        """
        Input: [L, N, C]
        """
        x = reshape_time_series_2_d(data, self.feature_index)
        L, N = x.shape
        mi_matrix = np.zeros((N, N), dtype=np.float32)
        
        for i in range(N):
            # mutual_info_regression can calculate MI between one target and multiple features
            # but for a symmetric matrix we can do it row by row 
            # to keep it simple and consistent with other generators.
            target = x[:, i]
            # MI is symmetric in theory, so we only need the upper triangle
            # but sklearn's estimation might have slight variations.
            for j in range(i + 1, N):
                mi = mutual_info_regression(
                    x[:, j].reshape(-1, 1), 
                    target, 
                    n_neighbors=self.n_neighbors
                )[0]
                mi_matrix[i, j] = mi_matrix[j, i] = mi
        
        # Fill diagonal with max MI (or just 1.0 for normalized MI)
        # Here we just leave it or fill with a large value
        np.fill_diagonal(mi_matrix, mi_matrix.max() if mi_matrix.size > 0 else 1.0)
        
        return mi_matrix
