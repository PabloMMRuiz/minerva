"""
Correlation-based matrix generation methods.
"""

import numpy as np
from typing import Dict, Optional, Any
from .base import MatrixGenerator
from src.data.preprocessing import reshape_time_series_2_d


class PearsonCorrelationGenerator(MatrixGenerator):
    """
    Generate adjacency matrix using Pearson correlation coefficient.

    Computes linear correlation between time series of all node pairs.
    """

    def __init__(self, feature_index: int = 0, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Pearson correlation generator.

        Args:
            feature_index: Which feature to use if input is 3D (default: 0)
            params: Additional parameters
        """
        params = params or {}
        params['feature_index'] = feature_index
        super().__init__("pearson_correlation", params)
        self.feature_index = feature_index

    def generate(self, time_series: np.ndarray) -> np.ndarray:
        """
        Calculate Pearson correlation matrix between all node pairs.

        Args:
            time_series: Shape [L, N] or [L, N, C]
                        L = time steps, N = nodes, C = features

        Returns:
            Correlation matrix with shape [N, N]
        """
        # Handle 3D input by extracting feature
        if time_series.ndim == 3:
            time_series = reshape_time_series_2_d(
                time_series,
                feature_index=self.feature_index
            )

        # Validate shape
        if time_series.ndim != 2:
            raise ValueError(
                f"Expected 2D array [L, N] after reshaping, got shape {time_series.shape}"
            )

        # Calculate Pearson correlation
        # np.corrcoef expects features (nodes) as rows, so transpose
        corr_matrix = np.corrcoef(time_series.T)

        return corr_matrix.astype(np.float32)
class PartialCorrelationGenerator(MatrixGenerator):
    """
    Calculates the Partial Correlation matrix.
    Measures the degree of association between two nodes, removing the effect 
    of all other nodes in the set.
    """

    def __init__(self, feature_index: int = 0, params: Optional[Dict[str, Any]] = None):
        super().__init__("partial_correlation", params)
        self.feature_index = feature_index

    def generate(self, data: np.ndarray) -> np.ndarray:
        """
        Input shape: [L, N, C]
        """
        # Reshape to [L, N]
        shaped_data = reshape_time_series_2_d(data, self.feature_index)
        
        # Calculate covariance matrix
        cov = np.cov(shaped_data, rowvar=False)
        
        # Calculate precision matrix (inverse covariance)
        # Use pseudo-inverse if singular
        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(cov)
            
        # Partial correlation: rho_ij.rest = -P_ij / sqrt(P_ii * P_jj)
        diag = np.diag(precision)
        outer_sqrt_diag = np.sqrt(np.outer(diag, diag))
        partial_corr = -precision / outer_sqrt_diag
        
        # Diagonal elements should be 1.0
        np.fill_diagonal(partial_corr, 1.0)
        
        return partial_corr.astype(np.float32)
