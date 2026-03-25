"""
Granger Causality matrix generator (directed).
"""

import numpy as np
from typing import Dict, Optional, Any
from statsmodels.tsa.stattools import grangercausalitytests
from .base import MatrixGenerator
from src.data.preprocessing import reshape_time_series_2_d


class GrangerCausalityGenerator(MatrixGenerator):
    """
    Calculates the Granger Causality matrix.
    Resulting matrix is asymmetric (directed graph).
    Shows if time series A is useful in forecasting time series B.
    """

    def __init__(self, maxlag: int = 4, feature_index: int = 0, params: Optional[Dict[str, Any]] = None):
        super().__init__("granger_causality", params)
        self.maxlag = maxlag
        self.feature_index = feature_index

    def generate(self, data: np.ndarray) -> np.ndarray:
        """
        Input: [L, N, C]
        Output: [N, N] where adj[i, j] is the p-value or test statistic.
        We will use 1 - p_value as a similarity measure (higher = more significant causality).
        """
        x = reshape_time_series_2_d(data, self.feature_index)
        L, N = x.shape
        causality_matrix = np.zeros((N, N), dtype=np.float32)
        
        # Granger causality requires stationarity normally, 
        # but here we follow the standard application of the test.
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                # Test if j causes i
                test_data = x[:, [i, j]]
                try:
                    result = grangercausalitytests(test_data, maxlag=self.maxlag, verbose=False)
                    # Use the SSR-based F-test p-value of the best lag
                    p_values = [v[0]['ssr_ftest'][1] for v in result.values()]
                    min_p = min(p_values)
                    causality_matrix[j, i] = 1.0 - min_p
                except Exception:
                    causality_matrix[j, i] = 0.0
        
        return causality_matrix
