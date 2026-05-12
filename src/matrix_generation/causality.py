import numpy as np
from typing import Dict, Optional, Any
from statsmodels.tsa.stattools import grangercausalitytests
from joblib import Parallel, delayed
from tqdm import tqdm

from .base import MatrixGenerator
from src.data.preprocessing import reshape_time_series_2_d

# =====================================================================
# CORE ESTIMATOR: Pairwise Granger Test
# =====================================================================


def _compute_granger_row(x: np.ndarray, i: int, maxlag: int) -> tuple:
    """
    Computes the causality for a specific target row i.
    Tests if feature j causes feature i.
    """
    L, N = x.shape
    row = np.zeros(N, dtype=np.float32)

    for j in range(N):
        if i == j:
            continue

        # Test if j causes i (Input order: [dependent, independent])
        test_data = x[:, [i, j]]
        try:
            # verbose=False is crucial for performance and clean logs
            result = grangercausalitytests(
                test_data, maxlag=maxlag)

            # Extract the SSR-based F-test p-value for each lag and take the minimum
            p_values = [v[0]['ssr_ftest'][1] for v in result.values()]
            min_p = min(p_values)

            # Convert p-value to a similarity-like score
            row[j] = 1.0 - min_p
        except Exception:
            row[j] = 0.0

    return i, row

# =====================================================================
# THE GENERATOR CLASS
# =====================================================================


class GrangerCausalityGenerator(MatrixGenerator):
    """
    Calculates the Granger Causality matrix with parallel execution.
    Resulting matrix is asymmetric: adj[j, i] represents j -> i.
    """

    def __init__(
        self,
        maxlag: int = 4,
        feature_index: int = 0,
        n_jobs: int = -1,
        params: Optional[Dict[str, Any]] = None
    ):
        super().__init__("granger_causality", params)
        self.maxlag = maxlag
        self.feature_index = feature_index
        self.n_jobs = n_jobs

    def generate(self, data: np.ndarray) -> np.ndarray:
        """
        Input: [L, N, C]
        Output: [N, N] where adj[j, i] is (1 - p_value).
        """
        x = reshape_time_series_2_d(data, self.feature_index)
        L, N = x.shape
        causality_matrix = np.zeros((N, N), dtype=np.float32)

        print(
            f"Calculating Granger Causality Matrix ({N}x{N}) with maxlag={self.maxlag}...")

        # Parallel execution over rows (target variables)
        # We use prefer="processes" because statsmodels/ols can be CPU intensive
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_compute_granger_row)(x, i, self.maxlag)
            for i in tqdm(range(N), desc="Analyzing Granger Paths")
        )

        # Assemble the matrix
        # Note: _compute_granger_row returns row[j] = j -> i
        for i, row in results:
            causality_matrix[:, i] = row

        # Fill diagonal with 1.0 (self-causality is perfect by definition)
        np.fill_diagonal(causality_matrix, 1.0)

        return causality_matrix
