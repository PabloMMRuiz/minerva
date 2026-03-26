import numpy as np
from typing import Dict, Optional, Any
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
from .base import MatrixGenerator
from src.data.preprocessing import reshape_time_series_2_d


import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
from joblib import Parallel, delayed
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
from typing import Optional, Dict, Any

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Optional, Dict, Any

# =====================================================================
# CORE ESTIMATOR: Vectorized Binary Search (Solution 2)
# =====================================================================


def _compute_mi_pair_vectorized(xi: np.ndarray, xj: np.ndarray, k: int) -> float:
    M = xi.shape[0]
    xy = np.column_stack((xi, xj))

    # Joint space kNN
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=k + 1)
    nn.fit(xy)
    distances, _ = nn.kneighbors(xy)
    eps = distances[:, k]

    # Binary search counting in marginals O(M log M)
    xi_sorted = np.sort(xi)
    xj_sorted = np.sort(xj)

    nx = np.searchsorted(xi_sorted, xi + eps, side='left') - \
        np.searchsorted(xi_sorted, xi - eps, side='right') - 1
    ny = np.searchsorted(xj_sorted, xj + eps, side='left') - \
        np.searchsorted(xj_sorted, xj - eps, side='right') - 1

    nx = np.maximum(nx, 0)
    ny = np.maximum(ny, 0)

    return float(digamma(k) + digamma(M) - np.mean(digamma(nx + 1) + digamma(ny + 1)))

# =====================================================================
# PARALLEL HELPERS
# =====================================================================


def _compute_row_custom(x: np.ndarray, i: int, k: int):
    N = x.shape[1]
    xi = x[:, i]
    row = np.zeros(N, dtype=np.float32)
    # Only compute upper triangle
    for j in range(i + 1, N):
        row[j] = _compute_mi_pair_vectorized(xi, x[:, j], k)
    return i, row


def _compute_row_sklearn(x: np.ndarray, i: int, k: int):
    target = x[:, i]
    # sklearn calculates MI for node i vs all others in one go (very fast)
    return i, mutual_info_regression(x, target, n_neighbors=k, n_jobs=1)

# =====================================================================
# THE GENERATOR CLASS
# =====================================================================


class MutualInformationGenerator(MatrixGenerator):
    def __init__(
        self,
        n_neighbors: int = 3,
        feature_index: int = 0,
        n_jobs: int = -1,
        symmetric: bool = True,
        use_custom_backend: bool = False,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("mutual_information", params)
        self.n_neighbors = n_neighbors
        self.feature_index = feature_index
        self.n_jobs = n_jobs
        self.symmetric = symmetric
        self.use_custom_backend = use_custom_backend

    def generate(self, data: np.ndarray) -> np.ndarray:
        """
        Input: [L, N, C] -> Output: [N, N]
        """
        x = reshape_time_series_2_d(data, self.feature_index)
        M, N = x.shape
        mi_matrix = np.zeros((N, N), dtype=np.float32)

        # Select backend
        task_func = _compute_row_custom if self.use_custom_backend else _compute_row_sklearn

        print(
            f"Calculating MI Matrix ({N}x{N}) using {'Custom' if self.use_custom_backend else 'Sklearn'} backend...")

        # Parallel execution with Progress Bar
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(task_func)(x, i, self.n_neighbors)
            for i in tqdm(range(N), desc="Processing Nodes")
        )

        for i, row in results:
            mi_matrix[i, :] = row

        if self.symmetric:
            # For sklearn, we average. For custom, we just add the transpose
            # because we only computed the upper triangle.
            if self.use_custom_backend:
                mi_matrix = mi_matrix + mi_matrix.T
            else:
                mi_matrix = (mi_matrix + mi_matrix.T) / 2.0

        # Fill diagonal
        max_val = mi_matrix.max() if mi_matrix.size > 0 else 1.0
        np.fill_diagonal(mi_matrix, max_val)

        return mi_matrix
