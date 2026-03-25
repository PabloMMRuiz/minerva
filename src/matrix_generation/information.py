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


def _compute_mi_pair(xi, xj, k):
    """
    Kraskov MI estimator for two 1D variables.
    """
    M = xi.shape[0]

    # Joint space
    xy = np.column_stack((xi, xj))

    # kNN in joint space (Chebyshev metric is standard here)
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=k + 1)
    nn.fit(xy)
    distances, _ = nn.kneighbors(xy)

    eps = distances[:, k]  # distance to k-th neighbor

    # Count neighbors in marginal spaces
    nx = np.array([
        np.sum(np.abs(xi - xi[i]) < eps[i]) - 1
        for i in range(M)
    ])
    ny = np.array([
        np.sum(np.abs(xj - xj[i]) < eps[i]) - 1
        for i in range(M)
    ])

    return (
        digamma(k)
        + digamma(M)
        - np.mean(digamma(nx + 1) + digamma(ny + 1))
    )


def _compute_row(x, i, k):
    """
    Compute MI between variable i and all others.
    """
    xi = x[:, i]
    N = x.shape[1]
    row = np.zeros(N, dtype=np.float32)

    for j in range(i, N):
        if i == j:
            continue
        xj = x[:, j]
        mi = _compute_mi_pair(xi, xj, k)
        row[j] = mi

    return i, row


def fast_kraskov_mi_matrix(x, k=3, n_jobs=-1, symmetric=True):
    """
    x: shape [M, N]
    returns: [N, N] MI matrix
    """
    M, N = x.shape
    mi_matrix = np.zeros((N, N), dtype=np.float32)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_compute_row)(x, i, k) for i in range(N)
    )

    for i, row in results:
        mi_matrix[i, :] = row

    # Symmetrize
    if symmetric:
        mi_matrix = mi_matrix + mi_matrix.T

    # Fill diagonal
    max_val = mi_matrix.max() if mi_matrix.size > 0 else 1.0
    np.fill_diagonal(mi_matrix, max_val)

    return mi_matrix


class MutualInformationGenerator(MatrixGenerator):
    def __init__(
        self,
        n_neighbors: int = 3,
        feature_index: int = 0,
        n_jobs: int = -1,
        symmetric: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("mutual_information", params)
        self.n_neighbors = n_neighbors
        self.feature_index = feature_index
        self.n_jobs = n_jobs
        self.symmetric = symmetric

    def _compute_row(self, x: np.ndarray, i: int) -> np.ndarray:
        target = x[:, i]
        return mutual_info_regression(
            x,
            target,
            n_neighbors=self.n_neighbors,
            n_jobs=1,  # avoid nested parallelism
        )

    def generate(self, data: np.ndarray) -> np.ndarray:
        """
        Input: [L, N, C]
        """
        x = reshape_time_series_2_d(data, self.feature_index)
        print('AAAAAAAAAAAAAAAAAA')
        return fast_kraskov_mi_matrix(
            x,
            k=self.n_neighbors,
            n_jobs=-1
        )
        L, N = x.shape

        # Parallel over targets
        rows = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._compute_row)(x, i) for i in range(N)
        )

        mi_matrix = np.vstack(rows).astype(np.float32)

        # Optional: enforce symmetry (MI should be symmetric but estimator noise breaks it)
        if self.symmetric:
            mi_matrix = 0.5 * (mi_matrix + mi_matrix.T)

        # Fill diagonal
        max_val = mi_matrix.max() if mi_matrix.size > 0 else 1.0
        np.fill_diagonal(mi_matrix, max_val)

        return mi_matrix
