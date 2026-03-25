"""
Dynamic Time Warping (DTW) matrix generator.
"""

import numpy as np
from typing import Dict, Optional, Any
from .base import MatrixGenerator
from src.data.preprocessing import reshape_time_series_2_d

import numpy as np
from numba import njit, prange
from typing import Optional, Dict, Any


@njit(fastmath=True)
def compute_envelopes(x, window):
    """Precomputes LB_Keogh envelopes: O(L * N)"""
    L, N = x.shape
    upper = np.zeros((L, N))
    lower = np.zeros((L, N))
    for i in range(N):
        for t in range(L):
            r = window
            # Get max/min in the sliding window
            upper[t, i] = np.max(x[max(0, t-r):min(L, t+r+1), i])
            lower[t, i] = np.min(x[max(0, t-r):min(L, t+r+1), i])
    return upper, lower


@njit(fastmath=True)
def lb_keogh_dist(s1, u2, l2):
    """Calculate LB_Keogh distance: O(L)"""
    dist_sq = 0.0
    for i in range(len(s1)):
        if s1[i] > u2[i]:
            dist_sq += (s1[i] - u2[i])**2
        elif s1[i] < l2[i]:
            dist_sq += (s1[i] - l2[i])**2
    return np.sqrt(dist_sq)


@njit(fastmath=True)
def _numba_dtw(s1, s2, window):
    """Optimized DTW with O(L) space."""
    n1, n2 = len(s1), len(s2)
    prev_line = np.full(n2 + 1, np.inf)
    curr_line = np.full(n2 + 1, np.inf)
    prev_line[0] = 0

    for i in range(1, n1 + 1):
        curr_line[0] = np.inf
        low = max(1, i - window)
        high = min(n2 + 1, i + window)
        for j in range(low, high):
            cost = (s1[i - 1] - s2[j - 1])**2
            curr_line[j] = cost + \
                min(prev_line[j], curr_line[j - 1], prev_line[j - 1])
        prev_line[:] = curr_line
    return np.sqrt(prev_line[n2])


@njit(parallel=True)
def _compute_matrix_with_pruning(x, upper, lower, window, threshold=None):
    L, N = x.shape
    dist_matrix = np.zeros((N, N), dtype=np.float32)

    for i in prange(N):
        s1 = x[:, i]
        for j in range(i + 1, N):
            # 1. Check LB_Keogh (Super fast O(L))
            lb = lb_keogh_dist(s1, upper[:, j], lower[:, j])

            # 2. Pruning: If LB is already too high, we can skip DTW
            # If threshold is None, we always compute DTW, but LB still helps logic
            if threshold is not None and lb > threshold:
                dist = lb  # Approximate or use a max_dist
            else:
                # 3. Compute full DTW only when necessary
                dist = _numba_dtw(s1, x[:, j], window)

            dist_matrix[i, j] = dist_matrix[j, i] = dist

    return 1.0 / (1.0 + dist_matrix)


class DTWGenerator(MatrixGenerator):
    def __init__(self, feature_index=0, window=None, pruning_threshold=None, params=None):
        super().__init__("dtw", params)
        self.feature_index = feature_index
        self.window = window
        self.pruning_threshold = pruning_threshold  # Distance beyond which we skip DTW

    def generate(self, data: np.ndarray) -> np.ndarray:
        x = reshape_time_series_2_d(data, self.feature_index)
        L, N = x.shape
        # 10% is a standard heuristic
        window = self.window if self.window is not None else L // 10

        # Step 1: Precompute envelopes
        upper, lower = compute_envelopes(x, window)

        # Step 2: Compute matrix with pruning
        return _compute_matrix_with_pruning(x, upper, lower, window, self.pruning_threshold)
