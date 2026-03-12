"""
Adjacency matrix preprocessing and normalization functions.

This module contains various methods for transforming adjacency matrices
into different representations suitable for GNN models.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple


# --------------------------------------------
# LAPLACIAN MATRICES

def calculate_scaled_laplacian(adj: np.ndarray, lambda_max: int = 2, undirected: bool = True) -> np.matrix:
    """
    Scale the normalized Laplacian for use in Chebyshev polynomials.

    Rescale the Laplacian matrix such that its eigenvalues are within the range [-1, 1].

    Args:
        adj (np.ndarray): Adjacency matrix A.
        lambda_max (int, optional): Maximum eigenvalue, defaults to 2.
        undirected (bool, optional): If True, treats the graph as undirected, defaults to True.

    Returns:
        np.matrix: Scaled Laplacian matrix.
    """

    if undirected:
        adj = np.maximum(adj, adj.T)

    laplacian = calculate_symmetric_normalized_laplacian(adj)

    if lambda_max is None:
        lambda_max, _ = sp.linalg.eigsh(laplacian, 1, which='LM')
        lambda_max = lambda_max[0]

    laplacian = sp.csr_matrix(laplacian)
    identity = sp.identity(
        laplacian.shape[0], format='csr', dtype=laplacian.dtype)

    scaled_laplacian = (2 / lambda_max) * laplacian - identity
    return scaled_laplacian


def calculate_symmetric_normalized_laplacian(adj: np.ndarray) -> np.matrix:
    """
    Calculate the symmetric normalized Laplacian.

    The symmetric normalized Laplacian matrix is given by:
    L^{Sym} = I - D^{-1/2} A D^{-1/2}, where L is the unnormalized Laplacian, 
    D is the degree matrix, and A is the adjacency matrix.

    Args:
        adj (np.ndarray): Adjacency matrix A.

    Returns:
        np.matrix: Symmetric normalized Laplacian L^{Sym}.
    """

    adj = sp.coo_matrix(adj)
    degree = np.array(adj.sum(1)).flatten()
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    matrix_degree_inv_sqrt = sp.diags(degree_inv_sqrt)

    laplacian = sp.eye(
        adj.shape[0]) - matrix_degree_inv_sqrt.dot(adj).dot(matrix_degree_inv_sqrt).tocoo()
    return laplacian


# ------------------------------------------------------------
# MESSAGE PASSING / DIFFUSION MATRICES

def calculate_symmetric_message_passing_adj(adj: np.ndarray) -> np.matrix:
    """
    Calculate the renormalized message-passing adjacency matrix as proposed in GCN.

    The message-passing adjacency matrix is defined as A' = D^{-1/2} (A + I) D^{-1/2}.

    Args:
        adj (np.ndarray): Adjacency matrix A.

    Returns:
        np.matrix: Renormalized message-passing adjacency matrix.
    """

    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    adj = sp.coo_matrix(adj)

    row_sum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = d_mat_inv_sqrt.dot(adj).transpose().dot(
        d_mat_inv_sqrt).astype(np.float32)

    return mp_adj


def calculate_transition_matrix(adj: np.ndarray) -> np.matrix:
    """
    Calculate the transition matrix as proposed in DCRNN and Graph WaveNet.

    The transition matrix is defined as P = D^{-1} A, where D is the degree matrix.

    Args:
        adj (np.ndarray): Adjacency matrix A.

    Returns:
        np.matrix: Transition matrix P.
    """

    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1)
    d_inv[np.isinf(d_inv)] = 0.0

    d_mat = sp.diags(d_inv)
    prob_matrix = d_mat.dot(adj).astype(np.float32).todense()

    return prob_matrix

# -----------------------------------------------
# UTILITY FUNCTIONS


def add_self_loops(adj_mx: np.ndarray, fill_value: float = 1.0) -> np.ndarray:
    """
    Add self-loops to adjacency matrix.

    Args:
        adj_mx: Adjacency matrix (N x N)
        fill_value: Value for self-loops (default: 1.0)

    Returns:
        Adjacency matrix with self-loops
    """
    adj_with_loops = adj_mx.copy()
    np.fill_diagonal(adj_with_loops, fill_value)
    return adj_with_loops


def remove_self_loops(adj_mx: np.ndarray) -> np.ndarray:
    """
    Remove self-loops from adjacency matrix.

    Args:
        adj_mx: Adjacency matrix (N x N)

    Returns:
        Adjacency matrix without self-loops
    """
    adj_no_loops = adj_mx.copy()
    np.fill_diagonal(adj_no_loops, 0)
    return adj_no_loops


def symmetrize_adjacency(adj_mx: np.ndarray, method: str = 'avg') -> np.ndarray:
    """
    Make adjacency matrix symmetric.

    Args:
        adj_mx: Adjacency matrix (N x N)
        method: Symmetrization method
                - 'avg': (A + A^T) / 2
                - 'max': element-wise max(A, A^T)
                - 'min': element-wise min(A, A^T)

    Returns:
        Symmetric adjacency matrix

    Raises:
        ValueError: If method is not recognized
    """
    if method == 'avg':
        return (adj_mx + adj_mx.T) / 2
    elif method == 'max':
        return np.maximum(adj_mx, adj_mx.T)
    elif method == 'min':
        return np.minimum(adj_mx, adj_mx.T)
    else:
        raise ValueError(f"Unknown symmetrization method: {method}. "
                         f"Choose from: 'avg', 'max', 'min'")


def reshape_time_series_2_d(data_array: np.ndarray, feature_index: int = 0) -> np.ndarray:
    """
    Extracts a single feature and reshapes the time series data for correlation analysis.

    The input shape is [L, N, C] (Time Steps, Nodes, Features).
    The output shape is [L, N] (Time Steps, Nodes), which is required by np.corrcoef.

    Args:
        data_array (np.ndarray): The time series data array with shape [L, N, C].
        feature_index (int, optional): The index of the feature (C) to use. Defaults to the first feature (0), which is usually the target.

    Returns:
        np.ndarray: The reshaped array with shape [L, N].
    """
    L, N, C = data_array.shape

    if not (0 <= feature_index < C):
        raise ValueError(
            f"Feature index {feature_index} is out of bounds. Valid range is 0 to {C-1}.")

    # (slicing keeps the [L, N, 1] shape)
    single_feature_data = data_array[:, :, feature_index]
    # .
    # so we squeeze it back in
    shaped_data = np.squeeze(single_feature_data)

    return shaped_data


# -------------------------------------------------------------
# DEGREE MATRIX OPERATIONS

def get_degree_matrix(adj_mx: np.ndarray) -> np.ndarray:
    """
    Compute degree matrix from adjacency matrix.

    Args:
        adj_mx: Adjacency matrix (N x N)
        sparse: Return as sparse diagonal matrix

    Returns:
        Degree matrix D where D[i,i] = sum of row i in adj_mx
    """
    degrees = np.sum(adj_mx, axis=1).A1 if sp.issparse(
        adj_mx) else np.sum(adj_mx, axis=1)
    return np.diag(degrees)


def get_inverse_degree_matrix(adj_mx: np.ndarray, power: float = -1.0, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute inverse (or fractional power) of degree matrix.

    Args:
        adj_mx: Adjacency matrix (N x N)
        power: Power for degree (e.g., -1 for inverse, -0.5 for D^(-1/2))
        sparse: Return as sparse diagonal matrix
        epsilon: Small value added to degrees to avoid division by zero

    Returns:
        D^power where D is degree matrix
    """
    degrees = np.sum(adj_mx, axis=1).A1 if sp.issparse(
        adj_mx) else np.sum(adj_mx, axis=1)

    # Add epsilon to avoid division by zero
    degrees = degrees + epsilon

    # Compute power
    degrees_power = np.power(degrees, power)

    return np.diag(degrees_power)


# --------------------------------------------------------------------
# VALIDATION

def validate_adjacency_matrix(adj_mx: np.ndarray) -> Tuple[bool, str]:
    """
    Validate adjacency matrix properties.

    Checks:
    - Square matrix
    - Non-negative values
    - No NaN/Inf values

    Args:
        adj_mx: Adjacency matrix to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if square
    if adj_mx.ndim != 2:
        return False, f"Matrix must be 2D, got {adj_mx.ndim}D"

    if adj_mx.shape[0] != adj_mx.shape[1]:
        return False, f"Matrix must be square, got shape {adj_mx.shape}"

    # Check for NaN or Inf
    if np.any(np.isnan(adj_mx)):
        return False, "Matrix contains NaN values"

    if np.any(np.isinf(adj_mx)):
        return False, "Matrix contains Inf values"

    # Check for non-negative (optional - some graphs allow negative weights)
    if np.any(adj_mx < 0):
        return False, "Matrix contains negative values"

    return True, "Valid adjacency matrix"
