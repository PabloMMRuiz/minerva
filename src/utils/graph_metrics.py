"""
Graph metrics calculation module

Contains methods for calculating diverse metrics and stats for graphs given their adjacency matrix
and for comparation of graphs
"""

import numpy as np
from scipy import stats


def calculate_graph_metrics(adj_matrix: np.ndarray, directed: bool = True):
    """
    Calculates structural and statistical metrics for a graph represented by an adjacency matrix.

    Args:
        adj_matrix (np.ndarray): N x N adjacency matrix (can be weighted).
        directed (bool): True if the graph is directed, False if undirected.

    Returns:
        dict: A dictionary of graph metrics.
    """
    # --- Input validation ---
    if not isinstance(adj_matrix, np.ndarray) or adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Input must be a square NumPy adjacency matrix.")

    # Ensure no negative weights
    adj_matrix = np.abs(adj_matrix)
    num_nodes = adj_matrix.shape[0]

    # --- Binary adjacency for unweighted measures ---
    binary_adj = (adj_matrix > 1e-6).astype(int)

    # --- Degrees ---
    if directed:
        out_degrees = np.sum(binary_adj, axis=1)
        in_degrees = np.sum(binary_adj, axis=0)
        degrees = out_degrees + in_degrees
    else:
        degrees = np.sum(binary_adj, axis=1)

    # --- Basic stats on degree distribution ---
    degree_stats = {
        "min_degree": int(np.min(degrees)),
        "max_degree": int(np.max(degrees)),
        "mean_degree": float(np.mean(degrees)),
        "median_degree": float(np.median(degrees)),
        "mode_degree": int(stats.mode(degrees, keepdims=True)[0][0]),
        "std_degree": float(np.std(degrees)),
        "q1_degree": float(np.percentile(degrees, 25)),
        "q3_degree": float(np.percentile(degrees, 75)),
    }

    total_edges = np.sum(binary_adj)
    if not directed:
        total_edges //= 2  # avoid double-counting

    # --- Weighted measures ---
    existing_weights = adj_matrix[adj_matrix > 0]
    avg_connection_strength = np.mean(
        existing_weights) if existing_weights.size > 0 else 0.0

    # --- Graph-level metrics ---
    possible_edges = num_nodes * (num_nodes - 1)
    if not directed:
        possible_edges /= 2

    density = total_edges / possible_edges if possible_edges > 0 else 0.0

    # Reciprocity (for directed graphs)
    reciprocity = None
    if directed:
        mutual_edges = np.sum((binary_adj > 0) & (binary_adj.T > 0)) // 2
        reciprocity = mutual_edges / total_edges if total_edges > 0 else 0.0

    # --- Clustering coefficient ---
    # Triangles = trace(A^3) / 6 (undirected), or /3 (directed)
    A3 = np.linalg.matrix_power(binary_adj, 3)
    triangles = np.trace(A3)
    if directed:
        clustering_coeff = triangles / \
            (num_nodes * (num_nodes - 1) * (num_nodes - 2)) if num_nodes > 2 else 0.0
    else:
        clustering_coeff = triangles / \
            (6 * total_edges) if total_edges > 0 else 0.0

    # --- Compile results ---
    metrics = {
        "num_nodes": num_nodes,
        "total_edges": int(total_edges),
        "density": float(density),
        "avg_connection_strength": float(avg_connection_strength),
        "clustering_coefficient": float(clustering_coeff),
        **degree_stats
    }

    if directed:
        metrics.update({
            "reciprocity": float(reciprocity),
            "mean_in_degree": float(np.mean(in_degrees)),
            "mean_out_degree": float(np.mean(out_degrees)),
        })

    return metrics


def compare_adjacency_matrices(A: np.ndarray, B: np.ndarray):
    """
    Compare two adjacency matrices A and B (possibly directed, weighted).
    """
    if A.shape != B.shape:
        raise ValueError("Adjacency matrices must have the same shape")

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    num_nodes = A.shape[0]

    A_bin = (A > 0).astype(int)
    B_bin = (B > 0).astype(int)

    # --- Basic structure ---
    num_edges_A = int(np.sum(A_bin))
    num_edges_B = int(np.sum(B_bin))
    avg_degree_A = float(num_edges_A / num_nodes)
    avg_degree_B = float(num_edges_B / num_nodes)

    # --- Error metrics ---
    abs_error_weighted = float(np.sum(np.abs(A - B)))
    abs_error_binary = float(np.sum(np.abs(A_bin - B_bin)))
    frobenius_error = float(np.linalg.norm(A - B, 'fro'))
    error_per_node_weighted = float(abs_error_weighted / num_nodes)
    error_per_node_binary = float(abs_error_binary / num_nodes)

    # --- Overlap metrics ---
    common_edges = int(np.sum((A_bin == 1) & (B_bin == 1)))
    union_edges = int(np.sum((A_bin == 1) | (B_bin == 1)))
    jaccard_similarity = float(common_edges / (union_edges + 1e-12))

    # --- Cosine & Pearson similarities ---
    cosine_similarity = float(
        np.dot(A.flatten(), B.flatten()) /
        (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12)
    )
    pearson_corr = float(np.corrcoef(A.flatten(), B.flatten())[0, 1])

    # --- Edge-level classification metrics ---
    tp = common_edges
    fp = int(np.sum((A_bin == 0) & (B_bin == 1)))
    fn = int(np.sum((A_bin == 1) & (B_bin == 0)))
    precision = float(tp / (tp + fp + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))
    f1_score = float(2 * precision * recall / (precision + recall + 1e-12))

    # --- Node-level degree correlations ---
    out_deg_A, in_deg_A = np.sum(A, axis=1), np.sum(A, axis=0)
    out_deg_B, in_deg_B = np.sum(B, axis=1), np.sum(B, axis=0)
    out_corr = float(np.corrcoef(out_deg_A, out_deg_B)[0, 1])
    in_corr = float(np.corrcoef(in_deg_A, in_deg_B)[0, 1])

    # --- Graph-level spectral measures ---
    def safe_eigvals(M):
        try:
            eigvals = np.linalg.eigvals(M)
            eigvals = np.real(eigvals)
            eigvals.sort()
            return eigvals
        except np.linalg.LinAlgError:
            return np.zeros(num_nodes)

    eig_A = safe_eigvals(A)
    eig_B = safe_eigvals(B)
    spectral_distance = float(np.linalg.norm(eig_A - eig_B))

    L_A = np.diag(np.sum(A, axis=1)) - A
    L_B = np.diag(np.sum(B, axis=1)) - B
    eig_LA = safe_eigvals(L_A)
    eig_LB = safe_eigvals(L_B)
    laplacian_spectral_distance = float(np.linalg.norm(eig_LA - eig_LB))
    graph_energy_diff = float(np.abs(np.sum(eig_A**2) - np.sum(eig_B**2)))

    return {
        # Basic info
        "num_nodes": int(num_nodes),
        "num_edges_A": num_edges_A,
        "num_edges_B": num_edges_B,
        "avg_degree_A": avg_degree_A,
        "avg_degree_B": avg_degree_B,

        # Error metrics
        "abs_error_weighted": abs_error_weighted,
        "abs_error_binary": abs_error_binary,
        "error_per_node_weighted": error_per_node_weighted,
        "error_per_node_binary": error_per_node_binary,
        "frobenius_error": frobenius_error,

        # Overlap / classification
        "common_edges": common_edges,
        "jaccard_similarity": jaccard_similarity,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,

        # Correlations
        "cosine_similarity": cosine_similarity,
        "pearson_corr": pearson_corr,
        "out_degree_corr": out_corr,
        "in_degree_corr": in_corr,

        # Spectral / global measures
        "spectral_distance": spectral_distance,
        "laplacian_spectral_distance": laplacian_spectral_distance,
        "graph_energy_diff": graph_energy_diff,
    }


def is_connected(adj_matrix: np.ndarray) -> bool:
    """
    Check if the undirected graph represented by the adjacency matrix is connected.

    Parameters:
        adj_matrix (np.ndarray): Square adjacency matrix (n x n)
                                 where adj_matrix[i, j] > 0 indicates an edge.

    Returns:
        bool: True if the graph is connected, False otherwise.
    """
    n = adj_matrix.shape[0]
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    # Use BFS or DFS — here we use BFS
    visited = np.zeros(n, dtype=bool)
    stack = [0]  # start from node 0
    visited[0] = True

    while stack:
        node = stack.pop()
        # Get neighbors (nonzero entries)
        neighbors = np.where(adj_matrix[node] > 0)[0]
        for neigh in neighbors:
            if not visited[neigh]:
                visited[neigh] = True
                stack.append(neigh)

    return visited.all()
