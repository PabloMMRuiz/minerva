"""
Boolean masks for edge selection
"""

import numpy as np
import heapq


def top_k_row_mask(corr_matrix: np.ndarray, k: int):
    """
    Selects the top k strongest correlations for each node and returns a boolean mask marking those positions.
    Args:
        corr_matrix (np.ndarray): N x N correlation or similarity matrix.
        total_edges (int): Desired total number of edges for each node
    Returns:
        np.ndarray: Boolean mask (N x N) where True indicates selected edges.
    """
    adj = np.abs(corr_matrix)
    np.fill_diagonal(adj, 0)  # Remove self-loops

    # Create a mask to keep only the top-k values for each row
    mask = np.zeros_like(adj, dtype=bool)

    # For each node (row)
    for i in range(corr_matrix.shape[0]):
        # Get the *indices* that would sort the row by weight (in descending order). np already does this
        # np.argsort returns ascending, so we use [::-1] to reverse it
        top_k_indices = np.argsort(adj[i, :])[::-1][:k]
        mask[i, top_k_indices] = True  # fill in the mask accordingly
    return mask


def global_top_e_mask(corr_matrix: np.ndarray, total_edges: int, directed: bool = True) -> np.ndarray:
    """
    Selects the globally strongest 'total_edges' correlations from the matrix
    and returns a boolean mask marking those positions.

    Args:
        corr_matrix (np.ndarray): N x N correlation or similarity matrix.
        total_edges (int): Desired total number of edges in the final adjacency.
        directed (bool): If False, ensures symmetry (i.e., undirected graph). Duplicates number of edges!
                         If True (default), treats matrix as directed and selects top E individually.

    Returns:
        np.ndarray: Boolean mask (N x N) where True indicates selected edges.
    """
    N = corr_matrix.shape[0]
    abs_corr = np.abs(corr_matrix).copy()
    np.fill_diagonal(abs_corr, 0)  # remove self-loops

    if directed:
        # Flatten the whole matrix
        flat_indices = np.argsort(abs_corr, axis=None)[::-1][:total_edges]
        mask = np.zeros_like(abs_corr, dtype=bool).flatten()
        mask[flat_indices] = True
        mask = mask.reshape(N, N)
    else:
        # Work only on upper triangle to avoid double counting
        triu_indices = np.triu_indices(N, k=1)
        edge_strengths = abs_corr[triu_indices]
        # Get indices of the top 'E' strongest edges
        top_e_idx = np.argsort(edge_strengths)[::-1][:total_edges]
        # Build mask
        mask = np.zeros_like(abs_corr, dtype=bool)
        top_rows = triu_indices[0][top_e_idx]
        top_cols = triu_indices[1][top_e_idx]
        mask[top_rows, top_cols] = True
        mask[top_cols, top_rows] = True  # make symmetric

    return mask


def degree_regularized_greedy_mask(
    corr_matrix: np.ndarray,
    total_edges: int,
    penalty_factor: float = 0.1,
    directed: bool = False
) -> np.ndarray:
    """

    Selects edges greedily based on correlation strength and degree regularization.



    Args:

        corr_matrix (np.ndarray): N x N correlation or similarity matrix.

        total_edges (int): Desired total number of edges in the graph.

        penalty_factor (float): Penalty factor for high-degree nodes.

                                Larger values = more uniform degree distribution. > 1 will cause top-k due to correlation being <1

        directed (bool): If True, treat matrix as directed; otherwise symmetric (default True).

        Duplicates number od edges in undirected graphs



    Returns:

        np.ndarray: Boolean mask (N x N) with True for selected edges.

    """
    N = corr_matrix.shape[0]
    W = np.abs(corr_matrix)

    # Avoid self loops
    np.fill_diagonal(W, 0)

    degrees = np.zeros(N, dtype=int)
    mask = np.zeros((N, N), dtype=bool)

    # We use a heap to increase efficiency
    edge_heap = []

    if directed:
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Initial score: W[i,j] - penalty * (0 + 0)
                score = W[i, j]
                heapq.heappush(edge_heap, (-score, i, j))
    else:
        # Symmetric case: only consider upper
        for i in range(N):
            for j in range(i + 1, N):
                score = W[i, j]
                heapq.heappush(edge_heap, (-score, i, j))

    selected_count = 0

    while selected_count < total_edges and edge_heap:
        neg_score, i, j = heapq.heappop(edge_heap)
        recorded_score = -neg_score

        # Calculate the actual current score based on updated degrees
        current_score = W[i, j] - penalty_factor * (degrees[i] + degrees[j])

        # If the score we popped is better than the real (updated) one, it might not be the maximum anymore
        # So we update it and rerun. If it was not better, it means it actually was the maximum (it is the updated store)
        # This entire thing only works because real scores are <= than recorded ones at all times.
        if recorded_score > current_score + 1e-9:
            heapq.heappush(edge_heap, (-current_score, i, j))
            continue

        # If we reach here, current_score is valid and is the global maximum
        mask[i, j] = True
        if not directed:
            mask[j, i] = True

        degrees[i] += 1
        degrees[j] += 1
        selected_count += 1

    return mask


def threshold_with_budget_mask(corr_matrix: np.ndarray, total_edges: int, directed: bool = True, tol: float = 0.02) -> np.ndarray:
    """
    Builds a mask by finding a correlation threshold that yields approximately
    'total_edges' edges.

    Args:
        corr_matrix (np.ndarray): N x N correlation or similarity matrix.
        total_edges (int): Desired total number of edges in the final graph.
        directed (bool): If False, ensure symmetry (will duplicate edge number)
        tol (float): Relative tolerance for number of edges (0 to 1).

    Returns:
        np.ndarray: Boolean mask (N x N) with True where edges are selected.
    """
    N = corr_matrix.shape[0]
    abs_corr = np.abs(corr_matrix).copy()
    # this could, in theory, still be selected, so we wil re-clear them
    np.fill_diagonal(abs_corr, 0)

    # Only use upper triangle for undirected
    if directed:
        values = abs_corr.flatten()
    else:
        values = abs_corr[np.triu_indices(N, k=1)]

    values.sort()
    values = values[::-1]  # descending order
    num_edges_available = len(values)

    # This method will crash if we ask for more edges than N^2
    if total_edges >= num_edges_available:
        mask = np.ones_like(abs_corr, dtype=bool)
        np.fill_diagonal(mask, 0)
        return mask

    # Binary search for threshold that yields near total_edges
    low, high = 0, 1
    best_tau = 0
    target_min = total_edges * (1 - tol)
    target_max = total_edges * (1 + tol)

    while low < high:
        tau = (low + high) / 2
        if directed:
            count = np.sum(abs_corr > tau)
        else:
            count = np.sum(np.triu(abs_corr, 1) > tau)

        if target_min <= count <= target_max:
            best_tau = tau
            break
        elif count > total_edges:
            low = tau  # too many edges, raise threshold
        else:
            high = tau  # too few edges, lower threshold

        # prevent infinite loop for discrete matrices
        if abs(high - low) < 1e-6:
            best_tau = tau
            break

    # Build final mask
    mask = abs_corr > best_tau
    if not directed:
        mask = np.triu(mask, 1)  # not really needed
        mask = mask | mask.T

    np.fill_diagonal(mask, 0)
    return mask


def knn_with_global_budget_mask(corr_matrix: np.ndarray, total_edges: int, k_local: int = 8, directed: bool = True,) -> np.ndarray:
    """
    Hybrid sparsification combining local K-NN and a global edge budget.

    Args:
        corr_matrix (np.ndarray): N x N correlation (or similarity) matrix.
        total_edges (int): Desired total number of edges.
        k_local (int): Number of local strongest neighbors per node (for KNN).
        directed (bool): If False, output is symmetric (undirected).

    Returns:
        np.ndarray: Boolean N x N mask of selected edges.
    """
    N = corr_matrix.shape[0]
    abs_corr = np.abs(corr_matrix).copy()
    np.fill_diagonal(abs_corr, 0)

    # select edges in each row
    knn_mask = np.zeros_like(abs_corr, dtype=bool)
    for i in range(N):
        top_idx = np.argsort(abs_corr[i, :])[::-1][:k_local]
        knn_mask[i, top_idx] = True

    if not directed:
        knn_mask = np.logical_or(knn_mask, knn_mask.T)

    # from the original selection trim until we have less than the global budget
    current_edges = np.count_nonzero(knn_mask)
    if not directed:
        current_edges //= 2  # account for symmetry

    if current_edges > total_edges:
        # Rank edges by correlation strength and keep top total_edges
        if directed:
            scores = abs_corr[knn_mask]
            threshold = np.partition(scores, -total_edges)[-total_edges]
            final_mask = np.logical_and(knn_mask, abs_corr >= threshold)
        else:
            triu_mask = np.triu(knn_mask, k=1)
            scores = abs_corr[triu_mask]
            top_idx = np.argsort(scores)[::-1][:total_edges]
            mask = np.zeros_like(abs_corr, dtype=bool)
            triu_indices = np.triu_indices(N, k=1)
            selected_rows = triu_indices[0][top_idx]
            selected_cols = triu_indices[1][top_idx]
            mask[selected_rows, selected_cols] = True
            mask[selected_cols, selected_rows] = True
            final_mask = mask
    else:
        final_mask = knn_mask

    return final_mask


def _kruskal_mst_edges(N: int, edge_list: np.ndarray) -> np.ndarray:
    """
    Kruskal MST (descending weights) to ensure connectivity.
    edge_list: array of shape (M, 3) -> (u, v, weight)
    Returns boolean mask edges_selected (M,) True if edge chosen.
    """
    # Union-Find
    parent = np.arange(N)
    rank = np.zeros(N, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[rb] < rank[ra]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    M = edge_list.shape[0]
    selected = np.zeros(M, dtype=bool)
    # edges already expected to be sorted by weight descending
    for idx in range(M):
        u, v, _w = edge_list[idx]
        if union(int(u), int(v)):
            selected[idx] = True
    return selected


def spectral_sparsification_proxy_mask(corr_matrix: np.ndarray, total_edges: int, ensure_connected: bool = True, eps: float = 1e-12,) -> np.ndarray:
    """
    Fast spectral-sparsification proxy: uses a resistance-like proxy
    score s_ij = W_ij * (1/(deg_i) + 1/(deg_j)) to rank edges, optionally
    adding an MST first to guarantee connectivity.

    Args:
        corr_matrix (np.ndarray): (N, N) symmetric correlation matrix.
        total_edges (int): Desired total number of undirected edges (E).
        ensure_connected (bool): If True, include an MST (N-1 edges) first.
        eps (float): small constant to avoid division by zero.

    Returns:
        mask (np.ndarray): boolean (N, N) symmetric mask of selected edges.
    """
    if corr_matrix.ndim != 2 or corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError("corr_matrix must be square (N x N).")

    N = corr_matrix.shape[0]
    if total_edges < 0:
        raise ValueError("total_edges must be non-negative.")
    # Maximum possible undirected edges
    max_edges = N * (N - 1) // 2
    if total_edges >= max_edges:
        mask = np.ones((N, N), dtype=bool)
        np.fill_diagonal(mask, False)
        return mask

    # 1) weights and basic prep
    W = np.abs(corr_matrix).astype(float).copy()
    np.fill_diagonal(W, 0.0)

    # 2) compute weighted degrees
    deg = W.sum(axis=1)  # shape (N,)
    # avoid zeros
    inv_deg = 1.0 / (deg + eps)

    # 3) build upper-triangle edge list
    iu, ju = np.triu_indices(N, k=1)
    weights = W[iu, ju]
    # compute proxy score s_ij = W_ij * (1/deg_i + 1/deg_j)
    scores = weights * (inv_deg[iu] + inv_deg[ju])

    # Build structured edge array: (u, v, weight, score)
    # We'll sort by score descending (ties broken by weight)
    M = len(iu)
    edge_arr = np.empty((M, 4), dtype=float)
    edge_arr[:, 0] = iu
    edge_arr[:, 1] = ju
    edge_arr[:, 2] = weights
    edge_arr[:, 3] = scores

    # Sort edges by score descending, then by weight descending, thus the '-'
    order = np.lexsort((-edge_arr[:, 2], -edge_arr[:, 3]))
    edge_arr = edge_arr[order]

    selected_mask = np.zeros((N, N), dtype=bool)
    edges_selected = 0
    cursor = 0

    # 4) Ensure connectivity via MST (optional)
    if ensure_connected:
        # For MST we want to pick edges with largest *weight* to connect graph backbone.
        # Sort by weight descending for Kruskal
        # Build a copy sorted by weight descending
        weight_order = np.argsort(edge_arr[:, 2])[::-1]
        edges_by_weight = edge_arr[weight_order][:, :3]  # (u, v, weight)
        # MST selection returns boolean mask relative to edges_by_weight
        mst_selected_flags = _kruskal_mst_edges(N, edges_by_weight)
        # Map back selected edges to original edge_arr indices
        mst_indices_in_edge_arr = weight_order[mst_selected_flags]
        for idx in mst_indices_in_edge_arr:
            u = int(edge_arr[idx, 0])
            v = int(edge_arr[idx, 1])
            if not selected_mask[u, v]:
                selected_mask[u, v] = True
                selected_mask[v, u] = True
                edges_selected += 1
                if edges_selected >= total_edges:
                    break
        # continue selection from top scores, skipping already selected edges
    # 5) Fill remaining edges by descending score
    while edges_selected < total_edges and cursor < M:
        u = int(edge_arr[cursor, 0])
        v = int(edge_arr[cursor, 1])
        if not selected_mask[u, v]:
            selected_mask[u, v] = True
            selected_mask[v, u] = True
            edges_selected += 1
        cursor += 1

    return selected_mask


def resistance_spectral_sparsify(
    corr_matrix: np.ndarray,
    m: int,
    directed: bool = True,
    reweight: bool = True,
    pinv_tol: float = 1e-12,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resistance-based spectral sparsification (exact effective resistances via pseudoinverse).

    Args:
        corr_matrix (np.ndarray): N x N correlation or similarity matrix.
        m (int): Number of edges to sample (total undirected edges if directed=False).
        directed (bool): If True, treat matrix as directed (sample ordered pairs).
        reweight (bool): If True, reweight sampled edges by w_e / (m * p_e).
        pinv_tol (float): Tolerance passed to np.linalg.pinv (SVD cutoff).
        seed (int|None): RNG seed for reproducibility.

    Returns:
        tuple: (adj_matrix, mask, counts)
            adj_matrix (np.ndarray): N x N weighted adjacency matrix.
            mask (np.ndarray): Boolean N x N matrix, True where edges were selected.
            counts (np.ndarray): Integer N x N matrix, how many times each edge was sampled.
    """
    rng = np.random.default_rng(seed)

    N = corr_matrix.shape[0]
    if corr_matrix.shape[1] != N:
        raise ValueError("corr_matrix must be square")
    if m <= 0:
        raise ValueError("m must be positive")

    # 1. Make symmetric, positive weights
    W = np.abs(corr_matrix).astype(float)
    np.fill_diagonal(W, 0.0)
    if not directed:
        W = (W + W.T) / 2.0

    # 2. Laplacian
    deg = W.sum(axis=1)
    L = np.diag(deg) - W

    # 3. Pseudoinverse (Moore–Penrose)
    Lplus = np.linalg.pinv(L, rcond=pinv_tol)
    Lplus_diag = np.diag(Lplus)

    # 4. Build edge list
    if directed:
        edges = [(i, j) for i in range(N) for j in range(N) if i != j]
    else:
        edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    E = len(edges)

    if E == 0:
        return np.zeros((N, N), dtype=np.float32), np.zeros((N, N), dtype=bool), np.zeros((N, N), dtype=int)

    i_idx = np.array([e[0] for e in edges])
    j_idx = np.array([e[1] for e in edges])

    # 5. Effective resistances
    R = Lplus_diag[i_idx] + Lplus_diag[j_idx] - 2 * Lplus[i_idx, j_idx]
    R = np.maximum(R, 0.0)
    w_edges = W[i_idx, j_idx]

    # 6. Sampling probabilities
    scores = w_edges * R
    total_score = scores.sum()
    if total_score <= 0:
        scores = (w_edges > 0).astype(float)
        total_score = scores.sum()
    probs = scores / total_score

    # 7. Sample m edges (with replacement)
    sampled_indices = rng.choice(np.arange(E), size=m, replace=True, p=probs)
    counts_flat = np.bincount(sampled_indices, minlength=E)

    # 8. Build adjacency
    adj = np.zeros((N, N), dtype=np.float64)
    counts = np.zeros((N, N), dtype=int)

    for idx, cnt in enumerate(counts_flat):
        if cnt == 0:
            continue
        i, j = edges[idx]
        p_e = probs[idx]
        w_e = w_edges[idx]
        if reweight:
            contrib = cnt * (w_e / (m * p_e))
        else:
            contrib = cnt * w_e
        adj[i, j] += contrib
        counts[i, j] = cnt
        if not directed:
            adj[j, i] += contrib
            counts[j, i] = cnt

    mask = adj > 0.0

    return adj.astype(np.float32), mask, counts


def dhondt_proportional_allocation_sum_correlation(corr_matrix: np.ndarray, total_edges: int, directed: bool = True) -> np.ndarray:
    """
    D'Hondt-style proportional allocation sparsification.
    Each node receives edges in proportion to its total correlation mass,
    balancing high-connectivity nodes with fair degree distribution.

    Args:
        corr_matrix (np.ndarray): N x N correlation or similarity matrix.
        total_edges (int): Total number of edges desired.
        directed (bool): Whether to treat graph as directed.

    Returns:
        np.ndarray: Boolean adjacency mask (N x N).
    """
    N = corr_matrix.shape[0]
    abs_corr = np.abs(corr_matrix).copy()
    np.fill_diagonal(abs_corr, 0)  # remove self links
    mask = np.zeros_like(abs_corr, dtype=bool)
    degrees = np.zeros(N, dtype=int)

    # Total connection strength for each node
    vote_weights = abs_corr.sum(axis=1)
    edge_count = 0

    # Neighbours (seats chosen) pre sorted
    sorted_neighbors = [np.argsort(abs_corr[i, :])[::-1] for i in range(N)]

    while edge_count < total_edges:
        # D’Hondt quotients
        quotients = vote_weights / (degrees + 1)

        # Pick node with highest quotient
        i = np.argmax(quotients)

        # Find its best available neighbor
        for j in sorted_neighbors[i]:
            if i == j or mask[i, j]:
                continue
            mask[i, j] = True
            if not directed:
                mask[j, i] = True
            degrees[i] += 1
            degrees[j] += 1
            edge_count += 1
            break  # assign one edge per iteration

        # Safety break to avoid infinite loops
        if edge_count >= total_edges or not np.any(~mask):
            break

    return mask


def dhondt_top_edge_allocation_mask(corr_matrix: np.ndarray, total_edges: int, directed: bool = True) -> np.ndarray:
    """
    D'Hondt-style proportional allocation using node top-edge priority.

    Each node's current "vote weight" is its highest remaining correlation edge.
    Once an edge is chosen, the node's next highest remaining edge becomes its new weight.

    Args:
        corr_matrix (np.ndarray): N x N correlation or similarity matrix.
        total_edges (int): Total number of edges desired.
        directed (bool): Whether to treat the graph as directed.

    Returns:
        np.ndarray: Boolean adjacency mask (N x N).
    """
    N = corr_matrix.shape[0]
    abs_corr = np.abs(corr_matrix).copy()
    np.fill_diagonal(abs_corr, 0)
    mask = np.zeros_like(abs_corr, dtype=bool)
    degrees = np.zeros(N, dtype=int)

    # Precompute sorted neighbor indices for each node
    sorted_neighbors = [list(np.argsort(abs_corr[i, :])[::-1])
                        for i in range(N)]

    edge_count = 0
    while edge_count < total_edges:
        # Compute each node's current top-edge
        top_edge_values = np.zeros(N)
        top_edge_neighbors = np.full(N, -1, dtype=int)
        for i in range(N):
            # Skip already chosen edges
            while sorted_neighbors[i] and mask[i, sorted_neighbors[i][0]]:
                sorted_neighbors[i].pop(0)
            if sorted_neighbors[i]:
                top_edge_neighbors[i] = sorted_neighbors[i][0]
                top_edge_values[i] = abs_corr[i, top_edge_neighbors[i]]
            else:
                top_edge_values[i] = 0  # no remaining edges

        # Compute D’Hondt quotients
        quotients = top_edge_values / (degrees + 1)
        if np.all(quotients == 0):
            break  # no more assignable edges

        # Pick node with highest quotient
        i_star = np.argmax(quotients)
        j_star = top_edge_neighbors[i_star]

        if j_star == -1:
            break  # no available neighbor

        # Assign edge
        mask[i_star, j_star] = True
        if not directed:
            mask[j_star, i_star] = True
        degrees[i_star] += 1
        degrees[j_star] += 1
        edge_count += 1

    return mask
