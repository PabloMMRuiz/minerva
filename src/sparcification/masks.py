"""
Sparsification methods implemented as Sparsifier classes.
"""

import numpy as np
import heapq
from typing import Dict, Optional, Any
from .base import Sparsifier


class TopKRowSparsifier(Sparsifier):
    """
    Selects the top k strongest correlations for each node.
    """

    def __init__(self, k_per_node: Optional[int] = None, params: Optional[Dict[str, Any]] = None):
        super().__init__("top_k_row", params)
        if k_per_node is not None:
            self.params['k_per_node'] = k_per_node

    def get_mask(self, matrix: np.ndarray, effective_k: int) -> np.ndarray:
        N = matrix.shape[0]
        adj = np.abs(matrix)
        np.fill_diagonal(adj, 0)
        
        k = self.params.get('k_per_node', effective_k // N if N > 0 else 0)
        mask = np.zeros_like(adj, dtype=bool)

        for i in range(N):
            top_k_indices = np.argsort(adj[i, :])[::-1][:k]
            mask[i, top_k_indices] = True
        return mask


class GlobalTopESparsifier(Sparsifier):
    """
    Selects the globally strongest correlations from the matrix.
    """

    def __init__(self, directed: bool = True, params: Optional[Dict[str, Any]] = None):
        super().__init__("global_top_e", params)
        self.directed = directed

    def get_mask(self, matrix: np.ndarray, effective_k: int) -> np.ndarray:
        N = matrix.shape[0]
        abs_corr = np.abs(matrix).copy()
        np.fill_diagonal(abs_corr, 0)

        if self.directed:
            flat_indices = np.argsort(abs_corr, axis=None)[::-1][:effective_k]
            mask = np.zeros_like(abs_corr, dtype=bool).flatten()
            mask[flat_indices] = True
            mask = mask.reshape(N, N)
        else:
            triu_indices = np.triu_indices(N, k=1)
            edge_strengths = abs_corr[triu_indices]
            top_e_idx = np.argsort(edge_strengths)[::-1][:effective_k]
            mask = np.zeros_like(abs_corr, dtype=bool)
            top_rows = triu_indices[0][top_e_idx]
            top_cols = triu_indices[1][top_e_idx]
            mask[top_rows, top_cols] = True
            mask[top_cols, top_rows] = True
        return mask


class DegreeRegularizedGreedySparsifier(Sparsifier):
    """
    Selects edges greedily based on correlation strength and degree regularization.
    """

    def __init__(self, penalty_factor: float = 0.1, directed: bool = True, params: Optional[Dict[str, Any]] = None):
        super().__init__("degree_regularized_greedy", params)
        self.penalty_factor = penalty_factor
        self.directed = directed

    def get_mask(self, matrix: np.ndarray, effective_k: int) -> np.ndarray:
        N = matrix.shape[0]
        W = np.abs(matrix)
        np.fill_diagonal(W, 0)

        degrees = np.zeros(N, dtype=int)
        mask = np.zeros((N, N), dtype=bool)
        edge_heap = []

        if self.directed:
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    heapq.heappush(edge_heap, (-W[i, j], i, j))
        else:
            for i in range(N):
                for j in range(i + 1, N):
                    heapq.heappush(edge_heap, (-W[i, j], i, j))

        selected_count = 0
        while selected_count < effective_k and edge_heap:
            neg_score, i, j = heapq.heappop(edge_heap)
            recorded_score = -neg_score
            current_score = W[i, j] - self.penalty_factor * (degrees[i] + degrees[j])

            if recorded_score > current_score + 1e-9:
                heapq.heappush(edge_heap, (-current_score, i, j))
                continue

            mask[i, j] = True
            if not self.directed:
                mask[j, i] = True
            degrees[i] += 1
            degrees[j] += 1
            selected_count += 1

        return mask


class ThresholdWithBudgetSparsifier(Sparsifier):
    """
    Finds a correlation threshold that yields approximately 'effective_k' edges.
    """

    def __init__(self, directed: bool = True, tol: float = 0.02, params: Optional[Dict[str, Any]] = None):
        super().__init__("threshold_with_budget", params)
        self.directed = directed
        self.tol = tol

    def get_mask(self, matrix: np.ndarray, effective_k: int) -> np.ndarray:
        N = matrix.shape[0]
        abs_corr = np.abs(matrix).copy()
        np.fill_diagonal(abs_corr, 0)

        if self.directed:
            values = abs_corr.flatten()
        else:
            values = abs_corr[np.triu_indices(N, k=1)]

        values.sort()
        values = values[::-1]
        if effective_k >= len(values):
            mask = np.ones_like(abs_corr, dtype=bool)
            np.fill_diagonal(mask, 0)
            return mask

        low, high = 0, 1
        best_tau = 0
        target_min = effective_k * (1 - self.tol)
        target_max = effective_k * (1 + self.tol)

        while low < high:
            tau = (low + high) / 2
            if self.directed:
                count = np.sum(abs_corr > tau)
            else:
                count = np.sum(np.triu(abs_corr, 1) > tau)

            if target_min <= count <= target_max:
                best_tau = tau
                break
            elif count > effective_k:
                low = tau
            else:
                high = tau
            if abs(high - low) < 1e-6:
                best_tau = tau
                break

        mask = abs_corr > best_tau
        if not self.directed:
            mask = mask | mask.T
        np.fill_diagonal(mask, 0)
        return mask


class KNNWithGlobalBudgetSparsifier(Sparsifier):
    """
    Hybrid sparsification combining local K-NN and a global edge budget.
    """

    def __init__(self, k_local: int = 8, directed: bool = True, params: Optional[Dict[str, Any]] = None):
        super().__init__("knn_with_global_budget", params)
        self.k_local = k_local
        self.directed = directed

    def get_mask(self, matrix: np.ndarray, effective_k: int) -> np.ndarray:
        N = matrix.shape[0]
        abs_corr = np.abs(matrix).copy()
        np.fill_diagonal(abs_corr, 0)

        knn_mask = np.zeros_like(abs_corr, dtype=bool)
        for i in range(N):
            top_idx = np.argsort(abs_corr[i, :])[::-1][:self.k_local]
            knn_mask[i, top_idx] = True

        if not self.directed:
            knn_mask = np.logical_or(knn_mask, knn_mask.T)

        current_edges = np.count_nonzero(knn_mask)
        if not self.directed:
            current_edges //= 2

        if current_edges > effective_k:
            if self.directed:
                scores = abs_corr[knn_mask]
                threshold = np.partition(scores, -effective_k)[-effective_k]
                final_mask = np.logical_and(knn_mask, abs_corr >= threshold)
            else:
                triu_mask = np.triu(knn_mask, k=1)
                scores = abs_corr[triu_mask]
                top_idx = np.argsort(scores)[::-1][:effective_k]
                mask = np.zeros_like(abs_corr, dtype=bool)
                triu_indices = np.triu_indices(N, k=1)
                mask[triu_indices[0][top_idx], triu_indices[1][top_idx]] = True
                mask = mask | mask.T
                final_mask = mask
        else:
            final_mask = knn_mask

        return final_mask


class SpectralProxySparsifier(Sparsifier):
    """
    Fast spectral-sparsification proxy using resistance-like scores.
    """

    def __init__(self, ensure_connected: bool = True, eps: float = 1e-12, params: Optional[Dict[str, Any]] = None):
        super().__init__("spectral_proxy", params)
        self.ensure_connected = ensure_connected
        self.eps = eps

    def _kruskal_mst_edges(self, N: int, edge_list: np.ndarray) -> np.ndarray:
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
        for idx in range(M):
            if union(int(edge_list[idx, 0]), int(edge_list[idx, 1])):
                selected[idx] = True
        return selected

    def get_mask(self, matrix: np.ndarray, effective_k: int) -> np.ndarray:
        N = matrix.shape[0]
        W = np.abs(matrix).copy()
        np.fill_diagonal(W, 0)

        deg = W.sum(axis=1)
        inv_deg = 1.0 / (deg + self.eps)

        iu, ju = np.triu_indices(N, k=1)
        weights = W[iu, ju]
        scores = weights * (inv_deg[iu] + inv_deg[ju])

        M = len(iu)
        edge_arr = np.empty((M, 4), dtype=float)
        edge_arr[:, 0], edge_arr[:, 1], edge_arr[:, 2], edge_arr[:, 3] = iu, ju, weights, scores

        order = np.lexsort((-edge_arr[:, 2], -edge_arr[:, 3]))
        edge_arr = edge_arr[order]

        mask = np.zeros((N, N), dtype=bool)
        edges_selected = 0

        if self.ensure_connected:
            weight_order = np.argsort(edge_arr[:, 2])[::-1]
            mst_selected = self._kruskal_mst_edges(N, edge_arr[weight_order][:, :3])
            for idx in weight_order[mst_selected]:
                u, v = int(edge_arr[idx, 0]), int(edge_arr[idx, 1])
                if not mask[u, v]:
                    mask[u, v] = mask[v, u] = True
                    edges_selected += 1
                if edges_selected >= effective_k:
                    break

        cursor = 0
        while edges_selected < effective_k and cursor < M:
            u, v = int(edge_arr[cursor, 0]), int(edge_arr[cursor, 1])
            if not mask[u, v]:
                mask[u, v] = mask[v, u] = True
                edges_selected += 1
            cursor += 1
        return mask


class ResistanceSpectralSparsifier(Sparsifier):
    """
    Resistance-based spectral sparsification (exact effective resistances).
    """

    def __init__(self, directed: bool = True, seed: Optional[int] = None, params: Optional[Dict[str, Any]] = None):
        super().__init__("resistance_spectral", params)
        self.directed = directed
        self.seed = seed

    def get_mask(self, matrix: np.ndarray, effective_k: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        N = matrix.shape[0]
        W = np.abs(matrix).astype(float)
        np.fill_diagonal(W, 0)
        if not self.directed:
            W = (W + W.T) / 2.0

        deg = W.sum(axis=1)
        L = np.diag(deg) - W
        Lplus = np.linalg.pinv(L, rcond=1e-12)
        Lplus_diag = np.diag(Lplus)

        if self.directed:
            edges = [(i, j) for i in range(N) for j in range(N) if i != j]
        else:
            edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        if not edges:
            return np.zeros((N, N), dtype=bool)

        i_idx, j_idx = np.array([e[0] for e in edges]), np.array([e[1] for e in edges])
        R = np.maximum(Lplus_diag[i_idx] + Lplus_diag[j_idx] - 2 * Lplus[i_idx, j_idx], 0.0)
        w_edges = W[i_idx, j_idx]
        scores = w_edges * R
        total_score = scores.sum()
        probs = scores / total_score if total_score > 0 else (w_edges > 0).astype(float) / (w_edges > 0).sum()

        sampled_indices = rng.choice(np.arange(len(edges)), size=effective_k, replace=True, p=probs)
        mask = np.zeros((N, N), dtype=bool)
        for idx in sampled_indices:
            i, j = edges[idx]
            mask[i, j] = True
            if not self.directed:
                mask[j, i] = True
        return mask


class DHondtSparsifier(Sparsifier):
    """
    D'Hondt-style proportional allocation sparsification.
    """

    def __init__(self, mode: str = 'sum_correlation', directed: bool = True, params: Optional[Dict[str, Any]] = None):
        super().__init__("dhondt", params)
        self.mode = mode
        self.directed = directed

    def get_mask(self, matrix: np.ndarray, effective_k: int) -> np.ndarray:
        N = matrix.shape[0]
        abs_corr = np.abs(matrix).copy()
        np.fill_diagonal(abs_corr, 0)
        mask = np.zeros_like(abs_corr, dtype=bool)
        degrees = np.zeros(N, dtype=int)
        edge_count = 0

        if self.mode == 'sum_correlation':
            vote_weights = abs_corr.sum(axis=1)
            sorted_neighbors = [np.argsort(abs_corr[i, :])[::-1] for i in range(N)]
            while edge_count < effective_k:
                quotients = vote_weights / (degrees + 1)
                i = np.argmax(quotients)
                for j in sorted_neighbors[i]:
                    if mask[i, j]:
                        continue
                    mask[i, j] = True
                    if not self.directed:
                        mask[j, i] = True
                    degrees[i] += 1
                    degrees[j] += 1
                    edge_count += 1
                    break
                if edge_count >= effective_k or not np.any(~mask):
                    break
        else:
            sorted_neighbors = [list(np.argsort(abs_corr[i, :])[::-1]) for i in range(N)]
            while edge_count < effective_k:
                top_vals = np.zeros(N)
                top_neighs = np.full(N, -1, dtype=int)
                for i in range(N):
                    while sorted_neighbors[i] and mask[i, sorted_neighbors[i][0]]:
                        sorted_neighbors[i].pop(0)
                    if sorted_neighbors[i]:
                        top_neighs[i] = sorted_neighbors[i][0]
                        top_vals[i] = abs_corr[i, top_neighs[i]]
                
                quotients = top_vals / (degrees + 1)
                if np.all(quotients == 0):
                    break
                i_star = np.argmax(quotients)
                j_star = top_neighs[i_star]
                if j_star == -1:
                    break
                mask[i_star, j_star] = True
                if not self.directed:
                    mask[j_star, i_star] = True
                degrees[i_star] += 1
                degrees[j_star] += 1
                edge_count += 1
        return mask


class MutualKNNSparsifier(Sparsifier):
    """
    Mutual k-Nearest Neighbors: Edge (i, j) exists only if 
    j is in top-k of i AND i is in top-k of j.
    Guarantees an undirected graph and reduces hub effects.
    """

    def __init__(self, k_per_node: Optional[int] = None, params: Optional[Dict[str, Any]] = None):
        super().__init__("mutual_knn", params)
        if k_per_node is not None:
            self.params['k_per_node'] = k_per_node

    def get_mask(self, matrix: np.ndarray, effective_k: int) -> np.ndarray:
        N = matrix.shape[0]
        adj = np.abs(matrix)
        np.fill_diagonal(adj, 0)
        
        k = self.params.get('k_per_node', effective_k // N if N > 0 else 0)
        
        # Get standard top-k masks for each row
        top_k_mask = np.zeros_like(adj, dtype=bool)
        for i in range(N):
            idx = np.argsort(adj[i, :])[::-1][:k]
            top_k_mask[i, idx] = True
            
        # Mutual k-NN: intersection of mask and its transpose
        mutual_mask = top_k_mask & top_k_mask.T
        return mutual_mask


class MSTSparsifier(Sparsifier):
    """
    Minimum Spanning Tree (MST) based sparsification.
    Ensures connectivity with minimum edges.
    Converts similarity to distance: dist = 1 - similarity.
    """

    def __init__(self, use_k_nn_hybrid: bool = False, k_hybrid: int = 5, params: Optional[Dict[str, Any]] = None):
        super().__init__("mst", params)
        self.use_k_nn_hybrid = use_k_nn_hybrid
        self.k_hybrid = k_hybrid

    def get_mask(self, matrix: np.ndarray, effective_k: int) -> np.ndarray:
        import networkx as nx
        N = matrix.shape[0]
        # Convert similarity to distance for MST
        similarity = np.abs(matrix)
        dist = 1.0 - (similarity / (similarity.max() + 1e-9))
        np.fill_diagonal(dist, 0)
        
        G = nx.from_numpy_array(dist)
        mst_edges = nx.minimum_spanning_edges(G, data=False)
        
        mask = np.zeros((N, N), dtype=bool)
        for u, v in mst_edges:
            mask[u, v] = mask[v, u] = True
            
        if self.use_k_nn_hybrid:
            # Add top-k edges to the MST to increase density/robustness
            for i in range(N):
                idx = np.argsort(similarity[i, :])[::-1][:self.k_hybrid]
                mask[i, idx] = mask[idx, i] = True
                
        return mask


class PMFGSparsifier(Sparsifier):
    """
    Planar Maximally Filtered Graph (PMFG).
    Heuristic implementation: greedily add edges if they don't break planarity.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("pmfg", params)

    def get_mask(self, matrix: np.ndarray, effective_k: int) -> np.ndarray:
        import networkx as nx
        from networkx.algorithms.planarity import check_planarity
        
        N = matrix.shape[0]
        similarity = np.abs(matrix)
        np.fill_diagonal(similarity, 0)
        
        # Sort edges by descending similarity
        iu, ju = np.triu_indices(N, k=1)
        weights = similarity[iu, ju]
        edge_order = np.argsort(weights)[::-1]
        
        G = nx.Graph()
        G.add_nodes_from(range(N))
        
        mask = np.zeros((N, N), dtype=bool)
        edges_added = 0
        max_edges = 3 * (N - 2) # Limit for planar graphs
        
        for idx in edge_order:
            u, v = iu[idx], ju[idx]
            G.add_edge(u, v)
            is_planar, _ = check_planarity(G)
            if is_planar:
                mask[u, v] = mask[v, u] = True
                edges_added += 1
                if edges_added >= max_edges:
                    break
            else:
                G.remove_edge(u, v)
                
        return mask
