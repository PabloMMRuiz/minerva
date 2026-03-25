import numpy as np
import networkx as nx
from typing import List
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import warnings

from .base import NodeBatcher

class KHopBatcher(NodeBatcher):
    """
    1. K-hop Neighborhood Batching
    Builds batches from nodes and their k-hop neighbors.
    We iterate over unassigned nodes and extract their k-hop neighborhood 
    to form a batch, until all nodes are assigned.
    """
    def __init__(self, k: int = 1, max_batch_size: int = None):
        self.k = k
        self.max_batch_size = max_batch_size

    def batch(self, adj_matrix: np.ndarray) -> List[List[int]]:
        self._validate_input(adj_matrix)
        G = nx.from_numpy_array(adj_matrix)
        nodes = list(G.nodes())
        unassigned = set(nodes)
        batches = []

        while unassigned:
            # Pick a seed
            seed = next(iter(unassigned))
            # Find k-hop neighbors
            lengths = nx.single_source_shortest_path_length(G, seed, cutoff=self.k)
            k_hop_group = set(lengths.keys())
            
            # Intersection with unassigned to avoid overlaps normally
            # unless we want overlapping batches (that's method 8)
            available = list(k_hop_group.intersection(unassigned))
            
            if self.max_batch_size and len(available) > self.max_batch_size:
                available = available[:self.max_batch_size]

            if not available:
                # Fallback to just the seed if somehow disconnected
                available = [seed]

            batches.append(available)
            unassigned.difference_update(available)

        return batches


class GreedyClusterBatcher(NodeBatcher):
    """
    2. Greedy Graph Clustering
    Iteratively grow batches by picking a seed and greedily adding its most 
    strongly connected unassigned neighbors until a size limit is reached.
    """
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size

    def batch(self, adj_matrix: np.ndarray) -> List[List[int]]:
        self._validate_input(adj_matrix)
        n_nodes = adj_matrix.shape[0]
        unassigned = set(range(n_nodes))
        batches = []

        while unassigned:
            seed = unassigned.pop()
            current_batch = [seed]
            
            while len(current_batch) < self.max_batch_size and unassigned:
                # Find strongest connection from current batch to unassigned
                best_node = None
                best_weight = -np.inf
                
                for node_in_batch in current_batch:
                    for un_node in unassigned:
                        w = adj_matrix[node_in_batch, un_node]
                        if w > best_weight:
                            best_weight = w
                            best_node = un_node
                            
                if best_node is not None and best_weight > 0:
                    current_batch.append(best_node)
                    unassigned.remove(best_node)
                else:
                    # No more connections
                    break
            
            batches.append(current_batch)
            
        # Distribute any remaining totally disconnected nodes
        for u in list(unassigned):
            batches.append([u])

        return batches


class LouvainBatcher(NodeBatcher):
    """
    3. Community Detection (Louvain)
    Uses the Louvain method to partition the graph into dense communities.
    """
    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution

    def batch(self, adj_matrix: np.ndarray) -> List[List[int]]:
        self._validate_input(adj_matrix)
        # B/c Louvain deals with symmetric graphs, we symmetrize softly
        sym_adj = (adj_matrix + adj_matrix.T) / 2
        G = nx.from_numpy_array(sym_adj)
        
        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(G, resolution=self.resolution, weight='weight')
            return [list(c) for c in communities]
        except AttributeError:
            warnings.warn("nx.community.louvain_communities not found. Falling back to greedy_modularity.")
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(G, weight='weight')
            return [list(c) for c in communities]


class StandardClusterBatcher(NodeBatcher):
    """
    4 & 5. Correlation/Hybrid Clustering
    Uses Agglomerative Clustering. This batcher expects the user to pass in the 
    correlation or hybrid matrix representing similarities. Features 4 & 5 are 
    enabled via this modular structure. 
    """
    def __init__(self, n_clusters: int = None, distance_threshold: float = None):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        if n_clusters is None and distance_threshold is None:
            self.n_clusters = 5 # default

    def batch(self, adj_matrix: np.ndarray) -> List[List[int]]:
        self._validate_input(adj_matrix)
        
        # Convert similarity to a pseudo-distance. 
        # We assume adj_matrix has values roughly in [-1, 1] or [0, 1].
        # Dist = max(adj) - adj_matrix 
        max_val = np.max(adj_matrix)
        dist_matrix = max_val - adj_matrix
        # Ensure non-negative diagonal
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = np.clip(dist_matrix, 0, None)

        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters, 
            distance_threshold=self.distance_threshold,
            metric='precomputed', 
            linkage='complete'
        )
        labels = clustering.fit_predict(dist_matrix)
        
        batches = {}
        for node_idx, label in enumerate(labels):
            if label not in batches:
                batches[label] = []
            batches[label].append(node_idx)
            
        return list(batches.values())


class SpectralBatcher(NodeBatcher):
    """
    6. Spectral Clustering
    Uses eigenvectors of the graph Laplacian to partition nodes into balanced clusters.
    """
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters

    def batch(self, adj_matrix: np.ndarray) -> List[List[int]]:
        self._validate_input(adj_matrix)
        # Symmetrize and ensure non-negative
        sym_adj = (adj_matrix + adj_matrix.T) / 2
        sym_adj = np.clip(sym_adj, 0, None)
        
        # Handle case where graph is too small for requested clusters
        n_clusters = min(self.n_clusters, sym_adj.shape[0])

        clustering = SpectralClustering(
            n_clusters=n_clusters, 
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )
        labels = clustering.fit_predict(sym_adj)
        
        batches = {}
        for node_idx, label in enumerate(labels):
            if label not in batches:
                batches[label] = []
            batches[label].append(node_idx)
            
        return list(batches.values())


class BalancedPartitionBatcher(NodeBatcher):
    """
    7. Balanced Graph Partitioning (METIS-style target)
    Attempts to divide the graph into n groups while forcing perfectly equal 
    sizes (or off by at most 1). We do this by sorting spectral embeddings.
    """
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters

    def batch(self, adj_matrix: np.ndarray) -> List[List[int]]:
        self._validate_input(adj_matrix)
        n_nodes = adj_matrix.shape[0]
        n_clusters = min(self.n_clusters, n_nodes)
        
        # Use spectral embedding to get 1D representation for sorting
        sym_adj = np.clip((adj_matrix + adj_matrix.T) / 2, 0, None)
        G = nx.from_numpy_array(sym_adj)
        
        # Fiedler vector (2nd smallest eigenvector of Laplacian)
        L = nx.laplacian_matrix(G).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Sort by eigenvalue, pick 2nd smallest (index 1 b/c 0 is the zero eigenvalue)
        if len(eigenvalues) > 1:
            fiedler = eigenvectors[:, 1]
        else:
            fiedler = np.zeros(n_nodes)
            
        # Sort nodes by their fiedler component
        sorted_nodes = np.argsort(fiedler)
        
        # Chunk into equal sizes
        chunk_size = int(np.ceil(n_nodes / n_clusters))
        batches = [sorted_nodes[i:i + chunk_size].tolist() for i in range(0, n_nodes, chunk_size)]
        return batches


class OverlappingBatcher(NodeBatcher):
    """
    8. Overlapping Subgraph Batching
    Allows nodes to appear in multiple batches by generating ego-graphs.
    Takes a set of seed nodes (e.g., every 3rd node) and fully expands their 
    neighborhood without removing nodes from a global unassigned pool.
    """
    def __init__(self, seed_fraction: float = 0.5, radius: int = 1):
        self.seed_fraction = seed_fraction
        self.radius = radius

    def batch(self, adj_matrix: np.ndarray) -> List[List[int]]:
        self._validate_input(adj_matrix)
        G = nx.from_numpy_array(adj_matrix)
        n_nodes = adj_matrix.shape[0]
        
        # Pick seeds (highest degree first makes good centers)
        degrees = [d for n, d in G.degree()]
        sorted_nodes = np.argsort(degrees)[::-1]
        n_seeds = max(1, int(n_nodes * self.seed_fraction))
        seeds = sorted_nodes[:n_seeds]
        
        batches = []
        for seed in seeds:
            ego = nx.ego_graph(G, seed, radius=self.radius)
            batches.append(list(ego.nodes()))
            
        # Ensure all nodes appear at least once
        covered = set(n for b in batches for n in b)
        uncovered = set(range(n_nodes)) - covered
        if uncovered:
            batches.append(list(uncovered))
            
        return batches


class DegreeAwareBatcher(NodeBatcher):
    """
    9. Degree-Aware Batching
    Adjusts batch composition based on node degree. High-degree hubs are 
    grouped into smaller batches so their message passing logic doesn't explode
    memory, while low-degree nodes go into larger batches.
    """
    def __init__(self, base_size: int = 32, scale_factor: float = 0.5):
        self.base_size = base_size
        self.scale_factor = scale_factor

    def batch(self, adj_matrix: np.ndarray) -> List[List[int]]:
        self._validate_input(adj_matrix)
        G = nx.from_numpy_array(adj_matrix)
        n_nodes = adj_matrix.shape[0]
        
        # Sort nodes by degree (descending)
        nodes_by_degree = sorted(G.degree(weight='weight'), key=lambda x: x[1], reverse=True)
        unassigned = [n for n, d in nodes_by_degree]
        
        batches = []
        while unassigned:
            seed = unassigned.pop(0)
            seed_degree = dict(G.degree(weight='weight'))[seed]
            
            # High degree -> smaller batch. Low degree -> larger batch.
            # Max batch size shrinks as degree grows
            max_size = int(max(2, self.base_size - (seed_degree * self.scale_factor)))
            
            current_batch = [seed]
            # Fill batch with highest connection neighbors
            while len(current_batch) < max_size and unassigned:
                best_node = None
                best_w = -np.inf
                for un_node in unassigned:
                    w = adj_matrix[seed, un_node] # Simplify to direct connected to seed
                    if w > best_w:
                        best_w = w
                        best_node = un_node
                
                if best_node is not None and best_w > 0:
                    current_batch.append(best_node)
                    unassigned.remove(best_node)
                else:
                    break
                    
            batches.append(current_batch)
            
        return batches
