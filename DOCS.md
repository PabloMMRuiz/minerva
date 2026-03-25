# Matrix Experimentation Library Documentation

This document provides a detailed overview of the modules, methods, and their usage in the Matrix Experimentation Library.

---

## 1. Matrix Generation (`src.matrix_generation`)

Generates similarity or adjacency matrices from multivariate time series data.

### Common Parameters
- **Input**: `data` (np.ndarray) - Shape `[L, N, C]` where `L` is length, `N` is nodes, `C` is features.
- **Output**: `np.ndarray` - Shape `[N, N]`.

### Methods
- **`PearsonCorrelationGenerator`**: Standard linear correlation.
- **`PartialCorrelationGenerator`**: Measures degree of association while removing effects of other variables.
- **`DTWGenerator`**: Dynamic Time Warping for phase-invariant similarity.
- **`GrangerCausalityGenerator(maxlag=3)`**: Statistical test for directed horizontal causal relationships.
- **`MutualInformationGenerator(n_neighbors=3)`**: Captures non-linear dependencies.
- **`EmbeddingCosineGenerator(hidden_dim, embedding_dim)`**: Learned latent similarity using a simple neural encoder.

---

## 2. Sparsification & Normalization (`src.sparcification`)

Refines raw similarity matrices into sparse, normalized adjacency matrices for GNNs.

### Sparsifiers
- **`TopKRowSparsifier(k_per_node)`**: Keeps the top K entries per row.
- **`MutualKNNSparsifier(k_per_node)`**: Keeps an edge only if both nodes are in each other's top K.
- **`MSTSparsifier()`**: Computes the Minimum Spanning Tree (guarantees connectivity).
- **`PMFGSparsifier()`**: Planar Maximally Filtered Graph (extension of MST).

### Normalizers
- **`RowL1Normalizer()`**: Rows sum to 1.
- **`RowSoftmaxNormalizer(temperature=1.0)`**: Applies softmax to rows.

---

## 3. Hybrid Matrix Fusion (`src.hybrid`)

Fuses multiple matrices into one.

### Methods
- **`WeightedAverageFuser(weights)`**: Linear combination of matrices.
- **`ElementWiseMaxFuser()` / `ElementWiseMinFuser()`**: Union or intersection of edges.
- **`RankAverageFuser()`**: Fuses based on the rank of edge weights.
- **`SNFDiffusionFuser(k_neighbors, n_iterations)`**: Similarity Network Fusion for robust integration.

---

## 4. Node Batching (`src.batching`)

Divides nodes into batches for mini-batch GNN training.

### Common Interface
- **Input**: `adj_matrix` (np.ndarray) - Shape `[N, N]`.
- **Output**: `List[List[int]]` - List of node index batches.

### Methods
| Method | Class | Description |
| :--- | :--- | :--- |
| **K-hop Neighborhood** | `KHopBatcher(k, max_batch_size)` | Nodes and their $k$-hop neighbors. |
| **Greedy Clustering** | `GreedyClusterBatcher(max_batch_size)` | Seed-based expansion using strongest edges. |
| **Louvain** | `LouvainBatcher(resolution)` | Community detection based on modularity. |
| **Standard Clustering** | `StandardClusterBatcher(n_clusters)` | Agglomerative clustering on similarity matrices. |
| **Spectral Clustering** | `SpectralBatcher(n_clusters)` | Clustering based on Laplacian eigenvectors. |
| **Balanced Partition** | `BalancedPartitionBatcher(n_clusters)` | Equal-sized groups using Fiedler vector sorting. |
| **Overlapping** | `OverlappingBatcher(seed_fraction, radius)` | Ego-network batching with allowed overlap. |
| **Degree-Aware** | `DegreeAwareBatcher(base_size, scale_factor)`| Shrinks batch size for high-degree hubs. |

---

## 5. Utilities (`src.utils`)

### Visualization
- `plot_adjacency_matrix_heatmap(adj)`
- `plot_spectral_analysis(adj)`: Eigenvalue distribution.
- `plot_matrix_spy(adj)`: Sparsity pattern.
- `plot_node_neighborhood(adj, node_idx)`: Local graph plot.

### Metrics
- `calculate_graph_metrics(adj)`: Returns density, sparsity, algebraic connectivity, spectral gap, and number of components.
