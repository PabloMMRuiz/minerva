"""
Verification script for the complete matrix-experimentation library,
including generators, sparsifiers, hybrid fusion, and advanced utilities.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from matrix_generation import (
    PearsonCorrelationGenerator,
    PartialCorrelationGenerator,
    DTWGenerator,
    GrangerCausalityGenerator,
    MutualInformationGenerator,
    EmbeddingCosineGenerator
)
from sparcification import (
    MatrixConstructionPipeline,
    make_adjacency_matrix,
    TopKRowSparsifier,
    RowL1Normalizer,
    MutualKNNSparsifier,
    MSTSparsifier,
    PMFGSparsifier
)
from hybrid import (
    WeightedAverageFuser,
    ElementWiseMaxFuser,
    ElementWiseMinFuser,
    RankAverageFuser,
    SNFDiffusionFuser,
    AttentionFuser
)
from sparcification.builder import make_hybrid_adjacency, FUSER_REGISTRY
from utils.graph_metrics import calculate_graph_metrics
from utils.visualization import (
    plot_adjacency_matrix_heatmap,
    plot_edge_weight_distribution,
    plot_spectral_analysis,
    plot_adjacency_diff,
    plot_matrix_spy,
    plot_node_neighborhood
)
from batching.algorithms import (
    KHopBatcher,
    GreedyClusterBatcher,
    LouvainBatcher,
    StandardClusterBatcher,
    SpectralBatcher,
    BalancedPartitionBatcher,
    OverlappingBatcher,
    DegreeAwareBatcher
)


def verify_generators():
    print("=" * 60)
    print("GENERATORS")
    print("=" * 60)
    L, N, C = 50, 6, 1
    data = np.random.randn(L, N, C).astype(np.float32)

    generators = [
        PearsonCorrelationGenerator(),
        PartialCorrelationGenerator(),
        DTWGenerator(),
        GrangerCausalityGenerator(maxlag=2),
        MutualInformationGenerator(n_neighbors=2),
        EmbeddingCosineGenerator(hidden_dim=8, embedding_dim=16)
    ]

    matrices = {}
    for gen in generators:
        name = gen.__class__.__name__
        print(f"  {name}...", end=" ")
        m = gen.generate(data)
        assert m.shape == (N, N)
        matrices[name] = m
        print("OK")

    return data, matrices


def verify_sparsifiers(base_sim):
    print("\n" + "=" * 60)
    print("SPARSIFIERS")
    print("=" * 60)
    N = base_sim.shape[0]

    sparsifiers = [
        ("TopKRow", TopKRowSparsifier(k_per_node=2)),
        ("MutualKNN", MutualKNNSparsifier(k_per_node=2)),
        ("MST", MSTSparsifier()),
        ("PMFG", PMFGSparsifier())
    ]

    for name, sparsifier in sparsifiers:
        print(f"  {name}...", end=" ")
        pipeline = MatrixConstructionPipeline(sparsifiers=[sparsifier])
        adj = pipeline.run(base_sim, k=10)
        assert adj.shape == (N, N)

        if name == "MutualKNN":
            assert np.allclose(adj, adj.T), "Mutual k-NN must be symmetric"
        if name == "MST":
            edge_count = np.count_nonzero(adj) // 2
            assert edge_count == N - 1, f"MST: expected {N-1} edges, got {edge_count}"
        print("OK")


def verify_hybrid_fusers(matrices):
    print("\n" + "=" * 60)
    print("HYBRID FUSERS")
    print("=" * 60)
    mat_list = list(matrices.values())[:3]
    N = mat_list[0].shape[0]

    print("  WeightedAverageFuser...", end=" ")
    result = WeightedAverageFuser().fuse(mat_list)
    assert result.shape == (N, N)
    print("OK")

    print("  SNFDiffusionFuser...", end=" ")
    result = SNFDiffusionFuser(k_neighbors=2, n_iterations=5).fuse(mat_list)
    assert result.shape == (N, N)
    print("OK")


def verify_utilities(matrices):
    print("\n" + "=" * 60)
    print("UTILITIES (Metrics & Visualization)")
    print("=" * 60)
    adj = list(matrices.values())[0]
    
    print("  calculate_graph_metrics...", end=" ")
    metrics = calculate_graph_metrics(adj, directed=True)
    assert "sparsity" in metrics
    assert "algebraic_connectivity" in metrics
    assert "spectral_gap" in metrics
    assert "num_components" in metrics
    print("OK")

    print("  Visualization functions (smoke test)...", end=" ")
    plt.switch_backend('Agg') # Don't show plots
    
    plot_edge_weight_distribution(adj, show=False)
    plot_spectral_analysis(adj, show=False)
    plot_adjacency_diff(adj, adj * 0.9, show=False)
    plot_matrix_spy(adj, show=False)
    plot_node_neighborhood(adj, node_idx=0, show=False)
    print("OK")

def verify_batchers(adj):
    print("\n" + "=" * 60)
    print("BATCHERS")
    print("=" * 60)
    
    batchers = [
        ("KHopBatcher", KHopBatcher(k=2, max_batch_size=3)),
        ("GreedyClusterBatcher", GreedyClusterBatcher(max_batch_size=3)),
        ("LouvainBatcher", LouvainBatcher()),
        ("StandardClusterBatcher", StandardClusterBatcher(n_clusters=2)),
        ("SpectralBatcher", SpectralBatcher(n_clusters=2)),
        ("BalancedPartitionBatcher", BalancedPartitionBatcher(n_clusters=2)),
        ("OverlappingBatcher", OverlappingBatcher(seed_fraction=0.5, radius=1)),
        ("DegreeAwareBatcher", DegreeAwareBatcher(base_size=5, scale_factor=0.5))
    ]
    
    for name, batcher in batchers:
        print(f"  {name}...", end=" ")
        batches = batcher.batch(adj)
        
        # Check that it returns a list of lists
        assert isinstance(batches, list), f"{name} returned {type(batches)}"
        if name != "OverlappingBatcher":
            # For partitioning, check that all nodes are covered
            nodes_in_batches = sum([len(b) for b in batches])
            assert nodes_in_batches == adj.shape[0], f"{name}: expected {adj.shape[0]} nodes, got {nodes_in_batches}"
        
        print(f"OK ({len(batches)} batches)")


def main():
    print("Starting comprehensive verification...\n")
    data, matrices = verify_generators()
    verify_sparsifiers(list(matrices.values())[0])
    verify_hybrid_fusers(matrices)
    verify_utilities(matrices)
    verify_batchers(list(matrices.values())[0])
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
