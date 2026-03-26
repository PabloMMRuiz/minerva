import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.loaders import load_pkl
from src.utils.visualization import compare_adjacency_matrices_heatmap
from src.utils.graph_viz import plot_node_batches_on_graph

def visualize_dataset_resources(dataset_name):
    print(f"\n--- Visualizing resources for {dataset_name} ---")
    
    resource_dir = os.path.join(project_root, 'scripts', 'traffic_experiment')
    matrix_dir = os.path.join(resource_dir, 'matrices')
    batch_dir = os.path.join(resource_dir, 'batches')
    plot_dir = os.path.join(resource_dir, 'plots', dataset_name.lower())
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Visualize Matrices
    matrix_names = ['original', 'pearson', 'mi', 'fused']
    matrices = []
    titles = []
    
    for name in matrix_names:
        path = os.path.join(matrix_dir, f"{dataset_name.lower()}_{name}.pkl")
        if os.path.exists(path):
            mat = load_pkl(path)
            matrices.append(mat)
            titles.append(f"{name.capitalize()} Matrix")
    
    if matrices:
        compare_adjacency_matrices_heatmap(
            matrices, 
            titles, 
            overall_title=f"{dataset_name} Adjacency Matrices Comparison",
            save_path=os.path.join(plot_dir, "matrices_comparison.png"),
            show=False
        )
        print(f"Matrix comparison plot saved for {dataset_name}")

    # 2. Visualize Batches
    batch_configs = ['louvain', 'khop', 'spectral']
    
    for m_name in matrix_names:
        # Load the base matrix for visualization (if it exists)
        m_path = os.path.join(matrix_dir, f"{dataset_name.lower()}_{m_name}.pkl")
        if not os.path.exists(m_path):
            continue
        
        base_adj = load_pkl(m_path)
        
        for b_name in batch_configs:
            # New File Naming: {dataset}_{matrix}_{batcher}.json
            path = os.path.join(batch_dir, f"{dataset_name.lower()}_{m_name}_{b_name}.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    batches = json.load(f)
                
                plot_node_batches_on_graph(
                    base_adj,
                    batches,
                    title=f"{dataset_name} - {m_name.capitalize()} Matrix - {b_name.capitalize()} Batches",
                    save_path=os.path.join(plot_dir, f"batches_{m_name}_{b_name}.png"),
                    show=False,
                    node_size=100
                )
                print(f"Batch visualization ({m_name}/{b_name}) saved for {dataset_name}")


if __name__ == "__main__":
    for ds in ["PEMS-BAY", "PEMS04"]:
        try:
            visualize_dataset_resources(ds)
        except Exception as e:
            print(f"Error visualizing {ds}: {e}")
            
    print("\nVisualization complete! Plots are in scripts/traffic_experiment/plots/")
