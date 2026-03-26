import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

def plot_node_batches_on_graph(
    adj_matrix: np.ndarray,
    batches: List[List[int]],
    title: str = "Node Batches Visualization",
    layout: str = 'spring',
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    node_size: int = 300
):
    """
    Plots the graph with nodes colored based on their batch assignment.
    
    Args:
        adj_matrix: N x N adjacency matrix.
        batches: List of lists containing node indices for each batch.
        title: Title of the plot.
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai').
        save_path: Path to save the image.
        show: Whether to display the plot.
        figsize: Size of the figure.
        node_size: Size of the nodes.
    """
    G = nx.Graph(adj_matrix)
    n_nodes = adj_matrix.shape[0]
    
    # Create a mapping from node to batch index
    node_to_batch = {}
    for i, batch in enumerate(batches):
        for node in batch:
            node_to_batch[node] = i
            
    # Assign colors to batches
    cmap = plt.cm.get_cmap('tab20', len(batches))
    colors = [cmap(node_to_batch.get(i, -1)) if i in node_to_batch else (0.5, 0.5, 0.5, 1.0) 
              for i in range(n_nodes)]
    
    # Compute layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
        
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.5, edge_color='gray', alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Batch visualization saved to {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_top_k_connections(
    adj_matrix: np.ndarray,
    node_idx: int,
    k: int = 5,
    title: str = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize a single node and its top K strongest connections.
    """
    n_nodes = adj_matrix.shape[0]
    row = adj_matrix[node_idx]
    
    # Get top K indices
    top_indices = np.argsort(row)[-k:]
    # Filter out zero weights if any
    top_indices = [i for i in top_indices if row[i] > 1e-6]
    
    # Create ego graph manually
    nodes = [node_idx] + list(top_indices)
    sub_adj = adj_matrix[np.ix_(nodes, nodes)]
    G = nx.Graph(sub_adj)
    
    mapping = {i: n for i, n in enumerate(nodes)}
    G = nx.relabel_nodes(G, mapping)
    
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    
    # Draw edges with width proportional to weight
    weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    if title is None:
        title = f"Top {len(top_indices)} connections for Node {node_idx}"
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
