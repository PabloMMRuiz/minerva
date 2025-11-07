"""
Contains all graphication functions for either graphs or series
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, Tuple, List
import networkx as nx
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------------------------------------
# Graph graphs...


def plot_adjacency_matrix_heatmap(  # I NEED to fit arguments in one line before it triggers my ocd
    adj_matrix: np.ndarray,
    title: str = "Adjacency Matrix Heatmap",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (16, 16),
    cmap: str = "viridis",
    return_fig: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plots the adjacency matrix as a heatmap to visualize graph structure.

    Args:
        adj_matrix: The N x N adjacency matrix.
        title: The title for the plot.
        save_path: Path to save the figure (e.g., 'results/heatmap.png'). 
                   If None, figure is not saved.
        show: If True, display the plot. Set to False to prevent blocking.
        figsize: Figure size as (width, height).
        cmap: Colormap for the heatmap.
        return_fig: If True, return (fig, ax) tuple for further customization.

    Returns:
        Tuple of (figure, axes) if return_fig=True, otherwise None.
    """
    if not isinstance(adj_matrix, np.ndarray) or adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        print("Error: Input must be a square NumPy adjacency matrix.")
        return None

    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(adj_matrix, cmap=cmap, cbar=True, ax=ax)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Node Index', fontsize=12)
    ax.set_ylabel('Node Index', fontsize=12)

    # Save if path provided
    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()

    # Return figure
    if return_fig:
        return fig, ax
    else:
        plt.close(fig)  # Clean up if not returning
        return None


def draw_graph_network(
    adj_matrix: np.ndarray,
    title: str = "Graph",
    directed: bool = True,
    layout: str = 'spring',
    node_size: int = 300,
    show_weights: bool = False,
    show_labels: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    return_fig: bool = False,
    **layout_kwargs
) -> Optional[Tuple[plt.Figure, plt.Axes, nx.Graph]]:
    """
    Draw a network graph from an adjacency matrix with flexible visualization options.

    Args:
        adj_matrix: The N x N adjacency matrix.
        title: Plot title.
        directed: If True, create directed graph.
        layout: Layout algorithm - 'spring', 'circular', 'kamada_kawai', 'spectral', 'random'
        node_size: Size of nodes.
        show_weights: If True, show edge weights as labels.
        show_labels: If True, show node labels.
        save_path: Path to save figure.
        show: If True, display plot.
        figsize: Figure size.
        return_fig: If True, return (fig, ax, graph).
        **layout_kwargs: Additional kwargs for layout algorithm (k, iterations for spring...).

    Returns:
        Tuple of (figure, axes, graph) if return_fig=True, otherwise None.
    """
    if not isinstance(adj_matrix, np.ndarray) or adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        print("Error: Input must be a square NumPy adjacency matrix.")
        return None

    # Create graph
    G = nx.DiGraph(adj_matrix) if directed else nx.Graph(adj_matrix)

    # Filter edges by threshold
    edges_to_draw = [(u, v) for u, v, _ in G.edges(data=True)]

    # Choose layout
    layout_functions = {
        'spring': nx.spring_layout,
        'circular': nx.circular_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'spectral': nx.spectral_layout,
        'random': nx.random_layout
    }

    layout_func = layout_functions.get(layout, nx.spring_layout)
    pos = layout_func(G, **layout_kwargs) if layout_kwargs else layout_func(G)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size,
                           alpha=0.9, ax=ax)

    # Draw edges
    if directed:
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw,
                               arrowsize=20, width=1.5, edge_color='gray',
                               alpha=0.6, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw,
                               width=1.5, edge_color='gray', alpha=0.6, ax=ax)

    # Draw labels
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

    # Draw edge weights
    if show_weights:
        edge_labels = {(u, v): f"{d['weight']:.2f}"
                       for u, v, d in G.edges(data=True) if (u, v) in edges_to_draw}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)

    ax.set_title(title, size=18)
    ax.axis('off')

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(
            save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    if return_fig:
        return fig, ax, G
    else:
        plt.close(fig)
        return None


def draw_graph_with_hubs(
    adj_matrix: np.ndarray,
    hub_threshold: Optional[float] = None,
    top_k_hubs: int = 10,
    title: str = "Graph with Hubs",
    directed: bool = True,
    layout: str = 'spring',
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    return_fig: bool = False
) -> Optional[Tuple]:
    """ 
    Visualize graph with hub nodes highlighted. Hubs will be either the top k degree nodes, or any node with more than k degree. threshold takes precedence 

    Args:
        adj_matrix: The N x N adjacency matrix. 
        hub_threshold: Nodes with degree > threshold are hubs (percentage of total nodes if < 1). 
        top_k_hubs: Number of top hubs to highlight, if no threshold has been provided 
        title: Plot title. 
        directed: If True, use out-degree 
        layout: Layout algorithm. 
        save_path: Path to save figure. 
        show: If True, display plot. 
        figsize: Figure size. 
        return_fig: If True, return (fig, ax, hub_indices). 

    Returns:
        Tuple of (figure, axes, hub_indices) if return_fig=True, otherwise None. 
    """
    # Calculate outbound degrees
    degrees = np.sum(adj_matrix, axis=1)

    # Identify hubs
    if hub_threshold is not None:
        if hub_threshold < 1:
            threshold_value = np.percentile(degrees, hub_threshold * 100)
        else:
            threshold_value = hub_threshold
        hub_indices = np.where(degrees > threshold_value)[0]
    else:
        hub_indices = np.argsort(degrees)[-top_k_hubs:]

    # Colors and sizes
    node_colors = ['red' if i in hub_indices else 'skyblue'
                   for i in range(len(adj_matrix))]
    node_sizes = [800 if i in hub_indices else 300
                  for i in range(len(adj_matrix))]

    # Build graph
    G = nx.DiGraph(adj_matrix) if directed else nx.Graph(adj_matrix)

    # Compute layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray',
                           alpha=0.6, arrows=directed, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

    ax.set_title(f"{title}\n(Hubs in red: {len(hub_indices)} nodes)", size=16)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig, ax, hub_indices.tolist()


def draw_graph_communities(
    adj_matrix: np.ndarray,
    title: str = "Graph with Communities",
    layout: str = 'spring',
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    return_fig: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes, List[set]]]:
    """
    Visualize graph with community (clique, densely connected regions) detection.

    Args:
        adj_matrix: The N x N adjacency matrix.
        title: Plot title.
        directed: If True, use directed graph (converts to undirected for community detection).
        layout: Layout algorithm.
        save_path: Path to save figure.
        show: If True, display plot.
        figsize: Figure size.
        return_fig: If True, return (fig, ax, communities).

    Returns:
        Tuple of (figure, axes, communities) if return_fig=True, otherwise None.
    """
    # Create graph (undirected for community detection)
    G = nx.Graph(adj_matrix)

    # Detect communities using Louvain algorithm
    try:
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(G)
    except:
        print("Warning: Community detection failed. Using connected components instead.")
        communities = list(nx.connected_components(G))

    # Assign colors to communities
    cmap = plt.cm.get_cmap('tab20', len(communities))
    node_colors = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_colors[node] = cmap(idx)

    color_list = [node_colors.get(i, (0.5, 0.5, 0.5, 1.0))
                  for i in range(len(adj_matrix))]

    # Draw graph
    fig, ax = plt.subplots(figsize=figsize)

    layout_functions = {
        'spring': nx.spring_layout,
        'circular': nx.circular_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'spectral': nx.spectral_layout
    }

    layout_func = layout_functions.get(layout, nx.spring_layout)
    pos = layout_func(G)

    nx.draw_networkx_nodes(G, pos, node_size=400, node_color=color_list,
                           alpha=0.9, ax=ax)
    nx.draw_networkx_edges(
        G, pos, width=1.5, edge_color='gray', alpha=0.4, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

    ax.set_title(
        f"{title}\n({len(communities)} communities detected)", size=16)
    ax.axis('off')

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(
            save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    if return_fig:
        return fig, ax, communities
    else:
        plt.close(fig)
        return None


def draw_graph_components(
    adj_matrix: np.ndarray,
    title: str = "Graph Connected Components",
    directed: bool = True,
    layout: str = 'spring',
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    return_fig: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes, List[set]]]:
    """
    Visualize graph with (strongly) connected components highlighted.

    Args:
        adj_matrix: The N x N adjacency matrix.
        title: Plot title.
        directed: If True, find strongly connected components.
        layout: Layout algorithm.
        save_path: Path to save figure.
        show: If True, display plot.
        figsize: Figure size.
        return_fig: If True, return (fig, ax, components).

    Returns:
        Tuple of (figure, axes, components) if return_fig=True, otherwise None.
    """
    # Create graph
    G = nx.DiGraph(adj_matrix) if directed else nx.Graph(adj_matrix)

    # Find components
    if directed:
        components = list(nx.strongly_connected_components(G))
        component_type = "strongly connected"
    else:
        components = list(nx.connected_components(G))
        component_type = "connected"

    # Assign colors
    cmap = plt.cm.get_cmap('Set3', len(components))
    node_colors = {}
    for idx, comp in enumerate(components):
        for node in comp:
            node_colors[node] = cmap(idx)

    color_list = [node_colors.get(i, (0.5, 0.5, 0.5, 1.0))
                  for i in range(len(adj_matrix))]

    # Draw
    fig, ax = plt.subplots(figsize=figsize)

    layout_functions = {
        'spring': nx.spring_layout,
        'circular': nx.circular_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'spectral': nx.spectral_layout
    }

    layout_func = layout_functions.get(layout, nx.spring_layout)
    pos = layout_func(G)

    nx.draw_networkx_nodes(G, pos, node_size=400, node_color=color_list,
                           alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray', alpha=0.6,
                           arrows=directed, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

    ax.set_title(
        f"{title}\n({len(components)} {component_type} components)", size=16)
    ax.axis('off')

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(
            save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    if return_fig:
        return fig, ax, components
    else:
        plt.close(fig)
        return None


def compare_adjacency_matrices_heatmap(
    matrices: List[np.ndarray],
    titles: List[str],
    overall_title: str = "Adjacency Matrix Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "viridis"
) -> Optional[Tuple[plt.Figure, np.ndarray]]:
    """
    Plot multiple adjacency matrices side by side for comparison.

    Args:
        matrices: List of N x N adjacency matrices.
        titles: List of titles for each matrix.
        overall_title: Overall figure title.
        save_path: Path to save figure.
        show: If True, display plot.
        figsize: Figure size (auto-calculated if None).
        cmap: Colormap.

    Returns:
        Tuple of (figure, axes) if requested, otherwise None.
    """
    n_matrices = len(matrices)

    if figsize is None:
        figsize = (6 * n_matrices, 6)

    fig, axes = plt.subplots(1, n_matrices, figsize=figsize)

    if n_matrices == 1:
        axes = [axes]

    for idx, (matrix, title) in enumerate(zip(matrices, titles)):
        sns.heatmap(matrix, cmap=cmap, cbar=True, ax=axes[idx])
        axes[idx].set_title(title, fontsize=14)
        axes[idx].set_xlabel('Node Index')
        axes[idx].set_ylabel('Node Index')

    fig.suptitle(overall_title, fontsize=18, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(
            save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return (fig, axes) if not show else None


def plot_degree_distribution(
    adj_matrix: np.ndarray,
    title: str = "Degree Distribution",
    directed: bool = True,
    log_scale: bool = False,  # if we are using this we are so cooked already
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    return_fig: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot degree distribution of the graph.

    Args:
        adj_matrix: The N x N adjacency matrix.
        title: Plot title.
        directed: If True, plot in-degree and out-degree separately.
        log_scale: If True, use log scale for y-axis.
        save_path: Path to save figure.
        show: If True, display plot.
        figsize: Figure size.
        return_fig: If True, return (fig, ax).

    Returns:
        Tuple of (figure, axes) if return_fig=True, otherwise None.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if directed:
        in_degrees = np.sum(adj_matrix, axis=0)
        out_degrees = np.sum(adj_matrix, axis=1)

        ax.hist(in_degrees, bins=30, alpha=0.5,
                label='In-degree', color='blue')
        ax.hist(out_degrees, bins=30, alpha=0.5,
                label='Out-degree', color='red')
        ax.legend()
    else:
        degrees = np.sum(adj_matrix, axis=1)
        ax.hist(degrees, bins=30, alpha=0.7, color='skyblue')

    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14)

    if log_scale:
        ax.set_yscale('log')

    ax.grid(alpha=0.3)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(
            save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    if return_fig:
        return fig, ax
    else:
        plt.close(fig)
        return None


# -------------------------------------------
# Time series graphs

def plot_node_time_series(
        data_array: np.ndarray,
        node_index: int,
        feature_index: int = 0,
        title: str = None,
        x_0: int = None,
        x_1: int = None,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (14, 6),
        return_fig: bool = False) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plots the time series data for a single node.

    Args:
        data_array (np.ndarray): The time series data array with shape [L, N, C].
        node_index (int): The index of the node (N) to plot.
        feature_index (int, optional): The index of the feature (C) to plot, defaults to 0.
        title (str, optional): A custom title for the plot.
        x_0 (int, optional): time step at which to start the plot
        x_1 (int, optional): time step at which to end the plot
        save_path: Path to save the figure (e.g., 'results/heatmap.png'). 
                   If None, figure is not saved.
        show: If True, display the plot. Set to False to prevent blocking.
        figsize: Figure size as (width, height).
        return_fig: If True, return (fig, ax) tuple for further customization.

    Returns:
        Tuple of (figure, axes) if return_fig=True, otherwise None.



    """
    L, N, C = data_array.shape

    if not (0 <= node_index < N):
        print(
            f"Error: Node index {node_index} is out of bounds. Valid range is 0 to {N-1}.")
        return

    if not (0 <= feature_index < C):
        print(
            f"Error: Feature index {feature_index} is out of bounds. Valid range is 0 to {C-1}.")
        return
    fig, ax = plt.subplots(figsize=figsize)
    # Extract the time series for the specified node and feature
    time_series = data_array[:, node_index, feature_index]
    if x_0 and x_1:
        time_series = time_series[x_0:x_1]
    elif x_0:
        time_series = time_series[x_0:]
    elif x_1:
        time_series = time_series[:x_1]
    # Create a DataFrame for easy plotting with seaborn
    df = pd.DataFrame(
        {'value': time_series, 'time_step': range(len(time_series))})

    sns.lineplot(data=df, x='time_step', y='value')

    if title is None:
        ax.set_title(
            f'Time Series for Node {node_index} (Feature {feature_index})', fontsize=16)
    else:
        ax.set_title(title, fontsize=16)

    ax.set_xlabel('Time steps', fontsize=12)

    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    if return_fig:
        return fig, ax
    else:
        plt.close(fig)
        return None


def plot_time_series_decomposition(
        data_array: np.ndarray,
        node_index: int,
        frequency: int,
        feature_index: int = 0,
        x_0: int = None,
        x_1: int = None,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (15, 12),
        return_fig: bool = False):
    """
    Performs and plots a time series decomposition for a single node.

    This function decomposes the time series into trend, seasonal, and residual components.
    It requires the frequency of the seasonal pattern in terms of time steps.

    Args:
        data_array (np.ndarray): The time series data array with shape [L, N, C].
        node_index (int): The index of the node to plot.
        frequency (int): The number of time steps per seasonal cycle (e.g., 288 for a daily cycle with 5-minute frequency).
        feature_index (int, optional): The index of the feature to plot, defaults to 0.

    """
    L, N, C = data_array.shape

    if not (0 <= node_index < N):
        print(
            f"Error: Node index {node_index} is out of bounds. Valid range is 0 to {N-1}.")
        return

    if not (0 <= feature_index < C):
        print(
            f"Error: Feature index {feature_index} is out of bounds. Valid range is 0 to {C-1}.")
        return

    if L < frequency * 2:
        print(
            f"Not enough data points ({L}) to perform decomposition with frequency {frequency}. At least two full cycles are recommended.")
        return

    # Extract the time series for the specified node and feature
    time_series = data_array[:, node_index, feature_index]
    if x_0 and x_1:
        time_series = time_series[x_0:x_1]
    elif x_0:
        time_series = time_series[x_0:]
    elif x_1:
        time_series = time_series[:x_1]
    # Perform additive decomposition
    # Additive model: Y(t) = T(t) + S(t) + R(t) is a good default for data with constant seasonal amplitude.
    try:
        result = seasonal_decompose(
            time_series, model='additive', period=frequency)
        # Plot the decomposed components
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            4, 1, figsize=figsize, sharex=True)
        ax1.plot(result.observed)
        ax1.set_title('Observed')
        ax2.plot(result.trend)
        ax2.set_title('Trend')
        ax3.plot(result.seasonal)
        ax3.set_title('Seasonal')
        ax4.plot(result.resid)
        ax4.set_title('Residual')

        plt.suptitle(
            f'Time Series Decomposition for Node {node_index}', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        if show:
            plt.show()

        if return_fig:
            return fig, (ax1, ax2, ax3, ax4)
        else:
            plt.close(fig)
            return None
    except ValueError as e:
        print(
            f"Decomposition failed. The data might not be suitable for the specified frequency. Error: {e}")
