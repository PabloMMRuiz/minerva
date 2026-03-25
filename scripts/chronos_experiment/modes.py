"""
Prediction modes for the Chronos experiment runner.

Three modes:
  - single_node:   Predict each node independently (univariate).
  - whole_matrix:   Predict all nodes as variates of one series (multivariate).
  - adj_neighbour: For each node, feed it + its adjacency neighbours as variates.
"""

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional


def _extract_median_prediction(forecast_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract the median (quantile index 1) prediction from a Chronos forecast.

    Handles both 4D [Series, Variates, Time, Quantiles] and
    3D [Variates, Time, Quantiles] output shapes.

    Returns:
        np.ndarray of shape [Variates, Time].
    """
    pred = forecast_tensor[0]  # remove batch dim
    if pred.ndim == 4:
        # [Series, Variates, Time, Quantiles]
        return pred[0, :, :, 1].cpu().numpy()
    elif pred.ndim == 3:
        # [Variates, Time, Quantiles]
        return pred[:, :, 1].cpu().numpy()
    elif pred.ndim == 2:
        # [Time, Quantiles]  (single variate)
        return pred[:, 1].cpu().numpy().reshape(1, -1)
    else:
        raise ValueError(f"Unexpected forecast shape: {pred.shape}")


def predict_single_node(
    data_mm: np.ndarray,
    pipeline,
    context_start: int,
    context_len: int,
    prediction_length: int,
    num_nodes: int,
    progress: bool = True,
) -> np.ndarray:
    """
    Predict each node independently as a univariate series.

    Args:
        data_mm: Full dataset memmap, shape [T, N, F].
        pipeline: Chronos2Pipeline instance.
        context_start: Index of the first context timestep.
        context_len: Number of context timesteps.
        prediction_length: Number of steps to predict.
        num_nodes: Total number of nodes.
        progress: Show progress bar.

    Returns:
        Predictions array of shape [num_nodes, prediction_length].
    """
    predictions = np.zeros((num_nodes, prediction_length), dtype=np.float32)
    iterator = range(num_nodes)
    if progress:
        iterator = tqdm(iterator, desc="Single-node prediction", leave=False)

    for node_idx in iterator:
        # Shape: [1, T] -> univariate
        context_slice = data_mm[context_start:context_start + context_len, node_idx, 0]
        context = torch.tensor(context_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # context shape: [1, 1, T]

        with torch.no_grad():
            forecast = pipeline.predict(context, prediction_length=prediction_length)

        pred = _extract_median_prediction(forecast)
        # pred shape: [1, prediction_length]
        predictions[node_idx, :] = pred[0, :prediction_length]

    return predictions


def predict_whole_matrix(
    data_mm: np.ndarray,
    pipeline,
    context_start: int,
    context_len: int,
    prediction_length: int,
    num_nodes: int,
    progress: bool = True,
) -> np.ndarray:
    """
    Predict all nodes simultaneously as variates of a single multivariate series.

    Args:
        data_mm: Full dataset memmap, shape [T, N, F].
        pipeline: Chronos2Pipeline instance.
        context_start: Index of the first context timestep.
        context_len: Number of context timesteps.
        prediction_length: Number of steps to predict.
        num_nodes: Total number of nodes.
        progress: Show progress bar.

    Returns:
        Predictions array of shape [num_nodes, prediction_length].
    """
    # Shape: [T, N] -> transpose to [N, T]
    context_slice = data_mm[context_start:context_start + context_len, :, 0].T
    # Chronos-2: [Series, Variates, Time] = [1, N, T]
    context = torch.tensor(context_slice, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        forecast = pipeline.predict(context, prediction_length=prediction_length)

    predictions = _extract_median_prediction(forecast)
    return predictions[:, :prediction_length]


def predict_adj_neighbour(
    data_mm: np.ndarray,
    pipeline,
    context_start: int,
    context_len: int,
    prediction_length: int,
    num_nodes: int,
    adj_mx: np.ndarray,
    progress: bool = True,
) -> np.ndarray:
    """
    Predict each node using its adjacency neighbours as additional variates.

    For each target node, we assemble a multivariate context consisting of:
      - The target node's own series (first variate)
      - Series of all its neighbours (nonzero adjacency entries)
    We then predict and extract only the first variate (the target).

    Args:
        data_mm: Full dataset memmap, shape [T, N, F].
        pipeline: Chronos2Pipeline instance.
        context_start: Index of the first context timestep.
        context_len: Number of context timesteps.
        prediction_length: Number of steps to predict.
        num_nodes: Total number of nodes.
        adj_mx: Raw adjacency matrix, shape [N, N].
        progress: Show progress bar.

    Returns:
        Predictions array of shape [num_nodes, prediction_length].
    """
    predictions = np.zeros((num_nodes, prediction_length), dtype=np.float32)
    iterator = range(num_nodes)
    if progress:
        iterator = tqdm(iterator, desc="Adj-neighbour prediction", leave=False)

    for node_idx in iterator:
        # Find neighbours: nonzero entries in this node's adjacency row
        neighbour_mask = adj_mx[node_idx] != 0
        neighbour_indices = np.where(neighbour_mask)[0]

        # Build list of variate indices: target node first, then neighbours
        all_indices = [node_idx]
        for ni in neighbour_indices:
            if ni != node_idx:
                all_indices.append(ni)
        all_indices = np.array(all_indices)

        # Context: [K, T] where K = num variates
        context_slice = data_mm[context_start:context_start + context_len, all_indices, 0].T
        context = torch.tensor(context_slice, dtype=torch.float32).unsqueeze(0)
        # context shape: [1, K, T]

        with torch.no_grad():
            forecast = pipeline.predict(context, prediction_length=prediction_length)

        pred = _extract_median_prediction(forecast)
        # pred shape: [K, prediction_length]; take the first variate (target node)
        predictions[node_idx, :] = pred[0, :prediction_length]

    return predictions


def get_avg_neighbours_count(adj_mx: np.ndarray) -> float:
    """
    Compute the average number of variates per node in adj_neighbour mode.
    Each node sees itself + its nonzero neighbours.
    """
    num_nodes = adj_mx.shape[0]
    total_variates = 0
    for i in range(num_nodes):
        neighbours = np.count_nonzero(adj_mx[i])
        # Include self even if diagonal is zero
        if adj_mx[i, i] == 0:
            total_variates += neighbours + 1
        else:
            total_variates += neighbours
    return total_variates / num_nodes


def compute_effective_context_length(
    base_context_length: int,
    window_strategy: str,
    mode: str,
    num_nodes: int,
    adj_mx: Optional[np.ndarray] = None,
) -> int:
    """
    Compute the effective context length given window strategy and mode.

    Args:
        base_context_length: The configured context length.
        window_strategy: 'absolute' or 'divided'.
        mode: 'single_node', 'whole_matrix', or 'adj_neighbour'.
        num_nodes: Total number of nodes.
        adj_mx: Adjacency matrix (needed for adj_neighbour divided mode).

    Returns:
        Effective context length (int, >= 1).
    """
    if window_strategy == "absolute":
        return base_context_length

    # Divided strategy
    if mode == "single_node":
        return base_context_length  # 1 series, full budget
    elif mode == "whole_matrix":
        return max(1, base_context_length // num_nodes)
    elif mode == "adj_neighbour":
        if adj_mx is not None:
            avg_vars = get_avg_neighbours_count(adj_mx)
            return max(1, int(base_context_length / avg_vars))
        else:
            return base_context_length
    else:
        return base_context_length
