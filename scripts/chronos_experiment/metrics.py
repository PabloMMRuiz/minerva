"""
Metric computation for the Chronos experiment runner.

Computes MAE, RMSE, MSE, MAPE at both per-step and aggregate levels.
"""

import numpy as np
from typing import Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate MAE, RMSE, MSE, MAPE across all nodes and time steps.

    Args:
        y_true: Ground truth array of shape [num_nodes, horizon].
        y_pred: Predictions array of shape [num_nodes, horizon].

    Returns:
        Dictionary with keys 'mae', 'rmse', 'mse', 'mape'.
    """
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    # MAPE with floor to avoid division by zero
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1.0, None))) * 100)
    return {"mae": mae, "rmse": rmse, "mse": mse, "mape": mape}


def calculate_per_horizon_step_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculate metrics for each individual horizon step (across all nodes).

    Args:
        y_true: Ground truth array of shape [num_nodes, horizon].
        y_pred: Predictions array of shape [num_nodes, horizon].

    Returns:
        Dictionary with keys 'mae', 'rmse', 'mse', 'mape',
        each containing a 1D array of length = horizon.
    """
    horizon = y_true.shape[1]
    mae = np.zeros(horizon)
    rmse = np.zeros(horizon)
    mse = np.zeros(horizon)
    mape = np.zeros(horizon)

    for h in range(horizon):
        true_h = y_true[:, h]
        pred_h = y_pred[:, h]
        mae[h] = np.mean(np.abs(true_h - pred_h))
        mse[h] = np.mean((true_h - pred_h) ** 2)
        rmse[h] = np.sqrt(mse[h])
        mape[h] = np.mean(np.abs((true_h - pred_h) / np.clip(true_h, 1.0, None))) * 100

    return {"mae": mae, "rmse": rmse, "mse": mse, "mape": mape}
