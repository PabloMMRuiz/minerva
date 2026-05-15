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

def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype("float32")
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype("float32")
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype("float32"), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)



def calculate_masked_metrics(y_true: np.ndarray, y_pred: np.ndarray, null_val=np.nan) -> Dict[str, float]:
    """
    Calculate masked MAE, RMSE, MSE, MAPE across all nodes and time steps.
    """
    mape = float(masked_mape_np(preds=y_pred, labels=y_true, null_val=null_val))
    mae = float(masked_mae_np(preds=y_pred, labels=y_true, null_val=null_val))
    rmse = float(masked_rmse_np(preds=y_pred, labels=y_true, null_val=null_val))
    mse = float(masked_mse_np(preds=y_pred, labels=y_true, null_val=null_val))
    return {"mae": mae, "rmse": rmse, "mse": mse, "mape": mape}


def probabilistic_metrics(
    forecast_df, true_df, id_column, timestamp_column, target_column
) -> Dict[str, float]:
    """
    Calculate probabilistic metrics: coverage and IQR stats.
    """
    import pandas as pd

    # Merge on sensor and timestamp
    merged = pd.merge(true_df, forecast_df, on=[id_column, timestamp_column])
    if merged.empty:
        return {"coverage": 0.0, "iqr_mean": 0.0, "iqr_median": 0.0, "iqr_std": 0.0}

    # Coverage: check if true value is within [q0.1, q0.9]
    # Chronos predict_df returns quantiles as columns named like '0.1', '0.5', '0.9'
    q_low = "0.1"
    q_high = "0.9"

    if q_low in merged.columns and q_high in merged.columns:
        coverage = (merged[target_column] >= merged[q_low]) & (
            merged[target_column] <= merged[q_high]
        )
        iqr = merged[q_high] - merged[q_low]
        return {
            "coverage": float(coverage.mean()),
            "iqr_mean": float(iqr.mean()),
            "iqr_median": float(iqr.median()),
            "iqr_std": float(iqr.std()),
        }
    else:
        # Fallback if quantiles are missing
        return {"coverage": 0.0, "iqr_mean": 0.0, "iqr_median": 0.0, "iqr_std": 0.0}


def evaluation(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Simple wrapper returning (mape, mae, rmse).
    """
    m = calculate_metrics(y_true, y_pred)
    return m["mape"], m["mae"], m["rmse"]
