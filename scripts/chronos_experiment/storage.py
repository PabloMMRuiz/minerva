"""
Result storage utilities for the Chronos experiment runner.

Handles saving:
  - Step-by-step predictions and metrics as .npz files
  - Per-step metrics as CSV
  - Run-level summary CSV
  - Config snapshot as YAML
"""

import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional


def create_output_dir(
    base_dir: str,
    dataset_name: str,
    timestamp: Optional[str] = None,
) -> str:
    """
    Create the output directory structure for a run.

    Returns:
        Path to the created run directory.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(base_dir, dataset_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_step_data(
    run_dir: str,
    mode: str,
    horizon: int,
    window_strategy: str,
    predictions_list: List[np.ndarray],
    ground_truth_list: List[np.ndarray],
    per_step_metrics_list: List[Dict[str, Any]],
    context_length: int,
) -> str:
    """
    Save step-by-step predictions, ground truth, and metrics.

    Args:
        run_dir: Base run output directory.
        mode: Prediction mode name.
        horizon: Prediction horizon.
        window_strategy: Window strategy used.
        predictions_list: List of prediction arrays, each [N, H].
        ground_truth_list: List of ground truth arrays, each [N, H].
        per_step_metrics_list: List of per-step metric dicts.
        context_length: Effective context length used.

    Returns:
        Path to the created subdirectory.
    """
    subdir_name = f"h{horizon}_{window_strategy[:3]}"
    subdir = os.path.join(run_dir, mode, subdir_name)
    os.makedirs(subdir, exist_ok=True)

    # Save predictions and ground truth as npz
    preds = np.array(predictions_list)  # [num_steps, N, H]
    gts = np.array(ground_truth_list)   # [num_steps, N, H]
    np.savez_compressed(
        os.path.join(subdir, "predictions.npz"),
        predictions=preds,
        ground_truth=gts,
        context_length=np.array(context_length),
    )

    # Save per-step metrics as CSV
    if per_step_metrics_list:
        metrics_df = pd.DataFrame(per_step_metrics_list)
        metrics_df.index.name = "step"
        metrics_df.to_csv(os.path.join(subdir, "step_metrics.csv"))

    return subdir


def save_summary_csv(
    run_dir: str,
    summary_rows: List[Dict[str, Any]],
) -> str:
    """
    Save the overall summary CSV with one row per experiment combination.

    Returns:
        Path to the summary CSV file.
    """
    csv_path = os.path.join(run_dir, "summary.csv")
    df = pd.DataFrame(summary_rows)
    df.to_csv(csv_path, index=False)
    return csv_path


def save_config_snapshot(run_dir: str, config: Dict[str, Any]) -> str:
    """
    Save a snapshot of the config used for this run.

    Returns:
        Path to the saved config file.
    """
    import yaml
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path
