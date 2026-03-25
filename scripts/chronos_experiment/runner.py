"""
Core experiment runner for the Chronos experiment.

Orchestrates: data loading, model loading, running prediction modes
across horizons and window strategies, metric computation, and result storage.
"""

import os
import sys
import time
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, List, Optional

from .metrics import calculate_metrics, calculate_per_horizon_step_metrics
from .modes import (
    predict_single_node,
    predict_whole_matrix,
    predict_adj_neighbour,
    compute_effective_context_length,
)
from .storage import (
    create_output_dir,
    save_step_data,
    save_summary_csv,
    save_config_snapshot,
)


def _load_data(config: Dict[str, Any]):
    """Load dataset and metadata using src.data.loaders."""
    # Add project root to path so we can import src
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.data.loaders import load_dataset_as_memmap, load_dataset_description, load_adj

    dataset_path = config["dataset"]
    metadata = load_dataset_description(dataset_path)
    if metadata is None:
        raise RuntimeError(f"Failed to load metadata from {dataset_path}")

    data, success = load_dataset_as_memmap(dataset_path)
    if not success:
        raise RuntimeError(f"Failed to load data from {dataset_path}")

    # Load adjacency matrices if needed
    adj_matrices = {}
    if "adj_neighbour" in config.get("modes", []):
        for adj_path in config.get("adjacency_files", []):
            abs_adj_path = adj_path
            if not os.path.isabs(adj_path):
                abs_adj_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", adj_path
                )
                abs_adj_path = os.path.abspath(abs_adj_path)
            _, adj_raw = load_adj(abs_adj_path, "original")
            adj_name = os.path.splitext(os.path.basename(adj_path))[0]
            adj_matrices[adj_name] = adj_raw

    return data, metadata, adj_matrices


def _load_pipeline(config: Dict[str, Any]):
    """Load the Chronos2Pipeline."""
    from chronos import Chronos2Pipeline

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype_str = config.get("dtype", "bfloat16")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"Loading Chronos pipeline '{config['model']}' on {device} ({dtype_str})...")
    pipeline = Chronos2Pipeline.from_pretrained(
        config["model"],
        device_map=device,
        torch_dtype=torch_dtype,
    )
    return pipeline


def _get_test_ratio(config: Dict[str, Any], metadata: Dict[str, Any]) -> float:
    """Get test ratio from config or desc.json."""
    if config.get("test_ratio") is not None:
        return config["test_ratio"]
    ratios = metadata.get("regular_settings", {}).get("TRAIN_VAL_TEST_RATIO", [0.7, 0.1, 0.2])
    return ratios[2]


def run_experiment(config: Dict[str, Any]) -> str:
    """
    Run the full experiment according to the configuration.

    Args:
        config: Experiment configuration dictionary.

    Returns:
        Path to the run output directory.
    """
    print("=" * 60)
    print("Chronos v2 Experiment Runner")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading data...")
    data, metadata, adj_matrices = _load_data(config)

    dataset_name = metadata.get("name", "unknown")
    shape = metadata.get("shape", list(data.shape))
    total_steps = shape[0]
    num_nodes = shape[1]

    print(f"  Dataset: {dataset_name}")
    print(f"  Shape: {shape}")
    print(f"  Nodes: {num_nodes}")

    # 2. Load model
    print("\n[2/4] Loading model...")
    pipeline = _load_pipeline(config)

    # 3. Prepare experiment parameters
    print("\n[3/4] Preparing experiment...")
    test_ratio = _get_test_ratio(config, metadata)
    test_start = int(total_steps * (1 - test_ratio))
    horizons = config.get("horizons", [3, 6, 12])
    max_horizon = max(horizons)
    modes = config.get("modes", ["single_node", "whole_matrix", "adj_neighbour"])
    window_strategy = config.get("window_strategy", "absolute")
    base_context_length = config.get("context_length", 12)
    num_runs = config.get("num_runs", 1)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = create_output_dir(config.get("output_dir", "../results/"), dataset_name, timestamp)
    save_config_snapshot(run_dir, config)

    print(f"  Test start index: {test_start}")
    print(f"  Horizons: {horizons}")
    print(f"  Modes: {modes}")
    print(f"  Window strategy: {window_strategy}")
    print(f"  Base context length: {base_context_length}")
    print(f"  Num runs: {num_runs}")
    print(f"  Output: {run_dir}")

    # Evaluation indices
    eval_indices = np.arange(test_start, total_steps - max_horizon, max_horizon)
    print(f"  Evaluation windows: {len(eval_indices)}")

    # 4. Run experiments
    print("\n[4/4] Running experiments...")
    summary_rows = []

    for mode in modes:
        print(f"\n{'='*40}")
        print(f"Mode: {mode}")
        print(f"{'='*40}")

        # Determine adjacency matrices to iterate over for adj_neighbour mode
        if mode == "adj_neighbour":
            adj_items = list(adj_matrices.items())
        else:
            adj_items = [(None, None)]

        for adj_name, adj_mx in adj_items:
            if adj_name:
                print(f"\n  Adjacency: {adj_name}")

            # Compute effective context length
            eff_ctx = compute_effective_context_length(
                base_context_length, window_strategy, mode, num_nodes, adj_mx
            )
            print(f"  Effective context length: {eff_ctx}")

            for run_idx in range(num_runs):
                if num_runs > 1:
                    print(f"\n  >>> Run {run_idx + 1}/{num_runs}")

                run_start_time = time.time()

                # Per-horizon results
                for horizon in horizons:
                    print(f"\n  Horizon: {horizon}")
                    h_start_time = time.time()

                    predictions_list = []
                    ground_truth_list = []
                    per_step_metrics_list = []

                    for idx in eval_indices:
                        context_start = idx - eff_ctx
                        if context_start < 0:
                            continue

                        # Ground truth: [N, H]
                        gt = data[idx:idx + horizon, :, 0].T

                        # Run prediction
                        if mode == "single_node":
                            preds = predict_single_node(
                                data, pipeline, context_start, eff_ctx,
                                horizon, num_nodes, progress=False,
                            )
                        elif mode == "whole_matrix":
                            preds = predict_whole_matrix(
                                data, pipeline, context_start, eff_ctx,
                                horizon, num_nodes, progress=False,
                            )
                        elif mode == "adj_neighbour":
                            preds = predict_adj_neighbour(
                                data, pipeline, context_start, eff_ctx,
                                horizon, num_nodes, adj_mx, progress=False,
                            )
                        else:
                            raise ValueError(f"Unknown mode: {mode}")

                        # Compute per-step metrics
                        step_metrics = calculate_metrics(gt, preds)
                        per_step_metrics_list.append(step_metrics)
                        predictions_list.append(preds)
                        ground_truth_list.append(gt)

                    h_duration = time.time() - h_start_time

                    # Aggregate metrics for this horizon
                    if per_step_metrics_list:
                        agg_metrics = {
                            k: float(np.mean([m[k] for m in per_step_metrics_list]))
                            for k in ["mae", "rmse", "mse", "mape"]
                        }
                    else:
                        agg_metrics = {"mae": 0, "rmse": 0, "mse": 0, "mape": 0}

                    print(f"    MAE: {agg_metrics['mae']:.4f}  "
                          f"RMSE: {agg_metrics['rmse']:.4f}  "
                          f"MAPE: {agg_metrics['mape']:.2f}%  "
                          f"Time: {h_duration:.1f}s")

                    # Save step data
                    mode_label = f"{mode}_{adj_name}" if adj_name else mode
                    save_step_data(
                        run_dir, mode_label, horizon, window_strategy,
                        predictions_list, ground_truth_list,
                        per_step_metrics_list, eff_ctx,
                    )

                    # Add summary row
                    summary_rows.append({
                        "dataset": dataset_name,
                        "mode": mode,
                        "adjacency": adj_name or "",
                        "window_strategy": window_strategy,
                        "context_length": eff_ctx,
                        "horizon": horizon,
                        "run": run_idx + 1,
                        "num_nodes": num_nodes,
                        "num_eval_windows": len(per_step_metrics_list),
                        "mae": agg_metrics["mae"],
                        "rmse": agg_metrics["rmse"],
                        "mse": agg_metrics["mse"],
                        "mape": agg_metrics["mape"],
                        "time_sec": h_duration,
                    })

    # Save summary
    csv_path = save_summary_csv(run_dir, summary_rows)
    print(f"\n{'='*60}")
    print(f"Experiment complete. Results saved to: {run_dir}")
    print(f"Summary CSV: {csv_path}")
    print(f"{'='*60}")

    return run_dir
