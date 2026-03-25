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
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any

from .metrics import calculate_metrics
from .modes import (
    predict_single_node,
    predict_whole_matrix,
    predict_adj_neighbour,
    predict_node_batches,
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

    # Progress bar for all combinations of modes and runs
    def get_comb_count(m):
        if m == "adj_neighbour":
            return len(adj_matrices)
        return 1
        
    total_combinations = sum([get_comb_count(m) for m in modes]) * num_runs
    pbar_outer = tqdm(total=total_combinations, desc="Total Progress")

    for mode in modes:
        # Determine items to iterate over for this mode
        if mode == "adj_neighbour":
            mode_items = list(adj_matrices.items())
        elif mode == "node_batches":
            mode_items = []
            
            # 1. Explicit batches from config
            explicit = config.get("node_batches")
            if explicit:
                # Check if it's List[List[int]] (one set) or List[List[List[int]]] (multiple sets)
                # We assume if the first element is a list, and its first element is also a list, it's multiple sets.
                # Actually, explicit node_batches is always List[List[int]] for ONE set.
                # If we have multiple, it's List[List[List[int]]].
                if isinstance(explicit[0][0], int):
                    mode_items.append(("batches_0", explicit))
                else:
                    for i, b_set in enumerate(explicit):
                        mode_items.append((f"batches_{i}", b_set))
            
            # 2. From JSON files
            for fpath in config.get("node_batches_files", []):
                import json
                try:
                    with open(fpath, "r") as f:
                        loaded = json.load(f)
                        mode_items.append((os.path.basename(fpath), loaded))
                except Exception as e:
                    print(f"Warning: Failed to load node_batches_file {fpath}: {e}")
            
            # 3. From automatic sizes
            for size in config.get("batch_sizes", []):
                batches = [list(range(i, min(i + size, num_nodes))) for i in range(0, num_nodes, size)]
                mode_items.append((f"size_{size}", batches))
            
            if not mode_items:
                # Fallback to whole matrix
                mode_items.append(("batches_default", [list(range(num_nodes))]))
                print("  Warning: No batching info for node_batches mode. Falling back to default batch.")
            
            print(f"  Initialized {len(mode_items)} batch configurations for node_batches mode.")
        else:
            mode_items = [(None, None)]

        for item_name, item_data in mode_items:
            mode_display = f"{mode} ({item_name})" if item_name else mode
            
            # Compute effective context length
            eff_ctx = compute_effective_context_length(
                base_context_length, 
                window_strategy, 
                mode, 
                num_nodes, 
                adj_mx=item_data if mode == "adj_neighbour" else None,
                batches=item_data if mode == "node_batches" else None
            )

            for run_idx in range(num_runs):
                suffix = f" (Run {run_idx+1})" if num_runs > 1 else ""
                
                # We initialize storage for ALL horizons in this run
                # dict of lists: horizon -> List[np.ndarray]
                run_predictions = {h: [] for h in horizons}
                run_ground_truth = {h: [] for h in horizons}
                run_step_metrics = {h: [] for h in horizons}
                
                start_time = time.time()
                
                # Eval index loop - predict ONCE for max_horizon
                pbar_inner = tqdm(eval_indices, desc=f"Mode: {mode_display}{suffix}", leave=False)
                for idx in pbar_inner:
                    context_start = idx - eff_ctx
                    if context_start < 0:
                        continue

                    # Predict max_horizon
                    if mode == "single_node":
                        preds_full = predict_single_node(
                            data, pipeline, context_start, eff_ctx,
                            max_horizon, num_nodes, progress=False,
                        )
                    elif mode == "whole_matrix":
                        preds_full = predict_whole_matrix(
                            data, pipeline, context_start, eff_ctx,
                            max_horizon, num_nodes, progress=False,
                        )
                    elif mode == "adj_neighbour":
                        preds_full = predict_adj_neighbour(
                            data, pipeline, context_start, eff_ctx,
                            max_horizon, num_nodes, item_data, progress=False,
                        )
                    elif mode == "node_batches":
                        preds_full = predict_node_batches(
                            data, pipeline, context_start, eff_ctx,
                            max_horizon, num_nodes, item_data, progress=False,
                        )
                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                    # Distribute slices to each horizon
                    for h in horizons:
                        preds_h = preds_full[:, :h]
                        gt_h = data[idx:idx + h, :, 0].T
                        
                        run_predictions[h].append(preds_h)
                        run_ground_truth[h].append(gt_h)
                        run_step_metrics[h].append(calculate_metrics(gt_h, preds_h))

                total_run_duration = time.time() - start_time
                
                # Consolidate and save for each horizon
                for h in horizons:
                    h_preds = run_predictions[h]
                    h_gts = run_ground_truth[h]
                    h_metrics = run_step_metrics[h]
                    
                    if not h_preds:
                        continue
                        
                    # Aggregate results
                    agg_metrics = {
                        k: float(np.mean([m[k] for m in h_metrics]))
                        for k in ["mae", "rmse", "mse", "mape"]
                    }
                    
                    # Save step data
                    mode_label = f"{mode}_{item_name}" if item_name else mode
                    save_step_data(
                        run_dir, mode_label, h, window_strategy,
                        h_preds, h_gts, h_metrics, eff_ctx,
                    )

                    # Add summary row
                    summary_rows.append({
                        "dataset": dataset_name,
                        "mode": mode,
                        "config_name": item_name or "",
                        "window_strategy": window_strategy,
                        "context_length": eff_ctx,
                        "horizon": h,
                        "run": run_idx + 1,
                        "num_nodes": num_nodes,
                        "num_eval_windows": len(h_metrics),
                        "mae": agg_metrics["mae"],
                        "rmse": agg_metrics["rmse"],
                        "mse": agg_metrics["mse"],
                        "mape": agg_metrics["mape"],
                        "time_sec": total_run_duration / len(horizons), # proportional time
                    })

                pbar_outer.update(1)

    pbar_outer.close()

    # Save summary
    csv_path = save_summary_csv(run_dir, summary_rows)
    print(f"\n{'='*60}")
    print(f"Experiment complete. Results saved to: {run_dir}")
    print(f"Summary CSV: {csv_path}")
    print(f"{'='*60}")

    return run_dir
