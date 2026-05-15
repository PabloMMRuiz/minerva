"""
Core experiment runner for the Chronos experiment.

Orchestrates: data loading, model loading, running prediction modes
across horizons and window strategies, metric computation, and result storage.
"""

import os
import sys
import time
import threading
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, List

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


class ResourceMonitor:
    """
    Background thread that samples process RSS memory, VRAM, and GPU utilization.

    Usage::

        monitor = ResourceMonitor(interval=0.5)
        monitor.start()
        # ... run experiment ...
        stats = monitor.stop()
    """

    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._ram_samples: List[float] = []
        self._vram_samples: List[float] = []
        self._gpu_util_samples: List[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        
        if _PSUTIL_AVAILABLE:
            self._process = psutil.Process(os.getpid())
            
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                # Use the current device if possible
                device_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            except Exception:
                self._gpu_handle = None
        else:
            self._gpu_handle = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                # RAM
                if _PSUTIL_AVAILABLE:
                    rss_mb = self._process.memory_info().rss / (1024 ** 2)
                    self._ram_samples.append(rss_mb)
                
                # VRAM
                if torch.cuda.is_available():
                    vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                    self._vram_samples.append(vram_mb)
                
                # GPU Utilization
                if self._gpu_handle:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle).gpu
                    self._gpu_util_samples.append(float(util))
                    
            except Exception:
                pass
            self._stop_event.wait(self._interval)

    def start(self) -> None:
        """Start sampling in the background."""
        self._ram_samples.clear()
        self._vram_samples.clear()
        self._gpu_util_samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        """
        Stop sampling and return aggregated stats.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        
        def agg(samples):
            if not samples:
                return 0.0, 0.0
            return float(max(samples)), float(sum(samples) / len(samples))

        ram_peak, ram_avg = agg(self._ram_samples)
        vram_peak, vram_avg = agg(self._vram_samples)
        gpu_peak, gpu_avg = agg(self._gpu_util_samples)

        return {
            "ram_peak_mb": ram_peak,
            "ram_avg_mb": ram_avg,
            "vram_peak_mb": vram_peak,
            "vram_avg_mb": vram_avg,
            "gpu_peak_util": gpu_peak,
            "gpu_avg_util": gpu_avg,
        }

from .metrics import (
    calculate_metrics,
    calculate_masked_metrics,
    probabilistic_metrics,
    evaluation,
    masked_mae_np,
    masked_rmse_np,
    masked_mse_np,
    masked_mape_np,
)
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


def _prepare_long_df(data: np.ndarray, metadata: Dict[str, Any], config: Dict[str, Any]):
    """
    Convert the [T, N, F] memmap into a long-format pandas DataFrame.

    Returns (df_long, base_start_time). The DataFrame is indexed by timestamp
    and sorted for fast .loc slicing.
    """
    num_time_steps = data.shape[0]
    num_nodes = data.shape[1]
    freq_mins = metadata.get("frequency (minutes)", 5)

    # Create timestamps (arbitrary start if not present)
    start_time = pd.Timestamp("2024-01-01")
    timestamps = pd.date_range(start=start_time, periods=num_time_steps, freq=f"{freq_mins}min")

    # Flatten data[:, :, 0] (the target value)
    target_data = np.asarray(data[:, :, 0])  # [T, N]

    id_col = "sensor_id"
    ts_col = "timestamp"
    target_col = "value"

    # Vectorised conversion: repeat timestamps for each node, tile node ids
    node_ids = np.repeat(np.arange(num_nodes), num_time_steps)
    ts_tiled = np.tile(timestamps, num_nodes)
    values = target_data.T.ravel()  # [N*T] — column-major per node

    df_long = pd.DataFrame({
        ts_col: ts_tiled,
        target_col: values,
        id_col: node_ids,
    })

    # Set index and sort for faster filtering
    df_long.set_index(ts_col, inplace=True)
    df_long.sort_index(inplace=True)
    return df_long, start_time


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
        dtype=torch_dtype,
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
    print("\n[1/5] Loading data...")
    data, metadata, adj_matrices = _load_data(config)

    dataset_name = metadata.get("name", "unknown")
    shape = metadata.get("shape", list(data.shape))
    total_steps = shape[0]
    num_nodes = shape[1]

    print(f"  Dataset: {dataset_name}")
    print(f"  Shape: {shape}")
    print(f"  Nodes: {num_nodes}")

    # 2. Load model
    print("\n[2/5] Loading model...")
    pipeline = _load_pipeline(config)

    # 3. Prepare experiment parameters
    print("\n[3/5] Preparing experiment...")
    test_ratio = _get_test_ratio(config, metadata)
    test_start = int(total_steps * (1 - test_ratio))
    horizons = config.get("horizons", [3, 6, 12])
    max_horizon = max(horizons)
    modes = config.get("modes", ["single_node", "whole_matrix", "adj_neighbour"])
    window_strategy = config.get("window_strategy", "absolute")
    base_context_length = config.get("context_length", 12)
    num_runs = config.get("num_runs", 1)
    null_val = metadata.get("regular_settings", {}).get("NULL_VAL", 0.0)

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
    print(f"  NULL_VAL: {null_val}")
    print(f"  Output: {run_dir}")

    # Evaluation indices
    windowing_mode = config.get("windowing_mode", "overlapping")
    eval_step = 1 if windowing_mode == "overlapping" else max_horizon
    eval_indices = np.arange(test_start, total_steps - max_horizon, eval_step)
    print(f"  Evaluation windows: {len(eval_indices)} ({windowing_mode})")

    # 4. Prepare long DataFrame (once for the whole experiment)
    print("\n[4/5] Converting data to long DataFrame...")
    ts_col = "timestamp"
    id_col = "sensor_id"
    target_col = "value"
    freq = metadata.get("frequency (minutes)", 5)

    df_long_full, base_start_time = _prepare_long_df(data, metadata, config)
    print(f"  DataFrame shape: {df_long_full.shape}")

    # Precompute time deltas
    stride_delta = pd.Timedelta(minutes=eval_step * freq)
    pred_delta = pd.Timedelta(minutes=max_horizon * freq)
    experiment_start_time = base_start_time + pd.Timedelta(minutes=test_start * freq)
    experiment_end_time = base_start_time + pd.Timedelta(minutes=(total_steps - 1) * freq)
    safe_experiment_end = experiment_end_time - pred_delta

    # 5. Run experiments
    print("\n[5/5] Running experiments...")
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

            # Filter df_long for node_batches mode
            if mode == "node_batches":
                relevant_ids = [idx for batch in item_data for idx in batch]
                df_long = df_long_full[df_long_full[id_col].isin(relevant_ids)]
            else:
                df_long = df_long_full

            for run_idx in range(num_runs):
                suffix = f" (Run {run_idx+1})" if num_runs > 1 else ""

                # Metrics storage per horizon
                horizon_metrics = {
                    h: {
                        "mae": [], "rmse": [], "mse": [], "mape": [],
                        "masked_mae": [], "masked_rmse": [], "masked_mse": [], "masked_mape": [],
                        "coverage": [], "iqr_mean": [], "iqr_median": [], "iqr_std": [],
                    } for h in horizons
                }

                res_monitor = ResourceMonitor(interval=0.5)
                res_monitor.start()
                start_time_exec = time.time()

                current_pred_time = experiment_start_time
                step_count = 0

                pbar_inner = tqdm(
                    total=len(eval_indices),
                    desc=f"Mode: {mode_display}{suffix}",
                    leave=False,
                )

                while current_pred_time <= safe_experiment_end:
                    # Build context DataFrame
                    ctx_start = current_pred_time - pd.Timedelta(minutes=eff_ctx * freq)
                    ctx_end = current_pred_time - pd.Timedelta(seconds=1)
                    context_df = df_long.loc[ctx_start:ctx_end].reset_index()

                    # Build test DataFrame
                    test_end = current_pred_time + pred_delta - pd.Timedelta(seconds=1)
                    test_df = df_long.loc[current_pred_time:test_end].reset_index()

                    # Make forecast via predict_df
                    forecast_df = pipeline.predict_df(
                        context_df,
                        prediction_length=max_horizon,
                        quantile_levels=[0.1, 0.5, 0.9],
                        id_column=id_col,
                        timestamp_column=ts_col,
                        target=target_col,
                        cross_learning=(mode != "single_node"),
                    )

                    # Extract 0.5 quantile as point prediction
                    if "0.5" in forecast_df.columns:
                        forecast_df["predictions"] = forecast_df["0.5"]
                    elif "predictions" not in forecast_df.columns:
                        forecast_df["predictions"] = forecast_df.iloc[:, -1]

                    # Evaluate each horizon
                    for h in horizons:
                        target_ts = current_pred_time + pd.Timedelta(minutes=h * freq)

                        forecast_h = forecast_df[forecast_df[ts_col] == target_ts]
                        true_h = test_df[test_df[ts_col] == target_ts]

                        if forecast_h.empty or true_h.empty:
                            continue

                        merged_h = pd.merge(true_h, forecast_h, on=[id_col, ts_col])
                        if merged_h.empty:
                            continue

                        y_true = merged_h[target_col].values.astype(np.float32)
                        y_pred = merged_h["predictions"].values.astype(np.float32)

                        # Standard metrics
                        mae = float(np.mean(np.abs(y_true - y_pred)))
                        mse_val = float(np.mean((y_true - y_pred) ** 2))
                        rmse_val = float(np.sqrt(mse_val))
                        mape_val = float(
                            np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1.0, None))) * 100
                        )

                        horizon_metrics[h]["mae"].append(mae)
                        horizon_metrics[h]["rmse"].append(rmse_val)
                        horizon_metrics[h]["mse"].append(mse_val)
                        horizon_metrics[h]["mape"].append(mape_val)

                        # Masked metrics
                        horizon_metrics[h]["masked_mae"].append(
                            float(masked_mae_np(y_pred, y_true, null_val=null_val))
                        )
                        horizon_metrics[h]["masked_rmse"].append(
                            float(masked_rmse_np(y_pred, y_true, null_val=null_val))
                        )
                        horizon_metrics[h]["masked_mse"].append(
                            float(masked_mse_np(y_pred, y_true, null_val=null_val))
                        )
                        horizon_metrics[h]["masked_mape"].append(
                            float(masked_mape_np(y_pred, y_true, null_val=null_val))
                        )

                        # Probabilistic metrics
                        prob = probabilistic_metrics(
                            forecast_h, true_h, id_col, ts_col, target_col
                        )
                        horizon_metrics[h]["coverage"].append(prob["coverage"])
                        horizon_metrics[h]["iqr_mean"].append(prob["iqr_mean"])
                        horizon_metrics[h]["iqr_median"].append(prob["iqr_median"])
                        horizon_metrics[h]["iqr_std"].append(prob["iqr_std"])

                    current_pred_time += stride_delta
                    step_count += 1
                    pbar_inner.update(1)

                pbar_inner.close()
                total_run_duration = time.time() - start_time_exec
                res_stats = res_monitor.stop()

                # Consolidate and save for each horizon
                for h in horizons:
                    h_m = horizon_metrics[h]
                    if not h_m["mae"]:
                        continue

                    agg_metrics = {k: float(np.mean(v)) for k, v in h_m.items()}

                    summary_rows.append({
                        "dataset": dataset_name,
                        "mode": mode,
                        "config_name": item_name or "",
                        "window_strategy": window_strategy,
                        "windowing_mode": windowing_mode,
                        "context_length": eff_ctx,
                        "horizon": h,
                        "run": run_idx + 1,
                        "num_nodes": num_nodes,
                        "num_eval_windows": len(h_m["mae"]),
                        "mae": agg_metrics["mae"],
                        "rmse": agg_metrics["rmse"],
                        "mse": agg_metrics["mse"],
                        "mape": agg_metrics["mape"],
                        "masked_mae": agg_metrics["masked_mae"],
                        "masked_rmse": agg_metrics["masked_rmse"],
                        "masked_mse": agg_metrics["masked_mse"],
                        "masked_mape": agg_metrics["masked_mape"],
                        "coverage": agg_metrics["coverage"],
                        "iqr_mean": agg_metrics["iqr_mean"],
                        "iqr_median": agg_metrics["iqr_median"],
                        "iqr_std": agg_metrics["iqr_std"],
                        "time_sec": total_run_duration / len(horizons),
                        "ram_peak_mb": res_stats["ram_peak_mb"],
                        "ram_avg_mb": res_stats["ram_avg_mb"],
                        "vram_peak_mb": res_stats["vram_peak_mb"],
                        "vram_avg_mb": res_stats["vram_avg_mb"],
                        "gpu_peak_util": res_stats["gpu_peak_util"],
                        "gpu_avg_util": res_stats["gpu_avg_util"],
                    })

                save_summary_csv(run_dir, summary_rows)
                pbar_outer.update(1)

    pbar_outer.close()
    
    print(f"\n{'='*60}")
    print(f"Experiment complete. Results saved to: {run_dir}")
    print(f"Summary CSV: {os.path.join(run_dir, 'summary.csv')}")
    print(f"{'='*60}")

    return run_dir
