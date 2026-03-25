"""
YAML configuration loader for the Chronos experiment runner.
"""

import yaml
from typing import Dict, Any, Optional


DEFAULT_CONFIG = {
    "dataset": "../data/PEMS-BAY/",
    "model": "amazon/chronos-2",
    "device": "cuda",
    "dtype": "bfloat16",
    "horizons": [3, 6, 12],
    "context_length": 12,
    "window_strategy": "absolute",
    "modes": ["single_node", "whole_matrix", "adj_neighbour"],
    "adjacency_files": [],
    "test_ratio": None,  # None means read from desc.json
    "num_runs": 1,
    "output_dir": "../results/",
}

VALID_MODES = {"single_node", "whole_matrix", "adj_neighbour"}
VALID_WINDOW_STRATEGIES = {"absolute", "divided"}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from a YAML file.

    Merges with defaults so all keys are present.
    """
    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f) or {}

    config = {**DEFAULT_CONFIG, **user_config}
    _validate_config(config)
    return config


def build_config_from_args(args) -> Dict[str, Any]:
    """
    Build a config dictionary from parsed CLI arguments.
    """
    config = dict(DEFAULT_CONFIG)  # start with defaults

    if args.dataset:
        config["dataset"] = args.dataset
    if args.model:
        config["model"] = args.model
    if args.device:
        config["device"] = args.device
    if args.horizons:
        config["horizons"] = args.horizons
    if args.context_length is not None:
        config["context_length"] = args.context_length
    if args.window_strategy:
        config["window_strategy"] = args.window_strategy
    if args.modes:
        config["modes"] = args.modes
    if args.adjacency_files:
        config["adjacency_files"] = args.adjacency_files
    if args.test_ratio is not None:
        config["test_ratio"] = args.test_ratio
    if args.num_runs is not None:
        config["num_runs"] = args.num_runs
    if args.output_dir:
        config["output_dir"] = args.output_dir

    _validate_config(config)
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate config values, raising ValueError on invalid input."""
    for mode in config.get("modes", []):
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Valid modes: {VALID_MODES}"
            )

    ws = config.get("window_strategy", "absolute")
    if ws not in VALID_WINDOW_STRATEGIES:
        raise ValueError(
            f"Invalid window_strategy '{ws}'. Valid: {VALID_WINDOW_STRATEGIES}"
        )

    if "adj_neighbour" in config.get("modes", []):
        adj_files = config.get("adjacency_files", [])
        if not adj_files:
            raise ValueError(
                "adj_neighbour mode requires at least one adjacency file. "
                "Provide --adjacency-files or adjacency_files in config."
            )
