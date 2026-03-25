"""
CLI entry point for the Chronos experiment runner.

Usage:
    # From config file:
    python -m scripts.chronos_experiment.cli --config scripts/chronos_experiment/sample_config.yaml

    # From CLI flags:
    python -m scripts.chronos_experiment.cli \\
        --dataset ../data/PEMS-BAY/ \\
        --modes whole_matrix single_node \\
        --horizons 3 6 12 \\
        --context-length 288 \\
        --window-strategy absolute

    # Adjacency-neighbour mode:
    python -m scripts.chronos_experiment.cli \\
        --dataset ../data/PEMS-BAY/ \\
        --modes adj_neighbour \\
        --adjacency-files ../data/PEMS-BAY/adj_mx.pkl \\
        --horizons 3 6 12
"""

import argparse
import sys

from .config import load_config, build_config_from_args
from .runner import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Chronos v2 Experiment Runner — test Chronos on time-series datasets "
                    "with multiple prediction modes, window strategies, and horizons.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Run from a YAML config file:
  python -m scripts.chronos_experiment.cli --config scripts/chronos_experiment/sample_config.yaml

  # Run all modes on PEMS-BAY with absolute window:
  python -m scripts.chronos_experiment.cli \\
      --dataset ../data/PEMS-BAY/ \\
      --modes single_node whole_matrix adj_neighbour \\
      --adjacency-files ../data/PEMS-BAY/adj_mx.pkl \\
      --horizons 3 6 12 \\
      --context-length 288

  # Run only whole_matrix mode with divided window:
  python -m scripts.chronos_experiment.cli \\
      --dataset ../data/PEMS-BAY/ \\
      --modes whole_matrix \\
      --window-strategy divided \\
      --context-length 288

  # Minimal run:
  python -m scripts.chronos_experiment.cli \\
      --dataset ../data/PEMS-BAY/ \\
      --modes whole_matrix \\
      --horizons 3
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file. When provided, all other flags are ignored.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the dataset folder (must contain desc.json and data.dat).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Chronos model identifier (default: amazon/chronos-2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda if available).",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Prediction horizons (default: 3 6 12).",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Context window length (default: 12).",
    )
    parser.add_argument(
        "--window-strategy",
        type=str,
        default=None,
        choices=["absolute", "divided"],
        help="Window strategy: 'absolute' (fixed) or 'divided' (budget/num_series).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=None,
        choices=["single_node", "whole_matrix", "adj_neighbour"],
        help="Prediction modes to run.",
    )
    parser.add_argument(
        "--adjacency-files",
        type=str,
        nargs="+",
        default=None,
        help="Paths to adjacency matrix pickle files (needed for adj_neighbour mode).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Fraction of data to use as test set (default: from desc.json).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Number of experiment runs for averaging (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: ../results/).",
    )

    args = parser.parse_args()

    # Build configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
    else:
        if not args.dataset:
            parser.error("--dataset is required when not using --config")
        if not args.modes:
            parser.error("--modes is required when not using --config")
        config = build_config_from_args(args)

    # Run the experiment
    try:
        run_dir = run_experiment(config)
        print(f"\nDone! Results at: {run_dir}")
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
