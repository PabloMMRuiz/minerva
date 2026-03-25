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

import json
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

  # Run node batches (manual partition)
  python -m scripts.chronos_experiment.cli \\
      --dataset data/PEMS-BAY/ \\
      --modes node_batches \\
      --node-batches "0,1,2|3,4,5"
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
        choices=["single_node", "whole_matrix", "adj_neighbour", "node_batches"],
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
    parser.add_argument(
        "--node-batches",
        type=str,
        default=None,
        help="Explicit node batches. Use '|' to separate batches and ';' to separate sets: '0,1|2;3,4|5'.",
    )
    parser.add_argument(
        "--node-batches-files",
        type=str,
        nargs="+",
        default=None,
        help="One or more JSON files containing node batches (list of lists).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="One or more sizes for automatic node partitioning.",
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
        
        # Parse node batches if provided as string
        if args.node_batches:
            try:
                # Support multiple sets separated by ';'
                sets_str = args.node_batches.split(';')
                all_sets = []
                for s in sets_str:
                    batches_str = s.strip().split('|')
                    all_sets.append([[int(i) for i in b.split(',')] for b in batches_str])
                # If only one set, we keep it as is (list of lists) but runner can handle list of sets
                args.node_batches = all_sets
            except Exception as e:
                parser.error(f"Invalid format for --node-batches: {e}")
        
        # Load node batches files if provided
        if args.node_batches_files:
            args.node_batches_files_data = []
            for fpath in args.node_batches_files:
                try:
                    with open(fpath, 'r') as f:
                        args.node_batches_files_data.append(json.load(f))
                except Exception as e:
                    parser.error(f"Failed to load --node-batches-files {fpath}: {e}")

        config = build_config_from_args(args)
        
        # transfer loaded file data to config if present
        if hasattr(args, 'node_batches_files_data'):
            # Combine with existing explicit batches
            if config.get('node_batches') is None:
                config['node_batches'] = args.node_batches_files_data
            else:
                config['node_batches'].extend(args.node_batches_files_data)

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
