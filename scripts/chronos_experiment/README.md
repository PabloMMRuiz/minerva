# Chronos v2 Experiment Runner

Test Chronos v2 on time-series datasets with three prediction modes,
two time-window strategies, multiple horizons, and full result storage.

## Quick Start

### From a config file

```bash
python -m scripts.chronos_experiment.cli --config scripts/chronos_experiment/sample_config.yaml
```

### From CLI flags

```bash
# All modes on PEMS-BAY with absolute window
python -m scripts.chronos_experiment.cli \
    --dataset ../data/PEMS-BAY/ \
    --modes single_node whole_matrix adj_neighbour \
    --adjacency-files ../data/PEMS-BAY/adj_mx.pkl \
    --horizons 3 6 12 \
    --context-length 288

# Only whole_matrix with divided window strategy
python -m scripts.chronos_experiment.cli \
    --dataset data/PEMS-BAY/ \
    --modes whole_matrix \
    --window-strategy divided \
    --context-length 288

# Node batches (automatic)
python -m scripts.chronos_experiment.cli \
    --dataset data/PEMS-BAY/ \
    --modes node_batches \
    --batch-size 50

# Node batches (explicit string)
python -m scripts.chronos_experiment.cli \
    --dataset data/PEMS-BAY/ \
    --modes node_batches \
    --node-batches "0,1,2|3,4,5"

# Minimal run
python -m scripts.chronos_experiment.cli \
    --dataset ../data/PEMS-BAY/ \
    --modes whole_matrix \
    --horizons 3
```

## Prediction Modes

| Mode | Description |
|---|---|
| `single_node` | Predict each node independently as a univariate series |
| `whole_matrix` | Predict all nodes simultaneously as variates of one multivariate series |
| `adj_neighbour` | For each node, feed it plus its adjacency neighbours as variates |
| `node_batches` | Predict nodes in partitioned batches (parallel multivariate) |

## Window Strategies

| Strategy | Description |
|---|---|
| `absolute` | Fixed context window for all modes |
| `divided` | Total budget divided by number of series used; equalises data budget across modes |

## Output Structure

```
results/<dataset_name>/<timestamp>/
├── config.yaml         # Config snapshot
├── summary.csv         # One row per (mode × horizon × run) combination
├── single_node/
│   ├── h3_abs/
│   │   ├── predictions.npz    # Arrays: predictions, ground_truth, context_length
│   │   └── step_metrics.csv   # Per-evaluation-window metrics
│   ├── h6_abs/
│   └── h12_abs/
├── whole_matrix/
│   └── ...
└── adj_neighbour_adj_mx/
    └── ...
```

### summary.csv columns

`dataset`, `mode`, `adjacency`, `window_strategy`, `context_length`, `horizon`,
`run`, `num_nodes`, `num_eval_windows`, `mae`, `rmse`, `mse`, `mape`, `time_sec`

### predictions.npz contents

- `predictions`: shape `[num_eval_windows, num_nodes, horizon]`
- `ground_truth`: shape `[num_eval_windows, num_nodes, horizon]`
- `context_length`: scalar

## Config File Reference

```yaml
dataset: "../data/PEMS-BAY/"           # Path to dataset folder
model: "amazon/chronos-2"             # Chronos model name
device: "cuda"                         # "cuda" or "cpu"
dtype: "bfloat16"                      # "bfloat16", "float16", or "float32"
horizons: [3, 6, 12]                   # Prediction horizons
context_length: 288                    # Context window size
window_strategy: "absolute"            # "absolute" or "divided"
modes: ["single_node", "node_batches"] # Modes to run
adjacency_files: ["data/PEMS-BAY/adj_mx.pkl"]

# Batch options (for node_batches mode)
node_batches: [[0,1,2], [3,4,5]]       # Explicit batches
# node_batches_file: "batches.json"    # Or path to JSON file
# batch_size: 50                       # Or auto-partition size

test_ratio: null                       # null = read from desc.json
num_runs: 1                            # Number of repeated runs
output_dir: "../results/"              # Output directory
```
