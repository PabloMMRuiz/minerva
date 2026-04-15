import nbformat as nbf

nb = nbf.v4.new_notebook()

text_cells = [
    "# Dataset Exploration\nThis notebook provides modular tools to analyze and visualize the time series datasets. We can study the full dataset, detect outliers (such as 0 variance series), and visualize specific slices.",
    "## Configuration & Loading"
]

code_cells = [
    "import sys\nimport os\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Add src to path\nsys.path.append(os.path.abspath(os.path.join('..')))\n\nfrom src.data.loaders import load_dataset\nfrom src.utils.visualization import plot_node_time_series, plot_time_series_decomposition\nfrom src.utils.graph_metrics import calculate_graph_metrics\n\nplt.style.use('seaborn-v0_8-whitegrid')",
    
    "# --- CONFIGURATION ---\nDATASET_NAME = 'PEMS07'  # Try 'PEMS04', 'PEMS08', etc.\nDATA_DIR = os.path.join('..', 'data', DATASET_NAME)\nFEATURE_IDX = 0  # We usually focus on the first feature\n\nprint(f'Loading dataset: {DATASET_NAME}')\ndataset_dict = load_dataset(DATA_DIR, load_adj_matrix=True)\ndata = dataset_dict.get('data')\nmetadata = dataset_dict.get('metadata')\nadj = dataset_dict.get('adj_raw')\n\nprint(f'\\nData shape: {data.shape} [Time Steps, Nodes, Features]')\nL, N, C = data.shape",

    "## 1. Outlier & Variance Detection\ndef detect_zero_variance_nodes(data_array, feature_idx=0, time_slice=None, std_threshold=1e-5):\n    \"\"\"\n    Detect nodes that have zero or near-zero variance in a given time slice.\n    \"\"\"\n    if time_slice is None:\n        slice_data = data_array[:, :, feature_idx]\n    else:\n        start, end = time_slice\n        slice_data = data_array[start:end, :, feature_idx]\n        \n    std_devs = np.std(slice_data, axis=0)\n    zero_var_nodes = np.where(std_devs <= std_threshold)[0]\n    \n    print(f'Found {len(zero_var_nodes)} nodes with ~0 variance.')\n    for idx in zero_var_nodes:\n        print(f'Node {idx}: std = {std_devs[idx]:.6f}')\n    \n    return zero_var_nodes, std_devs\n\n# Example: Check the first 1000 time steps\nprint('Checking for zero-variance nodes in the first 1000 time steps...')\nzero_nodes, stds = detect_zero_variance_nodes(data, feature_idx=FEATURE_IDX, time_slice=(0, 1000))\n",

    "## 2. Basic Dataset Statistics\ndef analyze_dataset_stats(data_array, feature_idx=0):\n    flat_data = data_array[:, :, feature_idx].flatten()\n    \n    print('--- Global Statistics ---')\n    print(f'Mean: {np.mean(flat_data):.4f}')\n    print(f'Std : {np.std(flat_data):.4f}')\n    print(f'Min : {np.min(flat_data):.4f}')\n    print(f'Max : {np.max(flat_data):.4f}')\n    print(f'0.1% percentile: {np.percentile(flat_data, 0.1):.4f}')\n    print(f'99.9% percentile: {np.percentile(flat_data, 99.9):.4f}')\n    \n    plt.figure(figsize=(10, 5))\n    sns.histplot(flat_data, bins=50, kde=True)\n    plt.title('Global Distribution of Feature Values')\n    plt.xlabel('Value')\n    plt.ylabel('Frequency')\n    plt.show()\n\nanalyze_dataset_stats(data, feature_idx=FEATURE_IDX)",

    "## 3. Node Visualization\n# Visualize a normal node and an outlier node if any\nnode_to_plot = 0\nif len(zero_nodes) > 0:\n    node_to_plot = zero_nodes[0]\n    print(f'Plotting outlier node: {node_to_plot}')\nelse:\n    print(f'Plotting normal node: {node_to_plot}')\n\nplot_node_time_series(data, node_index=node_to_plot, feature_index=FEATURE_IDX, show=True)\nplot_node_time_series(data, node_index=node_to_plot, feature_index=FEATURE_IDX, x_0=0, x_1=1000, title=f'Node {node_to_plot} (First 1000 Steps)')\n",

    "## 4. Covariance & Correlation Study\ndef study_correlation(data_array, feature_idx=0, nodes_subset=None):\n    \"\"\"\n    Study the correlation/covariance structure of a subset of nodes\n    \"\"\"\n    if nodes_subset is None:\n        # Just take first 50 nodes if not specified, to avoid huge matrices\n        nodes_subset = np.arange(min(50, data_array.shape[1]))\n        \n    subset_data = data_array[:, nodes_subset, feature_idx]\n    \n    # Correlation Matrix\n    corr_matrix = np.corrcoef(subset_data.T)\n    \n    plt.figure(figsize=(10, 8))\n    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)\n    plt.title(f'Correlation Matrix (Subset of {len(nodes_subset)} nodes)')\n    plt.show()\n    \n    return corr_matrix\n    \n# Let\\'s look at correlation of the first 50 nodes\ncorr_mx = study_correlation(data, feature_idx=FEATURE_IDX)",
    
    "## 5. Adjacency Matrix Analysis\nif adj is not None:\n    print('--- Adjacency Matrix Structure ---')\n    # Calculate basic graph metrics if adj is present\n    try:\n        metrics = calculate_graph_metrics(adj)\n        for k, v in metrics.items():\n            print(f'{k}: {v}')\n    except Exception as e:\n        print(f'Failed to compute graph metrics: {e}')\nelse:\n    print('No adjacency matrix found for this dataset.')"
]

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_cells[0]),
    nbf.v4.new_code_cell(code_cells[0]),
    nbf.v4.new_markdown_cell(text_cells[1]),
    nbf.v4.new_code_cell(code_cells[1]),
    nbf.v4.new_code_cell(code_cells[2]),
    nbf.v4.new_code_cell(code_cells[3]),
    nbf.v4.new_code_cell(code_cells[4]),
    nbf.v4.new_code_cell(code_cells[5]),
    nbf.v4.new_code_cell(code_cells[6])
]

with open('c:/Work/Minerva/gnns/matrix-experimentation/notebooks/dataset_explorer.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Notebook generated successfully.')
