"""
Data loading utilities for time series datasets.

This module handles loading of:
- Dataset metadata (desc.json)
- Time series data (memmap files)
- Adjacency matrices (pickle files)
"""

import os
import json
import pickle
import numpy as np
from typing import Tuple, Optional, Dict, Any, List


# ------------------------------------------------------
# METADATA LOADING

def load_dataset_description(folder_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads metadata from the 'desc.json' file in the specified folder.

    Args:
        folder_path: Path to the folder containing the 'desc.json' file.

    Returns:
        Dictionary containing dataset metadata, or None if loading fails.
    """
    json_path = os.path.join(folder_path, 'desc.json')
    if not os.path.exists(json_path):
        print(f"Error: desc.json not found in the folder '{folder_path}'")
        return None

    try:
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        return metadata

    except json.JSONDecodeError:
        print(f"Error: Could not decode the JSON file at '{json_path}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def print_dataset_description(folder_path: str) -> None:
    """
    Loads and prints key metadata from the 'desc.json' file.

    Args:
        folder_path: Path to the folder containing the 'desc.json' file.
    """
    metadata = load_dataset_description(folder_path)

    if metadata is None:
        return

    print("--- Dataset Description ---")

    # General Information
    print(f"Dataset Name: {metadata.get('name', 'N/A')}")
    print(f"Domain: {metadata.get('domain', 'N/A')}")
    shape = metadata.get('shape', ['N/A', 'N/A', 'N/A'])
    print(f"Number of Time Slices: {shape[0]}")
    print(f"Number of Nodes (Time Series): {shape[1]}")
    print(f"Number of features per Node: {shape[2]}")

    print("\n--- Feature and Structure ---")

    # Feature Descriptions
    features = metadata.get('feature_description', 'N/A')
    features_str = ', '.join(features) if isinstance(
        features, list) else features
    print(f"Features description: {features_str}")
    print(
        f"Time step length: {metadata.get('frequency (minutes)', 'N/A')} minutes")
    print(
        f"Prior Graph Structures: {'Present' if metadata.get('has_graph') else 'Not Present'}")

    print("\n--- Model Settings ---")

    # Regular Settings
    settings = metadata.get('regular_settings', {})
    print(f"Input Sequence Length: {settings.get('INPUT_LEN', 'N/A')}")
    print(f"Output Sequence Length: {settings.get('OUTPUT_LEN', 'N/A')}")

    # Data Splits
    ratios = settings.get('TRAIN_VAL_TEST_RATIO', ['N/A', 'N/A', 'N/A'])
    print(f"Training Data Ratio: {ratios[0]}")
    print(f"Validation Data Ratio: {ratios[1]}")
    print(f"Test Data Ratio: {ratios[2]}")

    # Normalization and Evaluation
    print(
        f"Individual Channel Normalization: {'Yes' if settings.get('NORM_EACH_CHANNEL', False) else 'No'}")
    print(
        f"Renormalize during Evaluation: {'Yes' if settings.get('RESCALE', False) else 'No'}")
    metrics = settings.get('METRICS', 'N/A')
    metrics_str = ', '.join(metrics) if isinstance(metrics, list) else metrics
    print(f"Evaluation Metrics: {metrics_str}")
    print(f"Outlier Handling: {settings.get('NULL_VAL', 'N/A')}")


# --------------------------------------------
# TIME SERIES DATA LOADING


def load_dataset_as_memmap(
    folder_path: str,
    data_file: str = 'data.dat'
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Loads the raw time series data from a numpy.memmap file.

    Args:
        folder_path: Path to the folder containing the data files.
        data_file: Name of the numpy.memmap file.

    Returns:
        Tuple of (data array, success flag)
        - data array: numpy.memmap object or None if failed
        - success flag: True if loaded successfully, False otherwise
    """
    json_path = os.path.join(folder_path, 'desc.json')
    data_path = os.path.join(folder_path, data_file)

    # Check if files exist
    if not os.path.exists(json_path):
        print(f"Error: desc.json not found at '{json_path}'")
        return None, False

    if not os.path.exists(data_path):
        print(f"Error: {data_file} not found at '{data_path}'")
        return None, False

    try:
        # Load metadata to get shape
        metadata = load_dataset_description(folder_path)

        if metadata is None:
            return None, False

        shape = metadata.get('shape')

        if shape is None:
            print("Error: 'shape' not found in desc.json. Cannot load data.")
            return None, False

        print(f"Attempting to load data with shape {shape}...")

        # Load the memmap file
        data = np.memmap(
            data_path,
            dtype='float32',
            mode='r',
            shape=tuple(shape)
        )
        print("Data loaded successfully as a numpy.memmap object.")
        print(f"Data shape: {data.shape}")

        return data, True

    except Exception as e:
        print(f"An unexpected error occurred while loading the data: {e}")
        return None, False


# ---------------------------------------
# ADJACENCY MATRIX LOADING

def load_pkl(pickle_file: str) -> Any:
    """
    Load data from a pickle file.

    Args:
        pickle_file: Path to the pickle file.

    Returns:
        Loaded object from the pickle file.

    Raises:
        Exception: If unable to load the pickle file.
    """
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f'Unable to load data from {pickle_file}: {e}')
        raise
    return pickle_data


def save_pkl(data_object: object, output_file: str):
    """
    Saves a Python object to a pickle file.

    Args:
        data_object: The object to be saved.
        output_file: The destination path for the pickle file.
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(data_object, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Object successfully saved to {output_file}")
    except Exception as e:
        print(f'Error: Unable to save data to {output_file}: {e}')
        raise


def load_adj(
    file_path: str,
    adj_type: str = 'original'
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load and preprocess an adjacency matrix.

    Args:
        file_path: Path to the pkl file.
        adj_type: Preprocessing type (original, scalap, normlap, symnadj, transition, doubletransition).

    Returns:
        Tuple of (processed list, raw matrix)
    """
    try:
        data = load_pkl(file_path)
        if isinstance(data, (list, tuple)) and len(data) == 3:
            adj_mx = data[2]
        else:
            adj_mx = data
    except Exception as e:
        print(f"Error reading adjacency from {file_path}: {e}")
        raise

    from .preprocessing import (
        calculate_scaled_laplacian,
        calculate_symmetric_normalized_laplacian,
        calculate_symmetric_message_passing_adj,
        calculate_transition_matrix
    )

    TRANSFORMS = {
        'original': lambda x: [x],
        'scalap': lambda x: [calculate_scaled_laplacian(x).astype(np.float32).todense()],
        'normlap': lambda x: [calculate_symmetric_normalized_laplacian(x).astype(np.float32).todense()],
        'symnadj': lambda x: [calculate_symmetric_message_passing_adj(x).astype(np.float32).todense()],
        'transition': lambda x: [calculate_transition_matrix(x).T],
        'doubletransition': lambda x: [calculate_transition_matrix(x).T, calculate_transition_matrix(x.T).T],
        'identity': lambda x: [np.diag(np.ones(x.shape[0])).astype(np.float32)]
    }

    if adj_type not in TRANSFORMS:
        raise ValueError(f"Unknown adj_type: {adj_type}. Choices: {list(TRANSFORMS.keys())}")

    return TRANSFORMS[adj_type](adj_mx), adj_mx


# --------------------------------------------------
# CONVENIENCE FUNCTIONS

def load_dataset(
    folder_path: str,
    load_adj_matrix: bool = True,
    adj_file: str = 'adj_mx.pkl',
    adj_type: str = 'original',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to load complete dataset.

    Args:
        folder_path: Path to dataset folder
        load_adj_matrix: Whether to load adjacency matrix
        adj_file: Name of adjacency matrix pickle file
        adj_type: Type of adjacency preprocessing
        verbose: Whether to print dataset description

    Returns:
        Dictionary containing:
        - 'data': Time series data (memmap)
        - 'metadata': Dataset metadata
        - 'adj': Processed adjacency matrix (if load_adj_matrix=True)
        - 'adj_raw': Raw adjacency matrix (if load_adj_matrix=True)
    """
    result = {}

    # Load metadata
    metadata = load_dataset_description(folder_path)
    result['metadata'] = metadata

    if verbose and metadata is not None:
        print_dataset_description(folder_path)

    # Load time series data
    data, success = load_dataset_as_memmap(folder_path)
    if success:
        result['data'] = data
    else:
        print("Warning: Failed to load time series data")

    # Load adjacency matrix if requested
    if load_adj_matrix:
        adj_path = os.path.join(folder_path, adj_file)
        if os.path.exists(adj_path):
            try:
                adj, adj_raw = load_adj(adj_path, adj_type)
                result['adj'] = adj
                result['adj_raw'] = adj_raw
                print(f"Loaded adjacency matrix with shape {adj_raw.shape}")
            except Exception as e:
                print(f"Warning: Failed to load adjacency matrix: {e}")
        else:
            print(f"Warning: Adjacency file not found at {adj_path}")

    return result


def load_gaussian_data(
    n_nodes: int = 10,
    n_timesteps: int = 100,
    n_features: int = 1,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generates synthetic Gaussian time series data.

    Returns:
        Tuple of (data, metadata)
    """
    np.random.seed(seed)
    data = np.random.randn(n_timesteps, n_nodes, n_features).astype(np.float32)
    metadata = {
        'name': 'Gaussian Synthetic',
        'shape': [n_timesteps, n_nodes, n_features],
        'domain': 'Synthetic',
        'has_graph': False
    }
    return data, metadata


def load_sinusoidal_data(
    n_nodes: int = 10,
    n_timesteps: int = 200,
    n_features: int = 1,
    noise_level: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generates synthetic sinusoidal data with some dependencies.
    Node 1 is a delayed version of Node 0 (causality).
    Node 2 is a non-linear transformation of Node 0.

    Returns:
        Tuple of (data, metadata)
    """
    np.random.seed(seed)
    t = np.linspace(0, 10, n_timesteps)
    data = np.zeros((n_timesteps, n_nodes, n_features), dtype=np.float32)

    # Base signals
    for i in range(n_nodes):
        freq = 1.0 + 0.5 * np.random.rand()
        phase = 2 * np.pi * np.random.rand()
        data[:, i, 0] = np.sin(freq * t + phase)

    # Inject dependencies for nodes 0, 1, 2 if they exist
    if n_nodes >= 3:
        data[:, 0, 0] = np.sin(t)
        data[:, 1, 0] = np.sin(t - 0.5)  # Lagged (Granger causality)
        data[:, 2, 0] = data[:, 0, 0]**2 # Non-linear dependency

    data += np.random.normal(0, noise_level, data.shape)

    metadata = {
        'name': 'Sinusoidal Synthetic',
        'shape': [n_timesteps, n_nodes, n_features],
        'domain': 'Synthetic',
        'has_graph': False,
        'description': 'Node 1 lagged Node 0; Node 2 non-linear on Node 0'
    }
    return data, metadata
