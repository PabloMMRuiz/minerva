"""
Data loading and preprocessing module.
"""

from .loaders import (
    load_dataset_description,
    print_dataset_description,
    load_dataset_as_memmap,
    load_pkl,
    load_adj,
    save_pkl,
    load_dataset,
    load_gaussian_data,
    load_sinusoidal_data
)

from .preprocessing import (
    calculate_scaled_laplacian,
    calculate_symmetric_normalized_laplacian,
    calculate_symmetric_message_passing_adj,
    calculate_transition_matrix,
    add_self_loops,
    remove_self_loops,
    symmetrize_adjacency,
    reshape_time_series_2_d,
)

__all__ = [
    # Loaders
    'load_dataset_description',
    'print_dataset_description',
    'load_dataset_as_memmap',
    'load_pkl',
    'load_adj',
    'load_dataset',
    'load_gaussian_data',
    'load_sinusoidal_data',
    # Preprocessing
    'calculate_scaled_laplacian',
    'calculate_symmetric_normalized_laplacian',
    'calculate_symmetric_message_passing_adj',
    'calculate_transition_matrix',
    'add_self_loops',
    'remove_self_loops',
    'symmetrize_adjacency',
    'reshape_time_series_2_d',
    # move this out of here
    'save_pkl'
]