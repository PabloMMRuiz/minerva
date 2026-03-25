from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

class NodeBatcher(ABC):
    """
    Abstract base class for all node batching (grouping/clustering) algorithms.
    
    A NodeBatcher takes an adjacency matrix as input and returns a list of batches,
    where each batch is a list of node indices. This makes the batching logic 
    independent of how the graph was generated (e.g., Pearson, DTW, Hybrid).
    """

    def __init__(self, **kwargs):
        """
        Initialize batching parameters specific to the algorithm.
        """
        pass

    @abstractmethod
    def batch(self, adj_matrix: np.ndarray) -> List[List[int]]:
        """
        Divide the nodes of the graph into batches based on the algorithm.
        
        Args:
            adj_matrix (np.ndarray): A square adjacency matrix (N x N) representing
                                     the graph structure or similarities.
                                     
        Returns:
            List[List[int]]: A list of batches. Each batch is a list of integer 
                             node indices.
        """
        pass

    def _validate_input(self, adj_matrix: np.ndarray):
        """
        Validates the input adjacency matrix.
        """
        if not isinstance(adj_matrix, np.ndarray):
            raise TypeError("Adjacency matrix must be a numpy array.")
            
        if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError(f"Adjacency matrix must be 2D square. Got shape {adj_matrix.shape}")
