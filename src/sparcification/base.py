"""
Base classes for sparsification and normalization.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class Sparsifier(ABC):
    """
    Abstract base class for matrix sparsification methods.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params if params is not None else {}

    @abstractmethod
    def get_mask(self, matrix: np.ndarray, effective_k: int) -> np.ndarray:
        """
        Generate a boolean mask for the matrix.

        Args:
            matrix: The N x N input matrix.
            effective_k: The target number of edges.

        Returns:
            Boolean mask (N x N) where True indicates selected edges.
        """
        pass

    def apply(self, matrix: np.ndarray, effective_k: int) -> np.ndarray:
        """
        Apply the sparsification mask to the matrix.
        """
        mask = self.get_mask(matrix, effective_k)
        return np.where(mask, matrix, 0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"


class Normalizer(ABC):
    """
    Abstract base class for matrix normalization methods.
    """

    def __init__(self, name: str, norm_strength: float = 1.0):
        self.name = name
        self.norm_strength = norm_strength

    @abstractmethod
    def normalize(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize the matrix.

        Args:
            matrix: The N x N input matrix.

        Returns:
            Normalized matrix.
        """
        pass

    def __call__(self, matrix: np.ndarray) -> np.ndarray:
        return self.normalize(matrix)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(strength={self.norm_strength})"
