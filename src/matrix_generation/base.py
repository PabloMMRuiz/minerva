"""
Base class for adjacency matrix generation methods.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class MatrixGenerator(ABC):
    """
    Abstract base class for matrix generation methods.

    All matrix generators should inherit from this class and implement
    the generate() method.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize matrix generator.

        Args:
            name: Name of the generator method
            params: Dictionary of parameters for the generator
        """
        self.name = name
        self.params = params if params is not None else {}

    @abstractmethod
    def generate(self, time_series: np.ndarray) -> np.ndarray:
        """
        Generate similarity/adjacency matrix from time series.

        Args:
            time_series: Time series data with shape [L, N] or [L, N, C]
                        where L=timesteps, N=nodes, C=features

        Returns:
            Adjacency matrix with shape [N, N]
        """
        pass

    def __repr__(self) -> str:
        """String representation of the generator."""
        if self.params:
            param_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.name}({param_str})"
        return self.name

    def __call__(self, time_series: np.ndarray) -> np.ndarray:
        """Allow generator to be called directly."""
        return self.generate(time_series)
