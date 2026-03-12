"""
Base class for hybrid matrix fusion methods.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class HybridFuser(ABC):
    """
    Abstract base class for matrix fusion methods.

    A HybridFuser takes a list of similarity/adjacency matrices
    (each [N, N]) and produces a single fused matrix [N, N].
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params if params is not None else {}

    @abstractmethod
    def fuse(self, matrices: List[np.ndarray]) -> np.ndarray:
        """
        Fuse multiple matrices into one.

        Args:
            matrices: List of [N, N] numpy arrays. All must share the same shape.

        Returns:
            A single fused [N, N] numpy array.
        """
        pass

    def _validate_inputs(self, matrices: List[np.ndarray]):
        if len(matrices) < 2:
            raise ValueError(f"Need at least 2 matrices to fuse, got {len(matrices)}")
        shape = matrices[0].shape
        if shape[0] != shape[1]:
            raise ValueError(f"Matrices must be square, got {shape}")
        for i, m in enumerate(matrices):
            if m.shape != shape:
                raise ValueError(
                    f"Shape mismatch: matrix 0 is {shape}, matrix {i} is {m.shape}"
                )

    def __repr__(self) -> str:
        if self.params:
            param_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.name}({param_str})"
        return self.name

    def __call__(self, matrices: List[np.ndarray]) -> np.ndarray:
        return self.fuse(matrices)
