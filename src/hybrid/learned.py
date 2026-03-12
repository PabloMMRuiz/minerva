"""
Learned Attention Fusion for hybrid matrix construction.

╔═══════════════════════════════════════════════════════════════════════╗
║  WARNING: This module requires integration with a GNN training loop. ║
║  The AttentionFuser is a PyTorch nn.Module that learns per-edge      ║
║  attention weights across multiple matrix views. It CANNOT be used   ║
║  standalone — it must be embedded in a model and trained end-to-end  ║
║  with a downstream loss (e.g., forecasting MSE).                     ║
║                                                                      ║
║  Usage pattern:                                                      ║
║    1. Generate K similarity matrices (views).                        ║
║    2. Pass them through AttentionFuser.forward() to get a fused adj. ║
║    3. Feed the fused adj into a GNN.                                 ║
║    4. Backpropagate through the entire pipeline.                     ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class AttentionFuser(nn.Module):
    """
    Learned per-edge attention fusion across K matrix views.

    Given K input matrices of shape [N, N], this module learns to weight
    each view's contribution at every edge (i, j) independently.

    Architecture:
        For each edge (i, j), the K similarity values are stacked into a
        vector of length K, passed through a small MLP, and a softmax
        produces attention weights. The fused value is the weighted sum.

    WARNING: This is a trainable module. It must be used inside a PyTorch
    training loop with a downstream loss signal. Calling it without
    training will produce random (untrained) attention weights.
    """

    def __init__(self, n_views: int, hidden_dim: int = 16):
        """
        Args:
            n_views: Number of input matrices (K).
            hidden_dim: Hidden dimension of the per-edge MLP.
        """
        super().__init__()
        self.n_views = n_views
        self.attention_mlp = nn.Sequential(
            nn.Linear(n_views, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_views)
        )

    def forward(self, matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse K matrices using learned attention.

        Args:
            matrices: List of K tensors, each [N, N].

        Returns:
            Fused [N, N] tensor.
        """
        if len(matrices) != self.n_views:
            raise ValueError(
                f"Expected {self.n_views} views, got {len(matrices)}"
            )

        # Stack: [K, N, N] -> permute to [N, N, K]
        stacked = torch.stack(matrices, dim=0)  # [K, N, N]
        stacked = stacked.permute(1, 2, 0)      # [N, N, K]

        # Per-edge attention weights
        attn_logits = self.attention_mlp(stacked)  # [N, N, K]
        attn_weights = F.softmax(attn_logits, dim=-1)  # [N, N, K]

        # Weighted sum
        fused = (attn_weights * stacked).sum(dim=-1)  # [N, N]

        return fused

    def fuse_numpy(self, matrices: List[np.ndarray]) -> np.ndarray:
        """
        Convenience method for inference (no gradient).
        Accepts numpy arrays, returns numpy.

        WARNING: Only meaningful after the module has been trained.
        """
        tensors = [torch.from_numpy(m).float() for m in matrices]
        with torch.no_grad():
            result = self.forward(tensors)
        return result.numpy()

    def get_attention_map(self, matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Return the raw attention weights for visualization/analysis.

        Returns:
            [N, N, K] tensor of attention weights (summing to 1 along K).
        """
        stacked = torch.stack(matrices, dim=0).permute(1, 2, 0)
        attn_logits = self.attention_mlp(stacked)
        return F.softmax(attn_logits, dim=-1)
