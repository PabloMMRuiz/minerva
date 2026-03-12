"""
Embedding-based Cosine Similarity matrix generator.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from .base import MatrixGenerator
from data.preprocessing import reshape_time_series_2_d


class DilatedCNNEncoder(nn.Module):
    """Simple Dilated CNN encoder for time series embeddings."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=4, dilation=4)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [N, C, L]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.adaptive_pool(x) # [N, output_dim, 1]
        return x.squeeze(-1)


class EmbeddingCosineGenerator(MatrixGenerator):
    """
    Calculates cosine similarity between learned embeddings of time series.
    Uses a simple Dilated CNN as the encoder.
    """

    def __init__(
        self, 
        hidden_dim: int = 32, 
        embedding_dim: int = 64, 
        feature_index: int = 0, 
        params: Optional[Dict[str, Any]] = None
    ):
        super().__init__("embedding_cosine", params)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.feature_index = feature_index

    def generate(self, data: np.ndarray) -> np.ndarray:
        """
        Input: [L, N, C]
        """
        # x: [L, N] -> [N, 1, L] for Conv1d
        x_raw = reshape_time_series_2_d(data, self.feature_index)
        L, N = x_raw.shape
        x_tensor = torch.from_numpy(x_raw.T).float().unsqueeze(1)
        
        # Initialize encoder (deterministic for this comparison purpose or random)
        # Note: In a real scenario, this might be pre-trained.
        encoder = DilatedCNNEncoder(input_dim=1, hidden_dim=self.hidden_dim, output_dim=self.embedding_dim)
        encoder.eval()
        
        with torch.no_grad():
            embeddings = encoder(x_tensor) # [N, embedding_dim]
            
        # Compute Cosine Similarity
        norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        embeddings_normalized = embeddings / (norm + 1e-8)
        cosine_sim = torch.mm(embeddings_normalized, embeddings_normalized.t())
        
        return cosine_sim.numpy().astype(np.float32)
