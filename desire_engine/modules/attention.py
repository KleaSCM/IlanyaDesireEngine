"""
Ilanya Desire Engine - Attention Module

Transformer-style attention mechanism for desire space modeling.
Provides contextual awareness and importance weighting.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ..config import DesireEngineConfig


class AttentionModule(nn.Module):
    """
    Multi-head attention module for desire space modeling.
    
    Provides:
    - Self-attention between desires
    - Contextual awareness
    - Importance weighting
    - Relationship modeling
    """
    
    def __init__(self, config: DesireEngineConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=config.desire_dim,
            num_heads=config.num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(config.desire_dim)
        self.layer_norm2 = nn.LayerNorm(config.desire_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.desire_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.desire_dim)
        )
        
    def forward(self, desire_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention to desire embeddings."""
        if desire_embeddings.size(0) == 0:
            return desire_embeddings, torch.empty(0, 0)
        
        # Self-attention
        attn_output, attention_weights = self.multihead_attn(
            desire_embeddings, desire_embeddings, desire_embeddings
        )
        
        # Residual connection and layer norm
        attn_output = self.layer_norm1(desire_embeddings + attn_output)
        
        # Feed-forward network
        ff_output = self.feed_forward(attn_output)
        
        # Residual connection and layer norm
        output = self.layer_norm2(attn_output + ff_output)
        
        return output, attention_weights 