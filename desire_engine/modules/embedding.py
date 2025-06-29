"""
Ilanya Desire Engine - Embedding Module

Multi-modal desire embedding system using neural networks.
Creates rich representations of desires from multiple modalities.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np

from ..models import Desire
from ..config import DesireEngineConfig


class DesireEmbeddingModule(nn.Module):
    """
    Multi-modal desire embedding module.
    
    Creates rich embeddings from multiple modalities:
    - Semantic embeddings (desire name and description)
    - Numerical embeddings (strength, reinforcement count)
    - Temporal embeddings (creation time, last reinforcement)
    - Trait embeddings (source traits)
    - State embeddings (desire state)
    """
    
    def __init__(self, config: DesireEngineConfig):
        super().__init__()
        self.config = config
        
        # Semantic embedding layers
        self.semantic_embedding = nn.Embedding(1000, config.desire_dim // 4)  # Simplified for demo
        
        # Numerical embedding layers
        self.strength_embedding = nn.Linear(1, config.desire_dim // 8)
        self.reinforcement_embedding = nn.Linear(1, config.desire_dim // 8)
        self.goal_potential_embedding = nn.Linear(1, config.desire_dim // 8)
        
        # Trait embedding layers
        self.trait_embedding = nn.Embedding(100, config.desire_dim // 4)  # Simplified for demo
        
        # State embedding layers
        self.state_embedding = nn.Embedding(5, config.desire_dim // 8)  # 5 possible states
        
        # Calculate total embedding size
        semantic_size = config.desire_dim // 4  # 16
        numerical_size = (config.desire_dim // 8) * 3  # 8 * 3 = 24
        trait_size = config.desire_dim // 4  # 16
        state_size = config.desire_dim // 8  # 8
        total_embedding_size = semantic_size + numerical_size + trait_size + state_size  # 16 + 24 + 16 + 8 = 64
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(total_embedding_size, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.desire_dim),
            nn.LayerNorm(config.desire_dim)
        )
        
    def forward(self, desires: List[Desire]) -> torch.Tensor:
        """Create embeddings for a list of desires."""
        if not desires:
            return torch.empty(0, self.config.desire_dim)
        
        # State mapping for embedding
        state_mapping = {
            "active": 0,
            "reinforced": 1,
            "goal_candidate": 2,
            "weakening": 3,
            "pruned": 4
        }
        
        embeddings = []
        
        for desire in desires:
            # Semantic embedding (simplified - using hash of name)
            name_hash = hash(desire.name) % 1000
            semantic_emb = self.semantic_embedding(torch.tensor(name_hash, dtype=torch.long))
            
            # Numerical embeddings
            strength_emb = self.strength_embedding(torch.tensor([[desire.strength]], dtype=torch.float32))
            reinforcement_emb = self.reinforcement_embedding(torch.tensor([[desire.reinforcement_count]], dtype=torch.float32))
            goal_potential_emb = self.goal_potential_embedding(torch.tensor([[desire.goal_potential]], dtype=torch.float32))
            
            # Trait embedding (simplified - using first trait)
            if desire.source_traits:
                trait_hash = hash(desire.source_traits[0]) % 100
                trait_emb = self.trait_embedding(torch.tensor(trait_hash, dtype=torch.long))
            else:
                trait_emb = self.trait_embedding(torch.tensor(0, dtype=torch.long))
            
            # State embedding
            state_idx = state_mapping.get(desire.state.value, 0)  # Default to active if unknown
            state_emb = self.state_embedding(torch.tensor(state_idx, dtype=torch.long))
            
            # Concatenate all embeddings
            combined_emb = torch.cat([
                semantic_emb,
                strength_emb.squeeze(),
                reinforcement_emb.squeeze(),
                goal_potential_emb.squeeze(),
                trait_emb,
                state_emb
            ])
            
            # Pass through fusion network
            final_emb = self.fusion_network(combined_emb)
            embeddings.append(final_emb)
        
        return torch.stack(embeddings) 