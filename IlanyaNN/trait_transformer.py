"""
Ilanya Trait Engine - Trait Transformer Neural Network

Main neural network architecture for trait processing using transformer-based models.
Implements TraitTransformer, TraitEmbedding, MultiHeadTraitAttention, and
PositionalEncoding for processing trait relationships and generating evolution signals.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class TraitTransformerConfig:
    
    # Configuration for the Trait Transformer.
    
    # Contains all hyperparameters and architectural choices for the trait
    # transformer neural network. This allows for easy experimentation and
    # configuration management.
    
    
    input_dim: int = 512                    # Input dimension for trait data
    hidden_dim: int = 1024                  # Hidden layer dimension
    num_layers: int = 6                     # Number of transformer layers
    num_heads: int = 8                      # Number of attention heads
    dropout: float = 0.1                    # Dropout rate for regularization
    max_seq_length: int = 1000              # Maximum sequence length
    num_traits: int = 20                    # Number of trait types
    trait_embedding_dim: int = 64           # Trait embedding dimension
    use_positional_encoding: bool = True    # Whether to use positional encoding
    activation: str = "gelu"                # Activation function (gelu/relu)


class TraitEmbedding(nn.Module):
    
    # Embedding layer for trait representations.
    
    # Converts trait data (type, value, confidence) into high-dimensional
    # embeddings suitable for transformer processing. Combines trait type
    # embeddings with value/confidence projections.
    
    
    def __init__(self, num_traits: int, embedding_dim: int, input_dim: int):
        
        # Initialize the trait embedding layer.
        
        # Args:
        #     num_traits: Number of different trait types
        #     embedding_dim: Dimension of the output embeddings
        #     input_dim: Input dimension (unused, kept for compatibility)
        
        super().__init__()
        self.num_traits = num_traits
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        
        # Trait type embeddings - learnable representations for each trait type
        self.trait_embeddings = nn.Embedding(num_traits, embedding_dim)
        
        # Value and confidence projections - converts [value, confidence] to embedding
        self.value_projection = nn.Linear(2, embedding_dim)
        
        # Combined projection - merges trait type and value embeddings
        self.combined_projection = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Layer normalization - stabilizes training
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, trait_values: torch.Tensor, trait_confidences: torch.Tensor, 
                trait_indices: torch.Tensor) -> torch.Tensor:
        
        # Forward pass for trait embedding.
        
        # Args:
        #     trait_values: Tensor of shape (batch_size, num_traits) - trait strength values
        #     trait_confidences: Tensor of shape (batch_size, num_traits) - confidence scores
        #     trait_indices: Tensor of shape (batch_size, num_traits) - trait type indices
            
        # Returns:
        #     Embedded traits of shape (batch_size, num_traits, embedding_dim)
        
        batch_size, num_traits = trait_values.shape
        
        # Get trait type embeddings from learned embedding table
        trait_type_embeddings = self.trait_embeddings(trait_indices)  # (batch_size, num_traits, embedding_dim)
        
        # Project values and confidences to embedding space
        value_conf = torch.stack([trait_values, trait_confidences], dim=-1)  # (batch_size, num_traits, 2)
        value_embeddings = self.value_projection(value_conf)  # (batch_size, num_traits, embedding_dim)
        
        # Combine trait type and value embeddings
        combined = torch.cat([trait_type_embeddings, value_embeddings], dim=-1)  # (batch_size, num_traits, embedding_dim * 2)
        embedded = self.combined_projection(combined)  # (batch_size, num_traits, embedding_dim)
        
        # Apply layer normalization for training stability
        embedded = self.layer_norm(embedded)
        
        return embedded


class PositionalEncoding(nn.Module):
    
    # Positional encoding for sequence position information.
    
    # Adds position information to embeddings using sinusoidal functions.
    # This allows the transformer to understand the order of traits in the sequence.
    
    
    def __init__(self, embedding_dim: int, max_seq_length: int = 1000):
        
        # Initialize positional encoding.
        
        # Args:
        #     embedding_dim: Dimension of the embeddings
        #     max_seq_length: Maximum sequence length to support
        
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Create positional encoding matrix using sinusoidal functions
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           (-math.log(10000.0) / embedding_dim))
        
        # Apply sinusoidal functions to create position encodings
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0).transpose(0, 1)  # Reshape for broadcasting
        
        # Register as buffer (not a parameter, but part of the model state)
        self.register_buffer('pe', pe)
        self.pe: torch.Tensor  # type: ignore
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        # Add positional encoding to input.
        
        # Args:
        #     x: Input tensor of shape (seq_len, batch_size, embedding_dim)
            
        # Returns:
        #     Input with positional encoding added
        
        return x + self.pe[:x.size(0), :]


class MultiHeadTraitAttention(nn.Module):
    
    # Multi-head attention mechanism for trait relationships.
    
    # Implements scaled dot-product attention with multiple heads to capture
    # different types of relationships between traits. This is the core mechanism
    # for understanding how traits influence each other.
    
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        
        # Initialize multi-head attention.
        
        # Args:
        #     embedding_dim: Dimension of input embeddings
        #     num_heads: Number of attention heads
        #     dropout: Dropout rate for attention weights
        
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads  # Dimension per attention head
        
        # Ensure embedding dimension is divisible by number of heads
        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"
        
        # Linear projections for query, key, value, and output
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)  # Scaling factor for attention scores
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Forward pass for multi-head attention.
        
        # Args:
        #     x: Input tensor of shape (batch_size, seq_len, embedding_dim)
        #     mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
            
        # Returns:
        #     Output tensor of shape (batch_size, seq_len, embedding_dim)
        
        batch_size, seq_len, embedding_dim = x.shape
        
        # Project input to query, key, value representations
        Q = self.query_projection(x)  # (batch_size, seq_len, embedding_dim)
        K = self.key_projection(x)    # (batch_size, seq_len, embedding_dim)
        V = self.value_projection(x)  # (batch_size, seq_len, embedding_dim)
        
        # Reshape for multi-head attention - split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores using scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply attention mask if provided (for masking future positions)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights and apply dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back to original format
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)
        
        # Final linear projection
        output = self.output_projection(context)
        
        return output


class TraitTransformerBlock(nn.Module):
    
    # Single transformer block for trait processing.
    
    # Combines multi-head attention with feed-forward network and residual
    # connections. This is the basic building block of the transformer architecture.
    
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1, activation: str = "gelu"):
        
        # Initialize transformer block.
        
        # Args:
        #     embedding_dim: Dimension of embeddings
        #     num_heads: Number of attention heads
        #     dropout: Dropout rate
        #     activation: Activation function for feed-forward network
        
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Multi-head self-attention with layer normalization
        self.attention = MultiHeadTraitAttention(embedding_dim, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(embedding_dim)
        
        # Feed-forward network with activation and dropout
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Expand dimension
            nn.GELU() if activation == "gelu" else nn.ReLU(),  # Activation
            nn.Dropout(dropout),  # Dropout
            nn.Linear(embedding_dim * 4, embedding_dim),  # Project back
            nn.Dropout(dropout)  # Dropout
        )
        self.ff_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Forward pass for transformer block.
        
        # Args:
        #     x: Input tensor of shape (batch_size, seq_len, embedding_dim)
        #     mask: Optional attention mask
            
        # Returns:
        #     Output tensor of shape (batch_size, seq_len, embedding_dim)
        # Self-attention with residual connection and layer normalization
        attn_output = self.attention(x, mask)
        x = self.attention_norm(x + attn_output)  # Residual connection
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.ff_norm(x + ff_output)  # Residual connection
        
        return x


class TraitTransformer(nn.Module):
    
    # Main trait transformer neural network.
    
    # Complete transformer architecture for processing trait data. Combines
    # embedding, positional encoding, multiple transformer blocks, and output
    # projections to generate trait predictions and evolution signals.
    # Includes network-level identity protection to preserve core identity traits.
    
    
    def __init__(self, config: TraitTransformerConfig):
        # Initialize trait transformer.
        
        # Args:
        #     config: Configuration object containing all hyperparameters
        
        super().__init__()
        self.config = config
        
        # Trait embedding layer - converts trait data to embeddings
        self.trait_embedding = TraitEmbedding(
            config.num_traits,
            config.trait_embedding_dim,
            config.input_dim
        )
        
        # Positional encoding - adds position information
        if config.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(config.trait_embedding_dim, config.max_seq_length)
        else:
            self.pos_encoding = None
        
        # Stack of transformer blocks - core processing layers
        self.transformer_blocks = nn.ModuleList([
            TraitTransformerBlock(
                config.trait_embedding_dim,
                config.num_heads,
                config.dropout,
                config.activation
            )
            for _ in range(config.num_layers)
        ])
        
        # Output projections for different tasks
        self.trait_output_projection = nn.Linear(config.trait_embedding_dim, 2)  # value + confidence
        self.evolution_output_projection = nn.Linear(config.trait_embedding_dim, 1)  # evolution signal
        self.interaction_output_projection = nn.Linear(config.trait_embedding_dim, config.num_traits)
        
        # IDENTITY PROTECTION: Identity preservation layer
        self.identity_preservation = nn.Linear(config.trait_embedding_dim, config.trait_embedding_dim)
        self.identity_gate = nn.Sigmoid()  # Gates identity preservation
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, trait_data: torch.Tensor, trait_indices: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Forward pass for trait transformer.
        
        # Args:
        #     trait_data: Tensor of shape (batch_size, num_traits, 2) with values and confidences
        #     trait_indices: Tensor of shape (batch_size, num_traits) with trait type indices
        #     mask: Optional attention mask
            
        # Returns:
        #     Dictionary containing:
        #     - trait_predictions: Predicted trait values and confidences
        #     - evolution_signals: Evolution signals for each trait
        #     - interaction_weights: Interaction weights between traits
        
        batch_size, num_traits, _ = trait_data.shape
        
        # Extract values and confidences from input tensor
        trait_values = trait_data[:, :, 0]  # (batch_size, num_traits)
        trait_confidences = trait_data[:, :, 1]  # (batch_size, num_traits)
        
        # Embed traits using the embedding layer
        embedded = self.trait_embedding(trait_values, trait_confidences, trait_indices)
        
        # Add positional encoding if enabled
        if self.pos_encoding is not None:
            embedded = embedded.transpose(0, 1)  # (num_traits, batch_size, embedding_dim)
            embedded = self.pos_encoding(embedded)
            embedded = embedded.transpose(0, 1)  # (batch_size, num_traits, embedding_dim)
        
        # Apply dropout for regularization
        embedded = self.dropout(embedded)
        
        # Pass through transformer blocks sequentially
        for block in self.transformer_blocks:
            embedded = block(embedded, mask)
        
        # IDENTITY PROTECTION: Apply identity preservation layer
        identity_preserved = self._apply_identity_protection(embedded, trait_indices)
        
        # Generate outputs using different projection layers
        trait_predictions = self.trait_output_projection(identity_preserved)  # (batch_size, num_traits, 2)
        evolution_signals = self.evolution_output_projection(identity_preserved).squeeze(-1)  # (batch_size, num_traits)
        interaction_weights = self.interaction_output_projection(identity_preserved)  # (batch_size, num_traits, num_traits)
        
        return {
            'trait_predictions': trait_predictions,
            'evolution_signals': evolution_signals,
            'interaction_weights': interaction_weights,
            'embeddings': identity_preserved
        }
    
    def get_trait_embeddings(self, trait_data: torch.Tensor, trait_indices: torch.Tensor) -> torch.Tensor:
        
        # Get trait embeddings without full forward pass.
        
        # Useful for extracting embeddings without computing all outputs.
        
        # Args:
        #     trait_data: Input trait data tensor
        #     trait_indices: Trait type indices
            
        # Returns:
        #     Trait embeddings tensor

        batch_size, num_traits, _ = trait_data.shape
        
        # Extract values and confidences
        trait_values = trait_data[:, :, 0]
        trait_confidences = trait_data[:, :, 1]
        
        # Get embeddings
        embedded = self.trait_embedding(trait_values, trait_confidences, trait_indices)
        
        # Add positional encoding if enabled
        if self.pos_encoding is not None:
            embedded = embedded.transpose(0, 1)
            embedded = self.pos_encoding(embedded)
            embedded = embedded.transpose(0, 1)
        
        return embedded

    def _apply_identity_protection(self, embedded: torch.Tensor, trait_indices: torch.Tensor) -> torch.Tensor:
        """
        Apply network-level identity protection.
        
        Uses learned identity preservation to protect core identity traits
        from being modified during neural network processing.
        
        Args:
            embedded: Embedded trait representations
            trait_indices: Trait type indices for identification
            
        Returns:
            Identity-protected embeddings
        """
        # Import here to avoid circular imports
        from ..trait_models.trait_types import PERMANENTLY_PROTECTED_TRAITS, PARTIALLY_PROTECTED_TRAITS, TraitType
        
        # Create protection masks based on trait indices
        permanent_mask = torch.zeros_like(trait_indices, dtype=torch.bool)
        partial_mask = torch.zeros_like(trait_indices, dtype=torch.bool)
        
        for i, trait_type in enumerate(TraitType):
            if trait_type in PERMANENTLY_PROTECTED_TRAITS:
                # Mark permanently protected traits for strongest protection
                permanent_mask |= (trait_indices == i)
            elif trait_type in PARTIALLY_PROTECTED_TRAITS:
                # Mark partially protected traits for moderate protection
                partial_mask |= (trait_indices == i)
        
        # Apply identity preservation to protected traits
        identity_preserved = embedded.clone()
        
        # PERMANENTLY PROTECTED TRAITS: Strongest protection
        if permanent_mask.any():
            # Get identity preservation signal for permanent traits
            identity_signal = self.identity_preservation(embedded)
            identity_gate = self.identity_gate(identity_signal)
            
            # For permanently protected traits, preserve original embeddings very strongly
            # identity_gate will be close to 1, preserving original almost completely
            identity_preserved[permanent_mask] = (
                embedded[permanent_mask] * 0.95 +  # 95% preserve original
                embedded[permanent_mask] * identity_gate[permanent_mask] * 0.05  # 5% allow minimal change
            )
        
        # PARTIALLY PROTECTED TRAITS: Moderate protection
        if partial_mask.any():
            # Get identity preservation signal for partial traits
            identity_signal = self.identity_preservation(embedded)
            identity_gate = self.identity_gate(identity_signal)
            
            # For partially protected traits, allow some evolution but preserve core
            # identity_gate will be moderate, allowing some change but preserving fundamentals
            identity_preserved[partial_mask] = (
                embedded[partial_mask] * 0.7 +  # 70% preserve original
                embedded[partial_mask] * identity_gate[partial_mask] * 0.3  # 30% allow evolution
            )
        
        return identity_preserved 