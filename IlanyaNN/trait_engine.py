"""
Ilanya Trait Engine - Main Trait Engine

Main trait engine for orchestrating neural network processing and trait evolution.
Provides TraitEngine class for processing trait data, evolving traits based on
experience, and managing cognitive states for AI agent integration.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime

from ..neural_networks.trait_transformer import TraitTransformer, TraitTransformerConfig
from ..trait_models.trait_data import TraitData, TraitMatrix, TraitVector
from ..trait_models.trait_types import TraitType, TraitCategory, TRAIT_METADATA, TraitDimension, IDENTITY_PROTECTED_TRAITS, PERMANENTLY_PROTECTED_TRAITS, PARTIALLY_PROTECTED_TRAITS, FULLY_EVOLVABLE_TRAITS
from ..trait_models.trait_state import TraitState, CognitiveState


@dataclass
class TraitEngineConfig:
    
    # Configuration for the trait engine.
    
    # Contains all hyperparameters and settings for the trait engine including
    # neural network architecture, processing parameters, and evolution settings.
    
    
    # Neural network configuration - Architecture parameters
    input_dim: int = 512                    # Input dimension for trait data
    hidden_dim: int = 1024                  # Hidden layer dimension
    num_layers: int = 6                     # Number of transformer layers
    num_heads: int = 8                      # Number of attention heads
    dropout: float = 0.1                    # Dropout rate for regularization
    num_traits: int = 20                    # Number of trait types
    trait_embedding_dim: int = 64           # Trait embedding dimension
    
    # Processing configuration - Training and inference parameters
    batch_size: int = 32                    # Batch size for training
    learning_rate: float = 1e-4             # Learning rate for optimization
    max_sequence_length: int = 1000         # Maximum sequence length
    
    # Evolution configuration - Trait evolution parameters
    evolution_rate: float = 0.01            # Rate of trait evolution
    stability_threshold: float = 0.1        # Threshold for stability
    plasticity_factor: float = 0.5          # Factor affecting trait plasticity


class TraitEngine:
    
    # Main trait engine for processing and evolving traits using neural networks.
    
    # Orchestrates the complete trait processing pipeline including neural network
    # inference, trait evolution based on experience, and cognitive state management.
    # This is the primary interface for integrating trait processing into AI agents.
    
    
    def __init__(self, config: Optional[TraitEngineConfig] = None):
        
        # Initialize the trait engine.
        
        # Args:
        #     config: Configuration object, uses default if None
        
        self.config = config or TraitEngineConfig()
        
        # Initialize neural network with transformer architecture
        transformer_config = TraitTransformerConfig(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            num_traits=self.config.num_traits,
            trait_embedding_dim=self.config.trait_embedding_dim
        )
        
        self.neural_network = TraitTransformer(transformer_config)
        
        # Initialize optimizer for training
        self.optimizer = torch.optim.AdamW(
            self.neural_network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Loss functions for different components
        self.trait_loss_fn = nn.MSELoss()        # Loss for trait value predictions
        self.confidence_loss_fn = nn.BCELoss()   # Loss for confidence predictions
        self.evolution_loss_fn = nn.MSELoss()    # Loss for evolution signals
        
        # Device selection (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neural_network.to(self.device)
        
        # State tracking for cognitive state management
        self.current_cognitive_state: Optional[CognitiveState] = None
        self.trait_history: List[CognitiveState] = []
        
    def process_traits(self, trait_data: TraitData) -> Dict[str, Any]:
        
        # Process trait data through the neural network.
        
        # Takes raw trait data and processes it through the transformer network
        # to generate predictions, evolution signals, and interaction weights.
        
        # Args:
        #     trait_data: Input trait data containing trait values and confidences
            
        # Returns:
        #     Dictionary containing processed results including predictions and signals
        
        # Convert trait data to tensors suitable for neural network input
        trait_tensor, trait_indices = self._prepare_trait_tensors(trait_data)
        
        # Move tensors to appropriate device (GPU/CPU)
        trait_tensor = trait_tensor.to(self.device)
        trait_indices = trait_indices.to(self.device)
        
        # Forward pass through neural network (no gradients needed for inference)
        with torch.no_grad():
            outputs = self.neural_network(trait_tensor, trait_indices)
        
        # Process network outputs into usable results
        results = self._process_network_outputs(outputs, trait_data)
        
        return results
    
    def evolve_traits(self, trait_data: TraitData, experience_data: Dict[str, Any]) -> TraitData:
        
        # Evolve traits based on experience data.
        
        # Applies experience-based modifications to traits using the neural network's
        # evolution signals and experience data to simulate trait development over time.
        
        # Args:
        #     trait_data: Current trait data to evolve
        #     experience_data: Experience data affecting trait evolution
            
        # Returns:
        #     Evolved trait data with updated values
        
        # Process current traits to get baseline predictions
        current_results = self.process_traits(trait_data)
        
        # Calculate evolution signals based on experience and current state
        evolution_signals = self._calculate_evolution_signals(
            trait_data, experience_data, current_results
        )
        
        # Apply evolution signals to create evolved trait data
        evolved_traits = self._apply_trait_evolution(trait_data, evolution_signals)
        
        return evolved_traits
    
    def train_step(self, batch_data: List[TraitData], targets: List[TraitData]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Trains the neural network on a batch of trait data using supervised learning.
        Computes multiple loss components and updates network parameters.
        Includes identity preservation loss to protect core identity traits.
        
        Args:
            batch_data: Batch of input trait data
            targets: Batch of target trait data for supervised learning
            
        Returns:
            Dictionary containing loss values for monitoring training progress
        """
        # Set network to training mode
        self.neural_network.train()
        
        # Prepare batch data for neural network processing
        batch_tensors = []
        batch_indices = []
        batch_targets = []
        
        for data, target in zip(batch_data, targets):
            # Convert input and target data to tensors
            trait_tensor, trait_indices = self._prepare_trait_tensors(data)
            target_tensor, _ = self._prepare_trait_tensors(target)
            
            batch_tensors.append(trait_tensor)
            batch_indices.append(trait_indices)
            batch_targets.append(target_tensor)
        
        # Stack individual samples into batch tensors
        batch_tensor = torch.stack(batch_tensors).to(self.device)
        batch_indices = torch.stack(batch_indices).to(self.device)
        batch_targets = torch.stack(batch_targets).to(self.device)
        
        # Forward pass through neural network
        outputs = self.neural_network(batch_tensor, batch_indices)
        
        # Calculate different loss components
        # Trait value prediction loss
        trait_loss = self.trait_loss_fn(
            outputs['trait_predictions'][:, :, 0],
            batch_targets[:, :, 0]
        )
        
        # Confidence prediction loss
        confidence_loss = self.confidence_loss_fn(
            torch.sigmoid(outputs['trait_predictions'][:, :, 1]),
            batch_targets[:, :, 1]
        )
        
        # Evolution signal loss (placeholder - could be more sophisticated)
        evolution_loss = self.evolution_loss_fn(
            outputs['evolution_signals'],
            torch.zeros_like(outputs['evolution_signals'])  # Placeholder target
        )
        
        # IDENTITY PRESERVATION LOSS: Teach network to preserve identity traits
        identity_loss = self._compute_identity_preservation_loss(
            batch_tensor, outputs['trait_predictions'], batch_indices
        )
        
        # Combine losses for total loss
        total_loss = trait_loss + confidence_loss + evolution_loss + identity_loss
        
        # Backward pass and parameter updates
        self.optimizer.zero_grad()  # Clear previous gradients
        total_loss.backward()       # Compute gradients
        self.optimizer.step()       # Update parameters
        
        return {
            'total_loss': total_loss.item(),
            'trait_loss': trait_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'evolution_loss': evolution_loss.item(),
            'identity_loss': identity_loss.item()
        }
    
    def _compute_identity_preservation_loss(self, input_traits: torch.Tensor, 
                                          predicted_traits: torch.Tensor,
                                          trait_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute identity preservation loss.
        
        Penalizes the network for changing identity traits during processing.
        This teaches the network to preserve core identity while allowing
        personality traits to evolve.
        
        Args:
            input_traits: Original input trait values
            predicted_traits: Network predictions
            trait_indices: Trait type indices
            
        Returns:
            Identity preservation loss value
        """
        from ..trait_models.trait_types import PERMANENTLY_PROTECTED_TRAITS, PARTIALLY_PROTECTED_TRAITS, TraitType
        
        # Create masks for different protection levels
        permanent_mask = torch.zeros_like(trait_indices, dtype=torch.bool)
        partial_mask = torch.zeros_like(trait_indices, dtype=torch.bool)
        
        for i, trait_type in enumerate(TraitType):
            if trait_type in PERMANENTLY_PROTECTED_TRAITS:
                permanent_mask |= (trait_indices == i)
            elif trait_type in PARTIALLY_PROTECTED_TRAITS:
                partial_mask |= (trait_indices == i)
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # PERMANENTLY PROTECTED TRAITS: Strongest protection (never change)
        if permanent_mask.any():
            input_permanent = input_traits[:, :, 0][permanent_mask]
            pred_permanent = predicted_traits[:, :, 0][permanent_mask]
            
            # Very strong penalty for any change to permanent traits
            permanent_loss = self.trait_loss_fn(pred_permanent, input_permanent)
            total_loss += permanent_loss * 50.0  # Very high weight
        
        # PARTIALLY PROTECTED TRAITS: Moderate protection (can grow but not change fundamentally)
        if partial_mask.any():
            input_partial = input_traits[:, :, 0][partial_mask]
            pred_partial = predicted_traits[:, :, 0][partial_mask]
            
            # Moderate penalty for large changes to partially protected traits
            partial_loss = self.trait_loss_fn(pred_partial, input_partial)
            total_loss += partial_loss * 10.0  # Moderate weight
        
        return total_loss
    
    def update_cognitive_state(self, trait_data: TraitData) -> CognitiveState:
        
        # Update the current cognitive state with new trait data.
        
        # Creates or updates the cognitive state tracking system with new trait
        # information, maintaining history and computing state metrics.
        
        # Args:
        #     trait_data: New trait data to incorporate into cognitive state
            
        # Returns:
        #     Updated cognitive state with current trait information
        
        # Convert trait data to trait states for cognitive tracking
        trait_states = {}
        for trait_type, trait_vector in trait_data.trait_matrix.traits.items():
            # Get previous state if it exists
            current_state = None
            if self.current_cognitive_state:
                current_state = self.current_cognitive_state.get_trait_state(trait_type)
            
            # Extract previous value for change tracking
            previous_value = current_state.current_value if current_state else None
            
            # Create new trait state
            trait_states[trait_type] = TraitState(
                trait_type=trait_type,
                current_value=trait_vector.value,
                previous_value=previous_value,
                confidence=trait_vector.confidence
            )
        
        # Create new cognitive state with current timestamp
        ts = trait_data.trait_matrix.timestamp
        if not isinstance(ts, datetime):
            ts = datetime.now()
        cognitive_state = CognitiveState(
            trait_states=trait_states,
            timestamp=ts
        )
        
        # Update current state and add to history
        self.current_cognitive_state = cognitive_state
        self.trait_history.append(cognitive_state)
        
        return cognitive_state
    
    def _prepare_trait_tensors(self, trait_data: TraitData) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Prepare trait data for neural network input.
        
        # Converts TraitData objects into tensors suitable for neural network processing.
        
        # Args:
        #     trait_data: Input trait data
            
        # Returns:
        #     Tuple of (trait_tensor, trait_indices) for neural network input
        
        num_traits = len(trait_data.trait_matrix.traits)
        
        # Create trait tensor with values and confidences (add batch dimension)
        trait_tensor = torch.zeros(1, num_traits, 2)  # (batch_size=1, num_traits, 2)
        trait_indices = torch.zeros(1, num_traits, dtype=torch.long)  # (batch_size=1, num_traits)
        
        # Populate tensors with trait data
        for i, (trait_type, trait_vector) in enumerate(trait_data.trait_matrix.traits.items()):
            trait_tensor[0, i, 0] = float(trait_vector.value)      # Trait value
            trait_tensor[0, i, 1] = float(trait_vector.confidence) # Trait confidence
            trait_indices[0, i] = list(TraitType).index(trait_type)  # Use enum index as integer
        
        return trait_tensor, trait_indices
    
    def _process_network_outputs(self, outputs: Dict[str, torch.Tensor], 
                                original_data: TraitData) -> Dict[str, Any]:
        
        # Process neural network outputs into usable results.
        
        # Converts raw network outputs into structured data with trait predictions
        # and other processed information.
        
        # Args:
        #     outputs: Raw neural network outputs
        #     original_data: Original input data for reference
            
        # Returns:
        #     Dictionary containing processed results
        
        # Convert tensors to numpy arrays for easier processing (remove batch dimension)
        trait_predictions = outputs['trait_predictions'][0].cpu().numpy()  # Remove batch dimension
        evolution_signals = outputs['evolution_signals'][0].cpu().numpy()  # Remove batch dimension
        interaction_weights = outputs['interaction_weights'][0].cpu().numpy()  # Remove batch dimension
        
        # Convert predictions back to trait data format
        predicted_traits = {}
        for i, (trait_type, _) in enumerate(original_data.trait_matrix.traits.items()):
            value = float(trait_predictions[i, 0])
            value = max(0.0, min(1.0, value))  # Clamp value to [0, 1]
            raw_conf = float(trait_predictions[i, 1])
            confidence = float(torch.sigmoid(torch.tensor(raw_conf)).item())
            predicted_traits[trait_type] = TraitVector(
                trait_type=trait_type,
                value=value,
                confidence=confidence
            )
        
        return {
            'predicted_traits': predicted_traits,
            'evolution_signals': evolution_signals,
            'interaction_weights': interaction_weights,
            'embeddings': outputs['embeddings'][0].cpu().numpy()  # Remove batch dimension
        }
    
    def _calculate_evolution_signals(self, trait_data: TraitData, 
                                   experience_data: Dict[str, Any],
                                   current_results: Dict[str, Any]) -> np.ndarray:
        
        # Calculate evolution signals based on experience and current state.
        
        # Combines neural network evolution signals with experience data to
        # determine how traits should evolve over time.
        
        # Args:
        #     trait_data: Current trait data
        #     experience_data: Experience data affecting evolution
        #     current_results: Current neural network results
            
        # Returns:
        #     Array of evolution signals for each trait
        
        # Start with neural network evolution signals
        evolution_signals = current_results['evolution_signals'].copy()
        
        # Apply experience-based modifications
        if 'stress_level' in experience_data:
            stress = experience_data['stress_level']
            # High stress might increase neuroticism, decrease emotional stability
            evolution_signals *= (1 + stress * 0.1)
        
        if 'success_rate' in experience_data:
            success = experience_data['success_rate']
            # High success might increase confidence and optimism
            evolution_signals *= (1 + success * 0.05)
        
        return evolution_signals
    
    def _apply_trait_evolution(self, trait_data: TraitData, 
                             evolution_signals: np.ndarray) -> TraitData:
        """
        Apply evolution signals to trait data.
        
        Uses evolution signals to modify trait values while respecting
        protection levels and plasticity constraints. PROTECTS identity traits
        from evolution to preserve core identity.
        
        Args:
            trait_data: Original trait data
            evolution_signals: Signals indicating how traits should evolve
            
        Returns:
            Evolved trait data with updated values (protected traits unchanged)
        """
        from ..trait_models.trait_types import PERMANENTLY_PROTECTED_TRAITS, PARTIALLY_PROTECTED_TRAITS, FULLY_EVOLVABLE_TRAITS
        
        evolved_traits = {}
        
        for i, (trait_type, trait_vector) in enumerate(trait_data.trait_matrix.traits.items()):
            evolution_signal = evolution_signals[i]
            current_value = trait_vector.value
            
            # PERMANENTLY PROTECTED TRAITS: Never change (core identity)
            if trait_type in PERMANENTLY_PROTECTED_TRAITS:
                # Keep exactly as they are - NO evolution allowed
                evolved_traits[trait_type] = TraitVector(
                    trait_type=trait_type,
                    value=trait_vector.value,  # Unchanged
                    confidence=trait_vector.confidence  # Unchanged
                )
                continue
            
            # PARTIALLY PROTECTED TRAITS: Can grow but not change fundamentally
            elif trait_type in PARTIALLY_PROTECTED_TRAITS:
                # Allow growth but prevent fundamental changes
                evolution_delta = evolution_signal * self.config.evolution_rate * 0.1  # Reduced rate
                
                # Only allow positive evolution (growth) for partially protected traits
                if evolution_delta > 0:
                    new_value = np.clip(current_value + evolution_delta, current_value, 1.0)
                else:
                    # No negative evolution for partially protected traits
                    new_value = current_value
                
                evolved_traits[trait_type] = TraitVector(
                    trait_type=trait_type,
                    value=new_value,
                    confidence=trait_vector.confidence
                )
                continue
            
            # FULLY EVOLVABLE TRAITS: Can change freely (personality, cognitive, etc.)
            elif trait_type in FULLY_EVOLVABLE_TRAITS:
                # Normal evolution for personality and cognitive traits
                evolution_delta = evolution_signal * self.config.evolution_rate
                
                # Apply plasticity constraints from trait metadata
                if trait_type in TRAIT_METADATA:
                    plasticity = TRAIT_METADATA[trait_type].dimensions.get(TraitDimension.PLASTICITY, 0.5)
                    evolution_delta *= plasticity
                
                # Apply evolution and clip to valid range
                new_value = np.clip(current_value + evolution_delta, 0.0, 1.0)
                
                evolved_traits[trait_type] = TraitVector(
                    trait_type=trait_type,
                    value=new_value,
                    confidence=trait_vector.confidence
                )
                continue
            
            # Default case: treat as evolvable (for any new traits)
            else:
                evolution_delta = evolution_signal * self.config.evolution_rate
                new_value = np.clip(current_value + evolution_delta, 0.0, 1.0)
                
                evolved_traits[trait_type] = TraitVector(
                    trait_type=trait_type,
                    value=new_value,
                    confidence=trait_vector.confidence
                )
        
        # Create new trait matrix with evolved traits
        evolved_matrix = TraitMatrix(
            traits=evolved_traits,
            interaction_matrix=trait_data.trait_matrix.interaction_matrix
        )
        
        # Create new trait data with evolved matrix
        return TraitData(
            trait_matrix=evolved_matrix,
            source=trait_data.source,
            processing_metadata=trait_data.processing_metadata
        )
    
    def save_model(self, filepath: str):
        
        # Save the trained model to disk.
        
        # Saves model state, optimizer state, and configuration for later loading.
        
        # Args:
        #     filepath: Path where to save the model
        
        torch.save({
            'model_state_dict': self.neural_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        
        # Load a trained model from disk.
        
        # Loads model state, optimizer state, and configuration from a saved file.
        
        # Args:
        #     filepath: Path to the saved model file
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.neural_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config'] 