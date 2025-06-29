"""
Ilanya Trait Engine - Test Suite

Comprehensive test suite for the Ilanya Trait Engine components.
Tests trait types, data structures, state management, and neural network
components to ensure reliability and correctness of the trait processing system.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import numpy as np

from src.trait_models.trait_types import TraitType, TraitCategory, TraitDimension
from src.trait_models.trait_data import TraitVector, TraitMatrix, TraitData, TraitDataBuilder
from src.trait_models.trait_state import TraitState, CognitiveState


class TestTraitTypes:
    """
    Test trait type definitions.
    
    Verifies that all trait types, categories, and dimensions are properly
    defined and accessible. Ensures the enum structure is correct.
    """
    
    def test_trait_type_enum(self):
        """
        Test that trait types are properly defined.
        
        Verifies that trait type enums have correct string values and
        can be accessed properly.
        """
        assert TraitType.OPENNESS.value == "openness"
        assert TraitType.CREATIVITY.value == "creativity"
        assert TraitType.ADAPTABILITY.value == "adaptability"
    
    def test_trait_categories(self):
        """
        Test trait categories.
        
        Verifies that trait categories are properly defined and
        have correct string representations.
        """
        assert TraitCategory.PERSONALITY.value == "personality"
        assert TraitCategory.COGNITIVE.value == "cognitive"
        assert TraitCategory.BEHAVIORAL.value == "behavioral"
    
    def test_trait_dimensions(self):
        """
        Test trait dimensions.
        
        Verifies that trait dimensions are properly defined and
        have correct string representations.
        """
        assert TraitDimension.INTENSITY.value == "intensity"
        assert TraitDimension.STABILITY.value == "stability"
        assert TraitDimension.PLASTICITY.value == "plasticity"


class TestTraitData:
    """
    Test trait data structures.
    
    Tests the core data structures including TraitVector, TraitMatrix,
    TraitData, and TraitDataBuilder. Verifies data validation, serialization,
    and basic operations.
    """
    
    def test_trait_vector_creation(self):
        """
        Test creating a trait vector.
        
        Verifies that TraitVector objects can be created with valid
        parameters and that the data is stored correctly.
        """
        trait_vector = TraitVector(
            trait_type=TraitType.OPENNESS,
            value=0.7,
            confidence=0.9
        )
        
        assert trait_vector.trait_type == TraitType.OPENNESS
        assert trait_vector.value == 0.7
        assert trait_vector.confidence == 0.9
    
    def test_trait_vector_validation(self):
        """
        Test trait vector validation.
        
        Verifies that TraitVector properly validates input values
        and raises appropriate errors for invalid data.
        """
        # Should raise error for invalid values
        with pytest.raises(ValueError):
            TraitVector(TraitType.OPENNESS, 1.5, 0.9)  # Value > 1.0
        
        with pytest.raises(ValueError):
            TraitVector(TraitType.OPENNESS, 0.7, -0.1)  # Confidence < 0.0
    
    def test_trait_matrix_creation(self):
        """
        Test creating a trait matrix.
        
        Verifies that TraitMatrix objects can be created with multiple
        traits and that the interaction matrix is properly initialized.
        """
        traits = {
            TraitType.OPENNESS: TraitVector(TraitType.OPENNESS, 0.7, 0.9),
            TraitType.CREATIVITY: TraitVector(TraitType.CREATIVITY, 0.8, 0.8)
        }
        
        trait_matrix = TraitMatrix(traits=traits)
        
        assert len(trait_matrix.traits) == 2
        assert trait_matrix.interaction_matrix is not None
        assert trait_matrix.interaction_matrix.shape == (2, 2)
    
    def test_trait_data_builder(self):
        """
        Test trait data builder.
        
        Verifies that TraitDataBuilder can construct TraitData objects
        step by step and that all metadata is properly set.
        """
        builder = TraitDataBuilder()
        builder.add_trait(TraitType.OPENNESS, 0.7, 0.9)
        builder.add_trait(TraitType.CREATIVITY, 0.8, 0.8)
        builder.set_source("test_source")
        
        trait_data = builder.build()
        
        assert trait_data.get_trait_count() == 2
        assert trait_data.source == "test_source"
        assert TraitType.OPENNESS in trait_data.trait_matrix.traits
        assert TraitType.CREATIVITY in trait_data.trait_matrix.traits


class TestTraitState:
    """
    Test trait state tracking.
    
    Tests the TraitState and CognitiveState classes that handle
    state tracking and cognitive state management over time.
    """
    
    def test_trait_state_creation(self):
        """
        Test creating a trait state.
        
        Verifies that TraitState objects can be created with valid
        parameters and that change rates are computed correctly.
        """
        trait_state = TraitState(
            trait_type=TraitType.OPENNESS,
            current_value=0.7,
            previous_value=0.6,
            confidence=0.9
        )
        
        assert trait_state.trait_type == TraitType.OPENNESS
        assert trait_state.current_value == 0.7
        assert trait_state.previous_value == 0.6
        assert trait_state.change_rate == 0.1  # 0.7 - 0.6
        assert trait_state.confidence == 0.9
    
    def test_cognitive_state_creation(self):
        """
        Test creating a cognitive state.
        
        Verifies that CognitiveState objects can be created with
        multiple trait states and cognitive metrics.
        """
        trait_states = {
            TraitType.OPENNESS: TraitState(TraitType.OPENNESS, 0.7, confidence=0.9),
            TraitType.CREATIVITY: TraitState(TraitType.CREATIVITY, 0.8, confidence=0.8)
        }
        
        cognitive_state = CognitiveState(
            trait_states=trait_states,
            overall_stability=0.8,
            cognitive_load=0.3,
            attention_focus=0.9,
            emotional_state=0.6
        )
        
        assert len(cognitive_state.trait_states) == 2
        assert cognitive_state.overall_stability == 0.8
        assert cognitive_state.cognitive_load == 0.3
        assert cognitive_state.attention_focus == 0.9
        assert cognitive_state.emotional_state == 0.6
    
    def test_cognitive_state_validation(self):
        """
        Test cognitive state validation.
        
        Verifies that CognitiveState properly validates input values
        and raises appropriate errors for invalid data.
        """
        trait_states = {
            TraitType.OPENNESS: TraitState(TraitType.OPENNESS, 0.7, confidence=0.9)
        }
        
        # Should raise error for invalid values
        with pytest.raises(ValueError):
            CognitiveState(trait_states, overall_stability=1.5)  # > 1.0
        
        with pytest.raises(ValueError):
            CognitiveState(trait_states, cognitive_load=-0.1)  # < 0.0


class TestNeuralNetworkComponents:
    """
    Test neural network components.
    
    Tests the neural network components including embeddings,
    positional encoding, and transformer blocks.
    """
    
    def test_trait_embedding(self):
        """
        Test trait embedding layer.
        
        Verifies that the TraitEmbedding layer can process trait data
        and produce embeddings of the correct shape.
        """
        from src.neural_networks.trait_transformer import TraitEmbedding
        
        embedding = TraitEmbedding(num_traits=20, embedding_dim=64, input_dim=512)
        
        # Test forward pass with sample data
        batch_size, num_traits = 2, 5
        trait_values = torch.randn(batch_size, num_traits)
        trait_confidences = torch.randn(batch_size, num_traits)
        trait_indices = torch.randint(0, 20, (batch_size, num_traits))
        
        output = embedding(trait_values, trait_confidences, trait_indices)
        
        assert output.shape == (batch_size, num_traits, 64)
    
    def test_positional_encoding(self):
        """
        Test positional encoding.
        
        Verifies that the PositionalEncoding layer adds position
        information to embeddings and produces different outputs.
        """
        from src.neural_networks.trait_transformer import PositionalEncoding
        
        pos_encoding = PositionalEncoding(embedding_dim=64, max_seq_length=100)
        
        # Test forward pass
        seq_len, batch_size, embedding_dim = 10, 2, 64
        x = torch.randn(seq_len, batch_size, embedding_dim)
        
        output = pos_encoding(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # Should be different due to positional encoding


if __name__ == "__main__":
    pytest.main([__file__]) 