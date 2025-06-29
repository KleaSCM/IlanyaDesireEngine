"""
Ilanya Trait Engine - Basic Usage Example

Comprehensive usage example demonstrating the full trait engine capabilities.
Shows trait processing, evolution, cognitive state management, and training
with the complete neural network architecture.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from src.trait_engine import TraitEngine, TraitEngineConfig
from src.trait_models.trait_data import TraitDataBuilder
from src.trait_models.trait_types import TraitType, TraitCategory


def create_sample_trait_data():
    """
    Create sample trait data for demonstration.
    
    Builds a comprehensive set of trait data covering all major trait categories
    including personality, cognitive, behavioral, and emotional traits. This
    provides a realistic dataset for testing the trait engine capabilities.
    
    Returns:
        TraitData object with sample trait information
    """
    builder = TraitDataBuilder()
    
    # Add personality traits - Big Five model
    builder.add_trait(TraitType.OPENNESS, 0.7, 0.9)           # High openness to experience
    builder.add_trait(TraitType.CONSCIENTIOUSNESS, 0.6, 0.8)  # Moderate conscientiousness
    builder.add_trait(TraitType.EXTRAVERSION, 0.5, 0.7)       # Moderate extraversion
    builder.add_trait(TraitType.AGREEABLENESS, 0.8, 0.9)      # High agreeableness
    builder.add_trait(TraitType.NEUROTICISM, 0.3, 0.6)        # Low neuroticism
    
    # Add cognitive traits - Mental processing abilities
    builder.add_trait(TraitType.CREATIVITY, 0.8, 0.9)              # High creativity
    builder.add_trait(TraitType.ANALYTICAL_THINKING, 0.7, 0.8)     # Good analytical skills
    builder.add_trait(TraitType.MEMORY_CAPACITY, 0.6, 0.7)         # Moderate memory
    builder.add_trait(TraitType.LEARNING_RATE, 0.9, 0.9)           # Very fast learner
    builder.add_trait(TraitType.ATTENTION_SPAN, 0.5, 0.6)          # Moderate attention span
    
    # Add behavioral traits - Action and decision patterns
    builder.add_trait(TraitType.RISK_TAKING, 0.4, 0.7)        # Conservative risk approach
    builder.add_trait(TraitType.PERSISTENCE, 0.8, 0.9)        # High persistence
    builder.add_trait(TraitType.ADAPTABILITY, 0.7, 0.8)       # Good adaptability
    builder.add_trait(TraitType.SOCIAL_SKILLS, 0.6, 0.7)      # Moderate social skills
    builder.add_trait(TraitType.LEADERSHIP, 0.5, 0.6)         # Moderate leadership potential
    
    # Add emotional traits - Affective and emotional responses
    builder.add_trait(TraitType.EMOTIONAL_STABILITY, 0.7, 0.8)  # Good emotional stability
    builder.add_trait(TraitType.EMPATHY, 0.8, 0.9)              # High empathy
    builder.add_trait(TraitType.OPTIMISM, 0.6, 0.7)             # Moderate optimism
    builder.add_trait(TraitType.RESILIENCE, 0.7, 0.8)           # Good resilience
    builder.add_trait(TraitType.SELF_AWARENESS, 0.8, 0.9)       # High self-awareness
    
    # Set metadata for the sample data
    builder.set_source("sample_data")
    builder.add_metadata("description", "Sample trait data for demonstration")
    
    return builder.build()


def main():
    """
    Main demonstration function.
    
    Runs a comprehensive demonstration of the Ilanya Trait Engine showing:
    1. Engine initialization and configuration
    2. Sample data creation
    3. Neural network processing
    4. Trait evolution based on experience
    5. Cognitive state management
    6. Training demonstration
    """
    print("=== Ilanya Trait Engine Demo ===\n")
    
    # Step 1: Create and configure the trait engine
    print("1. Initializing Trait Engine...")
    config = TraitEngineConfig(
        num_traits=20,              # Number of trait types to process
        trait_embedding_dim=64,     # Embedding dimension for traits
        num_layers=4,               # Number of transformer layers
        num_heads=4,                # Number of attention heads
        learning_rate=1e-4          # Learning rate for training
    )
    
    # Initialize the trait engine with configuration
    engine = TraitEngine(config)
    print(f"   - Device: {engine.device}")
    print(f"   - Number of parameters: {sum(p.numel() for p in engine.neural_network.parameters()):,}")
    print()
    
    # Step 2: Create sample trait data for demonstration
    print("2. Creating sample trait data...")
    trait_data = create_sample_trait_data()
    print(f"   - Number of traits: {trait_data.get_trait_count()}")
    print(f"   - Source: {trait_data.source}")
    print()
    
    # Step 3: Process traits through the neural network
    print("3. Processing traits through neural network...")
    results = engine.process_traits(trait_data)
    print(f"   - Predicted traits: {len(results['predicted_traits'])}")
    print(f"   - Evolution signals shape: {results['evolution_signals'].shape}")
    print(f"   - Interaction weights shape: {results['interaction_weights'].shape}")
    print()
    
    # Step 4: Show sample predictions from the neural network
    print("4. Sample trait predictions:")
    for i, (trait_type, predicted_trait) in enumerate(list(results['predicted_traits'].items())[:5]):
        original_trait = trait_data.trait_matrix.traits[trait_type]
        print(f"   {trait_type.value}:")
        print(f"     Original: {original_trait.value:.3f} (conf: {original_trait.confidence:.3f})")
        print(f"     Predicted: {predicted_trait.value:.3f} (conf: {predicted_trait.confidence:.3f})")
    print()
    
    # Step 5: Demonstrate trait evolution based on experience
    print("5. Evolving traits based on experience...")
    # Create experience data that would affect trait evolution
    experience_data = {
        'stress_level': 0.3,           # Moderate stress
        'success_rate': 0.7,           # Good success rate
        'social_interactions': 0.6,    # Moderate social activity
        'learning_opportunities': 0.8  # High learning opportunities
    }
    
    # Apply evolution to traits based on experience
    evolved_data = engine.evolve_traits(trait_data, experience_data)
    print(f"   - Evolution applied successfully")
    print()
    
    # Step 6: Show the results of trait evolution
    print("6. Trait evolution results:")
    for i, (trait_type, evolved_trait) in enumerate(list(evolved_data.trait_matrix.traits.items())[:5]):
        original_trait = trait_data.trait_matrix.traits[trait_type]
        change = evolved_trait.value - original_trait.value
        print(f"   {trait_type.value}: {original_trait.value:.3f} → {evolved_trait.value:.3f} (Δ: {change:+.3f})")
    print()
    
    # Step 7: Update and display cognitive state
    print("7. Updating cognitive state...")
    cognitive_state = engine.update_cognitive_state(evolved_data)
    print(f"   - Cognitive state updated")
    print(f"   - Overall stability: {cognitive_state.overall_stability:.3f}")
    print(f"   - Cognitive load: {cognitive_state.cognitive_load:.3f}")
    print(f"   - Attention focus: {cognitive_state.attention_focus:.3f}")
    print(f"   - Emotional state: {cognitive_state.emotional_state:.3f}")
    print()
    
    # Step 8: Demonstrate training capabilities
    print("8. Demonstrating training step...")
    # Create a simple training batch for demonstration
    batch_data = [trait_data, trait_data]  # Duplicate for demo purposes
    targets = [evolved_data, evolved_data]  # Use evolved data as targets
    
    try:
        # Perform a training step
        loss_info = engine.train_step(batch_data, targets)
        print(f"   - Training step completed")
        print(f"   - Total loss: {loss_info['total_loss']:.6f}")
        print(f"   - Trait loss: {loss_info['trait_loss']:.6f}")
        print(f"   - Confidence loss: {loss_info['confidence_loss']:.6f}")
        print(f"   - Evolution loss: {loss_info['evolution_loss']:.6f}")
    except Exception as e:
        print(f"   - Training step failed: {e}")
    print()
    
    # Summary and conclusion
    print("=== Demo completed successfully! ===")
    print("\nThe Ilanya Trait Engine is now ready for integration into your AI agent.")
    print("Key features demonstrated:")
    print("- Neural network-based trait processing")
    print("- Trait evolution based on experience")
    print("- Cognitive state tracking")
    print("- Multi-head attention for trait relationships")
    print("- Training capabilities")


if __name__ == "__main__":
    main() 