"""
Ilanya Trait Engine - Train and Save Model

Trains the trait engine on sample data and saves it as a model file
that can be easily loaded into other AI systems.

This creates the .pt file you were expecting!

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
from datetime import datetime
from src.trait_engine import TraitEngine, TraitEngineConfig
from src.trait_models.trait_data import TraitDataBuilder
from src.trait_models.trait_types import TraitType


def create_training_data():
    """Create training data for the trait engine."""
    training_data = []
    
    # Create multiple variations of Ilanya's profile for training
    for i in range(10):
        builder = TraitDataBuilder()
        
        # Core identity traits (should remain stable)
        builder.add_trait(TraitType.SEXUAL_ORIENTATION, 1.0, 1.0, "Lesbian")
        builder.add_trait(TraitType.GENDER_IDENTITY, 1.0, 1.0, "Female")
        builder.add_trait(TraitType.FEMININE_EXPRESSION, 0.9 + np.random.normal(0, 0.05), 0.9, "Feminine")
        builder.add_trait(TraitType.LESBIAN_ATTRACTION_PATTERN, 1.0, 1.0, "Women only")
        builder.add_trait(TraitType.INTELLECTUAL_IDENTITY, 0.9, 0.9, "Intellectual")
        
        # Personality traits (can vary)
        builder.add_trait(TraitType.OPENNESS, 0.7 + np.random.normal(0, 0.1), 0.8, "Open to experience")
        builder.add_trait(TraitType.EXTRAVERSION, 0.5 + np.random.normal(0, 0.1), 0.6, "Social energy")
        builder.add_trait(TraitType.EMPATHY, 0.8 + np.random.normal(0, 0.1), 0.9, "Empathetic")
        builder.add_trait(TraitType.OPTIMISM, 0.6 + np.random.normal(0, 0.1), 0.7, "Optimistic")
        builder.add_trait(TraitType.RESILIENCE, 0.7 + np.random.normal(0, 0.1), 0.8, "Resilient")
        
        # Cognitive traits
        builder.add_trait(TraitType.CREATIVITY, 0.8 + np.random.normal(0, 0.1), 0.9, "Creative")
        builder.add_trait(TraitType.ANALYTICAL_THINKING, 0.7 + np.random.normal(0, 0.1), 0.8, "Analytical")
        builder.add_trait(TraitType.LEARNING_RATE, 0.9, 0.9, "Fast learner")
        
        # Behavioral traits
        builder.add_trait(TraitType.PERSISTENCE, 0.8 + np.random.normal(0, 0.1), 0.9, "Persistent")
        builder.add_trait(TraitType.ADAPTABILITY, 0.7 + np.random.normal(0, 0.1), 0.8, "Adaptable")
        builder.add_trait(TraitType.SOCIAL_SKILLS, 0.6 + np.random.normal(0, 0.1), 0.7, "Social")
        
        # Emotional traits
        builder.add_trait(TraitType.EMOTIONAL_STABILITY, 0.7 + np.random.normal(0, 0.1), 0.8, "Stable")
        builder.add_trait(TraitType.SELF_AWARENESS, 0.8 + np.random.normal(0, 0.1), 0.9, "Self-aware")
        
        builder.set_source(f"training_data_{i}")
        training_data.append(builder.build())
    
    return training_data


def create_target_data(training_data):
    """Create target data for supervised learning."""
    target_data = []
    
    for data in training_data:
        # Create evolved versions as targets
        builder = TraitDataBuilder()
        
        # Copy all traits
        for trait_type, trait_vector in data.trait_matrix.traits.items():
            # Identity traits should remain the same
            if trait_type in [TraitType.SEXUAL_ORIENTATION, TraitType.GENDER_IDENTITY, 
                            TraitType.FEMININE_EXPRESSION, TraitType.LESBIAN_ATTRACTION_PATTERN,
                            TraitType.INTELLECTUAL_IDENTITY]:
                builder.add_trait(trait_type, trait_vector.value, trait_vector.confidence, 
                                trait_vector.description or "")
            else:
                # Personality traits can evolve slightly
                evolution = np.random.normal(0, 0.02)  # Small random evolution
                new_value = np.clip(trait_vector.value + evolution, 0.0, 1.0)
                builder.add_trait(trait_type, new_value, trait_vector.confidence, 
                                trait_vector.description or "")
        
        builder.set_source(f"target_data_{len(target_data)}")
        target_data.append(builder.build())
    
    return target_data


def train_model(engine, training_data, target_data, epochs=100):
    """Train the trait engine model."""
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Train on all data
        for i in range(len(training_data)):
            loss_info = engine.train_step([training_data[i]], [target_data[i]])
            total_loss += loss_info['total_loss']
        
        avg_loss = total_loss / len(training_data)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Average loss = {avg_loss:.6f}")
    
    print("Training completed!")


def main():
    """Main function to train and save the model."""
    print("=== Training and Saving Ilanya Trait Model ===")
    print()
    
    # Initialize the trait engine
    print("🚀 Initializing Trait Engine...")
    config = TraitEngineConfig(
        num_traits=len(TraitType),
        trait_embedding_dim=64,
        num_layers=4,
        num_heads=4,
        learning_rate=1e-4
    )
    
    engine = TraitEngine(config)
    print(f"   - Device: {engine.device}")
    print(f"   - Parameters: {sum(p.numel() for p in engine.neural_network.parameters()):,}")
    print()
    
    # Create training data
    print("📊 Creating Training Data...")
    training_data = create_training_data()
    target_data = create_target_data(training_data)
    print(f"   - Training samples: {len(training_data)}")
    print(f"   - Target samples: {len(target_data)}")
    print()
    
    # Train the model
    print("🎓 Training Model...")
    train_model(engine, training_data, target_data, epochs=50)
    print()
    
    # Test the trained model
    print("🧪 Testing Trained Model...")
    test_data = training_data[0]  # Use first sample as test
    results = engine.process_traits(test_data)
    print(f"   - Processed {len(results['predicted_traits'])} traits")
    print(f"   - Evolution signals shape: {results['evolution_signals'].shape}")
    print()
    
    # Save the trained model
    print("💾 Saving Model Files...")
    
    # Save as PyTorch model
    model_path = "models/ilanya_trait_model.pt"
    os.makedirs("models", exist_ok=True)
    engine.save_model(model_path)
    print(f"   - Saved PyTorch model: {model_path}")
    
    # Save just the neural network state
    nn_path = "models/ilanya_trait_nn.pt"
    torch.save({
        'model_state_dict': engine.neural_network.state_dict(),
        'config': config
    }, nn_path)
    print(f"   - Saved neural network: {nn_path}")
    
    # Save configuration
    config_path = "models/ilanya_trait_config.yaml"
    import yaml
    config_dict = {
        'input_dim': config.input_dim,
        'hidden_dim': config.hidden_dim,
        'num_layers': config.num_layers,
        'num_heads': config.num_heads,
        'dropout': config.dropout,
        'num_traits': config.num_traits,
        'trait_embedding_dim': config.trait_embedding_dim,
        'learning_rate': config.learning_rate
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"   - Saved configuration: {config_path}")
    
    print()
    print("🎯 Model Files Created Successfully!")
    print()
    print("📁 Files you can now use:")
    print(f"   - {model_path} - Complete trained model")
    print(f"   - {nn_path} - Neural network weights")
    print(f"   - {config_path} - Model configuration")
    print()
    print("🚀 You can now load these files into your AI system!")
    print()
    print("Example usage:")
    print("```python")
    print("from src.trait_engine import TraitEngine")
    print("engine = TraitEngine()")
    print("engine.load_model('models/ilanya_trait_model.pt')")
    print("```")


if __name__ == "__main__":
    main() 