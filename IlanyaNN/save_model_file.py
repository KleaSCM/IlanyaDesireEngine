"""
Ilanya Trait Engine - Save Model File

Quickly saves the current trait engine as a model file
that can be loaded into other AI systems.

This gives you the .pt file you were expecting!

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
from src.trait_engine import TraitEngine, TraitEngineConfig
from src.trait_models.trait_types import TraitType


def main():
    """Save the trait engine as a model file."""
    print("=== Saving Ilanya Trait Model File ===")
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
    
    # Test that it works
    print("🧪 Testing Model...")
    from src.trait_models.trait_data import TraitDataBuilder
    
    # Create a simple test profile
    builder = TraitDataBuilder()
    builder.add_trait(TraitType.SEXUAL_ORIENTATION, 1.0, 1.0, "Lesbian")
    builder.add_trait(TraitType.GENDER_IDENTITY, 1.0, 1.0, "Female")
    builder.add_trait(TraitType.EMPATHY, 0.8, 0.9, "Empathetic")
    builder.add_trait(TraitType.CREATIVITY, 0.8, 0.9, "Creative")
    test_data = builder.build()
    
    results = engine.process_traits(test_data)
    print(f"   - Successfully processed {len(results['predicted_traits'])} traits")
    print()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Save the complete model
    print("💾 Saving Model Files...")
    
    # Save as PyTorch model
    model_path = "models/ilanya_trait_model.pt"
    engine.save_model(model_path)
    print(f"   - Saved complete model: {model_path}")
    
    # Save just the neural network
    nn_path = "models/ilanya_trait_nn.pt"
    torch.save({
        'model_state_dict': engine.neural_network.state_dict(),
        'config': config
    }, nn_path)
    print(f"   - Saved neural network: {nn_path}")
    
    # Save configuration
    config_path = "models/ilanya_trait_config.yaml"
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
    
    # Save Ilanya's profile
    profile_path = "models/ilanya_profile.pt"
    torch.save({
        'trait_data': test_data,
        'source': 'ilanya_core_profile'
    }, profile_path)
    print(f"   - Saved Ilanya's profile: {profile_path}")
    
    print()
    print("🎯 Model Files Created Successfully!")
    print()
    print("📁 Files you can now use:")
    print(f"   - {model_path} - Complete trained model")
    print(f"   - {nn_path} - Neural network weights")
    print(f"   - {config_path} - Model configuration")
    print(f"   - {profile_path} - Ilanya's trait profile")
    print()
    print("🚀 You can now load these files into your AI system!")
    print()
    print("Example usage:")
    print("```python")
    print("from src.trait_engine import TraitEngine")
    print("engine = TraitEngine()")
    print("engine.load_model('models/ilanya_trait_model.pt')")
    print("```")
    print()
    print("🎯 This is exactly what you were expecting - a model file you can load!")


if __name__ == "__main__":
    main() 