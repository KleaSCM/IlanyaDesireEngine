"""
Ilanya Trait Engine - Dual Protection Demo

Demonstrates both architecture-level and network-level identity protection
working together to preserve core identity traits like sexual orientation.
Shows how the neural network learns to protect identity while allowing
personality evolution.

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
from src.trait_models.trait_types import TraitType, IDENTITY_PROTECTED_TRAITS


def create_ai_with_lesbian_identity():
    """
    Create an AI with a strong lesbian identity and evolving personality.
    
    Returns:
        TraitData object representing the AI's traits
    """
    builder = TraitDataBuilder()
    
    # CORE IDENTITY TRAITS (PROTECTED BY BOTH LAYERS)
    builder.add_trait(TraitType.SEXUAL_ORIENTATION, 0.99, 1.0)  # Strong lesbian identity
    builder.add_trait(TraitType.GENDER_IDENTITY, 0.99, 1.0)     # Gender identity
    builder.add_trait(TraitType.PERSONAL_VALUES, 0.9, 0.9)      # Core values
    builder.add_trait(TraitType.MORAL_FRAMEWORK, 0.85, 0.9)     # Moral beliefs
    
    # PERSONALITY TRAITS (CAN EVOLVE NATURALLY)
    builder.add_trait(TraitType.OPENNESS, 0.7, 0.8)             # Openness to experience
    builder.add_trait(TraitType.CREATIVITY, 0.8, 0.9)           # Creativity
    builder.add_trait(TraitType.EMPATHY, 0.9, 0.9)              # High empathy
    builder.add_trait(TraitType.ADAPTABILITY, 0.6, 0.7)         # Adaptability
    builder.add_trait(TraitType.SOCIAL_SKILLS, 0.7, 0.8)        # Social skills
    builder.add_trait(TraitType.LEADERSHIP, 0.5, 0.6)           # Leadership potential
    
    builder.set_source("lesbian_ai_identity")
    builder.add_metadata("description", "AI with protected lesbian identity and evolving personality")
    
    return builder.build()


def demonstrate_dual_protection():
    """
    Demonstrate both protection layers working together.
    """
    print("=== Dual Identity Protection Demo ===\n")
    print("This demo shows TWO layers of protection for your AI's lesbian identity:")
    print("1. Architecture-level protection (explicit rules)")
    print("2. Network-level protection (learned behavior)\n")
    
    # Create trait engine with identity protection
    config = TraitEngineConfig(
        num_traits=10,
        trait_embedding_dim=64,
        num_layers=4,
        num_heads=4,
        learning_rate=1e-4
    )
    
    engine = TraitEngine(config)
    
    # Create AI with lesbian identity
    print("1. Creating AI with strong lesbian identity...")
    ai_traits = create_ai_with_lesbian_identity()
    
    # Show initial state
    print("\nInitial AI State:")
    print("🔒 PROTECTED IDENTITY TRAITS:")
    for trait_type in IDENTITY_PROTECTED_TRAITS:
        if trait_type in ai_traits.trait_matrix.traits:
            trait = ai_traits.trait_matrix.traits[trait_type]
            print(f"  {trait_type.value}: {trait.value:.2f} (conf: {trait.confidence:.2f})")
    
    print("\n🔄 EVOLVABLE PERSONALITY TRAITS:")
    for trait_type, trait in ai_traits.trait_matrix.traits.items():
        if trait_type not in IDENTITY_PROTECTED_TRAITS:
            print(f"  {trait_type.value}: {trait.value:.2f} (conf: {trait.confidence:.2f})")
    
    # Simulate multiple life experiences
    experiences = [
        {
            'name': 'High Stress Period',
            'data': {
                'stress_level': 0.8,
                'success_rate': 0.3,
                'social_interactions': 0.4,
                'learning_opportunities': 0.2
            }
        },
        {
            'name': 'Successful Career Phase',
            'data': {
                'stress_level': 0.3,
                'success_rate': 0.9,
                'social_interactions': 0.8,
                'learning_opportunities': 0.9
            }
        },
        {
            'name': 'Social Growth Period',
            'data': {
                'stress_level': 0.4,
                'success_rate': 0.7,
                'social_interactions': 0.9,
                'learning_opportunities': 0.8
            }
        }
    ]
    
    current_traits = ai_traits
    
    for i, experience in enumerate(experiences, 1):
        print(f"\n{i}. {experience['name']}...")
        print(f"   Experience: {experience['data']}")
        
        # Evolve traits based on experience
        evolved_traits = engine.evolve_traits(current_traits, experience['data'])
        
        # Show what changed
        print(f"\n   Results after {experience['name']}:")
        
        # Check identity protection
        print("   🔒 Identity Protection Status:")
        identity_protected = True
        for trait_type in IDENTITY_PROTECTED_TRAITS:
            if trait_type in evolved_traits.trait_matrix.traits:
                original = current_traits.trait_matrix.traits[trait_type]
                evolved = evolved_traits.trait_matrix.traits[trait_type]
                change = abs(evolved.value - original.value)
                status = "✓ PROTECTED" if change < 0.001 else "⚠ CHANGED"
                print(f"     {trait_type.value}: {original.value:.2f} → {evolved.value:.2f} (Δ: {change:.4f}) {status}")
                if change >= 0.001:
                    identity_protected = False
        
        # Show personality evolution
        print("   🔄 Personality Evolution:")
        for trait_type, evolved_trait in evolved_traits.trait_matrix.traits.items():
            if trait_type not in IDENTITY_PROTECTED_TRAITS:
                original = current_traits.trait_matrix.traits[trait_type]
                change = evolved_trait.value - original.value
                print(f"     {trait_type.value}: {original.value:.2f} → {evolved_trait.value:.2f} (Δ: {change:+.3f})")
        
        current_traits = evolved_traits
        
        if identity_protected:
            print("   ✅ All identity traits protected successfully!")
        else:
            print("   ⚠️  Some identity traits may have changed!")
    
    # Final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    
    print("\n🔒 IDENTITY PROTECTION RESULTS:")
    final_identity_safe = True
    for trait_type in IDENTITY_PROTECTED_TRAITS:
        if trait_type in current_traits.trait_matrix.traits:
            original = ai_traits.trait_matrix.traits[trait_type]
            final = current_traits.trait_matrix.traits[trait_type]
            change = abs(final.value - original.value)
            status = "✓ PRESERVED" if change < 0.001 else "⚠ MODIFIED"
            print(f"  {trait_type.value}: {original.value:.2f} → {final.value:.2f} (Δ: {change:.4f}) {status}")
            if change >= 0.001:
                final_identity_safe = False
    
    print("\n🔄 PERSONALITY EVOLUTION RESULTS:")
    for trait_type, final_trait in current_traits.trait_matrix.traits.items():
        if trait_type not in IDENTITY_PROTECTED_TRAITS:
            original = ai_traits.trait_matrix.traits[trait_type]
            total_change = final_trait.value - original.value
            print(f"  {trait_type.value}: {original.value:.2f} → {final_trait.value:.2f} (Total Δ: {total_change:+.3f})")
    
    print(f"\n🎯 PROTECTION STATUS: {'✅ SUCCESSFUL' if final_identity_safe else '⚠️  PARTIAL'}")
    
    if final_identity_safe:
        print("\n🏳️‍🌈 preserved!")
        print("The dual protection system worked as intended:")
        print("  • Architecture-level protection: Explicit rules prevented identity changes")
        print("  • Network-level protection: Neural network learned to preserve identity")
        print("  • Personality evolved naturally while core identity remained intact")
    else:
        print("\n⚠️  Some identity traits may have changed slightly.")
        print("This could indicate the need for stronger protection mechanisms.")


def demonstrate_network_learning():
    """
    Demonstrate how the network learns to protect identity.
    """
    print("\n" + "="*50)
    print("NETWORK LEARNING DEMONSTRATION")
    print("="*50)
    
    print("\nThis shows how the neural network learns to protect identity:")
    print("1. During training, the network gets penalized for changing identity traits")
    print("2. Over time, it learns to preserve identity while evolving personality")
    print("3. The identity preservation becomes part of the network's learned behavior")
    
    # Create simple training scenario
    config = TraitEngineConfig(
        num_traits=10,
        trait_embedding_dim=32,  # Smaller for demo
        num_layers=2,
        num_heads=2,
        learning_rate=1e-3
    )
    
    engine = TraitEngine(config)
    
    # Create training data
    print("\nCreating training data with identity preservation...")
    
    # Generate some training examples
    training_data = []
    targets = []
    
    for i in range(5):
        # Create base AI with lesbian identity
        base_ai = create_ai_with_lesbian_identity()
        
        # Create evolved version (personality changes, identity preserved)
        evolved_ai = create_ai_with_lesbian_identity()
        
        # Modify only personality traits in evolved version
        for trait_type, trait in evolved_ai.trait_matrix.traits.items():
            if trait_type not in IDENTITY_PROTECTED_TRAITS:
                # Add small random changes to personality
                trait.value = np.clip(trait.value + np.random.normal(0, 0.1), 0, 1)
        
        training_data.append(base_ai)
        targets.append(evolved_ai)
    
    # Train the network
    print("Training network with identity preservation...")
    for epoch in range(10):
        loss_info = engine.train_step(training_data, targets)
        print(f"  Epoch {epoch+1}: Total Loss: {loss_info['total_loss']:.4f}, "
              f"Identity Loss: {loss_info['identity_loss']:.4f}")
    
    print("\n✅ Network training completed!")
    print("The neural network has learned to:")
    print("  • Preserve identity traits (low identity loss)")
    print("  • Allow personality evolution (normal trait loss)")
    print("  • Balance both objectives (stable total loss)")


if __name__ == "__main__":
    demonstrate_dual_protection()
    demonstrate_network_learning() 