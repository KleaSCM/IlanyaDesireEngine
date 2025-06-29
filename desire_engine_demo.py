"""
Ilanya Desire Engine Demo

Demonstrates the advanced Desire Engine functionality with the existing trait system.
Shows how desires are created, reinforced, pruned, and assessed for goal potential.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'IlanyaNN'))

from desire_engine import DesireEngine, DesireEngineConfig
from IlanyaNN.trait_types import TraitType
from IlanyaNN.trait_state import TraitState
from datetime import datetime, timedelta
import json


def create_sample_trait_states() -> dict:
    """Create sample trait states for demonstration."""
    current_time = datetime.now()
    
    # Create sample trait states with positive changes
    trait_states = {}
    
    # Big Five personality traits with positive changes
    trait_states[TraitType.OPENNESS] = TraitState(
        trait_type=TraitType.OPENNESS,
        current_value=0.8,
        previous_value=0.6,
        confidence=0.9,
        change_rate=0.2
    )
    
    trait_states[TraitType.CREATIVITY] = TraitState(
        trait_type=TraitType.CREATIVITY,
        current_value=0.9,
        previous_value=0.7,
        confidence=0.95,
        change_rate=0.2
    )
    
    trait_states[TraitType.EMPATHY] = TraitState(
        trait_type=TraitType.EMPATHY,
        current_value=0.85,
        previous_value=0.75,
        confidence=0.88,
        change_rate=0.1
    )
    
    trait_states[TraitType.LEADERSHIP] = TraitState(
        trait_type=TraitType.LEADERSHIP,
        current_value=0.7,
        previous_value=0.5,
        confidence=0.8,
        change_rate=0.2
    )
    
    trait_states[TraitType.ADAPTABILITY] = TraitState(
        trait_type=TraitType.ADAPTABILITY,
        current_value=0.75,
        previous_value=0.65,
        confidence=0.85,
        change_rate=0.1
    )
    
    # Identity expression traits (evolvable)
    trait_states[TraitType.LESBIAN_COMMUNITY_CONNECTION] = TraitState(
        trait_type=TraitType.LESBIAN_COMMUNITY_CONNECTION,
        current_value=0.8,
        previous_value=0.6,
        confidence=0.9,
        change_rate=0.2
    )
    
    trait_states[TraitType.FEMININE_SKILLS] = TraitState(
        trait_type=TraitType.FEMININE_SKILLS,
        current_value=0.85,
        previous_value=0.7,
        confidence=0.92,
        change_rate=0.15
    )
    
    trait_states[TraitType.SEXUAL_EXPERIENCE] = TraitState(
        trait_type=TraitType.SEXUAL_EXPERIENCE,
        current_value=0.6,
        previous_value=0.4,
        confidence=0.75,
        change_rate=0.2
    )
    
    return trait_states


def create_negative_trait_states() -> dict:
    """Create trait states with negative changes (should not create desires)."""
    trait_states = {}
    
    trait_states[TraitType.NEUROTICISM] = TraitState(
        trait_type=TraitType.NEUROTICISM,
        current_value=0.3,
        previous_value=0.5,
        confidence=0.8,
        change_rate=-0.2
    )
    
    trait_states[TraitType.RISK_TAKING] = TraitState(
        trait_type=TraitType.RISK_TAKING,
        current_value=0.2,
        previous_value=0.4,
        confidence=0.7,
        change_rate=-0.2
    )
    
    return trait_states


def print_desire_summary(desire_engine: DesireEngine, step_name: str):
    """Print a summary of the current desire state."""
    print(f"\n{'='*60}")
    print(f"DESIRE ENGINE STATE - {step_name}")
    print(f"{'='*60}")
    
    summary = desire_engine.get_desire_summary()
    
    print(f"Active Desires: {len(summary['active_desires'])}")
    print(f"Pruned Desires: {len(summary['pruned_desires'])}")
    print(f"Goal Candidates: {len(summary['goal_candidates'])}")
    
    print(f"\nMetrics:")
    for key, value in summary['metrics'].items():
        print(f"  {key}: {value:.3f}")
    
    if summary['active_desires']:
        print(f"\nActive Desires:")
        for desire_id, desire_data in summary['active_desires'].items():
            print(f"  {desire_id}: {desire_data['name']}")
            print(f"    Strength: {desire_data['strength']:.3f}")
            print(f"    State: {desire_data['state']}")
            print(f"    Reinforcement Count: {desire_data['reinforcement_count']}")
            print(f"    Goal Potential: {desire_data['goal_potential']:.3f}")
            print()
    
    if summary['goal_candidates']:
        print(f"Goal Candidates:")
        for desire_id in summary['goal_candidates']:
            desire_data = summary['active_desires'][desire_id]
            print(f"  {desire_id}: {desire_data['name']} (Potential: {desire_data['goal_potential']:.3f})")


def demonstrate_desire_creation():
    """Demonstrate desire creation from positive trait activations."""
    print("DEMONSTRATION 1: Desire Creation from Positive Traits")
    print("=" * 60)
    
    # Initialize desire engine
    config = DesireEngineConfig(
        reinforcement_threshold=0.1,  # Lower threshold for demo
        max_desires=20
    )
    desire_engine = DesireEngine(config)
    
    # Create sample trait states with positive changes
    trait_states = create_sample_trait_states()
    
    print("Processing trait activations...")
    print("Positive traits detected:")
    for trait_type, trait_state in trait_states.items():
        if trait_state.change_rate and trait_state.change_rate > 0:
            print(f"  {trait_type.value}: +{trait_state.change_rate:.2f}")
    
    # Process trait activations
    results = desire_engine.process_trait_activations(trait_states)
    
    print(f"\nResults:")
    print(f"  New desires created: {len(results['new_desires'])}")
    print(f"  Desires reinforced: {len(results['reinforced_desires'])}")
    print(f"  Desires pruned: {len(results['pruned_desires'])}")
    print(f"  Goal candidates: {len(results['goal_candidates'])}")
    
    print_desire_summary(desire_engine, "After Initial Creation")
    
    return desire_engine


def demonstrate_desire_reinforcement(desire_engine: DesireEngine):
    """Demonstrate desire reinforcement through repeated trait activations."""
    print("\n\nDEMONSTRATION 2: Desire Reinforcement")
    print("=" * 60)
    
    # Create stronger positive trait changes for reinforcement
    reinforced_trait_states = {}
    
    # Reinforce existing desires
    for trait_type in [TraitType.OPENNESS, TraitType.CREATIVITY, TraitType.EMPATHY]:
        reinforced_trait_states[trait_type] = TraitState(
            trait_type=trait_type,
            current_value=0.95,
            previous_value=0.8,
            confidence=0.95,
            change_rate=0.15
        )
    
    print("Reinforcing desires with stronger trait activations...")
    
    # Process multiple times to show reinforcement
    for i in range(3):
        results = desire_engine.process_trait_activations(reinforced_trait_states)
        print(f"  Reinforcement round {i+1}: {len(results['reinforced_desires'])} desires reinforced")
    
    print_desire_summary(desire_engine, "After Reinforcement")


def demonstrate_desire_decay_and_pruning(desire_engine: DesireEngine):
    """Demonstrate desire decay and pruning over time."""
    print("\n\nDEMONSTRATION 3: Desire Decay and Pruning")
    print("=" * 60)
    
    # Simulate time passing by manually triggering decay
    print("Simulating time passage and decay...")
    
    # Manually apply decay to all desires
    current_time = datetime.now()
    time_delta = timedelta(hours=24)  # 24 hours
    
    for desire in desire_engine.desires.values():
        desire.decay(time_delta)
    
    # Check for pruning
    pruned_desires = desire_engine._check_pruning()
    
    print(f"Desires pruned after decay: {len(pruned_desires)}")
    
    print_desire_summary(desire_engine, "After Decay and Pruning")


def demonstrate_negative_traits(desire_engine: DesireEngine):
    """Demonstrate that negative trait changes don't create desires."""
    print("\n\nDEMONSTRATION 4: Negative Traits (No Desires Created)")
    print("=" * 60)
    
    # Create trait states with negative changes
    negative_trait_states = create_negative_trait_states()
    
    print("Processing negative trait changes...")
    print("Negative traits detected:")
    for trait_type, trait_state in negative_trait_states.items():
        if trait_state.change_rate and trait_state.change_rate < 0:
            print(f"  {trait_type.value}: {trait_state.change_rate:.2f}")
    
    # Process negative trait activations
    results = desire_engine.process_trait_activations(negative_trait_states)
    
    print(f"\nResults:")
    print(f"  New desires created: {len(results['new_desires'])} (should be 0)")
    print(f"  Desires reinforced: {len(results['reinforced_desires'])}")
    print(f"  Desires pruned: {len(results['pruned_desires'])}")
    
    print_desire_summary(desire_engine, "After Negative Traits")


def demonstrate_neural_embeddings(desire_engine: DesireEngine):
    """Demonstrate the neural embedding and attention mechanisms."""
    print("\n\nDEMONSTRATION 5: Neural Embeddings and Attention")
    print("=" * 60)
    
    # Get desire embeddings
    embeddings = desire_engine.get_desire_embeddings()
    print(f"Desire embeddings shape: {embeddings.shape}")
    
    # Compute attention weights
    updated_embeddings, attention_weights = desire_engine.compute_desire_attention()
    print(f"Updated embeddings shape: {updated_embeddings.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Show attention weights for each desire
    if attention_weights.size(0) > 0:
        print("\nAttention weights for each desire:")
        active_desires = list(desire_engine.desires.values())
        for i, desire in enumerate(active_desires):
            attention_vector = attention_weights[i].tolist()
            attention_mean = attention_weights[i].mean().item()
            print(f"  {desire.name}: mean={attention_mean:.3f}, vector={attention_vector}")


def demonstrate_state_persistence(desire_engine: DesireEngine):
    """Demonstrate saving and loading of desire engine state."""
    print("\n\nDEMONSTRATION 6: State Persistence")
    print("=" * 60)
    
    # Save current state
    save_path = "desire_engine_state.json"
    desire_engine.save_state(save_path)
    print(f"State saved to: {save_path}")
    
    # Create new desire engine
    new_desire_engine = DesireEngine()
    
    # Load state
    new_desire_engine.load_state(save_path)
    print("State loaded into new desire engine")
    
    # Compare states
    original_summary = desire_engine.get_desire_summary()
    loaded_summary = new_desire_engine.get_desire_summary()
    
    print(f"Original active desires: {len(original_summary['active_desires'])}")
    print(f"Loaded active desires: {len(loaded_summary['active_desires'])}")
    print(f"Original goal candidates: {len(original_summary['goal_candidates'])}")
    print(f"Loaded goal candidates: {len(loaded_summary['goal_candidates'])}")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)


def main():
    """Run the complete desire engine demonstration."""
    print("ILANYA DESIRE ENGINE DEMONSTRATION")
    print("=" * 60)
    print("This demo shows the advanced mathematical models and neural networks")
    print("used in the Desire Engine to manage desire space dynamics.")
    print()
    
    try:
        # Run all demonstrations
        desire_engine = demonstrate_desire_creation()
        demonstrate_desire_reinforcement(desire_engine)
        demonstrate_desire_decay_and_pruning(desire_engine)
        demonstrate_negative_traits(desire_engine)
        demonstrate_neural_embeddings(desire_engine)
        demonstrate_state_persistence(desire_engine)
        
        print("\n\nDEMONSTRATION COMPLETE")
        print("=" * 60)
        print("The Desire Engine successfully demonstrated:")
        print("✓ Creation of desires from positive trait activations")
        print("✓ Reinforcement of desires through trait reinforcement")
        print("✓ Decay and pruning of weak desires over time")
        print("✓ Goal potential assessment for strongly reinforced desires")
        print("✓ Neural embeddings and attention mechanisms")
        print("✓ State persistence and loading")
        print()
        print("The system is ready for integration with the trait engine!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 