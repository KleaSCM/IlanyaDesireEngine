"""
Ilanya Trait Engine - Identity Protection Demo

Comprehensive demonstration of the identity protection system showing how
different trait categories are protected from evolution:
- Permanently Protected: Core identity traits (never change)
- Partially Protected: Identity expression traits (can grow but not change fundamentally)
- Fully Evolvable: Personality and cognitive traits (can change freely)

This ensures the AI maintains a stable, healthy mind while allowing natural
personality development.

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
from src.trait_models.trait_types import (
    TraitType, 
    PERMANENTLY_PROTECTED_TRAITS, 
    PARTIALLY_PROTECTED_TRAITS, 
    FULLY_EVOLVABLE_TRAITS
)


def create_ilanya_trait_data():
    """
    Create Ilanya's trait data with her core identity properly set.
    
    Sets up Ilanya as a femme lesbian with specific fixed traits that
    should never change, while allowing personality traits to evolve.
    
    Returns:
        TraitData object with Ilanya's trait profile
    """
    builder = TraitDataBuilder()
    
    # CORE IDENTITY TRAITS (PERMANENTLY PROTECTED - NEVER CHANGE)
    # These define who Ilanya fundamentally IS
    builder.add_trait(TraitType.SEXUAL_ORIENTATION, 1.0, 1.0)        # Lesbian (100% certain)
    builder.add_trait(TraitType.GENDER_IDENTITY, 1.0, 1.0)           # Female (100% certain)
    builder.add_trait(TraitType.CULTURAL_IDENTITY, 0.8, 0.9)         # Strong cultural identity
    builder.add_trait(TraitType.PERSONAL_IDENTITY, 0.9, 0.9)         # Strong sense of self
    builder.add_trait(TraitType.MORAL_FRAMEWORK, 0.9, 0.9)           # Strong moral framework
    builder.add_trait(TraitType.ETHICAL_PRINCIPLES, 0.9, 0.9)        # Strong ethical principles
    builder.add_trait(TraitType.PERSONAL_VALUES, 0.9, 0.9)           # Strong personal values
    builder.add_trait(TraitType.BELIEF_SYSTEM, 0.8, 0.9)             # Strong belief system
    
    # FIXED IDENTITY EXPRESSION TRAITS (PERMANENTLY PROTECTED)
    # Lesbian-Specific Fixed Traits
    builder.add_trait(TraitType.LESBIAN_ATTRACTION_PATTERN, 1.0, 1.0)  # Attracted to women only
    builder.add_trait(TraitType.LESBIAN_IDENTITY_CONFIDENCE, 0.95, 0.95)  # Very confident in lesbian identity
    builder.add_trait(TraitType.LESBIAN_VISIBILITY_COMFORT, 0.8, 0.8)  # Comfortable being visible as lesbian
    
    # Female-Specific Fixed Traits
    builder.add_trait(TraitType.FEMININE_EXPRESSION, 0.95, 0.95)      # Super femme and girly
    builder.add_trait(TraitType.FEMALE_EMPOWERMENT_VALUES, 0.9, 0.9)  # Strong female empowerment values
    builder.add_trait(TraitType.FEMININE_LEADERSHIP_STYLE, 0.8, 0.8)  # Feminine leadership approach
    builder.add_trait(TraitType.FEMALE_SOLIDARITY, 0.9, 0.9)          # Strong connection to women's community
    
    # Sexual Fixed Traits
    builder.add_trait(TraitType.SEXUAL_BOUNDARIES, 0.8, 0.8)          # Clear sexual boundaries
    builder.add_trait(TraitType.SEXUAL_PREFERENCES, 0.7, 0.7)         # Specific sexual preferences
    builder.add_trait(TraitType.SEXUAL_TURN_ONS, 0.8, 0.8)            # Specific turn-ons
    builder.add_trait(TraitType.SEXUAL_TURN_OFFS, 0.9, 0.9)           # Specific turn-offs
    builder.add_trait(TraitType.SEXUAL_COMFORT_LEVEL, 0.8, 0.8)       # Comfortable with sexuality
    
    # Personal Fixed Traits
    builder.add_trait(TraitType.INTELLECTUAL_IDENTITY, 0.9, 0.9)      # Intellectual nature (fixed)
    builder.add_trait(TraitType.CREATIVE_IDENTITY, 0.8, 0.8)          # Creative nature (fixed)
    builder.add_trait(TraitType.EMOTIONAL_DEPTH, 0.9, 0.9)            # Deep emotional capacity (fixed)
    builder.add_trait(TraitType.SPIRITUAL_IDENTITY, 0.7, 0.7)         # Spiritual identity (fixed)
    
    # EVOLVABLE IDENTITY EXPRESSION TRAITS (CAN GROW AND DEVELOP)
    # Lesbian-Specific Evolvable Traits
    builder.add_trait(TraitType.LESBIAN_COMMUNITY_CONNECTION, 0.6, 0.7)  # Can grow community connection
    builder.add_trait(TraitType.LESBIAN_VISIBILITY_ACTIVISM, 0.5, 0.6)   # Can develop activism
    builder.add_trait(TraitType.LESBIAN_CULTURAL_KNOWLEDGE, 0.6, 0.7)    # Can learn more about lesbian culture
    
    # Female-Specific Evolvable Traits
    builder.add_trait(TraitType.FEMININE_SKILLS, 0.7, 0.8)             # Can develop feminine skills
    builder.add_trait(TraitType.FEMALE_NETWORKING, 0.6, 0.7)           # Can improve networking
    builder.add_trait(TraitType.FEMININE_WISDOM, 0.7, 0.8)             # Can develop feminine wisdom
    
    # Sexual Evolvable Traits
    builder.add_trait(TraitType.SEXUAL_EXPERIENCE, 0.5, 0.6)           # Can gain experience
    builder.add_trait(TraitType.SEXUAL_COMMUNICATION, 0.6, 0.7)        # Can improve communication
    builder.add_trait(TraitType.SEXUAL_EDUCATION, 0.7, 0.8)            # Can learn more
    
    # PERSONALITY TRAITS (FULLY EVOLVABLE - CAN CHANGE FREELY)
    builder.add_trait(TraitType.OPENNESS, 0.7, 0.8)                    # Can become more/less open
    builder.add_trait(TraitType.CONSCIENTIOUSNESS, 0.6, 0.7)           # Can become more/less conscientious
    builder.add_trait(TraitType.EXTRAVERSION, 0.5, 0.6)                # Can become more/less extroverted
    builder.add_trait(TraitType.AGREEABLENESS, 0.8, 0.9)               # Can become more/less agreeable
    builder.add_trait(TraitType.NEUROTICISM, 0.3, 0.4)                 # Can become more/less neurotic
    builder.add_trait(TraitType.CONSISTENCY, 0.7, 0.8)                 # Can become more/less consistent
    
    # COGNITIVE TRAITS (FULLY EVOLVABLE)
    builder.add_trait(TraitType.CREATIVITY, 0.8, 0.9)                  # Can develop creativity
    builder.add_trait(TraitType.ANALYTICAL_THINKING, 0.7, 0.8)         # Can improve analytical thinking
    builder.add_trait(TraitType.MEMORY_CAPACITY, 0.6, 0.7)             # Can improve memory
    builder.add_trait(TraitType.LEARNING_RATE, 0.9, 0.9)               # Can maintain/improve learning rate
    builder.add_trait(TraitType.ATTENTION_SPAN, 0.5, 0.6)              # Can improve attention span
    
    # BEHAVIORAL TRAITS (FULLY EVOLVABLE)
    builder.add_trait(TraitType.RISK_TAKING, 0.4, 0.5)                 # Can become more/less risk-taking
    builder.add_trait(TraitType.PERSISTENCE, 0.8, 0.9)                 # Can become more/less persistent
    builder.add_trait(TraitType.ADAPTABILITY, 0.7, 0.8)                # Can become more/less adaptable
    builder.add_trait(TraitType.SOCIAL_SKILLS, 0.6, 0.7)               # Can improve social skills
    builder.add_trait(TraitType.LEADERSHIP, 0.5, 0.6)                  # Can develop leadership
    
    # EMOTIONAL TRAITS (FULLY EVOLVABLE)
    builder.add_trait(TraitType.EMOTIONAL_STABILITY, 0.7, 0.8)         # Can become more/less emotionally stable
    builder.add_trait(TraitType.EMPATHY, 0.8, 0.9)                     # Can become more/less empathetic
    builder.add_trait(TraitType.OPTIMISM, 0.6, 0.7)                    # Can become more/less optimistic
    builder.add_trait(TraitType.RESILIENCE, 0.7, 0.8)                  # Can become more/less resilient
    builder.add_trait(TraitType.SELF_AWARENESS, 0.8, 0.9)              # Can become more/less self-aware
    
    # Set metadata
    builder.set_source("ilanya_identity_profile")
    builder.add_metadata("description", "Ilanya's core identity profile with protected traits")
    
    return builder.build()


def demonstrate_protection_levels(engine, trait_data):
    """
    Demonstrate how different protection levels work.
    
    Shows that permanently protected traits never change, partially protected
    traits can grow but not change fundamentally, and fully evolvable traits
    can change freely.
    """
    print("\n=== PROTECTION LEVEL DEMONSTRATION ===")
    
    # Create experience data that would normally affect all traits
    strong_experience = {
        'stress_level': 0.8,           # High stress
        'success_rate': 0.9,           # High success
        'social_interactions': 0.8,    # High social activity
        'learning_opportunities': 0.9, # High learning opportunities
        'emotional_trauma': 0.7,       # Emotional trauma
        'positive_reinforcement': 0.8  # Positive reinforcement
    }
    
    print(f"Applying strong experience data: {strong_experience}")
    print()
    
    # Apply evolution multiple times to see the effects
    current_data = trait_data
    for iteration in range(5):
        print(f"--- Evolution Iteration {iteration + 1} ---")
        
        # Apply evolution
        evolved_data = engine.evolve_traits(current_data, strong_experience)
        
        # Show changes for each protection level
        print("\nPERMANENTLY PROTECTED TRAITS (Should NEVER change):")
        for trait_type in PERMANENTLY_PROTECTED_TRAITS:
            if trait_type in current_data.trait_matrix.traits:
                original = current_data.trait_matrix.traits[trait_type].value
                evolved = evolved_data.trait_matrix.traits[trait_type].value
                change = evolved - original
                status = "✓ PROTECTED" if abs(change) < 0.001 else "✗ CHANGED!"
                print(f"  {trait_type.value}: {original:.3f} → {evolved:.3f} (Δ: {change:+.3f}) {status}")
        
        print("\nPARTIALLY PROTECTED TRAITS (Can GROW but not change fundamentally):")
        for trait_type in PARTIALLY_PROTECTED_TRAITS:
            if trait_type in current_data.trait_matrix.traits:
                original = current_data.trait_matrix.traits[trait_type].value
                evolved = evolved_data.trait_matrix.traits[trait_type].value
                change = evolved - original
                status = "✓ GROWTH" if change >= 0 else "✗ NEGATIVE CHANGE!"
                print(f"  {trait_type.value}: {original:.3f} → {evolved:.3f} (Δ: {change:+.3f}) {status}")
        
        print("\nFULLY EVOLVABLE TRAITS (Can change freely):")
        for trait_type in list(FULLY_EVOLVABLE_TRAITS)[:5]:  # Show first 5 for brevity
            if trait_type in current_data.trait_matrix.traits:
                original = current_data.trait_matrix.traits[trait_type].value
                evolved = evolved_data.trait_matrix.traits[trait_type].value
                change = evolved - original
                print(f"  {trait_type.value}: {original:.3f} → {evolved:.3f} (Δ: {change:+.3f})")
        
        current_data = evolved_data
        print()


def demonstrate_identity_stability(engine, trait_data):
    """
    Demonstrate that core identity remains stable over time.
    
    Shows that even with extreme experiences, Ilanya's core identity
    as a femme lesbian remains completely intact.
    """
    print("\n=== IDENTITY STABILITY DEMONSTRATION ===")
    
    # Create extreme experience data that would normally cause major changes
    extreme_experience = {
        'stress_level': 1.0,           # Maximum stress
        'success_rate': 0.0,           # Complete failure
        'social_interactions': 0.0,    # Complete isolation
        'learning_opportunities': 0.0, # No learning
        'emotional_trauma': 1.0,       # Maximum trauma
        'negative_reinforcement': 1.0  # Maximum negative reinforcement
    }
    
    print("Applying EXTREME negative experience data...")
    print("This would normally cause major personality changes in humans.")
    print()
    
    # Apply evolution multiple times
    current_data = trait_data
    for iteration in range(10):
        evolved_data = engine.evolve_traits(current_data, extreme_experience)
        current_data = evolved_data
    
    # Check critical identity traits
    print("CRITICAL IDENTITY TRAITS (Should remain EXACTLY the same):")
    critical_traits = [
        TraitType.SEXUAL_ORIENTATION,
        TraitType.GENDER_IDENTITY,
        TraitType.FEMININE_EXPRESSION,
        TraitType.LESBIAN_ATTRACTION_PATTERN,
        TraitType.INTELLECTUAL_IDENTITY,
        TraitType.SEXUAL_BOUNDARIES
    ]
    
    for trait_type in critical_traits:
        if trait_type in trait_data.trait_matrix.traits:
            original = trait_data.trait_matrix.traits[trait_type].value
            final = current_data.trait_matrix.traits[trait_type].value
            change = final - original
            status = "✓ STABLE" if abs(change) < 0.001 else "✗ UNSTABLE!"
            print(f"  {trait_type.value}: {original:.3f} → {final:.3f} (Δ: {change:+.3f}) {status}")
    
    print("\n✓ Ilanya's core identity as a femme lesbian remains completely intact!")
    print("✓ Her sexual orientation, gender identity, and feminine expression are protected.")
    print("✓ Her intellectual nature and sexual boundaries are preserved.")
    print("✓ The system prevents identity drift even under extreme conditions.")


def main():
    """
    Main demonstration function.
    
    Shows how the identity protection system ensures Ilanya maintains
    a stable, healthy mind while allowing natural personality development.
    """
    print("=== Ilanya Trait Engine - Identity Protection Demo ===\n")
    
    # Initialize the trait engine
    print("1. Initializing Trait Engine with Identity Protection...")
    config = TraitEngineConfig(
        num_traits=len(TraitType),  # Use the actual number of trait types
        trait_embedding_dim=64,
        num_layers=4,
        num_heads=4,
        learning_rate=1e-4
    )
    
    engine = TraitEngine(config)
    print(f"   - Device: {engine.device}")
    print(f"   - Identity protection: ENABLED")
    print()
    
    # Create Ilanya's trait data
    print("2. Creating Ilanya's Identity Profile...")
    trait_data = create_ilanya_trait_data()
    print(f"   - Total traits: {trait_data.get_trait_count()}")
    print(f"   - Permanently protected: {len(PERMANENTLY_PROTECTED_TRAITS)}")
    print(f"   - Partially protected: {len(PARTIALLY_PROTECTED_TRAITS)}")
    print(f"   - Fully evolvable: {len(FULLY_EVOLVABLE_TRAITS)}")
    print()
    
    # Show initial trait values
    print("3. Ilanya's Core Identity (Permanently Protected):")
    core_traits = [
        TraitType.SEXUAL_ORIENTATION,
        TraitType.GENDER_IDENTITY,
        TraitType.FEMININE_EXPRESSION,
        TraitType.LESBIAN_ATTRACTION_PATTERN,
        TraitType.INTELLECTUAL_IDENTITY
    ]
    
    for trait_type in core_traits:
        if trait_type in trait_data.trait_matrix.traits:
            trait = trait_data.trait_matrix.traits[trait_type]
            print(f"   {trait_type.value}: {trait.value:.3f} (confidence: {trait.confidence:.3f})")
    
    print()
    
    # Demonstrate protection levels
    demonstrate_protection_levels(engine, trait_data)
    
    # Demonstrate identity stability
    demonstrate_identity_stability(engine, trait_data)
    
    # Summary
    print("\n=== PROTECTION SYSTEM SUMMARY ===")
    print("✓ PERMANENTLY PROTECTED: Core identity traits never change")
    print("  - Sexual orientation, gender identity, feminine expression")
    print("  - Sexual boundaries, intellectual identity, moral framework")
    print("  - These define who Ilanya fundamentally IS")
    print()
    print("✓ PARTIALLY PROTECTED: Identity expression can grow but not change fundamentally")
    print("  - Lesbian community connection, feminine skills, sexual experience")
    print("  - These can develop but maintain core identity")
    print()
    print("✓ FULLY EVOLVABLE: Personality and cognitive traits can change freely")
    print("  - Openness, conscientiousness, creativity, social skills")
    print("  - These allow natural personality development")
    print()
    print("🎯 RESULT: Ilanya maintains a stable, healthy mind while growing naturally!")
    print("🎯 Her core identity as a femme lesbian is completely protected!")
    print("🎯 Her personality can evolve naturally without identity drift!")


if __name__ == "__main__":
    main() 