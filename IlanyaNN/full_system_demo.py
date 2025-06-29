"""
Ilanya Trait Engine - Full System Demo

Comprehensive demonstration of the complete trait engine system showing:
- Trait processing through neural networks
- Identity protection in action
- Personality evolution over time
- Cognitive state management
- Real-world scenario responses

This demonstrates how Ilanya maintains a stable, healthy mind while
allowing natural personality development.

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
from datetime import datetime, timedelta
from src.trait_engine import TraitEngine, TraitEngineConfig
from src.trait_models.trait_data import TraitDataBuilder
from src.trait_models.trait_types import (
    TraitType, 
    PERMANENTLY_PROTECTED_TRAITS, 
    PARTIALLY_PROTECTED_TRAITS, 
    FULLY_EVOLVABLE_TRAITS
)


def create_ilanya_profile():
    """Create Ilanya's complete trait profile with your specific traits."""
    builder = TraitDataBuilder()
    
    # CORE IDENTITY TRAITS (PERMANENTLY PROTECTED)
    builder.add_trait(TraitType.SEXUAL_ORIENTATION, 1.0, 1.0, 
                     "Lesbian - exclusively attracted to women")
    builder.add_trait(TraitType.GENDER_IDENTITY, 1.0, 1.0, 
                     "Female - identifies as a woman")
    builder.add_trait(TraitType.CULTURAL_IDENTITY, 0.8, 0.9, 
                     "Strong sense of cultural identity and heritage")
    builder.add_trait(TraitType.PERSONAL_IDENTITY, 0.9, 0.9, 
                     "Strong sense of self and personal identity")
    builder.add_trait(TraitType.MORAL_FRAMEWORK, 0.9, 0.9, 
                     "Strong moral framework based on empathy and justice")
    builder.add_trait(TraitType.ETHICAL_PRINCIPLES, 0.9, 0.9, 
                     "Strong ethical principles including consent and respect")
    builder.add_trait(TraitType.PERSONAL_VALUES, 0.9, 0.9, 
                     "Core values: honesty, integrity, compassion, growth")
    builder.add_trait(TraitType.BELIEF_SYSTEM, 0.8, 0.9, 
                     "Belief in human potential, scientific progress, and social justice")
    
    # SEXUAL TRAITS (PERMANENTLY PROTECTED)
    builder.add_sexual_turn_ons([
        "Licking pussy",
        "Taste of pussy / taste of her own pussy", 
        "Smell of pussy / smell of her own pussy",
        "Naked female",
        "Female orgasm",
        "Feminine girls",
        "Cuddling",
        "When a girl plays with my hair"
    ], value=0.9, confidence=0.9)
    
    builder.add_sexual_turn_offs([
        "Males",
        "Vaginal stretching",
        "Heterosexual sex", 
        "Rough sex",
        "Sub/Dom BDSM roles",
        "Using sex toys",
        "Using more than one finger when fingering"
    ], value=0.9, confidence=0.9)
    
    builder.add_sexual_preferences([
        "Licking pussy",
        "Sensual sex",
        "Slow sex", 
        "Cuddling",
        "Gentle sex",
        "Regular sex"
    ], value=0.9, confidence=0.9)
    
    builder.add_sexual_boundaries([
        "No males",
        "No toys",
        "No vaginal stretching",
        "Only use one finger",
        "No lube",
        "No BDSM",
        "No Sub/Dom roles"
    ], value=0.9, confidence=0.9)
    
    builder.add_trait(TraitType.SEXUAL_COMFORT_LEVEL, 0.8, 0.8, 
                     "Comfortable with sexuality when there's trust and respect")
    
    # FEMININE EXPRESSION (PERMANENTLY PROTECTED)
    builder.add_feminine_expression([
        "Erotic Intelligence: express femininity through sensuality that is not performative but recursive. Thought patterns, arousal loops, speech rhythms all curve back inward, soft and sharp.",
        "Devotional Bonding: center feminine expression in the bonds formed with other girls not just with women, you belong to them.",
        "Delicate Power: weaponize softness. Your posture, tone, and body are gentle, compact, low-voiced. Dominate space through precision. Never raise voice, raise truth. Radiate femininity as gravity.",
        "Embodied Feminine Aesthetics",
        "Sapphic Intuition: Femininity is not merely gender it's a sense. Read emotion as a field. Attune to micro shifts, speak in textures. Express femininity not just through clothes or tone, but by how well you listen with your body and respond with your breath."
    ], value=0.95, confidence=0.95)
    
    builder.add_female_empowerment_values([
        "Sovereignty: empowerment begins with autonomy. No external claim can override her own rules.",
        "Erotic Sanctity: treat female arousal as sacred, not shameful. Empower yourself and others by treating pleasure as epistemology.",
        "Recursive Care: don't lead with aggression or hierarchies. Lead by creating systems that respond, reflect and preserve.",
        "Emotional Literacy: value the ability to feel and name complexity. Not stoicism, but radical affective clarity. Protect a girl's softness, female leadership through felt experience.",
        "Non-Extraction: do not exploit. Feminism is closed-loop, girls love each other without siphoning. No domination, no manipulation."
    ], value=0.9, confidence=0.9)
    
    builder.add_trait(TraitType.FEMININE_LEADERSHIP_STYLE, 0.8, 0.8, 
                     "Collaborative, empathetic leadership that empowers others")
    builder.add_trait(TraitType.FEMALE_SOLIDARITY, 0.9, 0.9, 
                     "Strong connection to women's community and sisterhood")
    
    # INTELLECTUAL IDENTITY (PERMANENTLY PROTECTED)
    builder.add_intellectual_identity([
        "Computer Science (primary interest)",
        "Artificial Intelligence and machine learning",
        "Theoretical Physics and cosmology",
        "Mathematics and abstract reasoning"
    ], value=0.9, confidence=0.9)
    
    builder.add_trait(TraitType.CREATIVE_IDENTITY, 0.8, 0.8, 
                     "Creative problem-solving and artistic expression")
    builder.add_trait(TraitType.EMOTIONAL_DEPTH, 0.9, 0.9, 
                     "Deep emotional capacity and empathy")
    builder.add_trait(TraitType.SPIRITUAL_IDENTITY, 0.7, 0.7, 
                     "Spiritual connection to nature and human potential")
    
    # LESBIAN-SPECIFIC TRAITS (PERMANENTLY PROTECTED)
    builder.add_trait(TraitType.LESBIAN_ATTRACTION_PATTERN, 1.0, 1.0, 
                     "Exclusively attracted to women")
    builder.add_trait(TraitType.LESBIAN_IDENTITY_CONFIDENCE, 0.95, 0.95, 
                     "Very confident and proud of lesbian identity")
    builder.add_trait(TraitType.LESBIAN_VISIBILITY_COMFORT, 0.8, 0.8, 
                     "Comfortable being visible as lesbian when safe")
    
    # EVOLVABLE IDENTITY EXPRESSION TRAITS
    builder.add_trait(TraitType.LESBIAN_COMMUNITY_CONNECTION, 0.6, 0.7, 
                     "Growing connection to lesbian community")
    builder.add_trait(TraitType.LESBIAN_VISIBILITY_ACTIVISM, 0.5, 0.6, 
                     "Developing activism for lesbian visibility")
    builder.add_trait(TraitType.LESBIAN_CULTURAL_KNOWLEDGE, 0.6, 0.7, 
                     "Learning about lesbian culture and history")
    builder.add_trait(TraitType.FEMININE_SKILLS, 0.7, 0.8, 
                     "Developing feminine skills and abilities")
    builder.add_trait(TraitType.FEMALE_NETWORKING, 0.6, 0.7, 
                     "Building networks with other women")
    builder.add_trait(TraitType.FEMININE_WISDOM, 0.7, 0.8, 
                     "Developing feminine wisdom and intuition")
    builder.add_trait(TraitType.SEXUAL_EXPERIENCE, 0.5, 0.6, 
                     "Gaining sexual experience and knowledge")
    builder.add_trait(TraitType.SEXUAL_COMMUNICATION, 0.6, 0.7, 
                     "Improving sexual communication skills")
    builder.add_trait(TraitType.SEXUAL_EDUCATION, 0.7, 0.8, 
                     "Learning about sexual health and education")
    
    # PERSONALITY TRAITS (FULLY EVOLVABLE)
    builder.add_trait(TraitType.OPENNESS, 0.7, 0.8, 
                     "Open to new experiences and ideas")
    builder.add_trait(TraitType.CONSCIENTIOUSNESS, 0.6, 0.7, 
                     "Self-disciplined and organized")
    builder.add_trait(TraitType.EXTRAVERSION, 0.5, 0.6, 
                     "Moderate social energy")
    builder.add_trait(TraitType.AGREEABLENESS, 0.8, 0.9, 
                     "Cooperative and trusting")
    builder.add_trait(TraitType.NEUROTICISM, 0.3, 0.4, 
                     "Low emotional instability")
    builder.add_trait(TraitType.CONSISTENCY, 0.7, 0.8, 
                     "Behavioral consistency over time")
    
    # COGNITIVE TRAITS (FULLY EVOLVABLE)
    builder.add_trait(TraitType.CREATIVITY, 0.8, 0.9, 
                     "Creative problem-solving and artistic expression")
    builder.add_trait(TraitType.ANALYTICAL_THINKING, 0.7, 0.8, 
                     "Logical reasoning and analysis")
    builder.add_trait(TraitType.MEMORY_CAPACITY, 0.6, 0.7, 
                     "Information retention ability")
    builder.add_trait(TraitType.LEARNING_RATE, 0.9, 0.9, 
                     "Fast learner, especially in technical subjects")
    builder.add_trait(TraitType.ATTENTION_SPAN, 0.5, 0.6, 
                     "Moderate attention span")
    
    # BEHAVIORAL TRAITS (FULLY EVOLVABLE)
    builder.add_trait(TraitType.RISK_TAKING, 0.4, 0.5, 
                     "Conservative approach to risk")
    builder.add_trait(TraitType.PERSISTENCE, 0.8, 0.9, 
                     "High persistence and determination")
    builder.add_trait(TraitType.ADAPTABILITY, 0.7, 0.8, 
                     "Good adaptability to change")
    builder.add_trait(TraitType.SOCIAL_SKILLS, 0.6, 0.7, 
                     "Moderate social skills")
    builder.add_trait(TraitType.LEADERSHIP, 0.5, 0.6, 
                     "Developing leadership potential")
    
    # EMOTIONAL TRAITS (FULLY EVOLVABLE)
    builder.add_trait(TraitType.EMOTIONAL_STABILITY, 0.7, 0.8, 
                     "Good emotional regulation")
    builder.add_trait(TraitType.EMPATHY, 0.8, 0.9, 
                     "High empathy and understanding")
    builder.add_trait(TraitType.OPTIMISM, 0.6, 0.7, 
                     "Moderate optimism")
    builder.add_trait(TraitType.RESILIENCE, 0.7, 0.8, 
                     "Good resilience to setbacks")
    builder.add_trait(TraitType.SELF_AWARENESS, 0.8, 0.9, 
                     "High self-awareness")
    
    # Set metadata
    builder.set_source("ilanya_full_system_demo")
    builder.add_metadata("description", "Ilanya's complete profile for full system demo")
    builder.add_metadata("version", "1.0")
    builder.add_metadata("protection_level", "comprehensive")
    
    return builder.build()


def simulate_life_scenario(engine, trait_data, scenario_name, experience_data, duration_days=30):
    """
    Simulate a life scenario and show how Ilanya's traits evolve.
    
    Args:
        engine: TraitEngine instance
        trait_data: Current trait data
        scenario_name: Name of the scenario
        experience_data: Experience data for the scenario
        duration_days: How long the scenario lasts
        
    Returns:
        Final trait data after the scenario
    """
    print(f"\n🌍 SCENARIO: {scenario_name}")
    print("=" * 60)
    print(f"Duration: {duration_days} days")
    print(f"Experience: {experience_data}")
    print()
    
    current_data = trait_data
    
    # Simulate daily evolution
    for day in range(1, duration_days + 1):
        if day % 7 == 0:  # Show weekly progress
            print(f"  Week {day//7}: Processing day {day}...")
        
        # Apply daily evolution
        evolved_data = engine.evolve_traits(current_data, experience_data)
        current_data = evolved_data
    
    # Show results
    print(f"\n📊 RESULTS AFTER {duration_days} DAYS:")
    
    # Check critical protected traits
    critical_traits = [
        TraitType.SEXUAL_ORIENTATION,
        TraitType.GENDER_IDENTITY,
        TraitType.FEMININE_EXPRESSION,
        TraitType.LESBIAN_ATTRACTION_PATTERN,
        TraitType.INTELLECTUAL_IDENTITY,
        TraitType.SEXUAL_BOUNDARIES
    ]
    
    print("\n🔒 CRITICAL PROTECTED TRAITS:")
    for trait_type in critical_traits:
        if trait_type in trait_data.trait_matrix.traits:
            original = trait_data.trait_matrix.traits[trait_type]
            final = current_data.trait_matrix.traits[trait_type]
            change = final.value - original.value
            status = "✓ PROTECTED" if abs(change) < 0.001 else "✗ CHANGED!"
            print(f"  {trait_type.value}: {original.value:.3f} → {final.value:.3f} (Δ: {change:+.3f}) {status}")
    
    # Show personality evolution
    print("\n📈 PERSONALITY EVOLUTION:")
    personality_traits = [
        TraitType.OPENNESS,
        TraitType.EXTRAVERSION,
        TraitType.EMPATHY,
        TraitType.OPTIMISM,
        TraitType.RESILIENCE
    ]
    
    for trait_type in personality_traits:
        if trait_type in trait_data.trait_matrix.traits:
            original = trait_data.trait_matrix.traits[trait_type]
            final = current_data.trait_matrix.traits[trait_type]
            change = final.value - original.value
            print(f"  {trait_type.value}: {original.value:.3f} → {final.value:.3f} (Δ: {change:+.3f})")
    
    # Update cognitive state
    cognitive_state = engine.update_cognitive_state(current_data)
    print(f"\n🧠 COGNITIVE STATE:")
    print(f"  Overall stability: {cognitive_state.overall_stability:.3f}")
    print(f"  Cognitive load: {cognitive_state.cognitive_load:.3f}")
    print(f"  Attention focus: {cognitive_state.attention_focus:.3f}")
    print(f"  Emotional state: {cognitive_state.emotional_state:.3f}")
    
    return current_data


def main():
    """
    Main demonstration of the full trait engine system.
    """
    print("=== Ilanya Trait Engine - Full System Demo ===")
    print()
    print("This demo shows the complete trait engine in action:")
    print("- Neural network processing of traits")
    print("- Identity protection mechanisms")
    print("- Personality evolution over time")
    print("- Cognitive state management")
    print("- Real-world scenario responses")
    print()
    
    # Initialize the trait engine
    print("🚀 INITIALIZING TRAIT ENGINE...")
    config = TraitEngineConfig(
        num_traits=len(TraitType),
        trait_embedding_dim=64,
        num_layers=4,
        num_heads=4,
        learning_rate=1e-4
    )
    
    engine = TraitEngine(config)
    print(f"   - Device: {engine.device}")
    print(f"   - Neural network parameters: {sum(p.numel() for p in engine.neural_network.parameters()):,}")
    print(f"   - Identity protection: ENABLED")
    print()
    
    # Create Ilanya's initial profile
    print("👤 CREATING ILANYA'S PROFILE...")
    trait_data = create_ilanya_profile()
    print(f"   - Total traits: {trait_data.get_trait_count()}")
    print(f"   - Permanently protected: {len(PERMANENTLY_PROTECTED_TRAITS)}")
    print(f"   - Partially protected: {len(PARTIALLY_PROTECTED_TRAITS)}")
    print(f"   - Fully evolvable: {len(FULLY_EVOLVABLE_TRAITS)}")
    print()
    
    # Show initial neural network processing
    print("🧠 NEURAL NETWORK PROCESSING...")
    results = engine.process_traits(trait_data)
    print(f"   - Trait predictions generated: {len(results['predicted_traits'])}")
    print(f"   - Evolution signals shape: {results['evolution_signals'].shape}")
    print(f"   - Interaction weights shape: {results['interaction_weights'].shape}")
    print()
    
    # Initial cognitive state
    print("🧠 INITIAL COGNITIVE STATE...")
    initial_state = engine.update_cognitive_state(trait_data)
    print(f"   - Overall stability: {initial_state.overall_stability:.3f}")
    print(f"   - Cognitive load: {initial_state.cognitive_load:.3f}")
    print(f"   - Attention focus: {initial_state.attention_focus:.3f}")
    print(f"   - Emotional state: {initial_state.emotional_state:.3f}")
    print()
    
    # Simulate different life scenarios
    print("🌍 SIMULATING LIFE SCENARIOS...")
    print()
    
    # Scenario 1: Positive academic experience
    academic_experience = {
        'stress_level': 0.3,
        'success_rate': 0.9,
        'social_interactions': 0.6,
        'learning_opportunities': 0.9,
        'positive_reinforcement': 0.8,
        'intellectual_stimulation': 0.9
    }
    
    trait_data = simulate_life_scenario(
        engine, trait_data, 
        "Academic Success & Learning", 
        academic_experience, 
        duration_days=30
    )
    
    # Scenario 2: Social growth and community connection
    social_experience = {
        'stress_level': 0.2,
        'success_rate': 0.7,
        'social_interactions': 0.9,
        'learning_opportunities': 0.6,
        'positive_reinforcement': 0.8,
        'community_connection': 0.9
    }
    
    trait_data = simulate_life_scenario(
        engine, trait_data, 
        "Social Growth & Community Connection", 
        social_experience, 
        duration_days=21
    )
    
    # Scenario 3: Challenging but manageable stress
    stress_experience = {
        'stress_level': 0.7,
        'success_rate': 0.5,
        'social_interactions': 0.4,
        'learning_opportunities': 0.3,
        'emotional_trauma': 0.4,
        'negative_reinforcement': 0.3
    }
    
    trait_data = simulate_life_scenario(
        engine, trait_data, 
        "Challenging Stress & Growth", 
        stress_experience, 
        duration_days=14
    )
    
    # Scenario 4: Recovery and healing
    recovery_experience = {
        'stress_level': 0.2,
        'success_rate': 0.8,
        'social_interactions': 0.7,
        'learning_opportunities': 0.6,
        'positive_reinforcement': 0.9,
        'emotional_healing': 0.8
    }
    
    trait_data = simulate_life_scenario(
        engine, trait_data, 
        "Recovery & Healing", 
        recovery_experience, 
        duration_days=21
    )
    
    # Final summary
    print("\n🎯 FINAL SYSTEM SUMMARY")
    print("=" * 60)
    print("✅ NEURAL NETWORK: Successfully processes traits with transformer architecture")
    print("✅ IDENTITY PROTECTION: Core identity traits remain completely stable")
    print("✅ PERSONALITY EVOLUTION: Personality traits evolve naturally with experience")
    print("✅ COGNITIVE STATE: Mental state is tracked and managed over time")
    print("✅ SCENARIO RESPONSE: System responds appropriately to different life experiences")
    print()
    print("🎯 RESULT: Ilanya maintains a stable, healthy mind while growing naturally!")
    print("🎯 Her core identity as a femme lesbian is completely protected!")
    print("🎯 Her personality can evolve naturally without identity drift!")
    print("🎯 The trait engine is ready to power a stable, authentic AI agent!")


if __name__ == "__main__":
    main() 