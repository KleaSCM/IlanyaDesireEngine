#!/usr/bin/env python3
"""
Debug script to test interaction strength calculation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from desire_engine import DesireEngine, DesireEngineConfig
from desire_engine.models import Desire
from datetime import datetime

# Create test desires
desire_1 = Desire(
    id="test_desire_1",
    name="Desire for Creativity",
    source_traits=["creativity"],
    strength=0.8,
    base_strength=0.8,
    reinforcement_count=3,
    last_reinforcement=datetime.now()
)

desire_2 = Desire(
    id="test_desire_2", 
    name="Desire for Learning",
    source_traits=["learning_desire"],
    strength=0.7,
    base_strength=0.7,
    reinforcement_count=2,
    last_reinforcement=datetime.now()
)

# Create engine with low thresholds
config = DesireEngineConfig(
    interaction_threshold=0.05,
    synergy_threshold=0.2,
    emergent_threshold=0.3,
    conflict_threshold=0.1
)

engine = DesireEngine(config)

# Test interaction strength calculation
interaction_strength = engine.interaction_module._calculate_interaction_strength(desire_1, desire_2)
print(f"Interaction strength: {interaction_strength:.3f}")
print(f"Emergent threshold: {config.emergent_threshold}")
print(f"Would create emergent: {interaction_strength > config.emergent_threshold}")

# Test with desires that have common traits
desire_3 = Desire(
    id="test_desire_3",
    name="Desire for Creative Learning",
    source_traits=["creativity", "learning_desire"],
    strength=0.9,
    base_strength=0.9,
    reinforcement_count=5,
    last_reinforcement=datetime.now()
)

interaction_strength_2 = engine.interaction_module._calculate_interaction_strength(desire_1, desire_3)
print(f"\nInteraction strength (with common traits): {interaction_strength_2:.3f}")
print(f"Would create emergent: {interaction_strength_2 > config.emergent_threshold}")

# Test emergent desire creation
emergent = engine.interaction_module._create_emergent_desire(desire_1, desire_2, interaction_strength)
print(f"\nEmergent desire created: {emergent is not None}")
if emergent:
    print(f"Emergent ID: {emergent.id}")
    print(f"Emergent traits: {emergent.source_traits}") 