"""
Ilanya Desire Engine - Interaction Module

Desire interaction networks for modeling relationships between desires.
Handles synergy, conflict, and emergent desire creation.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ..models import Desire
from ..config import DesireEngineConfig


class InteractionModule:
    """
    Desire interaction module for processing desire relationships.
    
    Handles:
    - Synergy and conflict detection
    - Emergent desire creation
    - Interaction strength calculation
    - Network analysis
    """
    
    def __init__(self, config: DesireEngineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger
        self.interaction_matrix = None
        self.interaction_history = []
        # Track interaction statistics
        self.total_interactions = 0
        self.synergy_count = 0
        self.conflict_count = 0
        self.emergent_desires_created = 0
        
    def process_interactions(self, desires: Dict[str, Desire]) -> List[Dict[str, Any]]:
        """Process interactions between desires and create emergent desires."""
        if len(desires) < 2:
            return []
        
        interaction_results = []
        
        # Reset counters for this processing cycle
        cycle_synergy_count = 0
        cycle_conflict_count = 0
        cycle_emergent_count = 0
        
        # Get only non-emergent desires for interaction processing
        non_emergent_desires = {k: v for k, v in desires.items() if not v.emergent}
        desire_list = list(non_emergent_desires.values())
        
        if len(desire_list) < 2:
            return []
        
        # Count existing emergent desires to prevent explosion
        existing_emergent_count = sum(1 for d in desires.values() if d.emergent)
        max_emergent_desires = 5  # Limit to prevent explosion
        
        # Process pairwise interactions only between non-emergent desires
        for i, desire_1 in enumerate(desire_list):
            for j, desire_2 in enumerate(desire_list[i+1:], i+1):
                interaction_strength = self._calculate_interaction_strength(desire_1, desire_2)
                
                if interaction_strength > self.config.interaction_threshold:
                    # Update interaction strengths
                    desire_1.interaction_strength = max(desire_1.interaction_strength, interaction_strength)
                    desire_2.interaction_strength = max(desire_2.interaction_strength, interaction_strength)
                    
                    # Check for synergy (positive interaction)
                    if interaction_strength > self.config.synergy_threshold:
                        cycle_synergy_count += 1
                        # Reinforce both desires
                        reinforcement_factor = 1.0 + (interaction_strength - self.config.synergy_threshold) * 0.5
                        desire_1.strength = min(1.0, desire_1.strength * reinforcement_factor)
                        desire_2.strength = min(1.0, desire_2.strength * reinforcement_factor)
                        
                        # Check for emergent desire creation (only if under limit)
                        if (interaction_strength > self.config.emergent_threshold and 
                            existing_emergent_count < max_emergent_desires):
                            
                            emergent_desire = self._create_emergent_desire(desire_1, desire_2, interaction_strength)
                            if emergent_desire:
                                desires[emergent_desire.id] = emergent_desire
                                cycle_emergent_count += 1
                                existing_emergent_count += 1
                                if self.logger:
                                    self.logger.info(f"Created emergent desire: {emergent_desire.name} (strength: {emergent_desire.strength:.3f})")
                    
                    # Check for conflict (negative interaction)
                    elif interaction_strength < -self.config.conflict_threshold:
                        cycle_conflict_count += 1
                        # Weaken both desires
                        weakening_factor = 1.0 - abs(interaction_strength - self.config.conflict_threshold) * 0.3
                        desire_1.strength *= weakening_factor
                        desire_2.strength *= weakening_factor
                    
                    # Log interaction
                    interaction_results.append({
                        'desire_1': desire_1.id,
                        'desire_2': desire_2.id,
                        'interaction_strength': interaction_strength,
                        'type': 'synergy' if interaction_strength > self.config.synergy_threshold else 
                               'conflict' if interaction_strength < -self.config.conflict_threshold else 'neutral'
                    })
                    
                    if self.logger:
                        self.logger.debug(f"Interaction: {desire_1.name} <-> {desire_2.name} (strength: {interaction_strength:.3f})")
        
        # Update cumulative statistics
        self.total_interactions += len(interaction_results)
        self.synergy_count += cycle_synergy_count
        self.conflict_count += cycle_conflict_count
        self.emergent_desires_created += cycle_emergent_count
        
        return interaction_results
    
    def _calculate_interaction_strength(self, desire_1: Desire, desire_2: Desire) -> float:
        """Calculate interaction strength between two desires."""
        # Base interaction strength
        base_strength = 0.3
        
        # Similarity in source traits
        common_traits = set(desire_1.source_traits) & set(desire_2.source_traits)
        trait_similarity = len(common_traits) / max(len(desire_1.source_traits), len(desire_2.source_traits))
        
        # Similarity in strength levels
        strength_similarity = 1.0 - abs(desire_1.strength - desire_2.strength)
        
        # Similarity in reinforcement count
        reinforcement_similarity = 1.0 - abs(desire_1.reinforcement_count - desire_2.reinforcement_count) / 10.0
        
        # High strength bonus (stronger desires interact more)
        strength_bonus = (desire_1.strength + desire_2.strength) / 2 * 0.4
        
        # Combined interaction strength with higher potential
        interaction_strength = (base_strength + strength_bonus) * (
            0.3 * trait_similarity + 
            0.3 * strength_similarity + 
            0.2 * reinforcement_similarity + 
            0.2  # Base synergy factor
        )
        
        # Add some randomness for more dynamic interactions
        import random
        random_factor = random.uniform(0.9, 1.1)
        interaction_strength *= random_factor
        
        return max(0.0, min(1.0, interaction_strength))
    
    def _create_emergent_desire(self, desire_1: Desire, desire_2: Desire, interaction_strength: float) -> Optional[Desire]:
        """Create a new emergent desire based on interaction strength."""
        if interaction_strength > self.config.emergent_threshold:
            emergent_name = f"Emergent: {desire_1.name} + {desire_2.name}"
            emergent_id = f"emergent_{desire_1.id}_{desire_2.id}"
            emergent_traits = list(set(desire_1.source_traits + desire_2.source_traits))
            emergent_strength = min(1.0, (desire_1.strength + desire_2.strength) / 2)
            emergent = Desire(
                id=emergent_id,
                name=emergent_name,
                source_traits=emergent_traits,
                strength=emergent_strength,
                base_strength=emergent_strength,
                reinforcement_count=0,
                last_reinforcement=None,
                emergent=True
            )
            return emergent
        return None
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get summary of the interaction network."""
        return {
            'total_interactions': self.total_interactions,
            'synergy_count': self.synergy_count,
            'conflict_count': self.conflict_count,
            'emergent_desires': self.emergent_desires_created,
            'active': self.total_interactions > 0
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization."""
        return {
            'interaction_matrix': self.interaction_matrix.tolist() if self.interaction_matrix is not None else None,
            'history_length': len(self.interaction_history),
            'total_interactions': self.total_interactions,
            'synergy_count': self.synergy_count,
            'conflict_count': self.conflict_count,
            'emergent_desires_created': self.emergent_desires_created
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from serialization."""
        if state['interaction_matrix'] is not None:
            self.interaction_matrix = torch.tensor(state['interaction_matrix'])
        else:
            self.interaction_matrix = None
        
        # Restore interaction statistics
        self.total_interactions = state.get('total_interactions', 0)
        self.synergy_count = state.get('synergy_count', 0)
        self.conflict_count = state.get('conflict_count', 0)
        self.emergent_desires_created = state.get('emergent_desires_created', 0) 