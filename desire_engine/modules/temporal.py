"""
Ilanya Desire Engine - Temporal Module

Time-based modeling for desires including decay, memory, and temporal patterns.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import math
from typing import Dict, Any
from datetime import datetime, timedelta

from ..models import Desire, DesireState
from ..config import DesireEngineConfig


class TemporalModule:
    """
    Temporal modeling module for desires.
    
    Handles time-based effects including:
    - Advanced decay models
    - Temporal attention
    - Seasonal patterns
    - Memory consolidation
    """
    
    def __init__(self, config: DesireEngineConfig):
        self.config = config
        self.temporal_memory = []
        self.seasonal_patterns = {}
        
    def update_desires(self, desires: Dict[str, Desire]):
        """Update desires with temporal information."""
        current_time = datetime.now()
        
        # Store temporal state
        temporal_state = {
            'timestamp': current_time,
            'desire_states': {desire_id: {
                'strength': desire.strength,
                'reinforcement_count': desire.reinforcement_count,
                'state': desire.state.value
            } for desire_id, desire in desires.items()}
        }
        
        self.temporal_memory.append(temporal_state)
        
        # Limit memory size
        if len(self.temporal_memory) > self.config.temporal_memory_size:
            self.temporal_memory.pop(0)
    
    def apply_decay(self, desires: Dict[str, Desire], time_delta: timedelta) -> int:
        """Apply advanced temporal decay to desires."""
        decayed_count = 0
        
        for desire in desires.values():
            if desire.state == DesireState.PRUNED:
                continue
            
            # Calculate decay factor based on time passed
            hours_passed = time_delta.total_seconds() / 3600.0
            
            # Adaptive decay rate based on desire properties
            adaptive_decay_rate = desire.decay_rate * self._get_adaptive_decay_multiplier(desire)
            
            # Apply decay
            decay_factor = math.exp(-adaptive_decay_rate * hours_passed)
            old_strength = desire.strength
            desire.strength *= decay_factor
            
            # Count if strength was reduced
            if desire.strength < old_strength:
                decayed_count += 1
            
            # Update state based on decay
            if desire.strength < 0.1:
                desire.state = DesireState.WEAKENING
            
            # Update goal potential
            if desire.strength < 0.3:
                desire.goal_potential *= self.config.goal_potential_decay
        
        return decayed_count
    
    def _get_adaptive_decay_multiplier(self, desire: Desire) -> float:
        """Get adaptive decay multiplier based on desire properties."""
        # Higher reinforcement count = slower decay
        reinforcement_factor = 1.0 / (1.0 + desire.reinforcement_count * 0.1)
        
        # Higher goal potential = slower decay
        goal_factor = 1.0 - desire.goal_potential * 0.3
        
        # Higher interaction strength = slower decay
        interaction_factor = 1.0 - desire.interaction_strength * 0.2
        
        return reinforcement_factor * goal_factor * interaction_factor
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of temporal state."""
        return {
            'memory_size': len(self.temporal_memory),
            'temporal_attention_enabled': self.config.temporal_attention_enabled,
            'seasonal_patterns_count': len(self.seasonal_patterns)
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization."""
        return {
            'temporal_memory_size': len(self.temporal_memory),
            'seasonal_patterns': self.seasonal_patterns
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from serialization."""
        self.seasonal_patterns = state.get('seasonal_patterns', {}) 