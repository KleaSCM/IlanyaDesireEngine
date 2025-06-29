"""
Ilanya Desire Engine - Threshold Module

Adaptive threshold management for desire reinforcement, pruning, and goal candidacy.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

from typing import Dict, Any
from datetime import datetime

from ..models import Desire
from ..config import DesireEngineConfig


class ThresholdModule:
    """
    Adaptive threshold module for desire management.
    
    Provides dynamic thresholds for:
    - Reinforcement thresholds
    - Pruning thresholds
    - Goal candidate thresholds
    """
    
    def __init__(self, config: DesireEngineConfig):
        self.config = config
        self.adaptive_thresholds = {}
        self.threshold_history = []
        
    def get_reinforcement_threshold(self, trait_type: str, desires: Dict[str, Desire], trait_state: Dict[str, Any]) -> float:
        """Get adaptive reinforcement threshold."""
        if not self.config.adaptive_threshold_enabled:
            return self.config.reinforcement_threshold
        
        # Base threshold
        base_threshold = self.config.reinforcement_threshold
        
        # Adjust based on desire space density
        density_factor = len(desires) / self.config.max_desires
        density_adjustment = density_factor * 0.2  # Higher density = higher threshold
        
        # Adjust based on trait type
        trait_adjustment = self._get_trait_type_adjustment(trait_type)
        
        # Adjust based on current trait value
        current_value = trait_state.get('current_value', 0.5)
        value_adjustment = (current_value - 0.5) * 0.1  # Higher values = lower threshold
        
        adaptive_threshold = base_threshold + density_adjustment + trait_adjustment + value_adjustment
        
        # Store threshold history
        self.threshold_history.append({
            'timestamp': datetime.now(),
            'trait_type': trait_type,
            'threshold': adaptive_threshold,
            'base_threshold': base_threshold
        })
        
        return max(0.1, min(0.8, adaptive_threshold))
    
    def get_pruning_threshold(self, desire: Desire, desires: Dict[str, Desire]) -> float:
        """Get adaptive pruning threshold."""
        if not self.config.adaptive_threshold_enabled:
            return self.config.pruning_threshold
        
        # Base threshold
        base_threshold = self.config.pruning_threshold
        
        # Adjust based on desire age
        age_hours = (datetime.now() - desire.creation_time).total_seconds() / 3600.0
        age_factor = min(1.0, age_hours / 24.0)  # Older desires get higher threshold
        
        # Adjust based on reinforcement count
        reinforcement_factor = desire.reinforcement_count / 10.0  # More reinforced = higher threshold
        
        # Adjust based on goal potential
        goal_factor = desire.goal_potential * 0.3  # Higher goal potential = higher threshold
        
        adaptive_threshold = base_threshold + (age_factor + reinforcement_factor + goal_factor) * 0.1
        
        return max(0.01, min(0.3, adaptive_threshold))
    
    def _get_trait_type_adjustment(self, trait_type: str) -> float:
        """Get adjustment based on trait type."""
        # Personality traits get lower thresholds (easier to reinforce)
        if any(word in trait_type.lower() for word in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']):
            return -0.1
        
        # Identity traits get higher thresholds (harder to reinforce)
        if any(word in trait_type.lower() for word in ['identity', 'orientation', 'gender']):
            return 0.1
        
        return 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization."""
        return {
            'adaptive_thresholds': self.adaptive_thresholds,
            'threshold_history_length': len(self.threshold_history)
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from serialization."""
        self.adaptive_thresholds = state.get('adaptive_thresholds', {}) 