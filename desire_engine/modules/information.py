"""
Ilanya Desire Engine - Information Module

Information theory metrics for desire space analysis including entropy, complexity, and stability.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import numpy as np
from typing import Dict, List, Any
from scipy.stats import entropy
from scipy.spatial.distance import pdist

from ..models import Desire
from ..config import DesireEngineConfig


class InformationModule:
    """
    Information theory module for desire space analysis.
    
    Provides metrics for:
    - Entropy calculations
    - Complexity measures
    - Information content
    - System stability
    """
    
    def __init__(self, config: DesireEngineConfig):
        self.config = config
        
    def compute_metrics(self, desires: Dict[str, Desire]) -> Dict[str, float]:
        """Compute comprehensive information theory metrics."""
        if not desires:
            return {
                'average_strength': 0.0,
                'total_reinforcement': 0,
                'space_density': 0.0,
                'goal_potential': 0.0,
                'entropy': 0.0,
                'complexity': 0.0,
                'stability': 1.0
            }
        
        # Basic metrics
        strengths = [d.strength for d in desires.values()]
        reinforcements = [d.reinforcement_count for d in desires.values()]
        goal_potentials = [d.goal_potential for d in desires.values()]
        
        # Information theory metrics
        entropy_val = self._compute_entropy(strengths) if self.config.entropy_calculation_enabled else 0.0
        complexity = self._compute_complexity(desires) if self.config.complexity_metrics_enabled else 0.0
        stability = self._compute_stability(desires)
        
        return {
            'average_strength': float(np.mean(strengths)),
            'total_reinforcement': sum(reinforcements),
            'space_density': len(desires) / self.config.max_desires,
            'goal_potential': float(np.mean(goal_potentials)),
            'entropy': entropy_val,
            'complexity': complexity,
            'stability': stability
        }
    
    def _compute_entropy(self, values: List[float]) -> float:
        """Compute entropy of desire strength distribution."""
        if not values:
            return 0.0
        
        # Normalize values to create a probability distribution
        total = sum(values)
        if total == 0:
            return 0.0
        
        probabilities = [v / total for v in values]
        
        # Compute entropy
        try:
            return float(entropy(probabilities))
        except:
            return 0.0
    
    def _compute_complexity(self, desires: Dict[str, Desire]) -> float:
        """Compute complexity of the desire space."""
        if len(desires) < 2:
            return 0.0
        
        # Extract features for complexity calculation
        features = []
        for desire in desires.values():
            features.append([
                desire.strength,
                desire.reinforcement_count,
                desire.goal_potential,
                desire.interaction_strength
            ])
        
        features = np.array(features)
        
        # Compute pairwise distances
        try:
            distances = pdist(features)
            complexity = float(np.mean(distances))
        except:
            complexity = 0.0
        
        return complexity
    
    def _compute_stability(self, desires: Dict[str, Desire]) -> float:
        """Compute stability of the desire space."""
        if not desires:
            return 1.0
        
        # Count desires in different states
        state_counts = {}
        for desire in desires.values():
            state = desire.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Stability is higher when more desires are in stable states
        stable_states = ['active', 'reinforced']
        stable_count = sum(state_counts.get(state, 0) for state in stable_states)
        total_count = len(desires)
        
        stability = stable_count / total_count if total_count > 0 else 1.0
        
        return float(stability)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization."""
        return {
            'entropy_calculation_enabled': self.config.entropy_calculation_enabled,
            'complexity_metrics_enabled': self.config.complexity_metrics_enabled
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from serialization."""
        # No state to load for this module
        pass 