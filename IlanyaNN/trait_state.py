"""
Ilanya Trait Engine - Trait State Management

State tracking and cognitive state management for trait evolution over time.
Provides TraitState and CognitiveState classes for tracking trait changes,
computing stability metrics, and managing cognitive load and attention.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from .trait_types import TraitType, TraitCategory


@dataclass
class TraitState:
    """
    State tracking for individual traits over time.
    
    Tracks the current and previous values of a trait, computes change rates,
    and maintains confidence levels. This enables monitoring trait evolution
    and stability over time.
    """
    
    trait_type: TraitType                    # Type of trait being tracked
    current_value: float                     # Current trait value (0-1 scale)
    previous_value: Optional[float] = None   # Previous trait value for change tracking
    confidence: float = 1.0                  # Confidence in current measurement
    change_rate: Optional[float] = None      # Rate of change from previous to current
    stability_score: Optional[float] = None  # Computed stability metric
    
    def __post_init__(self):
        """
        Compute derived metrics after initialization.
        
        Calculates change rate and stability score based on current and
        previous values. These metrics help understand trait dynamics.
        """
        # Calculate change rate if previous value is available
        if self.previous_value is not None:
            self.change_rate = self.current_value - self.previous_value
        
        # Calculate stability score based on change rate
        if self.change_rate is not None:
            # Stability is inversely proportional to absolute change rate
            self.stability_score = max(0.0, 1.0 - abs(self.change_rate))
        else:
            # If no previous value, assume high stability
            self.stability_score = 1.0
    
    def update(self, new_value: float, new_confidence: float = 1.0):
        """
        Update trait state with new values.
        
        Moves current value to previous and updates with new values,
        then recomputes derived metrics.
        
        Args:
            new_value: New trait value
            new_confidence: New confidence score
        """
        # Store current values as previous
        self.previous_value = self.current_value
        
        # Update with new values
        self.current_value = new_value
        self.confidence = new_confidence
        
        # Recompute derived metrics
        self.__post_init__()
    
    def get_change_direction(self) -> Optional[str]:
        """
        Get the direction of trait change.
        
        Returns:
            'increasing', 'decreasing', 'stable', or None if no previous value
        """
        if self.change_rate is None:
            return None
        
        if abs(self.change_rate) < 0.01:  # Small threshold for stability
            return 'stable'
        elif self.change_rate > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation for serialization.
        
        Returns:
            Dictionary containing all trait state data
        """
        return {
            'trait_type': self.trait_type.value,
            'current_value': self.current_value,
            'previous_value': self.previous_value,
            'confidence': self.confidence,
            'change_rate': self.change_rate,
            'stability_score': self.stability_score
        }


@dataclass
class CognitiveState:
    """
    Comprehensive cognitive state tracking for the AI agent.
    
    Manages the overall cognitive state including trait states, stability
    metrics, cognitive load, attention focus, and emotional state. This
    provides a holistic view of the AI's cognitive functioning.
    """
    
    trait_states: Dict[TraitType, TraitState]  # Individual trait states
    timestamp: datetime = field(default_factory=datetime.now)  # When state was recorded
    
    # Overall cognitive metrics
    overall_stability: float = 0.8            # Overall system stability (0-1)
    cognitive_load: float = 0.3               # Current cognitive load (0-1)
    attention_focus: float = 0.9              # Attention focus level (0-1)
    emotional_state: float = 0.6              # Emotional state (0-1, neutral=0.5)
    
    # Additional cognitive metrics
    processing_speed: float = 1.0             # Information processing speed
    memory_availability: float = 0.8          # Available memory capacity
    decision_confidence: float = 0.7          # Confidence in current decisions
    
    def __post_init__(self):
        """
        Validate cognitive state values and compute derived metrics.
        
        Ensures all values are within valid ranges and computes overall
        stability from individual trait stabilities.
        """
        # Validate all values are within 0-1 range
        for attr_name in ['overall_stability', 'cognitive_load', 'attention_focus', 
                         'emotional_state', 'processing_speed', 'memory_availability', 
                         'decision_confidence']:
            value = getattr(self, attr_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr_name} must be between 0 and 1, got {value}")
        
        # Compute overall stability from trait stabilities if not provided
        if self.overall_stability == 0.8 and self.trait_states:
            stability_scores = [state.stability_score for state in self.trait_states.values() 
                              if state.stability_score is not None]
            if stability_scores:
                self.overall_stability = float(np.mean(stability_scores))
    
    def get_trait_state(self, trait_type: TraitType) -> Optional[TraitState]:
        """
        Get state for a specific trait.
        
        Args:
            trait_type: The trait to retrieve state for
            
        Returns:
            TraitState if found, None otherwise
        """
        return self.trait_states.get(trait_type)
    
    def get_traits_by_category(self, category: TraitCategory) -> Dict[TraitType, TraitState]:
        """
        Get trait states filtered by category.
        
        Args:
            category: The category to filter by
            
        Returns:
            Dictionary of trait states belonging to the specified category
        """
        from .trait_types import TRAIT_METADATA
        return {
            trait_type: state for trait_type, state in self.trait_states.items()
            if trait_type in TRAIT_METADATA and TRAIT_METADATA[trait_type].category == category
        }
    
    def get_most_changed_traits(self, limit: int = 5) -> List[Tuple[TraitType, float]]:
        """
        Get traits with the highest change rates.
        
        Args:
            limit: Maximum number of traits to return
            
        Returns:
            List of (trait_type, change_rate) tuples sorted by absolute change rate
        """
        # Filter traits with change rates
        changed_traits = [
            (trait_type, state.change_rate) 
            for trait_type, state in self.trait_states.items()
            if state.change_rate is not None
        ]
        
        # Sort by absolute change rate (descending)
        changed_traits.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return changed_traits[:limit]
    
    def get_least_stable_traits(self, limit: int = 5) -> List[Tuple[TraitType, float]]:
        """
        Get traits with the lowest stability scores.
        
        Args:
            limit: Maximum number of traits to return
            
        Returns:
            List of (trait_type, stability_score) tuples sorted by stability
        """
        # Filter traits with stability scores
        stable_traits = [
            (trait_type, state.stability_score) 
            for trait_type, state in self.trait_states.items()
            if state.stability_score is not None
        ]
        
        # Sort by stability score (ascending - least stable first)
        stable_traits.sort(key=lambda x: x[1])
        
        return stable_traits[:limit]
    
    def compute_cognitive_load(self) -> float:
        """
        Compute cognitive load based on current state.
        
        Cognitive load increases with:
        - High number of changing traits
        - Low attention focus
        - High emotional intensity
        - Low memory availability
        
        Returns:
            Computed cognitive load value (0-1)
        """
        # Base load from changing traits
        changing_traits = sum(1 for state in self.trait_states.values() 
                            if state.change_rate and abs(state.change_rate) > 0.01)
        trait_load = min(1.0, changing_traits / len(self.trait_states) * 2)
        
        # Attention focus impact (low focus = higher load)
        attention_load = 1.0 - self.attention_focus
        
        # Emotional state impact (extreme emotions = higher load)
        emotional_load = abs(self.emotional_state - 0.5) * 2  # Distance from neutral
        
        # Memory availability impact (low memory = higher load)
        memory_load = 1.0 - self.memory_availability
        
        # Combine factors with weights
        total_load = (
            trait_load * 0.4 +
            attention_load * 0.3 +
            emotional_load * 0.2 +
            memory_load * 0.1
        )
        
        return min(1.0, total_load)
    
    def update_cognitive_load(self):
        """
        Update cognitive load based on current state.
        
        Automatically computes and updates the cognitive_load attribute
        based on the current cognitive state.
        """
        self.cognitive_load = self.compute_cognitive_load()
    
    def is_stable(self, threshold: float = 0.7) -> bool:
        """
        Check if the cognitive state is stable.
        
        Args:
            threshold: Stability threshold (0-1)
            
        Returns:
            True if overall stability is above threshold
        """
        return self.overall_stability >= threshold
    
    def is_overloaded(self, threshold: float = 0.8) -> bool:
        """
        Check if cognitive load is too high.
        
        Args:
            threshold: Overload threshold (0-1)
            
        Returns:
            True if cognitive load is above threshold
        """
        return self.cognitive_load >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation for serialization.
        
        Returns:
            Dictionary containing all cognitive state data
        """
        return {
            'trait_states': {trait_type.value: state.to_dict() 
                           for trait_type, state in self.trait_states.items()},
            'timestamp': self.timestamp.isoformat(),
            'overall_stability': self.overall_stability,
            'cognitive_load': self.cognitive_load,
            'attention_focus': self.attention_focus,
            'emotional_state': self.emotional_state,
            'processing_speed': self.processing_speed,
            'memory_availability': self.memory_availability,
            'decision_confidence': self.decision_confidence
        } 