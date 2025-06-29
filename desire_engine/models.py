"""
Ilanya Desire Engine - Models

Core data models for the desire engine including Desire and DesireState.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any
import math


class DesireState(Enum):
    """States a desire can be in within the desire space."""
    ACTIVE = "active"           # Currently active desire
    REINFORCED = "reinforced"   # Strongly reinforced desire
    GOAL_CANDIDATE = "goal_candidate"  # Ready to become a goal
    WEAKENING = "weakening"     # Losing strength, may be pruned
    PRUNED = "pruned"          # Has been pruned


@dataclass
class Desire:
    """
    Represents a single desire in the desire space.
    
    Each desire has mathematical properties that determine its behavior
    in the desire space including strength, reinforcement history,
    and interaction with other desires.
    """
    
    # Core identification
    id: str                     # Unique identifier
    name: str                   # Human-readable name
    source_traits: List[str]    # Traits that created this desire
    
    # Mathematical properties
    strength: float = 0.5       # Current strength (0-1)
    base_strength: float = 0.5  # Base strength without reinforcement
    reinforcement_count: int = 0  # Number of times reinforced
    last_reinforcement: Optional[datetime] = None  # Last reinforcement time
    
    # State and dynamics
    state: DesireState = DesireState.ACTIVE
    creation_time: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.1     # Rate at which strength decays over time
    
    # Interaction properties
    attention_weight: float = 1.0  # How much attention this desire receives
    interaction_strength: float = 0.5  # Strength of interactions with other desires
    
    # Goal potential
    goal_potential: float = 0.0  # Likelihood of becoming a goal (0-1)
    goal_threshold: float = 0.8  # Threshold for goal candidacy
    
    emergent: bool = False  # Marks if this is an emergent desire
    
    def __post_init__(self):
        """Validate and compute initial properties."""
        self.strength = max(0.0, min(1.0, self.strength))
        self.base_strength = max(0.0, min(1.0, self.base_strength))
        self.decay_rate = max(0.0, min(1.0, self.decay_rate))
        self.attention_weight = max(0.0, min(2.0, self.attention_weight))
        self.interaction_strength = max(0.0, min(1.0, self.interaction_strength))
        self.goal_potential = max(0.0, min(1.0, self.goal_potential))
    
    def reinforce(self, reinforcement_strength: float = 1.0):
        """Reinforce this desire, increasing its strength."""
        self.reinforcement_count += 1
        self.last_reinforcement = datetime.now()
        
        # Calculate reinforcement bonus using sigmoid function
        reinforcement_bonus = reinforcement_strength * (1.0 / (1.0 + math.exp(-self.reinforcement_count + 3)))
        self.strength = min(1.0, self.strength + reinforcement_bonus)
        
        # Update goal potential based on reinforcement
        self.goal_potential = min(1.0, self.reinforcement_count / 10.0)
        
        # Update state based on strength and goal potential
        self._update_state()
    
    def decay(self, time_delta: timedelta):
        """Apply time-based decay to desire strength."""
        if self.state == DesireState.PRUNED:
            return
        
        # Calculate decay factor based on time passed
        hours_passed = time_delta.total_seconds() / 3600.0
        decay_factor = math.exp(-self.decay_rate * hours_passed)
        
        # Apply decay to strength
        self.strength *= decay_factor
        
        # If strength drops too low, mark for pruning
        if self.strength < 0.1:
            self.state = DesireState.WEAKENING
        
        # Update goal potential (decreases with decay)
        if self.strength < 0.3:
            self.goal_potential *= 0.9
        
        self._update_state()
    
    def _update_state(self):
        """Update desire state based on current properties."""
        if self.state == DesireState.PRUNED:
            return
        
        if self.goal_potential >= self.goal_threshold:
            self.state = DesireState.GOAL_CANDIDATE
        elif self.strength >= 0.7:
            self.state = DesireState.REINFORCED
        elif self.strength < 0.1:
            self.state = DesireState.WEAKENING
        else:
            self.state = DesireState.ACTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'source_traits': self.source_traits,
            'strength': self.strength,
            'base_strength': self.base_strength,
            'reinforcement_count': self.reinforcement_count,
            'last_reinforcement': self.last_reinforcement.isoformat() if self.last_reinforcement else None,
            'state': self.state.value,
            'creation_time': self.creation_time.isoformat(),
            'decay_rate': self.decay_rate,
            'attention_weight': self.attention_weight,
            'interaction_strength': self.interaction_strength,
            'goal_potential': self.goal_potential,
            'goal_threshold': self.goal_threshold,
            'emergent': self.emergent
        } 