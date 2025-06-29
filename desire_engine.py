"""
Ilanya Desire Engine - Advanced Desire Space Management

A sophisticated mathematical system for managing desires based on trait activations.
Uses PyTorch tensors, attention mechanisms, and advanced mathematical models to:
- Create desires from positive trait activations
- Reinforce desires through trait reinforcement
- Prune weak desires over time
- Assess goal potential for strongly reinforced desires

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
import json

# Import from existing trait system
from IlanyaNN.trait_types import TraitType, TraitCategory
from IlanyaNN.trait_state import TraitState, CognitiveState


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
    source_traits: List[TraitType]  # Traits that created this desire
    
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
            'source_traits': [trait.value for trait in self.source_traits],
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
            'goal_threshold': self.goal_threshold
        }


class DesireSpaceAttention(nn.Module):
    """
    Attention mechanism for the desire space.
    
    Uses transformer-style attention to model interactions between desires
    and determine attention weights for each desire.
    """
    
    def __init__(self, desire_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.desire_dim = desire_dim
        self.num_heads = num_heads
        self.head_dim = desire_dim // num_heads
        
        # Attention layers
        self.query = nn.Linear(desire_dim, desire_dim)
        self.key = nn.Linear(desire_dim, desire_dim)
        self.value = nn.Linear(desire_dim, desire_dim)
        self.output = nn.Linear(desire_dim, desire_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(desire_dim)
        
    def forward(self, desire_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and updated embeddings.
        
        Args:
            desire_embeddings: Tensor of shape (num_desires, desire_dim)
            
        Returns:
            Tuple of (updated_embeddings, attention_weights)
        """
        batch_size = desire_embeddings.size(0)
        
        # Compute Q, K, V
        Q = self.query(desire_embeddings).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(desire_embeddings).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(desire_embeddings).view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        attended_values = attended_values.view(batch_size, self.desire_dim)
        
        # Output projection and residual connection
        output = self.output(attended_values)
        output = self.layer_norm(output + desire_embeddings)
        
        return output, attention_weights.mean(dim=1)  # Average across heads


class DesireEmbedding(nn.Module):
    """
    Neural network for embedding desires into a mathematical space.
    
    Converts desire properties into high-dimensional embeddings that
    capture the complex relationships between desires.
    """
    
    def __init__(self, desire_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.desire_dim = desire_dim
        
        # Embedding layers for different desire properties
        self.strength_embedding = nn.Linear(1, desire_dim // 4)
        self.reinforcement_embedding = nn.Linear(1, desire_dim // 4)
        self.age_embedding = nn.Linear(1, desire_dim // 4)
        self.state_embedding = nn.Embedding(len(DesireState), desire_dim // 4)
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(desire_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, desire_dim),
            nn.LayerNorm(desire_dim)
        )
        
    def forward(self, desires: List[Desire]) -> torch.Tensor:
        """
        Convert list of desires to embeddings.
        
        Args:
            desires: List of Desire objects
            
        Returns:
            Tensor of shape (num_desires, desire_dim)
        """
        if not desires:
            return torch.empty(0, self.desire_dim)
        
        embeddings = []
        current_time = datetime.now()
        
        for desire in desires:
            # Calculate age in hours
            age_hours = (current_time - desire.creation_time).total_seconds() / 3600.0
            
            # Generate embeddings for each feature
            strength_emb = self.strength_embedding(torch.tensor([[desire.strength]], dtype=torch.float32))
            reinforcement_emb = self.reinforcement_embedding(torch.tensor([[desire.reinforcement_count]], dtype=torch.float32))
            age_emb = self.age_embedding(torch.tensor([[age_hours]], dtype=torch.float32))
            state_emb = self.state_embedding(torch.tensor([list(DesireState).index(desire.state)]))
            
            # Concatenate embeddings
            combined_emb = torch.cat([strength_emb, reinforcement_emb, age_emb, state_emb], dim=1)
            embeddings.append(combined_emb)
        
        # Stack all embeddings
        desire_embeddings = torch.cat(embeddings, dim=0)
        
        # Apply fusion network
        fused_embeddings = self.fusion_network(desire_embeddings)
        
        return fused_embeddings


@dataclass
class DesireEngineConfig:
    """Configuration for the Desire Engine."""
    
    # Neural network parameters
    desire_dim: int = 64
    hidden_dim: int = 128
    num_attention_heads: int = 8
    
    # Desire management parameters
    min_desire_strength: float = 0.1
    max_desires: int = 50
    reinforcement_threshold: float = 0.3
    pruning_threshold: float = 0.05
    
    # Goal assessment parameters
    goal_candidate_threshold: float = 0.8
    goal_potential_decay: float = 0.95
    
    # Time-based parameters
    decay_check_interval: int = 3600  # seconds
    pruning_check_interval: int = 7200  # seconds


class DesireEngine:
    """
    Advanced Desire Engine for managing desire space dynamics.
    
    Uses sophisticated mathematical models and neural networks to:
    - Create desires from positive trait activations
    - Reinforce desires through trait reinforcement
    - Prune weak desires over time
    - Assess goal potential for strongly reinforced desires
    """
    
    def __init__(self, config: Optional[DesireEngineConfig] = None):
        """Initialize the Desire Engine."""
        self.config = config or DesireEngineConfig()
        
        # Neural network components
        self.desire_embedding = DesireEmbedding(
            desire_dim=self.config.desire_dim,
            hidden_dim=self.config.hidden_dim
        )
        self.attention_mechanism = DesireSpaceAttention(
            desire_dim=self.config.desire_dim,
            num_heads=self.config.num_attention_heads
        )
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.desire_embedding.to(self.device)
        self.attention_mechanism.to(self.device)
        
        # Desire storage
        self.desires: Dict[str, Desire] = {}
        self.pruned_desires: Dict[str, Desire] = {}
        self.goal_candidates: Set[str] = set()
        
        # State tracking
        self.last_decay_check = datetime.now()
        self.last_pruning_check = datetime.now()
        self.desire_counter = 0
        
        # Trait to desire mapping
        self.trait_desire_mapping: Dict[TraitType, List[str]] = {}
        
    def process_trait_activations(self, trait_states: Dict[TraitType, TraitState]) -> Dict[str, Any]:
        """
        Process trait activations and update desire space.
        
        Args:
            trait_states: Dictionary of trait states from the trait engine
            
        Returns:
            Dictionary containing processing results
        """
        # Find positive trait changes
        positive_traits = self._identify_positive_traits(trait_states)
        
        # Create or reinforce desires based on positive traits
        new_desires = []
        reinforced_desires = []
        
        for trait_type, trait_state in positive_traits.items():
            if trait_state.change_rate and trait_state.change_rate > self.config.reinforcement_threshold:
                # Create new desire or reinforce existing
                desire_id = self._get_or_create_desire(trait_type, trait_state)
                if desire_id in self.desires:
                    self.desires[desire_id].reinforce(trait_state.change_rate)
                    reinforced_desires.append(desire_id)
                else:
                    new_desires.append(desire_id)
        
        # Apply time-based decay
        self._apply_decay()
        
        # Check for pruning
        pruned_desires = self._check_pruning()
        
        # Update goal candidates
        self._update_goal_candidates()
        
        # Compute desire space metrics
        metrics = self._compute_desire_space_metrics()
        
        return {
            'new_desires': new_desires,
            'reinforced_desires': reinforced_desires,
            'pruned_desires': pruned_desires,
            'goal_candidates': list(self.goal_candidates),
            'metrics': metrics,
            'active_desires': len(self.desires),
            'total_desires': len(self.desires) + len(self.pruned_desires)
        }
    
    def _identify_positive_traits(self, trait_states: Dict[TraitType, TraitState]) -> Dict[TraitType, TraitState]:
        """Identify traits with positive changes that could create desires."""
        positive_traits = {}
        
        for trait_type, trait_state in trait_states.items():
            # Skip permanently protected traits
            if trait_type in [TraitType.SEXUAL_ORIENTATION, TraitType.GENDER_IDENTITY, 
                            TraitType.CULTURAL_IDENTITY, TraitType.PERSONAL_IDENTITY]:
                continue
            
            # Check for positive change
            if (trait_state.change_rate and 
                trait_state.change_rate > 0 and 
                trait_state.current_value > 0.3):
                positive_traits[trait_type] = trait_state
        
        return positive_traits
    
    def _get_or_create_desire(self, trait_type: TraitType, trait_state: TraitState) -> str:
        """Get existing desire for trait or create new one."""
        # Check if desire already exists for this trait
        if trait_type in self.trait_desire_mapping:
            for desire_id in self.trait_desire_mapping[trait_type]:
                if desire_id in self.desires:
                    return desire_id
        
        # Create new desire
        self.desire_counter += 1
        desire_id = f"desire_{self.desire_counter}"
        
        # Create desire name from trait
        desire_name = f"Desire for {trait_type.value.replace('_', ' ').title()}"
        
        # Calculate initial strength based on trait value and change rate
        change_rate = trait_state.change_rate or 0.0
        initial_strength = min(1.0, trait_state.current_value * (1 + change_rate))
        
        # Create new desire
        new_desire = Desire(
            id=desire_id,
            name=desire_name,
            source_traits=[trait_type],
            strength=initial_strength,
            base_strength=initial_strength,
            reinforcement_count=1,
            last_reinforcement=datetime.now()
        )
        
        # Store desire
        self.desires[desire_id] = new_desire
        
        # Update trait mapping
        if trait_type not in self.trait_desire_mapping:
            self.trait_desire_mapping[trait_type] = []
        self.trait_desire_mapping[trait_type].append(desire_id)
        
        return desire_id
    
    def _apply_decay(self):
        """Apply time-based decay to all desires."""
        current_time = datetime.now()
        time_delta = current_time - self.last_decay_check
        
        if time_delta.total_seconds() >= self.config.decay_check_interval:
            for desire in self.desires.values():
                desire.decay(time_delta)
            
            self.last_decay_check = current_time
    
    def _check_pruning(self) -> List[str]:
        """Check for desires that should be pruned."""
        current_time = datetime.now()
        time_delta = current_time - self.last_pruning_check
        
        if time_delta.total_seconds() < self.config.pruning_check_interval:
            return []
        
        pruned_desires = []
        
        for desire_id, desire in list(self.desires.items()):
            if (desire.state == DesireState.WEAKENING and 
                desire.strength < self.config.pruning_threshold):
                
                # Move to pruned desires
                self.pruned_desires[desire_id] = desire
                del self.desires[desire_id]
                pruned_desires.append(desire_id)
                
                # Remove from trait mapping
                for trait_type in desire.source_traits:
                    if trait_type in self.trait_desire_mapping:
                        self.trait_desire_mapping[trait_type] = [
                            d for d in self.trait_desire_mapping[trait_type] 
                            if d != desire_id
                        ]
        
        self.last_pruning_check = current_time
        return pruned_desires
    
    def _update_goal_candidates(self):
        """Update the set of goal candidates."""
        self.goal_candidates.clear()
        
        for desire_id, desire in self.desires.items():
            if desire.state == DesireState.GOAL_CANDIDATE:
                self.goal_candidates.add(desire_id)
    
    def _compute_desire_space_metrics(self) -> Dict[str, float]:
        """Compute mathematical metrics of the desire space."""
        if not self.desires:
            return {
                'average_strength': 0.0,
                'total_reinforcement': 0,
                'space_density': 0.0,
                'goal_potential': 0.0
            }
        
        strengths = [d.strength for d in self.desires.values()]
        reinforcements = [d.reinforcement_count for d in self.desires.values()]
        goal_potentials = [d.goal_potential for d in self.desires.values()]
        
        return {
            'average_strength': float(np.mean(strengths)),
            'total_reinforcement': sum(reinforcements),
            'space_density': len(self.desires) / self.config.max_desires,
            'goal_potential': float(np.mean(goal_potentials))
        }
    
    def get_desire_embeddings(self) -> torch.Tensor:
        """Get neural embeddings for all active desires."""
        active_desires = list(self.desires.values())
        if not active_desires:
            return torch.empty(0, self.config.desire_dim)
        
        embeddings = self.desire_embedding(active_desires)
        return embeddings
    
    def compute_desire_attention(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights for the desire space."""
        embeddings = self.get_desire_embeddings()
        if embeddings.size(0) == 0:
            return torch.empty(0, self.config.desire_dim), torch.empty(0, 0)
        
        updated_embeddings, attention_weights = self.attention_mechanism(embeddings)
        return updated_embeddings, attention_weights
    
    def get_desire_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the desire space."""
        return {
            'active_desires': {
                desire_id: desire.to_dict() 
                for desire_id, desire in self.desires.items()
            },
            'pruned_desires': {
                desire_id: desire.to_dict() 
                for desire_id, desire in self.pruned_desires.items()
            },
            'goal_candidates': list(self.goal_candidates),
            'metrics': self._compute_desire_space_metrics(),
            'trait_mapping': {
                trait.value: desire_ids 
                for trait, desire_ids in self.trait_desire_mapping.items()
            }
        }
    
    def save_state(self, filepath: str):
        """Save the current state of the desire engine."""
        state = {
            'desires': {k: v.to_dict() for k, v in self.desires.items()},
            'pruned_desires': {k: v.to_dict() for k, v in self.pruned_desires.items()},
            'goal_candidates': list(self.goal_candidates),
            'trait_desire_mapping': {
                trait.value: desire_ids 
                for trait, desire_ids in self.trait_desire_mapping.items()
            },
            'desire_counter': self.desire_counter,
            'last_decay_check': self.last_decay_check.isoformat(),
            'last_pruning_check': self.last_pruning_check.isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load the state of the desire engine."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Clear current state
        self.desires.clear()
        self.pruned_desires.clear()
        self.goal_candidates.clear()
        self.trait_desire_mapping.clear()
        
        # Load desires
        for desire_id, desire_data in state['desires'].items():
            desire = self._desire_from_dict(desire_data)
            self.desires[desire_id] = desire
        
        # Load pruned desires
        for desire_id, desire_data in state['pruned_desires'].items():
            desire = self._desire_from_dict(desire_data)
            self.pruned_desires[desire_id] = desire
        
        # Load other state
        self.goal_candidates = set(state['goal_candidates'])
        self.desire_counter = state['desire_counter']
        self.last_decay_check = datetime.fromisoformat(state['last_decay_check'])
        self.last_pruning_check = datetime.fromisoformat(state['last_pruning_check'])
        
        # Load trait mapping
        for trait_value, desire_ids in state['trait_desire_mapping'].items():
            trait_type = TraitType(trait_value)
            self.trait_desire_mapping[trait_type] = desire_ids
    
    def _desire_from_dict(self, desire_data: Dict[str, Any]) -> Desire:
        """Create a Desire object from dictionary data."""
        desire = Desire(
            id=desire_data['id'],
            name=desire_data['name'],
            source_traits=[TraitType(trait_value) for trait_value in desire_data['source_traits']],
            strength=desire_data['strength'],
            base_strength=desire_data['base_strength'],
            reinforcement_count=desire_data['reinforcement_count'],
            state=DesireState(desire_data['state']),
            creation_time=datetime.fromisoformat(desire_data['creation_time']),
            decay_rate=desire_data['decay_rate'],
            attention_weight=desire_data['attention_weight'],
            interaction_strength=desire_data['interaction_strength'],
            goal_potential=desire_data['goal_potential'],
            goal_threshold=desire_data['goal_threshold']
        )
        
        if desire_data['last_reinforcement']:
            desire.last_reinforcement = datetime.fromisoformat(desire_data['last_reinforcement'])
        
        return desire 