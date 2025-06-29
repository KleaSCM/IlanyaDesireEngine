"""
Ilanya Desire Engine - Core Module

Core desire engine functionality with modular architecture for easy expansion.
Provides the main DesireEngine class and basic desire management.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging

from .models import Desire, DesireState
from .config import DesireEngineConfig
from .modules import (
    DesireEmbeddingModule,
    AttentionModule,
    InteractionModule,
    TemporalModule,
    ThresholdModule,
    InformationModule
)


class DesireEngine:
    """
    Modular Desire Engine for managing desire space dynamics.
    
    Uses a modular architecture to allow easy addition of advanced features:
    - Multi-modal embeddings
    - Information theory metrics
    - Adaptive thresholds
    - Temporal modeling
    - Desire interaction networks
    """
    
    def __init__(self, config: Optional[DesireEngineConfig] = None):
        """Initialize the Desire Engine with modular components."""
        self.config = config or DesireEngineConfig()
        
        # Set up logging
        self.logger = self.config.setup_logging()
        self.logger.info("Initializing Desire Engine with modular architecture")
        
        # Initialize modular components
        self.embedding_module = DesireEmbeddingModule(self.config)
        self.attention_module = AttentionModule(self.config)
        self.interaction_module = InteractionModule(self.config, self.logger)
        self.temporal_module = TemporalModule(self.config)
        self.threshold_module = ThresholdModule(self.config)
        self.information_module = InformationModule(self.config)
        
        self.logger.info("All modules initialized successfully")
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self._move_modules_to_device()
        
        # Desire storage
        self.desires: Dict[str, Desire] = {}
        self.pruned_desires: Dict[str, Desire] = {}
        self.goal_candidates: Set[str] = set()
        
        # State tracking
        self.last_decay_check = datetime.now()
        self.last_pruning_check = datetime.now()
        self.desire_counter = 0
        
        # Trait to desire mapping
        self.trait_desire_mapping: Dict[str, List[str]] = {}
        
        self.logger.info("Desire Engine initialization complete")
        
    def _move_modules_to_device(self):
        """Move all neural modules to the appropriate device."""
        self.embedding_module.to(self.device)
        self.attention_module.to(self.device)
        # Non-PyTorch modules don't need .to() - they're regular Python classes
    
    def process_trait_activations(self, trait_states: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process trait activations and update desire space using modular components.
        
        Args:
            trait_states: Dictionary of trait states from the trait engine
            
        Returns:
            Dictionary containing processing results
        """
        self.logger.info(f"Processing {len(trait_states)} trait activations")
        
        # Find positive trait changes
        positive_traits = self._identify_positive_traits(trait_states)
        self.logger.info(f"Found {len(positive_traits)} positive traits")
        
        # Create or reinforce desires based on positive traits
        new_desires = []
        reinforced_desires = []
        
        for trait_type, trait_state in positive_traits.items():
            # Use adaptive threshold module
            threshold = self.threshold_module.get_reinforcement_threshold(
                trait_type, self.desires, trait_state
            )
            
            if trait_state.get('change_rate', 0) > threshold:
                # Create new desire or reinforce existing
                desire_id = self._get_or_create_desire(trait_type, trait_state)
                if desire_id in self.desires:
                    self.desires[desire_id].reinforce(trait_state.get('change_rate', 0))
                    reinforced_desires.append(desire_id)
                    if self.config.log_desire_reinforcement:
                        self.logger.info(f"Reinforced desire {desire_id} (strength: {self.desires[desire_id].strength:.3f})")
                else:
                    new_desires.append(desire_id)
                    if self.config.log_desire_creation:
                        self.logger.info(f"Created new desire {desire_id} from trait {trait_type}")
        
        # Apply temporal modeling
        self.temporal_module.update_desires(self.desires)
        if self.config.log_desire_decay:
            self.logger.debug("Applied temporal modeling to desires")
        
        # Apply desire interactions
        interaction_results = self.interaction_module.process_interactions(self.desires)
        if self.config.log_interactions:
            interaction_count = len(interaction_results) if interaction_results else 0
            self.logger.info(f"Processed {interaction_count} desire interactions")
        
        # Apply time-based decay
        decayed_count = self._apply_decay()
        if decayed_count > 0 and self.config.log_desire_decay:
            self.logger.info(f"Applied decay to {decayed_count} desires")
        
        # Check for pruning with adaptive thresholds
        pruned_desires = self._check_pruning()
        if pruned_desires and self.config.log_desire_pruning:
            self.logger.info(f"Pruned {len(pruned_desires)} desires: {pruned_desires}")
        
        # Update goal candidates
        old_goal_count = len(self.goal_candidates)
        self._update_goal_candidates()
        new_goal_count = len(self.goal_candidates)
        if new_goal_count != old_goal_count and self.config.log_goal_candidates:
            self.logger.info(f"Goal candidates updated: {old_goal_count} -> {new_goal_count}")
        
        # Compute metrics using information theory module
        metrics = self.information_module.compute_metrics(self.desires)
        if self.config.log_metrics:
            self.logger.info(f"Computed metrics: entropy={metrics.get('entropy', 0):.3f}, "
                           f"complexity={metrics.get('complexity', 0):.3f}")
        
        return {
            'new_desires': new_desires,
            'reinforced_desires': reinforced_desires,
            'pruned_desires': pruned_desires,
            'goal_candidates': list(self.goal_candidates),
            'metrics': metrics,
            'active_desires': len(self.desires),
            'total_desires': len(self.desires) + len(self.pruned_desires)
        }
    
    def _identify_positive_traits(self, trait_states: Dict[str, Any]) -> Dict[str, Any]:
        """Identify traits with positive changes that could create desires."""
        positive_traits = {}
        
        for trait_type, trait_state in trait_states.items():
            # Skip permanently protected traits
            if trait_type in ['sexual_orientation', 'gender_identity', 
                            'cultural_identity', 'personal_identity']:
                continue
            
            # Check for positive change
            change_rate = trait_state.get('change_rate', 0)
            current_value = trait_state.get('current_value', 0)
            
            if change_rate > 0 and current_value > 0.3:
                positive_traits[trait_type] = trait_state
        
        return positive_traits
    
    def _get_or_create_desire(self, trait_type: str, trait_state: Dict[str, Any]) -> str:
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
        desire_name = f"Desire for {trait_type.replace('_', ' ').title()}"
        
        # Calculate initial strength based on trait value and change rate
        change_rate = trait_state.get('change_rate', 0)
        current_value = trait_state.get('current_value', 0.5)
        initial_strength = min(1.0, current_value * (1 + change_rate))
        
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
    
    def _apply_decay(self) -> int:
        """Apply time-based decay to all desires using temporal module."""
        current_time = datetime.now()
        time_delta = current_time - self.last_decay_check
        
        if time_delta.total_seconds() >= self.config.decay_check_interval:
            decayed_count = self.temporal_module.apply_decay(self.desires, time_delta)
            self.last_decay_check = current_time
            return decayed_count
        return 0
    
    def _check_pruning(self) -> List[str]:
        """Check for desires that should be pruned using adaptive thresholds."""
        current_time = datetime.now()
        time_delta = current_time - self.last_pruning_check
        
        if time_delta.total_seconds() < self.config.pruning_check_interval:
            return []
        
        pruned_desires = []
        
        for desire_id, desire in list(self.desires.items()):
            # Use adaptive threshold for pruning
            pruning_threshold = self.threshold_module.get_pruning_threshold(
                desire, self.desires
            )
            
            if (desire.state == DesireState.WEAKENING and 
                desire.strength < pruning_threshold):
                
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
    
    def get_desire_embeddings(self) -> torch.Tensor:
        """Get multi-modal embeddings for all active desires."""
        active_desires = list(self.desires.values())
        if not active_desires:
            return torch.empty(0, self.config.desire_dim)
        
        embeddings = self.embedding_module(active_desires)
        return embeddings
    
    def compute_desire_attention(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights for the desire space."""
        embeddings = self.get_desire_embeddings()
        if embeddings.size(0) == 0:
            return torch.empty(0, self.config.desire_dim), torch.empty(0, 0)
        
        updated_embeddings, attention_weights = self.attention_module(embeddings)
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
            'metrics': self.information_module.compute_metrics(self.desires),
            'trait_mapping': self.trait_desire_mapping,
            'interaction_network': self.interaction_module.get_network_summary(),
            'temporal_state': self.temporal_module.get_state_summary()
        }
    
    def save_state(self, filepath: str):
        """Save the current state of the desire engine."""
        state = {
            'desires': {k: v.to_dict() for k, v in self.desires.items()},
            'pruned_desires': {k: v.to_dict() for k, v in self.pruned_desires.items()},
            'goal_candidates': list(self.goal_candidates),
            'trait_desire_mapping': self.trait_desire_mapping,
            'desire_counter': self.desire_counter,
            'last_decay_check': self.last_decay_check.isoformat(),
            'last_pruning_check': self.last_pruning_check.isoformat(),
            'module_states': {
                'interaction': self.interaction_module.get_state(),
                'temporal': self.temporal_module.get_state(),
                'threshold': self.threshold_module.get_state(),
                'information': self.information_module.get_state()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Saved desire engine state to {filepath}")
    
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
        self.trait_desire_mapping = state['trait_desire_mapping']
        
        # Load module states
        if 'module_states' in state:
            self.interaction_module.load_state(state['module_states']['interaction'])
            self.temporal_module.load_state(state['module_states']['temporal'])
            self.threshold_module.load_state(state['module_states']['threshold'])
            self.information_module.load_state(state['module_states']['information'])
        
        self.logger.info(f"Loaded desire engine state from {filepath}")
    
    def _desire_from_dict(self, desire_data: Dict[str, Any]) -> Desire:
        """Create a Desire object from dictionary data."""
        desire = Desire(
            id=desire_data['id'],
            name=desire_data['name'],
            source_traits=desire_data['source_traits'],
            strength=desire_data['strength'],
            base_strength=desire_data['base_strength'],
            reinforcement_count=desire_data['reinforcement_count'],
            state=DesireState(desire_data['state']),
            creation_time=datetime.fromisoformat(desire_data['creation_time']),
            decay_rate=desire_data['decay_rate'],
            attention_weight=desire_data['attention_weight'],
            interaction_strength=desire_data['interaction_strength'],
            goal_potential=desire_data['goal_potential'],
            goal_threshold=desire_data['goal_threshold'],
            emergent=desire_data.get('emergent', False)
        )
        
        if desire_data['last_reinforcement']:
            desire.last_reinforcement = datetime.fromisoformat(desire_data['last_reinforcement'])
        
        return desire 