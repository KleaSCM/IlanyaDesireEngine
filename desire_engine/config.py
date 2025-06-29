"""
Ilanya Desire Engine - Configuration

Configuration classes for the modular desire engine.
"""

from dataclasses import dataclass, field
import os
import logging


@dataclass
class DesireEngineConfig:
    """Configuration for the modular Desire Engine."""
    
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
    
    # Module-specific parameters
    # Embedding module
    multi_modal_enabled: bool = True
    embedding_fusion_layers: int = 2
    
    # Attention module
    attention_dropout: float = 0.1
    attention_layer_norm: bool = True
    
    # Interaction module
    interaction_graph_layers: int = 3
    interaction_attention_heads: int = 4
    
    # Temporal module
    temporal_memory_size: int = 100
    temporal_attention_enabled: bool = True
    
    # Interaction parameters
    interaction_threshold: float = 0.3
    synergy_threshold: float = 0.7
    conflict_threshold: float = 0.5
    emergent_threshold: float = 0.8
    
    # Threshold parameters
    adaptive_threshold_enabled: bool = True
    adaptive_threshold_base: float = 0.1
    threshold_learning_rate: float = 0.01
    
    # Information theory parameters
    entropy_calculation_enabled: bool = True
    complexity_metrics_enabled: bool = True
    entropy_window: int = 100
    complexity_weight: float = 0.5
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "log/desire_engine.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_rotation: str = "midnight"
    log_backup_count: int = 7
    
    # Logging categories
    log_desire_creation: bool = True
    log_desire_reinforcement: bool = True
    log_desire_decay: bool = True
    log_desire_pruning: bool = True
    log_interactions: bool = True
    log_emergent_desires: bool = True
    log_goal_candidates: bool = True
    log_metrics: bool = True
    
    def __post_init__(self):
        """Ensure log directory exists."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("DesireEngine")
        logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(self.log_format)
        
        # File handler with rotation
        from logging.handlers import TimedRotatingFileHandler
        file_handler = TimedRotatingFileHandler(
            self.log_file,
            when=self.log_rotation,
            backupCount=self.log_backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger 