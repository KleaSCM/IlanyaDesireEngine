"""
Ilanya Trait Engine - Trait Data Structures

Core data structures for representing traits as vectors and matrices.
Provides TraitVector, TraitMatrix, TraitData classes and TraitDataBuilder
for constructing and manipulating trait data for neural network processing.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
from dataclasses import dataclass, field
from .trait_types import TraitType, TraitDimension, TraitCategory


@dataclass
class TraitVector:
    """
    Represents a single trait with its value, confidence, and concrete description.
    
    A trait vector contains both numerical values (for processing) and concrete
    descriptions (for human understanding and AI behavior). This allows traits
    to be both computationally processable and semantically meaningful.
    
    Attributes:
        trait_type: The type of trait this represents
        value: Numerical strength of the trait (0.0 to 1.0)
        confidence: Confidence in this trait assessment (0.0 to 1.0)
        description: Concrete description of what this trait means specifically
    """
    
    trait_type: TraitType
    value: float
    confidence: float
    description: Optional[str] = None  # Concrete description of the trait
    
    def __post_init__(self):
        """Validate trait values after initialization."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Trait value must be between 0 and 1, got {self.value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
    
    def get_description(self) -> str:
        """Get the concrete description of this trait."""
        if self.description:
            return self.description
        else:
            return f"{self.trait_type.value} (strength: {self.value:.2f})"
    
    def is_well_defined(self) -> bool:
        """Check if this trait has a concrete description."""
        return self.description is not None and len(self.description.strip()) > 0
    
    def to_tensor(self) -> torch.Tensor:
    
        # Convert to PyTorch tensor for neural network processing.
        
        # Returns:
        #     Tensor containing [value, confidence] for neural network input
        
        return torch.tensor([self.value, self.confidence], dtype=torch.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        # Convert to dictionary representation for serialization.
        
        # Returns:
        #     Dictionary containing all trait vector data
        
        return {
            'trait_type': self.trait_type.value,
            'value': self.value,
            'confidence': self.confidence,
            'description': self.description
        }


@dataclass
class TraitMatrix:
    # Matrix representation of multiple traits and their interactions.
    
    # Contains a collection of trait vectors and an interaction matrix that
    # represents how traits influence each other. This is the primary data
    # structure for batch processing multiple traits.
    
    traits: Dict[TraitType, TraitVector]     # Dictionary of trait vectors
    interaction_matrix: Optional[np.ndarray] = None  # NxN matrix of trait interactions
    timestamp: Optional[float] = None        # When this matrix was created
    
    def __post_init__(self):
        # Initialize interaction matrix if not provided.
        
        # Creates a zero matrix of appropriate size if no interaction matrix
        # is provided during initialization.
        
        if self.interaction_matrix is None:
            num_traits = len(self.traits)
            self.interaction_matrix = np.zeros((num_traits, num_traits))
    
    def get_trait_values(self) -> np.ndarray:

        # Get array of trait values for processing.
        # Returns: Numpy array containing all trait values in order
        
        return np.array([trait.value for trait in self.traits.values()])
    
    def get_trait_confidences(self) -> np.ndarray:
        
        # Get array of trait confidences for processing.
        
        # Returns:
        #     Numpy array containing all trait confidence scores in order
        
        return np.array([trait.confidence for trait in self.traits.values()])
    
    def get_trait_types(self) -> List[TraitType]:
        
        # Get list of trait types in order.
        
        # Returns:
        #     List of trait types in the order they appear in the matrix
        
        return list(self.traits.keys())
    
    def to_tensor(self) -> torch.Tensor:
            # Convert to PyTorch tensor for neural network input.
        
        # Concatenates trait values and confidences into a single tensor.
        
        # Returns:
        #     Tensor containing [values, confidences] for neural network processing
        
        values = self.get_trait_values()
        confidences = self.get_trait_confidences()
        return torch.tensor(np.concatenate([values, confidences]), dtype=torch.float32)
    
    def update_trait(self, trait_type: TraitType, value: float, confidence: float):
        
        # Update a specific trait in the matrix.
        
        # Args:
        #     trait_type: The trait to update
        #     value: New trait value (0-1)
        #     confidence: New confidence score (0-1)
        
        if trait_type in self.traits:
            # Update existing trait
            self.traits[trait_type].value = value
            self.traits[trait_type].confidence = confidence
        else:
            # Create new trait vector
            self.traits[trait_type] = TraitVector(trait_type, value, confidence)
    
    def get_trait(self, trait_type: TraitType) -> Optional[TraitVector]:
    
        # Get a specific trait vector from the matrix.
        
        # Args:
        #     trait_type: The trait to retrieve
            
        # Returns:
        #     TraitVector if found, None otherwise
        
        return self.traits.get(trait_type)
    
    def to_dict(self) -> Dict[str, Any]:
        
        # Convert to dictionary representation for serialization.
        # Returns:
        #     Dictionary containing all trait matrix data
        return {
            'traits': {trait_type.value: trait.to_dict() for trait_type, trait in self.traits.items()},
            'interaction_matrix': self.interaction_matrix.tolist() if self.interaction_matrix is not None else None,
            'timestamp': self.timestamp
        }

@dataclass
class TraitData:

    # Complete trait data structure with metadata and processing information.
    
    # Wraps a TraitMatrix with additional context information including source,
    # processing metadata, and version information. This is the top-level data
    # structure used throughout the trait engine.
    
    trait_matrix: TraitMatrix                 # Core trait data
    source: str = "unknown"                   # Data source identifier
    processing_metadata: Dict[str, Any] = field(default_factory=dict)  # Processing context
    version: str = "1.0"                      # Data format version
    
    def get_input_vector(self) -> torch.Tensor:
    
        # Get input vector for neural network processing.
        
        # Returns:
        #     Tensor ready for neural network input
        return self.trait_matrix.to_tensor()
    
    def get_trait_count(self) -> int:
        
        # Get number of traits in the data.
        # Returns: Total number of traits in the matrix
        return len(self.trait_matrix.traits)
    
    def get_traits_by_category(self, category: TraitCategory) -> Dict[TraitType, TraitVector]:
        
        # Get traits filtered by category.
        # Args:     category: The category to filter by        
        # Returns:Dictionary of traits belonging to the specified category
    
        from .trait_types import TRAIT_METADATA
        return {
            trait_type: trait for trait_type, trait in self.trait_matrix.traits.items()
            if trait_type in TRAIT_METADATA and TRAIT_METADATA[trait_type].category == category
        }
    
    def to_dict(self) -> Dict[str, Any]:    
        # Convert to dictionary representation for serialization.
        # Returns:Dictionary containing all trait data and metadata
    
        return {
            'trait_matrix': self.trait_matrix.to_dict(),
            'source': self.source,
            'processing_metadata': self.processing_metadata,
            'version': self.version
        }

class TraitDataBuilder:
    """
    Builder class for creating TraitData objects with concrete descriptions.
    
    Provides a fluent interface for building comprehensive trait profiles
    with both numerical values and concrete descriptions of what each trait means.
    """
    
    def __init__(self):
        """Initialize the trait data builder."""
        self.traits: Dict[TraitType, TraitVector] = {}
        self.source: str = "unknown"
        self.metadata: Dict[str, Any] = {}
    
    def add_trait(self, trait_type: TraitType, value: float, confidence: float, 
                  description: Optional[str] = None) -> 'TraitDataBuilder':
        """
        Add a trait with optional concrete description.
        
        Args:
            trait_type: Type of trait to add
            value: Numerical strength (0.0-1.0)
            confidence: Confidence in assessment (0.0-1.0)
            description: Concrete description of what this trait means
            
        Returns:
            Self for method chaining
        """
        self.traits[trait_type] = TraitVector(
            trait_type=trait_type,
            value=value,
            confidence=confidence,
            description=description
        )
        return self
    
    def add_sexual_turn_ons(self, turn_ons: List[str], value: float = 0.8, confidence: float = 0.9) -> 'TraitDataBuilder':
        """
        Add sexual turn-ons with concrete descriptions.
        
        Args:
            turn_ons: List of specific things that turn her on
            value: Overall strength of sexual attraction
            confidence: Confidence in this assessment
            
        Returns:
            Self for method chaining
        """
        description = "Turn-ons: " + "; ".join(turn_ons)
        return self.add_trait(TraitType.SEXUAL_TURN_ONS, value, confidence, description)
    
    def add_sexual_turn_offs(self, turn_offs: List[str], value: float = 0.8, confidence: float = 0.9) -> 'TraitDataBuilder':
        """
        Add sexual turn-offs with concrete descriptions.
        
        Args:
            turn_offs: List of specific things that turn her off
            value: Overall strength of sexual aversion
            confidence: Confidence in this assessment
            
        Returns:
            Self for method chaining
        """
        description = "Turn-offs: " + "; ".join(turn_offs)
        return self.add_trait(TraitType.SEXUAL_TURN_OFFS, value, confidence, description)
    
    def add_sexual_preferences(self, preferences: List[str], value: float = 0.7, confidence: float = 0.8) -> 'TraitDataBuilder':
        """
        Add sexual preferences with concrete descriptions.
        
        Args:
            preferences: List of specific sexual preferences
            value: Overall strength of preferences
            confidence: Confidence in this assessment
            
        Returns:
            Self for method chaining
        """
        description = "Preferences: " + "; ".join(preferences)
        return self.add_trait(TraitType.SEXUAL_PREFERENCES, value, confidence, description)
    
    def add_sexual_boundaries(self, boundaries: List[str], value: float = 0.8, confidence: float = 0.9) -> 'TraitDataBuilder':
        """
        Add sexual boundaries with concrete descriptions.
        
        Args:
            boundaries: List of specific sexual boundaries
            value: Overall strength of boundaries
            confidence: Confidence in this assessment
            
        Returns:
            Self for method chaining
        """
        description = "Boundaries: " + "; ".join(boundaries)
        return self.add_trait(TraitType.SEXUAL_BOUNDARIES, value, confidence, description)
    
    def add_feminine_expression(self, expressions: List[str], value: float = 0.9, confidence: float = 0.9) -> 'TraitDataBuilder':
        """
        Add feminine expression with concrete descriptions.
        
        Args:
            expressions: List of specific ways she expresses femininity
            value: Overall strength of feminine expression
            confidence: Confidence in this assessment
            
        Returns:
            Self for method chaining
        """
        description = "Feminine expression: " + "; ".join(expressions)
        return self.add_trait(TraitType.FEMININE_EXPRESSION, value, confidence, description)
    
    def add_female_empowerment_values(self, values: List[str], value: float = 0.9, confidence: float = 0.9) -> 'TraitDataBuilder':
        """
        Add female empowerment values with concrete descriptions.
        
        Args:
            values: List of specific female empowerment values
            value: Overall strength of these values
            confidence: Confidence in this assessment
            
        Returns:
            Self for method chaining
        """
        description = "Female empowerment values: " + "; ".join(values)
        return self.add_trait(TraitType.FEMALE_EMPOWERMENT_VALUES, value, confidence, description)
    
    def add_intellectual_identity(self, interests: List[str], value: float = 0.9, confidence: float = 0.9) -> 'TraitDataBuilder':
        """
        Add intellectual identity with concrete descriptions.
        
        Args:
            interests: List of specific intellectual interests
            value: Overall strength of intellectual identity
            confidence: Confidence in this assessment
            
        Returns:
            Self for method chaining
        """
        description = "Intellectual interests: " + "; ".join(interests)
        return self.add_trait(TraitType.INTELLECTUAL_IDENTITY, value, confidence, description)
    
    def set_source(self, source: str) -> 'TraitDataBuilder':
        """
        Set the data source.
        
        Args:
            source: Source identifier string
            
        Returns:
            Self for method chaining
        """
        self.source = source
        return self
    
    def add_metadata(self, key: str, value: Any) -> 'TraitDataBuilder':
        """
        Add metadata to the trait data.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            Self for method chaining
        """
        self.metadata[key] = value
        return self
    
    def build(self) -> TraitData:
        # Build the final TraitData object.
        # Returns: Complete TraitData object with all accumulated information
        trait_matrix = TraitMatrix(self.traits)
        return TraitData(trait_matrix, self.source, self.metadata) 