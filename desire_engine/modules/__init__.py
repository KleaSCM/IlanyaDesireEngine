"""
Ilanya Desire Engine - Modules Package

Modular components for the Desire Engine:
- EmbeddingModule: Multi-modal desire embeddings
- AttentionModule: Transformer-style attention
- InteractionModule: Desire interaction networks
- TemporalModule: Time-based modeling
- ThresholdModule: Adaptive thresholds
- InformationModule: Information theory metrics

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

from .embedding import DesireEmbeddingModule
from .attention import AttentionModule
from .interaction import InteractionModule
from .temporal import TemporalModule
from .threshold import ThresholdModule
from .information import InformationModule

__all__ = [
    'DesireEmbeddingModule',
    'AttentionModule', 
    'InteractionModule',
    'TemporalModule',
    'ThresholdModule',
    'InformationModule'
] 