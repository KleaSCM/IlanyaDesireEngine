"""
Ilanya Desire Engine - Main Package

Modular desire engine for advanced desire space management.
"""

from .core import DesireEngine
from .config import DesireEngineConfig
from .models import Desire, DesireState

__version__ = "0.1.0"
__author__ = "KleaSCM"
__email__ = "KleaSCM@gmail.com"

__all__ = [
    "DesireEngine",
    "DesireEngineConfig", 
    "Desire",
    "DesireState"
] 