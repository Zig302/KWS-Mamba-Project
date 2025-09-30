"""
Mamba-based Keyword Spotting (KWS) Package

This package implements Mamba-based neural networks for keyword spotting tasks,
providing efficient sequence modeling with linear time complexity.
"""

__version__ = "0.1.0"
__author__ = "Alex Makarov, Ran Levi"
__email__ = "your-email@example.com"

from .models import MambaKWS
from .data import SpeechCommands, WaveToSpec, Augment
from .utils import *

__all__ = [
    "MambaKWS",
    "SpeechCommands", 
    "WaveToSpec",
    "Augment"
]