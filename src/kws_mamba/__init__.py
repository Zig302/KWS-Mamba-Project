"""
KWS-Mamba: Keyword Spotting with Mamba State Space Models

A PyTorch implementation of efficient keyword spotting using Mamba's 
selective state-space models with linear time complexity O(L).
"""

__version__ = "0.1.0"
__author__ = "Alex Makarov, Ran Levi"

# Core model imports
from .models.mamba_mel import build_small as mamba_mel_small, build_medium as mamba_mel_medium, build_large as mamba_mel_large
from .models.mamba_mfcc import build_small as mamba_mfcc_small, build_medium as mamba_mfcc_medium, build_large as mamba_mfcc_large
from .models.cnn import build_mobilenet_v2 as mobilenet_v2_kws
from .models.retnet import RetNetKWS

# Data processing imports
from .data.audio import WaveToSpec, Augment, collate_seq, collate_image
from .data.dataset import SpeechCommands

__all__ = [
    # Mamba models (Mel spectrogram)
    "mamba_mel_small", "mamba_mel_medium", "mamba_mel_large",
    # Mamba models (MFCC)
    "mamba_mfcc_small", "mamba_mfcc_medium", "mamba_mfcc_large", 
    # Baseline models
    "mobilenet_v2_kws", "RetNetKWS",
    # Data processing
    "WaveToSpec", "Augment", "SpeechCommands", "collate_seq", "collate_image"
]