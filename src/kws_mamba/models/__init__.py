"""
Model architectures for KWS-Mamba.

Contains Mamba-based models and baseline architectures.
"""

from .mamba_mel import MambaMelKWS, build_small as mamba_mel_small, build_medium as mamba_mel_medium, build_large as mamba_mel_large
from .mamba_mfcc import MambaKWS, build_small as mamba_mfcc_small, build_medium as mamba_mfcc_medium, build_large as mamba_mfcc_large
from .cnn import build_mobilenet_v2 as mobilenet_v2_kws
from .retnet import RetNetKWS

__all__ = [
    # Mamba Mel models
    "MambaMelKWS",
    "mamba_mel_small",
    "mamba_mel_medium",
    "mamba_mel_large",
    # Mamba MFCC models
    "MambaKWS",
    "mamba_mfcc_small",
    "mamba_mfcc_medium",
    "mamba_mfcc_large",
    # Baseline models
    "mobilenet_v2_kws",
    "RetNetKWS",
]
