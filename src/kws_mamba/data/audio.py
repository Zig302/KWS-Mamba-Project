"""
Audio preprocessing and augmentation utilities

Contains WaveToSpec for feature extraction, Augment for data augmentation,
and collate_fn for batching variable-length sequences.
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from typing import Optional, List, Tuple
import random


class WaveToSpec(nn.Module):
    """
    Audio preprocessing pipeline: waveform → spectrogram/MFCC
    
    Supports both Mel spectrograms and MFCC features with optional SpecAugment.
    
    Args:
        sample_rate (int): Audio sample rate (default: 16000)
        n_fft (int): FFT window size
        hop_length (int): Hop length for STFT  
        n_mels (int): Number of Mel frequency bins
        n_mfcc (Optional[int]): Number of MFCC coefficients (if None, returns log-Mel)
        apply_mask (bool): Whether to apply SpecAugment masking during training
        freq_mask_param (int): Frequency masking parameter for SpecAugment
        time_mask_param (int): Time masking parameter for SpecAugment
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 40,
        n_mfcc: Optional[int] = None,
        apply_mask: bool = False,
        freq_mask_param: int = 8,
        time_mask_param: int = 10,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.apply_mask = apply_mask
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        
        # MFCC transform if requested
        if n_mfcc is not None:
            self.mfcc_transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    'n_fft': n_fft,
                    'hop_length': hop_length,
                    'n_mels': n_mels,
                    'normalized': True
                }
            )
        else:
            self.mfcc_transform = None
        
        # SpecAugment transforms
        if apply_mask:
            self.freq_mask = T.FrequencyMasking(freq_mask_param)
            self.time_mask = T.TimeMasking(time_mask_param)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to features
        
        Args:
            waveform: Input waveform tensor of shape (L,) or (1, L)
            
        Returns:
            features: Log-Mel spectrogram or MFCC of shape (C, F, T)
        """
        # Ensure waveform has batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Apply MFCC or Mel spectrogram
        if self.mfcc_transform is not None:
            features = self.mfcc_transform(waveform)
        else:
            features = self.mel_transform(waveform)
            # Convert to log scale for better numerical stability
            features = torch.log(features + 1e-8)
        
        # Apply SpecAugment if enabled (only during training)
        if self.apply_mask and self.training:
            # Apply frequency masking twice as mentioned in your paper
            features = self.freq_mask(features)
            features = self.freq_mask(features)
            # Apply time masking twice for stronger regularization
            features = self.time_mask(features)
            features = self.time_mask(features)
        
        return features


class Augment(nn.Module):
    """
    Waveform-level augmentation pipeline
    
    Applies random time shifting and noise injection to raw audio waveforms
    before spectral feature extraction.
    
    Args:
        time_shift_range (int): Maximum time shift in samples (±range)
        noise_std_range (Tuple[float, float]): Range for additive noise std dev
    """
    
    def __init__(
        self,
        time_shift_range: int = 1600,  # ±0.1s at 16kHz
        noise_std_range: Tuple[float, float] = (0.001, 0.01),
    ):
        super().__init__()
        self.time_shift_range = time_shift_range
        self.noise_std_range = noise_std_range
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to waveform
        
        Args:
            waveform: Input waveform tensor
            
        Returns:
            augmented_waveform: Augmented waveform tensor
        """
        if not self.training:
            return waveform
        
        # Time shifting
        if self.time_shift_range > 0:
            shift = random.randint(-self.time_shift_range, self.time_shift_range)
            if shift > 0:
                # Shift right: pad left, truncate right
                waveform = torch.cat([torch.zeros(shift), waveform[:-shift]])
            elif shift < 0:
                # Shift left: truncate left, pad right
                waveform = torch.cat([waveform[-shift:], torch.zeros(-shift)])
        
        # Additive noise injection
        if self.noise_std_range[1] > 0:
            noise_std = random.uniform(*self.noise_std_range)
            noise = torch.randn_like(waveform) * noise_std
            waveform = waveform + noise
        
        return waveform


def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for variable-length sequences
    
    Handles padding of feature sequences to uniform length within each batch
    while preserving the true sequence lengths for masked operations.
    
    Args:
        batch: List of (features, label) tuples where features are (T, F)
        
    Returns:
        Tuple of:
            - padded_features: Padded features tensor (B, T_max, F)
            - labels: Label tensor (B,)
            - lengths: True sequence lengths tensor (B,)
    """
    features_list, labels = zip(*batch)
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Get dimensions
    batch_size = len(features_list)
    max_time = max(feat.shape[0] for feat in features_list)
    n_features = features_list[0].shape[1]
    
    # Initialize padded tensor
    padded_features = torch.zeros(batch_size, max_time, n_features)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill padded tensor and record lengths
    for i, feat in enumerate(features_list):
        seq_len = feat.shape[0]
        padded_features[i, :seq_len] = feat
        lengths[i] = seq_len
    
    return padded_features, labels, lengths