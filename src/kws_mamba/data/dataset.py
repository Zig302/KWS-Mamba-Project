"""
SpeechCommands dataset wrapper with normalization

Handles loading Google Speech Commands dataset with preprocessing pipeline
and proper normalization using precomputed statistics.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Any, Tuple


class SpeechCommands(Dataset):
    """
    Google Speech Commands Dataset Wrapper
    
    Wraps the raw dataset from HuggingFace datasets library and applies
    the complete preprocessing pipeline including augmentation, feature
    extraction, and normalization.
    
    Args:
        dataset: Raw dataset from datasets library (train/validation/test split)
        augment: Optional augmentation pipeline (Augment class)
        frontend: Audio preprocessing pipeline (WaveToSpec class) 
        mean: Precomputed normalization mean tensor
        std: Precomputed normalization std tensor
    """
    
    def __init__(
        self,
        dataset: Any,
        augment: Optional[Callable] = None,
        frontend: Optional[Callable] = None,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        self.dataset = dataset
        self.augment = augment
        self.frontend = frontend
        
        # Use provided normalization stats or defaults
        self.mean = mean if mean is not None else torch.tensor(0.0)
        self.std = std if std is not None else torch.tensor(1.0)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get preprocessed sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of:
                - features: Processed features tensor (T, F)  
                - label: Class label integer
        """
        # Get raw sample from dataset
        sample = self.dataset[idx]
        
        # Extract audio and label
        # Assuming dataset has 'audio' dict with 'array' key and 'label' key
        wav = torch.tensor(sample['audio']['array'], dtype=torch.float32)
        label = sample['label']
        
        # Apply waveform-level augmentation if provided (training only)
        if self.augment is not None:
            wav = self.augment(wav)
        
        # Apply frontend processing (WaveToSpec)
        if self.frontend is not None:
            features = self.frontend(wav)  # Shape: (C, F, T)
        else:
            # Fallback: use raw waveform (shouldn't happen in practice)
            features = wav.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        
        # Normalize using precomputed statistics
        # features shape: (C=1, F, T) -> normalize across F and T dimensions
        features = (features - self.mean) / (self.std + 1e-6)
        
        # Squeeze channel dimension and transpose for sequence processing
        # (C=1, F, T) -> (F, T) -> (T, F) for input to models
        features = features.squeeze(0).transpose(0, 1)
        
        return features, label


def compute_normalization_stats(dataset, frontend, device='cpu', max_samples=None):
    """
    Compute normalization statistics from training dataset
    
    Args:
        dataset: Training dataset  
        frontend: WaveToSpec frontend
        device: Device to compute on
        max_samples: Maximum samples to use (None = all)
        
    Returns:
        Tuple of (mean, std) tensors for normalization
    """
    print("Computing normalization statistics...")
    
    frontend = frontend.to(device)
    frontend.eval()
    
    # Collect features from training set
    all_features = []
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    with torch.no_grad():
        for i in range(num_samples):
            if i % 1000 == 0:
                print(f"Processing {i}/{num_samples}")
            
            sample = dataset[i]
            wav = torch.tensor(sample['audio']['array'], dtype=torch.float32).to(device)
            
            # Apply frontend
            features = frontend(wav)  # (C, F, T)
            all_features.append(features.cpu())
    
    # Concatenate along time dimension and compute stats
    all_features = torch.cat(all_features, dim=-1)  # (C, F, T_total)
    
    mean = all_features.mean(dim=(0, 2), keepdim=True)  # (1, F, 1)
    std = all_features.std(dim=(0, 2), keepdim=True)   # (1, F, 1)
    
    print(f"Normalization stats computed from {num_samples} samples")
    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    
    return mean, std