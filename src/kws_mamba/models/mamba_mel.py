"""
Mamba-based KWS model with Mel spectrogram input

Implements the hybrid CNN + Mamba architecture specifically designed for
Mel spectrogram features with small/medium/large model factories.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# TODO: Import actual Mamba layer when mamba-ssm is available
# from mamba_ssm import Mamba


class MambaMelKWS(nn.Module):
    """
    Mamba-based Keyword Spotting with Mel Spectrogram Input
    
    Hybrid architecture combining:
    1. Convolutional front-end for local pattern extraction
    2. Mamba blocks for efficient sequence modeling  
    3. Masked pooling and classification head
    
    Args:
        n_classes (int): Number of output classes (35 for Google Speech Commands)
        d_model (int): Model dimension for Mamba blocks
        n_layers (int): Number of Mamba layers
        d_state (int): State dimension for SSM (typically 16)
        expand (int): Expansion factor for Mamba blocks (typically 2)
        in_ch (int): Input channels (1 for mono audio)
    """
    
    def __init__(
        self,
        n_classes: int,
        d_model: int = 128,
        n_layers: int = 10,
        d_state: int = 16,
        expand: int = 2,
        in_ch: int = 1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Convolutional front-end for 2D feature extraction
        # Input: (B, C=1, F=40, T) -> Output: (B, 64, F', T')
        self.conv_embed = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2, 1)),  # Downsample frequency, keep time
            
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d((2, 1)),  # Downsample frequency again
        )
        
        # Calculate flattened dimension after conv layers
        # After 2 freq poolings: 40 -> 20 -> 10, so 64 * 10 = 640
        flattened_dim = 64 * 10  # 64 channels * 10 freq bins
        
        # Projection layer: flatten conv output to Mamba input
        self.proj = nn.Sequential(
            nn.Linear(flattened_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )
        
        # Mamba blocks with residual connections
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(d_model),
                # TODO: Replace with actual Mamba layer
                # Mamba(d_model=d_model, d_state=d_state, expand=expand),
                nn.Linear(d_model, d_model),  # Placeholder
                nn.Dropout(0.1)
            ))
        
        # Classification head
        self.classifier = nn.Linear(d_model, n_classes)
        
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, T, F) - already preprocessed features
            lengths: Optional sequence lengths for masked pooling
            
        Returns:
            logits: Output logits (B, n_classes)
        """
        B, T, F = x.shape
        
        # Reshape for conv processing: (B, T, F) -> (B, 1, F, T)
        x = x.transpose(1, 2).unsqueeze(1)  # (B, 1, F, T)
        
        # Convolutional feature extraction
        x = self.conv_embed(x)  # (B, 64, F', T)
        
        # Reshape for sequence processing: (B, 64, F', T) -> (B, T, 64*F')
        B, C, F_new, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, -1)  # (B, T, C*F')
        
        # Project to model dimension
        x = self.proj(x)  # (B, T, d_model)
        
        # Process through Mamba blocks with residual connections
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
        
        # Masked mean pooling
        if lengths is not None:
            # Create mask for variable lengths
            mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1)  # (B, T, 1)
            
            # Apply mask and compute mean
            x_masked = x * mask
            x_sum = x_masked.sum(dim=1)  # (B, d_model)
            lengths_expanded = lengths.unsqueeze(-1).float()  # (B, 1)
            pooled = x_sum / (lengths_expanded + 1e-6)
        else:
            # Simple mean pooling if no lengths provided
            pooled = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


# Model factories for different sizes
def mamba_mel_small(n_classes: int = 35) -> MambaMelKWS:
    """
    Small Mamba-Mel model (~404K parameters)
    
    Args:
        n_classes: Number of output classes
        
    Returns:
        MambaMelKWS model instance
    """
    return MambaMelKWS(
        n_classes=n_classes,
        d_model=64,
        n_layers=8,
        d_state=16,
        expand=2
    )


def mamba_mel_medium(n_classes: int = 35) -> MambaMelKWS:
    """
    Medium Mamba-Mel model (~1.34M parameters)
    
    Args:
        n_classes: Number of output classes
        
    Returns:
        MambaMelKWS model instance
    """
    return MambaMelKWS(
        n_classes=n_classes,
        d_model=128,
        n_layers=10,
        d_state=16,
        expand=2
    )


def mamba_mel_large(n_classes: int = 35) -> MambaMelKWS:
    """
    Large Mamba-Mel model (~3.17M parameters)
    
    Args:
        n_classes: Number of output classes
        
    Returns:
        MambaMelKWS model instance
    """
    return MambaMelKWS(
        n_classes=n_classes,
        d_model=192,
        n_layers=12,
        d_state=16,
        expand=2
    )