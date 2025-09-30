"""
Mamba-based KWS model with MFCC input

Implements Mamba architecture specifically designed for MFCC features
as an alternative to Mel spectrograms.
"""

import torch
import torch.nn as nn
from typing import Optional

# TODO: Import actual Mamba layer when mamba-ssm is available
# from mamba_ssm import Mamba


class MambaMFCCKWS(nn.Module):
    """
    Mamba-based Keyword Spotting with MFCC Input
    
    Simplified architecture that directly processes MFCC features with
    linear projection followed by Mamba blocks. This approach uses the
    DCT-decorrelated MFCC coefficients which partially substitute for
    early local feature mixing.
    
    Args:
        n_classes (int): Number of output classes
        d_model (int): Model dimension for Mamba blocks
        n_layers (int): Number of Mamba layers
        d_state (int): State dimension for SSM
        expand (int): Expansion factor for Mamba blocks
        n_mfcc (int): Number of MFCC coefficients (typically 13 or 40)
    """
    
    def __init__(
        self,
        n_classes: int,
        d_model: int = 128,
        n_layers: int = 10,
        d_state: int = 16,
        expand: int = 2,
        n_mfcc: int = 40,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_mfcc = n_mfcc
        
        # Direct linear projection from MFCC to model dimension
        # No CNN needed since MFCC is already decorrelated via DCT
        self.proj = nn.Sequential(
            nn.Linear(n_mfcc, d_model),
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
            x: Input MFCC features (B, T, n_mfcc)
            lengths: Optional sequence lengths for masked pooling
            
        Returns:
            logits: Output logits (B, n_classes)
        """
        B, T, _ = x.shape
        
        # Project MFCC to model dimension
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
def mamba_mfcc_small(n_classes: int = 35, n_mfcc: int = 40) -> MambaMFCCKWS:
    """
    Small Mamba-MFCC model
    
    Args:
        n_classes: Number of output classes
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        MambaMFCCKWS model instance
    """
    return MambaMFCCKWS(
        n_classes=n_classes,
        d_model=64,
        n_layers=8,
        d_state=16,
        expand=2,
        n_mfcc=n_mfcc
    )


def mamba_mfcc_medium(n_classes: int = 35, n_mfcc: int = 40) -> MambaMFCCKWS:
    """
    Medium Mamba-MFCC model
    
    Args:
        n_classes: Number of output classes
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        MambaMFCCKWS model instance
    """
    return MambaMFCCKWS(
        n_classes=n_classes,
        d_model=128,
        n_layers=10,
        d_state=16,
        expand=2,
        n_mfcc=n_mfcc
    )


def mamba_mfcc_large(n_classes: int = 35, n_mfcc: int = 40) -> MambaMFCCKWS:
    """
    Large Mamba-MFCC model
    
    Args:
        n_classes: Number of output classes  
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        MambaMFCCKWS model instance
    """
    return MambaMFCCKWS(
        n_classes=n_classes,
        d_model=192,
        n_layers=12,
        d_state=16,
        expand=2,
        n_mfcc=n_mfcc
    )