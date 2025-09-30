"""
RetNet baseline model for keyword spotting

Implements RetNet (Retentive Network) as a baseline comparison to Mamba,
providing another O(L) complexity alternative to attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# TODO: Import actual RetNet when yet-another-retnet is available
# from retnet import RetNetModel


class MultiScaleRetention(nn.Module):
    """
    Multi-Scale Retention mechanism
    
    Core component of RetNet that replaces attention with retention.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Retention decay parameter
        self.register_parameter('gamma', nn.Parameter(torch.ones(n_heads)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-scale retention
        
        Args:
            x: Input tensor (B, T, d_model)
            
        Returns:
            output: Retained features (B, T, d_model)
        """
        B, T, d = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)
        
        # Transpose for head-wise processing
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Simplified retention computation (placeholder)
        # In practice, this would use the full retention mechanism
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_weights.masked_fill_(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        out = self.out_proj(out)
        
        return out


class RetNetBlock(nn.Module):
    """
    RetNet block with retention and feed-forward layers
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.retention = MultiScaleRetention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RetNet block
        
        Args:
            x: Input tensor (B, T, d_model)
            
        Returns:
            output: Processed tensor (B, T, d_model)
        """
        # Retention with residual connection
        x = x + self.retention(self.norm1(x))
        
        # Feed-forward with residual connection
        x = x + self.ffn(self.norm2(x))
        
        return x


class RetNetKWS(nn.Module):
    """
    RetNet-based Keyword Spotting Model
    
    Uses the same convolutional front-end as Mamba models but replaces
    Mamba blocks with RetNet blocks for sequence modeling.
    
    Args:
        n_classes (int): Number of output classes
        d_model (int): Model dimension
        n_layers (int): Number of RetNet layers
        n_heads (int): Number of attention heads
        dropout (float): Dropout rate
        in_ch (int): Input channels
    """
    
    def __init__(
        self,
        n_classes: int = 35,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        in_ch: int = 1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Shared convolutional front-end (same as Mamba models)
        self.conv_embed = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2, 1)),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), 
            nn.SiLU(),
            nn.MaxPool2d((2, 1)),
        )
        
        # Calculate flattened dimension
        flattened_dim = 64 * 10  # Same as Mamba models
        
        # Projection layer
        self.proj = nn.Sequential(
            nn.Linear(flattened_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # RetNet blocks
        self.blocks = nn.ModuleList([
            RetNetBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
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
            x: Input tensor (B, T, F) - preprocessed features
            lengths: Optional sequence lengths for masked pooling
            
        Returns:
            logits: Output logits (B, n_classes)
        """
        B, T, F = x.shape
        
        # Reshape for conv processing: (B, T, F) -> (B, 1, F, T)
        x = x.transpose(1, 2).unsqueeze(1)  # (B, 1, F, T)
        
        # Convolutional feature extraction
        x = self.conv_embed(x)  # (B, 64, F', T)
        
        # Reshape for sequence processing
        B, C, F_new, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, -1)  # (B, T, C*F')
        
        # Project to model dimension
        x = self.proj(x)  # (B, T, d_model)
        
        # Process through RetNet blocks
        for block in self.blocks:
            x = block(x)
        
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


def retnet_kws(
    n_classes: int = 35,
    d_model: int = 128,
    n_layers: int = 6,
    n_heads: int = 8
) -> RetNetKWS:
    """
    Create RetNet-based KWS model
    
    Args:
        n_classes: Number of output classes
        d_model: Model dimension
        n_layers: Number of RetNet layers
        n_heads: Number of attention heads
        
    Returns:
        RetNetKWS model instance
    """
    return RetNetKWS(
        n_classes=n_classes,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads
    )