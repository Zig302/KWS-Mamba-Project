# src/kws_mamba/models/mamba_mfcc.py

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from mamba_ssm import Mamba


__all__ = [
    "MambaKWS",          # class exactly as in the original MFCC notebooks
    "build_small",
    "build_medium",
    "build_large",
]


class MambaKWS(nn.Module):
    """
    Keyword Spotting with MFCC front-end -> Linear(40 -> d_model) -> Mamba x n_layers -> Classifier.

    Stays faithful to the original Colab MFCC implementations:
      - 40-dim MFCC features are projected to d_model via Linear + LayerNorm + SiLU + Dropout.
      - Stack of n_layers blocks, each: LayerNorm -> Mamba(d_model, d_state, expand) -> Dropout,
        with residual connections and the same per-layer dropout schedule.
      - Pre-classifier LayerNorm, then a 2-layer MLP head.
      - Mask-aware mean pooling over time using the original (non-downsampled) lengths.

    Args:
        num_classes: number of output classes.
        d_model: width of the Mamba blocks.
        d_state: Mamba state size 
        expand: Mamba expansion factor 
        n_layers: number of stacked Mamba blocks.
        feature_dim: input feature dimension (40 for MFCC).
        p_drop: dropout used in the projection block.
    """
    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        d_state: int = 16,
        expand: int = 2,
        n_layers: int = 8,
        feature_dim: int = 40,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        # Linear(40 -> d_model) + LN + SiLU + Dropout
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(p_drop),
        )

        # Stack of Mamba blocks with per-layer dropout schedule: max(0.02, 0.05 - 0.005*i)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "mamba": Mamba(d_model=d_model, d_state=d_state, expand=expand),
                "dropout": nn.Dropout(max(0.02, 0.05 - (i * 0.005))),
            }) for i in range(n_layers)
        ])

        self.pre_classifier_norm = nn.LayerNorm(d_model)

        # Classifier head: Dropout -> Linear(d_model -> d_model//2) -> SiLU -> Dropout -> Linear(... -> num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, F]  (F = feature_dim, i.e., 40 for MFCC)
        lengths: optional [B] tensor with valid sequence lengths (no time downsampling here).
        """
        # Project MFCC frames into model width
        x = self.proj(x)  # [B, T, d_model]

        # Mamba blocks with residuals
        for blk in self.blocks:
            residual = x
            x = blk["norm"](x)
            x = blk["mamba"](x)
            x = blk["dropout"](x)
            x = residual + x

        x = self.pre_classifier_norm(x)

        # Mask-aware mean pooling over time (no time downsampling)
        if lengths is not None:
            Tprime = x.size(1)
            mask = (torch.arange(Tprime, device=x.device)[None, :] < lengths[:, None]).float()  # [B, T]
            x_sum = (x * mask.unsqueeze(-1)).sum(dim=1)                                         # [B, d_model]
            denom = mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)                                # [B, 1]
            pooled = x_sum / denom
        else:
            pooled = x.mean(dim=1)

        # Logits
        return self.classifier(pooled)  # [B, num_classes]


# ---- Factory helpers mirroring three MFCC variants ----

def build_small(num_classes: int) -> MambaKWS:
    """
    Small: d_model=64, n_layers=8, d_state=16, expand=2
    """
    return MambaKWS(num_classes, d_model=64, n_layers=8, d_state=16, expand=2)

def build_medium(num_classes: int) -> MambaKWS:
    """
    Medium: d_model=128, n_layers=10, d_state=16, expand=2
    """
    return MambaKWS(num_classes, d_model=128, n_layers=10, d_state=16, expand=2)

def build_large(num_classes: int) -> MambaKWS:
    """
    Large: d_model=192, n_layers=12, d_state=16, expand=2
    """
    return MambaKWS(num_classes, d_model=192, n_layers=12, d_state=16, expand=2)
