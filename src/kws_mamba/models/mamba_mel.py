# src/kws_mamba/models/mamba_mel.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MambaMelKWS(nn.Module):
    """
    Mel-spectrogram Mamba KWS model.

    Input:
        x: [B, T, F]  (time-major features; use collate_seq)
        lengths: Optional[B] original T lengths (for mask-aware pooling)

    Architecture
      Conv2d front-end → Linear projection to d_model →
      [Mamba block * n_layers] with pre-norm + residual + dropout →
      LayerNorm → mask-aware mean pooling over time → classifier.
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int = 128,        # width (set by variant)
        d_state: int = 16,         # state dim (fixed across variants)
        expand: int = 2,
        n_layers: int = 10,        # depth (set by variant)
        in_ch: int = 1,
        feature_dim: int = 128,    # Mel bins
    ):
        super().__init__()

        # ----- Conv embedding (2× blocks, keep T on 2nd pool) -----
        self.conv_embed = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.MaxPool2d((2, 1)),   # downsample F, preserve T
        )

        # ----- Projection to Mamba width -----
        freq_dim_after_conv = feature_dim // 4
        flattened_dim = 64 * freq_dim_after_conv
        self.proj = nn.Sequential(
            nn.Linear(flattened_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        # ----- Mamba blocks (pre-norm + residual + dropout schedule) -----
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "mamba": Mamba(d_model=d_model, d_state=d_state, expand=expand),
                "dropout": nn.Dropout(max(0.02, 0.05 - (i * 0.005))),
            }) for i in range(n_layers)
        ])

        # ----- Head -----
        self.pre_classifier_norm = nn.LayerNorm(d_model)
        self.classifier_dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, F]
        lengths: time lengths pre-downsample (for mask-aware pooling)
        """
        # [B, T, F] -> [B, 1, F, T] for Conv2d
        x = x.permute(0, 2, 1).unsqueeze(1)

        # Conv front-end
        x = self.conv_embed(x)                              # [B, 64, F', T']

        # Flatten per time step & project
        x = x.permute(0, 3, 1, 2).contiguous().flatten(2)   # [B, T', 64*F']
        x = self.proj(x)                                    # [B, T', d_model]

        # Mamba stack with residuals
        for blk in self.blocks:
            residual = x
            x = blk["norm"](x)
            x = blk["mamba"](x)
            x = blk["dropout"](x)
            x = residual + x

        x = self.pre_classifier_norm(x)

        # Mask-aware mean pooling over time (first pool halves T)
        if lengths is not None:
            t_lens = torch.div(lengths, 2, rounding_mode="floor").clamp(min=1).to(x.device)
            Tprime = x.size(1)
            mask = (torch.arange(Tprime, device=x.device)[None, :] < t_lens[:, None]).float().unsqueeze(-1)
            x_sum = (x * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = x_sum / denom
        else:
            pooled = x.mean(dim=1)

        return self.classifier(self.classifier_dropout(pooled))


# -----------------
# Variant builders
# -----------------
def build_small(num_classes: int) -> MambaMelKWS:
    """
    Small: d_model=64, n_layers=8, d_state=16, expand=2
    """
    return MambaMelKWS(num_classes=num_classes, d_model=64, n_layers=8, d_state=16, expand=2)


def build_medium(num_classes: int) -> MambaMelKWS:
    """
    Medium: d_model=128, n_layers=10, d_state=16, expand=2
    """
    return MambaMelKWS(num_classes=num_classes, d_model=128, n_layers=10, d_state=16, expand=2)


def build_large(num_classes: int) -> MambaMelKWS:
    """
    Large: d_model=192, n_layers=12, d_state=16, expand=2
    """
    return MambaMelKWS(num_classes=num_classes, d_model=192, n_layers=12, d_state=16, expand=2)


__all__ = [
    "MambaMelKWS",
    "build_small",
    "build_medium",
    "build_large",
]
