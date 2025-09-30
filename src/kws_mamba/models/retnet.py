# src/kws_mamba/models/retnet.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from yet_another_retnet.retention import MultiScaleRetention


class ChannelGroupNorm(nn.Module):
    """
    GroupNorm that operates on sequence tensors shaped [B, T, D] by
    transposing to [B, D, T] and back.
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, D]
        return self.gn(x.transpose(1, 2)).transpose(1, 2)


class RetNetBlock(nn.Module):
    """Pre-norm residual block with MultiScaleRetention (parallel form)."""
    def __init__(self, d_model: int, n_heads: int, drop: float):
        super().__init__()
        self.norm = ChannelGroupNorm(num_groups=n_heads, num_channels=d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.retention = MultiScaleRetention(
            embed_dim=d_model,
            num_heads=n_heads,
            relative_position=False,
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop)

        self.ffn_norm = ChannelGroupNorm(num_groups=n_heads, num_channels=d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, D]
        # Retention sublayer
        residual = x
        x = self.norm(x)
        q = F.normalize(self.q_proj(x), dim=-1)
        k = F.normalize(self.k_proj(x), dim=-1)
        v = self.v_proj(x)
        y, _ = self.retention.forward_parallel(q, k, v)
        y = self.out_proj(y)
        x = residual + self.dropout(y)

        # FFN sublayer
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)
        return x


class RetNetKWS(nn.Module):
    """
    RetNet keyword spotting model.

    Expected input:
        x: [B, T, F]   (time-major features from collate_seq)
        lengths: Optional[B] (raw T lengths before conv downsampling)

    Architecture:
      Conv2d embed (2x blocks with MaxPool) →
      Linear proj to d_model →
      N * RetNetBlock →
      pre-classifier LayerNorm →
      mask-aware mean pooling over time →
      classifier
    """
    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        in_ch: int = 1,
        feature_dim: int = 128,
    ):
        super().__init__()

        # Convolutional embedding (shared with Mamba models)
        self.conv_embed = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),           # downsample F and T by 2

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d((2, 1)),      # downsample F by 2 (T unchanged)
        )

        # Project [64 * (feature_dim/4)] → d_model
        freq_dim_after_conv = feature_dim // 4
        flattened_dim = 64 * freq_dim_after_conv
        self.proj = nn.Sequential(
            nn.Linear(flattened_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        # RetNet blocks (slightly decaying dropout across depth)
        self.blocks = nn.ModuleList([
            RetNetBlock(d_model=d_model, n_heads=n_heads, drop=max(0.02, 0.03 - (i * 0.003)))
            for i in range(n_layers)
        ])

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
        lengths: original T lengths before conv downsampling (optional, for mask-aware pooling)
        """
        # [B, T, F] → [B, 1, F, T]
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.conv_embed(x)                             # [B, 64, F', T']
        # [B, 64, F', T'] → [B, T', 64*F']
        x = x.permute(0, 3, 1, 2).contiguous().flatten(2)
        x = self.proj(x)                                   # [B, T', d_model]

        for blk in self.blocks:
            x = blk(x)

        x = self.pre_classifier_norm(x)

        # Mask-aware mean pooling over time (T')
        if lengths is not None:
            # Time was downsampled by 2 in the first MaxPool2d(2)
            t_lens = torch.div(lengths, 2, rounding_mode='floor').clamp(min=1).to(x.device)
            Tprime = x.size(1)
            mask = (torch.arange(Tprime, device=x.device)[None, :] < t_lens[:, None]).float().unsqueeze(-1)
            x_sum = (x * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = x_sum / denom
        else:
            pooled = x.mean(dim=1)

        logits = self.classifier(self.classifier_dropout(pooled))
        return logits
