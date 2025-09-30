# src/kws_mamba/models/cnn.py
"""
CNN models for KWS (as in the original code).

Includes:
- KeywordCNN: small CNN baseline with time-preserving pooling
- build_mobilenet_v2: MobileNetV2 baseline adapted for 1-channel inputs
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KeywordCNN(nn.Module):
    """Small CNN baseline with time-preserving pool and dropout head.

    Expected input: (B, 1, F, T) spectrogram/MFCC image
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # pool only along frequency axis (keep temporal resolution early)
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # pool both dims

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=1),  # -> (B, 128, 1, 1)
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, F, T)
        x = self.features(x).flatten(1)  # (B, 128)
        x = self.dropout(x)
        return self.fc(x)  # (B, n_classes)


def build_mobilenet_v2(num_classes: int, alpha: float = 0.75, pretrained: bool = False) -> nn.Module:
    """
    MobileNetV2 baseline adapted for single-channel inputs and classifier head.

    Args:
        num_classes: number of output classes (e.g., 35 or 36)
        alpha: width multiplier
        pretrained: if True, load torchvision's default MobileNetV2 weights
    """
    import torchvision.models as tvm

    net = tvm.mobilenet_v2(
        width_mult=alpha,
        weights=None if not pretrained else tvm.MobileNet_V2_Weights.DEFAULT,
    )

    # Adapt first conv to 1 input channel (keep all other params)
    first_conv = net.features[0][0]
    net.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )

    # Replace classifier with simple dropout + linear
    in_feats = net.classifier[-1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feats, num_classes),
    )
    return net


__all__ = ["KeywordCNN", "build_mobilenet_v2"]
