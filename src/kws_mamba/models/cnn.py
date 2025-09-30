"""
CNN baseline model (MobileNetV2) for keyword spotting

Implements the efficient CNN baseline using MobileNetV2 architecture
adapted for audio spectrograms.
"""

import torch
import torch.nn as nn
from typing import Optional

# TODO: Import torchvision when available
# import torchvision.models as models


class MobileNetV2KWS(nn.Module):
    """
    MobileNetV2-based Keyword Spotting Model
    
    Adapts the efficient MobileNetV2 architecture for keyword spotting
    by modifying the input layer for single-channel spectrograms and
    the output layer for classification.
    
    Args:
        n_classes (int): Number of output classes
        width_mult (float): Width multiplier for MobileNetV2 (default: 0.75)
        pretrained (bool): Whether to use pretrained weights (not applicable for audio)
    """
    
    def __init__(
        self,
        n_classes: int = 35,
        width_mult: float = 0.75,
        pretrained: bool = False,
    ):
        super().__init__()
        
        self.n_classes = n_classes
        
        # TODO: Replace with actual MobileNetV2 when torchvision is available
        # For now, create a simplified CNN that mimics MobileNetV2 structure
        self.features = self._make_mobilenet_features(width_mult)
        
        # Adaptive pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, n_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_mobilenet_features(self, width_mult: float) -> nn.Module:
        """
        Create MobileNetV2-like feature extractor
        
        This is a simplified version - replace with actual MobileNetV2
        when torchvision is available.
        """
        
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        input_channel = 32
        last_channel = 1280
        
        input_channel = _make_divisible(input_channel * width_mult, 8)
        last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)
        
        features = []
        
        # First layer: adapted for single-channel input
        features.append(nn.Conv2d(1, input_channel, 3, stride=2, padding=1, bias=False))
        features.append(nn.BatchNorm2d(input_channel))
        features.append(nn.ReLU6(inplace=True))
        
        # Simplified inverted residual blocks
        # In practice, you'd use the full MobileNetV2 architecture
        block_configs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        for t, c, n, s in block_configs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.extend(self._make_inverted_residual(
                    input_channel, output_channel, stride, expand_ratio=t
                ))
                input_channel = output_channel
        
        # Final layer
        features.append(nn.Conv2d(input_channel, last_channel, 1, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))
        
        return nn.Sequential(*features)
    
    def _make_inverted_residual(self, inp, oup, stride, expand_ratio):
        """Create simplified inverted residual block"""
        hidden_dim = int(round(inp * expand_ratio))
        use_residual = stride == 1 and inp == oup
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        
        conv = nn.Sequential(*layers)
        
        if use_residual:
            return [ResidualBlock(conv)]
        else:
            return [conv]
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, T, F) - preprocessed features
            
        Returns:
            logits: Output logits (B, n_classes)
        """
        B, T, F = x.shape
        
        # Reshape for conv processing: (B, T, F) -> (B, 1, F, T)
        x = x.transpose(1, 2).unsqueeze(1)  # (B, 1, F, T)
        
        # Feature extraction
        x = self.features(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class ResidualBlock(nn.Module):
    """Simple residual block wrapper"""
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
    
    def forward(self, x):
        return x + self.conv(x)


def mobilenet_v2_kws(n_classes: int = 35, width_mult: float = 0.75) -> MobileNetV2KWS:
    """
    Create MobileNetV2-based KWS model
    
    Args:
        n_classes: Number of output classes
        width_mult: Width multiplier for channel scaling
        
    Returns:
        MobileNetV2KWS model instance
    """
    return MobileNetV2KWS(
        n_classes=n_classes,
        width_mult=width_mult,
        pretrained=False
    )