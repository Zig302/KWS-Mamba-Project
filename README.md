# Mamba-based Keyword Spotting (KWS)

A PyTorch implementation of Mamba-based neural networks for keyword spotting with linear time complexity O(L).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements efficient keyword spotting using Mamba's selective state-space models (SSMs), achieving competitive accuracy while maintaining linear computational complexity with respect to sequence length. Our hybrid architecture combines convolutional feature extraction with Mamba's sequence modeling capabilities.

## Features

- **Linear Time Complexity**: O(L) computational complexity vs O(L²) for attention-based models
- **Hardware-Aware Design**: Optimized for both CPU and GPU deployment
- **Multiple Model Variants**: Small, Medium, and Large configurations for different accuracy/efficiency trade-offs
- **Comprehensive Benchmarking**: Latency, throughput, and memory analysis tools
- **Easy Integration**: Clean Python package structure with simple APIs

## Architecture

Our MambaKWS model combines:
1. **Convolutional Front-end**: Extracts local patterns from Mel spectrograms
2. **Mamba Blocks**: Process sequences with selective state-space modeling
3. **Classifier Head**: Masked pooling and classification for variable-length inputs

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Zig302/KWS-Mamba-Project.git
cd KWS-Mamba-Project
```

### Basic Usage

```python
import torch
from kws_mamba import mamba_mel_medium, SpeechCommands, WaveToSpec

# Initialize model
model = mamba_mel_medium(n_classes=35)

# Load and preprocess data (example)
frontend = WaveToSpec(n_mels=128, n_fft=2048, hop_length=256)
dataset = SpeechCommands(data, None, frontend)

# Training loop
for batch in dataloader:
    inputs, targets, lengths = batch
    outputs = model(inputs, lengths)
    loss = criterion(outputs, targets)
    # ... training code
```

### Model Variants (Mamba-Mel)

| Model | Parameters | d_model | n_layers | Accuracy |
|-------|------------|---------|----------|----------|
| Small | 404K | 64 | 8 | 97.42% |
| Medium | 1.34M | 128 | 10 | 97.59% |
| Large | 3.17M | 192 | 12 | 97.75% |

## Dataset

This project uses the Google Speech Commands V2-35 dataset:
- 35 keyword classes
- 1-second audio clips
- 16kHz sampling rate
- ~105K training samples

## Benchmarks

### Memory Usage

All models demonstrate linear memory scaling O(L) with sequence length, making them suitable for processing longer audio contexts without quadratic memory growth.

## Project Structure

```
├── src/kws_mamba/          # Main package
│   ├── models/             # Model implementations
│   │   ├── mamba_mel.py    # Mamba with Mel spectrogram input
│   │   ├── mamba_mfcc.py   # Mamba with MFCC input
│   │   ├── cnn.py          # MobileNetV2 baseline
│   │   └── retnet.py       # RetNet baseline
│   ├── data/               # Dataset and preprocessing
│   │   ├── audio.py        # WaveToSpec, Augment, collate_fn
│   │   └── dataset.py      # SpeechCommands wrapper
│   └── utils/              # Training and benchmarking utilities
│       ├── metrics.py      # Accuracy, F1, confusion matrix
│       └── seed.py         # Reproducibility and AMP helpers
├── notebooks/              # Jupyter notebooks and experiments
├── assets/                 # Figures, diagrams, results
├── configs/                # Configuration files
├── scripts/                # Training and evaluation scripts
└── tests/                  # Unit tests
```

## Training

```bash
# Train with default configuration
python -m kws_mamba.train

# Train with custom config
python -m kws_mamba.train --config configs/mamba_medium.yaml

# Benchmark trained model
python -m kws_mamba.benchmark --model-path checkpoints/best_model.pt
```

## Comparison with Baselines

| Architecture | Parameters | Accuracy | Notes |
|-------------|------------|----------|-------|
| MobileNetV2 | 450K | 96.5% | CNN baseline |
| RetNet-KWS | 1.6M | 97.2% | Alternative O(L) model |
| **Mamba-S** | **404K** | **97.42%** | **Our approach** |
| **Mamba-M** | **1.34M** | **97.59%** | **Our approach** |
| **Mamba-L** | **3.17M** | **97.75%** | **Our approach** |

## Documentation


## Citation

If you use this work in your research, please cite:

```bibtex
@misc{makarov2024mamba,
  title={Mamba-based Network Design for Keyword Spotting},
  author={Makarov, Alex and Levi, Ran},
  year={2025},
  institution={Bar-Ilan University}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Google Speech Commands Dataset](https://arxiv.org/abs/1804.03209)
- Bar-Ilan University Engineering Faculty
- Academic supervision by Dr. Leonid Yavits
- Mentorship by Zuher Jahshan

## Related Work

- [Mamba SSM](https://github.com/state-spaces/mamba)
- [Speech Commands Dataset](https://github.com/tensorflow/datasets/tree/master/docs/catalog/speech_commands)
- [RetNet](https://github.com/Jamie-Stirling/RetNet)
