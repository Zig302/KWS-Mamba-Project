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

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,retnet]"
```

### Basic Usage

```python
import torch
from mamba_kws import MambaKWS, SpeechCommands, WaveToSpec

# Initialize model
model = MambaKWS(
    n_classes=35,
    d_model=128,
    n_layers=10,
    d_state=16
)

# Load and preprocess data
frontend = WaveToSpec(n_mels=40, n_fft=512, hop_length=160)
dataset = SpeechCommands(data, None, frontend)

# Training loop
for batch in dataloader:
    inputs, targets, lengths = batch
    outputs = model(inputs, lengths)
    loss = criterion(outputs, targets)
    # ... training code
```

### Model Variants

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

### Inference Performance

| Device | Model | Latency (ms) | Throughput (samples/s) |
|--------|-------|--------------|------------------------|
| CPU (i5) | Small | 7.5 | 850 |
| CPU (i5) | Medium | 11.0 | 620 |
| GPU (T4) | Small | 0.8 | 3700 |
| GPU (T4) | Medium | 1.2 | 2400 |
| Raspberry Pi 5 | Small | 17.0 | 58 |

### Memory Usage

All models demonstrate linear memory scaling O(L) with sequence length, making them suitable for processing longer audio contexts without quadratic memory growth.

## Project Structure

```
├── src/mamba_kws/          # Main package
│   ├── models/             # Model implementations
│   ├── data/               # Dataset and preprocessing
│   └── utils/              # Training and benchmarking utilities
├── scripts/                # Training and evaluation scripts
├── notebooks/              # Jupyter notebooks and experiments
├── configs/                # Configuration files
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom config
python scripts/train.py --config configs/mamba_medium.yaml

# Benchmark trained model
python scripts/benchmark.py --model-path checkpoints/best_model.pt
```

## Comparison with Baselines

| Architecture | Parameters | Accuracy | Notes |
|-------------|------------|----------|-------|
| MobileNetV2 | 450K | 96.5% | CNN baseline |
| RetNet-KWS | 1.6M | 97.2% | Alternative O(L) model |
| **Mamba-S** | **404K** | **97.42%** | **Our approach** |
| **Mamba-M** | **1.34M** | **97.59%** | **Our approach** |

## Documentation

- [Model Architecture](docs/architecture.md)
- [Training Guide](docs/training.md)
- [Benchmarking](docs/benchmarking.md)
- [API Reference](docs/api.md)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{makarov2024mamba,
  title={Mamba-based Network Design for Keyword Spotting},
  author={Makarov, Alex and Levi, Ran},
  year={2024},
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