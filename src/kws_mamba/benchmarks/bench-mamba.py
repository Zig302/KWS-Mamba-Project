# benchmarks/bench-mamba.py
#!/usr/bin/env python3
"""
Mamba KWS Inference Benchmark

Benchmarks small/medium/large Mamba models for keyword spotting.
Measures latency (single-sample), throughput (various batch sizes),
and CUDA memory (if available).
"""
from __future__ import annotations

import time
import statistics
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

IS_COLAB = 'google.colab' in str(globals().get('get_ipython', lambda: None)())
HAS_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if HAS_CUDA else "cpu")

print(f"Environment: {'Colab' if IS_COLAB else 'Local'}")
print(f"CUDA available: {HAS_CUDA}")
print(f"Device: {device}")

from mamba_ssm import Mamba

# ---------- Minimal frontend (not used for timing; kept for parity) ----------
class WaveToSpec:
    def __init__(self,
                 feature_type: str = "mel",
                 sample_rate: int = 16_000,
                 n_fft: int = 2048,
                 hop_length: int = 256,
                 n_mels: int = 128,
                 n_mfcc: int = 40,
                 top_db: int | None = 80,
                 apply_mask: bool = True,
                 freq_mask_param: int = 15,
                 time_mask_param: int = 10):
        self.feature_type = feature_type.lower(); assert self.feature_type in {"mel", "mfcc"}
        self.apply_mask = apply_mask and self.feature_type == "mel"

        if self.feature_type == "mel":
            self.spec = T.MelSpectrogram(sample_rate, n_fft, hop_length, n_mels, power=2)
            self.to_db = T.AmplitudeToDB(stype="power", top_db=top_db)
            if self.apply_mask:
                self.freq_mask = T.FrequencyMasking(freq_mask_param)
                self.time_mask = T.TimeMasking(time_mask_param)
        else:
            self.spec = T.MFCC(sample_rate, n_mfcc,
                                melkwargs=dict(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels))
            self.to_db = None
            self.freq_mask = self.time_mask = None

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        feats = self.spec(wav)
        if self.to_db is not None:
            feats = self.to_db(feats.clamp(min=1e-10))
        return feats

# ---------- Model ----------
class MambaKWS(nn.Module):
    def __init__(self, num_classes: int, d_model=256, d_state=32, expand=2, n_layers=8, in_ch=1, feature_dim=128):
        super().__init__()
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
            nn.MaxPool2d((2, 1)),
        )

        freq_dim_after_conv = feature_dim // 4
        flattened_dim = 64 * freq_dim_after_conv

        self.proj = nn.Sequential(
            nn.Linear(flattened_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(0.1)
        )

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "mamba": Mamba(d_model=d_model, d_state=d_state, expand=expand),
                "dropout": nn.Dropout(max(0.02, 0.05 - (i * 0.005)))
            }) for i in range(n_layers)
        ])
        self.pre_classifier_norm = nn.LayerNorm(d_model)

        self.classifier_dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, lengths: torch.Tensor | None = None):
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.conv_embed(x)
        x = x.permute(0, 3, 1, 2).contiguous().flatten(2)
        x = self.proj(x)

        for blk in self.blocks:
            residual = x
            x = blk["norm"](x)
            x = blk["mamba"](x)
            x = blk["dropout"](x)
            x = residual + x

        x = self.pre_classifier_norm(x)

        if lengths is not None:
            t_lens = torch.div(lengths, 2, rounding_mode='floor').clamp(min=1).to(x.device)
            Tprime = x.size(1)
            mask = (torch.arange(Tprime, device=x.device)[None, :] < t_lens[:, None]).float().unsqueeze(-1)
            x_sum = (x * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = x_sum / denom
        else:
            pooled = x.mean(dim=1)

        return self.classifier(self.classifier_dropout(pooled))

# ---------- Model configurations ----------
MODEL_CONFIGS = {
    'small': {
        'model_path': 'content/small_kws_mel_97.42.pt',
        'd_model': 64,
        'd_state': 16,
        'n_layers': 8,
        'expected_classes': 35
    },
    'medium': {
        'model_path': 'content/medium_kws_melSpec_97.58.pt',
        'd_model': 128,
        'd_state': 16,
        'n_layers': 10,
        'expected_classes': 35
    },
    'large': {
        'model_path': 'content/large_kws_melSpec_97.75.pt',
        'd_model': 192,
        'd_state': 16,
        'n_layers': 12,
        'expected_classes': 35
    }
}

# ---------- Utilities ----------
def load_model(model_size: str, device: torch.device) -> MambaKWS:
    cfg = MODEL_CONFIGS[model_size]
    model = MambaKWS(
        num_classes=cfg['expected_classes'],
        d_model=cfg['d_model'],
        d_state=cfg['d_state'],
        n_layers=cfg['n_layers'],
        feature_dim=128
    )
    model_path = Path(cfg['model_path'])
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=True)
            print(f"Loaded weights: {model_path}")
        except Exception as e:
            print(f"Could not load weights for {model_size}: {e} (using random weights)")
    else:
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
        print(f"Using random weights for {model_size}")
    model.to(device).eval()
    return model

def create_dummy_input(batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    T, F = 63, 128
    features = torch.randn(batch_size, T, F, device=device)
    lengths = torch.full((batch_size,), T, dtype=torch.long, device=device)
    return features, lengths

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_memory_usage(model: nn.Module, device: torch.device,
                           batch_sizes: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1028, 2056]) -> Dict[str, Dict[str, float]]:
    if device.type != 'cuda':
        return {}
    model.eval()
    results = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()

    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated()

        x, l = create_dummy_input(batch_size, device)
        with torch.no_grad():
            _ = model(x, l)
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated()
        inference_memory = peak - baseline_memory
        activation_memory = peak - memory_before

        results[f"batch_{batch_size}"] = {
            'baseline_mb': baseline_memory / (1024**2),
            'peak_mb': peak / (1024**2),
            'inference_mb': inference_memory / (1024**2),
            'activation_mb': activation_memory / (1024**2),
            'memory_per_sample_mb': activation_memory / batch_size / (1024**2)
        }

        del x, l
        torch.cuda.empty_cache()

    return results

def benchmark_latency(model: nn.Module, device: torch.device, num_runs: int = 1000) -> Dict[str, float]:
    model.eval()
    x, l = create_dummy_input(1, device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, l)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            x, l = create_dummy_input(1, device)
            start = time.perf_counter()
            _ = model(x, l)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

    return {
        'mean_ms': statistics.mean(times),
        'median_ms': statistics.median(times),
        'std_ms': statistics.stdev(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99))
    }

def benchmark_throughput(model: nn.Module, device: torch.device,
                         batch_sizes: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1028, 2056]) -> Dict[int, Dict[str, float]]:
    model.eval()
    results = {}
    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size}...")
        x, l = create_dummy_input(batch_size, device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(x, l)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        num_runs = max(10, 100 // batch_size)
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                x, l = create_dummy_input(batch_size, device)
                start = time.perf_counter()
                _ = model(x, l)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)

        avg_time = statistics.mean(times)
        results[batch_size] = {
            'avg_time_s': avg_time,
            'throughput_samples_per_s': batch_size / avg_time,
            'time_per_sample_ms': (avg_time / batch_size) * 1000
        }
    return results

def run_full_benchmark():
    print("Starting Mamba KWS Inference Benchmark")
    print(f"Device: {device}")
    print("=" * 60)

    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'cuda_available': HAS_CUDA,
        'models': {}
    }

    for model_size in ['small', 'medium', 'large']:
        print(f"\nBenchmarking {model_size.upper()} model...")
        model = load_model(model_size, device)
        param_count = count_parameters(model)
        print(f"Model parameters: {param_count:,}")

        print("Running latency benchmark (1000 runs)...")
        latency_results = benchmark_latency(model, device)

        print("Running throughput benchmark...")
        throughput_results = benchmark_throughput(model, device)

        memory_results = {}
        if device.type == 'cuda':
            print("Running memory usage benchmark...")
            memory_results = benchmark_memory_usage(model, device)

        results['models'][model_size] = {
            'parameters': param_count,
            'config': MODEL_CONFIGS[model_size],
            'latency': latency_results,
            'throughput': throughput_results,
            'memory': memory_results
        }

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"benchmark_results_mamba_{ts}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")

    return results

if __name__ == "__main__":
    _ = run_full_benchmark()
