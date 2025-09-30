# benchmarks/bench-retnet.py
#!/usr/bin/env python3
"""
RetNet KWS Inference Benchmark (Parallel + Recurrent)

Runs both parallel and recurrent paths for a RetNet KWS model size,
mirroring your original architecture and benchmarking approach.
"""
from __future__ import annotations

import time
import statistics
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from yet_another_retnet.retention import MultiScaleRetention
USE_RETNET_LIB = True

# ---------- Minimal frontend (not used for timing; parity only) ----------
class WaveToSpec:
    def __init__(self,
                 feature_type: str = "mel",
                 sample_rate: int = 16_000,
                 n_fft: int = 2048,
                 hop_length: int = 256,
                 n_mels: int = 128,
                 n_mfcc: int = 40,
                 top_db: int | None = 80,
                 apply_mask: bool = False,
                 freq_mask_param: int = 12,
                 time_mask_param: int = 20):
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
        if self.apply_mask:
            feats = self.freq_mask(feats); feats = self.time_mask(feats)
        if self.to_db is not None:
            feats = self.to_db(feats.clamp(min=1e-10))
        return feats

# ---------- RetNet blocks with parallel + recurrent paths ----------
class ChannelGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)
    def forward(self, x):  # x: [B, T, D]
        return self.gn(x.transpose(1, 2)).transpose(1, 2)

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.lin_q = nn.Linear(d_model, d_model, bias=False)
        self.lin_k = nn.Linear(d_model, d_model, bias=False)
        self.lin_v = nn.Linear(d_model, d_model, bias=False)
        self.proj  = nn.Linear(d_model, d_model, bias=False)

    def forward_parallel(self, q, k, v):  # [B, T, D]
        attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))
        y = torch.matmul(F.softmax(attn, dim=-1), v)
        return self.proj(y), None

    def forward_recurrent(self, q_t, k_t, v_t, idx: int, state: Optional[Dict[str, torch.Tensor]]):
        if state is None:
            state = {'K': k_t, 'KV': torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1))}
        else:
            state['K'] = state['K'] + k_t
            state['KV'] = state['KV'] + torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1))
        denom = (torch.sum(q_t * state['K'], dim=-1, keepdim=True) + 1e-6)
        y_t = torch.bmm(state['KV'], q_t.unsqueeze(2)).squeeze(2) / denom
        return self.proj(y_t), state

class RetNetBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, drop: float):
        super().__init__()
        self.n_heads = n_heads
        self.norm = ChannelGroupNorm(num_groups=n_heads, num_channels=d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.retention = MultiScaleRetention(embed_dim=d_model, num_heads=n_heads, relative_position=False)

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

    def forward_parallel(self, x):  # x: [B, T, D]
        residual = x
        x = self.norm(x)
        q = F.normalize(self.q_proj(x), dim=-1)
        k = F.normalize(self.k_proj(x), dim=-1)
        v = self.v_proj(x)
        y, _ = self.retention.forward_parallel(q, k, v)
        y = self.out_proj(y)
        x = residual + self.dropout(y)

        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)
        return x

    def step_recurrent(self, h_t, t_idx: int, state=None):
        residual = h_t
        x1 = self.norm(h_t.unsqueeze(1)).squeeze(1)
        q_t = F.normalize(self.q_proj(x1), dim=-1)
        k_t = F.normalize(self.k_proj(x1), dim=-1)
        v_t = self.v_proj(x1)

        new_state = None
        if USE_RETNET_LIB and hasattr(self.retention, "forward_recurrent"):
            y_t, new_state = self.retention.forward_recurrent(q_t, k_t, v_t, t_idx, state)
        else:
            y1, _ = self.retention.forward_parallel(q_t.unsqueeze(1), k_t.unsqueeze(1), v_t.unsqueeze(1))
            y_t = y1.squeeze(1)
            new_state = state

        y_t = self.out_proj(y_t)
        h_t = residual + self.dropout(y_t)

        residual = h_t
        x2 = self.ffn_norm(h_t.unsqueeze(1)).squeeze(1)
        h_t = residual + self.ffn(x2)
        return h_t, new_state

class RetNetKWS(nn.Module):
    def __init__(self, num_classes: int, d_model=256, n_layers=8, n_heads=8, in_ch=1, feature_dim=128):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model

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
            RetNetBlock(d_model=d_model, n_heads=n_heads, drop=max(0.02, 0.03 - (i * 0.003)))
            for i in range(n_layers)
        ])
        self.pre_classifier_norm = nn.LayerNorm(d_model)
        self.classifier_dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(d_model // 2, num_classes)
        )

    def _embed(self, x):  # x: [B, T, F]
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.conv_embed(x)
        x = x.permute(0, 3, 1, 2).contiguous().flatten(2)
        x = self.proj(x)
        return x

    def forward(self, x, lengths: torch.Tensor | None = None, mode: str = "parallel"):
        x = self._embed(x)
        B, Tprime, D = x.shape

        if mode == "parallel":
            for blk in self.blocks:
                x = blk.forward_parallel(x)
            x = self.pre_classifier_norm(x)

            if lengths is not None:
                t_lens = torch.div(lengths, 2, rounding_mode='floor').clamp(min=1).to(x.device)
                mask = (torch.arange(Tprime, device=x.device)[None, :] < t_lens[:, None]).float().unsqueeze(-1)
                x_sum = (x * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled = x_sum / denom
            else:
                pooled = x.mean(dim=1)

            return self.classifier(self.classifier_dropout(pooled))

        elif mode == "recurrent":
            states: List[Optional[object]] = [None] * self.n_layers
            if lengths is not None:
                t_lens = torch.div(lengths, 2, rounding_mode='floor').clamp(min=1).to(x.device)
            else:
                t_lens = torch.full((B,), Tprime, dtype=torch.long, device=x.device)
            run_sum = torch.zeros(B, D, device=x.device)
            denom = torch.zeros(B, 1, device=x.device)

            for t in range(Tprime):
                h_t = x[:, t, :]
                for li, blk in enumerate(self.blocks):
                    h_t, states[li] = blk.step_recurrent(h_t, t_idx=t, state=states[li])
                h_t = self.pre_classifier_norm(h_t)
                use_t = (t < t_lens).float().unsqueeze(1)
                run_sum = run_sum + h_t * use_t
                denom = denom + use_t

            pooled = run_sum / denom.clamp(min=1.0)
            return self.classifier(self.classifier_dropout(pooled))

        else:
            raise ValueError(f"Unknown mode: {mode}")

# ---------- Model configs (update paths to your checkpoints if needed) ----------
MODEL_CONFIGS = {
    'medium':  {'model_path': '/content/drive/MyDrive/kws_models/best_kws_retnet-small.pt',
                'd_model': 128, 'n_layers': 6, 'n_heads': 8, 'expected_classes': 36},
}

# ---------- Utilities & Bench code ----------
def load_model(model_size: str, device: torch.device) -> RetNetKWS:
    cfg = MODEL_CONFIGS[model_size]
    model = RetNetKWS(
        num_classes=cfg['expected_classes'],
        d_model=cfg['d_model'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        feature_dim=128
    ).to(device)
    model.eval()

    mp = Path(cfg['model_path'])
    if mp.exists():
        try:
            ckpt = torch.load(mp, map_location=device)
            if isinstance(ckpt, dict):
                if 'model_state_dict' in ckpt:
                    state_dict = ckpt['model_state_dict']
                elif 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights: {mp}")
        except Exception as e:
            print(f"Could not load weights for {model_size}: {e} (using random weights)")
    else:
        print(f"Model file not found for {model_size}: {mp}. Using random weights.")
    return model

def create_dummy_input(batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    T, F = 63, 128
    x = torch.randn(batch_size, T, F, device=device)
    lengths = torch.full((batch_size,), T, dtype=torch.long, device=device)
    return x, lengths

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _forward_once(model, x, lengths, mode: str):
    return model(x, lengths=lengths, mode=mode)

def benchmark_memory_usage(model: nn.Module, device: torch.device,
                           mode: str, batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]) -> Dict[str, Dict[str, float]]:
    if device.type != 'cuda':
        return {}
    model.eval()
    results = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated()

        x, l = create_dummy_input(bs, device)
        with torch.no_grad():
            _ = _forward_once(model, x, l, mode=mode)
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated()
        inference_memory = peak - baseline_memory
        activation_memory = peak - memory_before

        results[f"batch_{bs}"] = {
            'baseline_mb': baseline_memory / (1024**2),
            'peak_mb': peak / (1024**2),
            'inference_mb': inference_memory / (1024**2),
            'activation_mb': activation_memory / (1024**2),
            'memory_per_sample_mb': activation_memory / bs / (1024**2)
        }
        del x, l
        torch.cuda.empty_cache()

    return results

def benchmark_latency(model: nn.Module, device: torch.device, mode: str, num_runs: int = 1000) -> Dict[str, float]:
    model.eval()
    x, l = create_dummy_input(1, device)
    with torch.no_grad():
        for _ in range(10):
            _ = _forward_once(model, x, l, mode=mode)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            x, l = create_dummy_input(1, device)
            start = time.perf_counter()
            _ = _forward_once(model, x, l, mode=mode)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    return {
        'mean_ms': statistics.mean(times_ms),
        'median_ms': statistics.median(times_ms),
        'std_ms': statistics.stdev(times_ms),
        'min_ms': min(times_ms),
        'max_ms': max(times_ms),
        'p95_ms': float(np.percentile(times_ms, 95)),
        'p99_ms': float(np.percentile(times_ms, 99)),
    }

def benchmark_throughput(model: nn.Module, device: torch.device, mode: str,
                         batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]) -> Dict[int, Dict[str, float]]:
    model.eval()
    results = {}
    for bs in batch_sizes:
        print(f"[{mode}] Testing batch size {bs}...")
        x, l = create_dummy_input(bs, device)
        with torch.no_grad():
            for _ in range(5):
                _ = _forward_once(model, x, l, mode=mode)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        num_runs = max(10, 100 // bs)
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                x, l = create_dummy_input(bs, device)
                start = time.perf_counter()
                _ = _forward_once(model, x, l, mode=mode)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)

        avg_time = statistics.mean(times)
        results[bs] = {
            'avg_time_s': avg_time,
            'throughput_samples_per_s': bs / avg_time,
            'time_per_sample_ms': (avg_time / bs) * 1000.0
        }
    return results

def run_full_benchmark():
    print("Starting RetNet KWS Inference Benchmark (Parallel + Recurrent)")
    print(f"Device: {device} | using_retnet_lib={USE_RETNET_LIB}")
    print("=" * 64)

    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'cuda_available': HAS_CUDA,
        'using_retnet_lib': USE_RETNET_LIB,
        'models': {}
    }

    for model_size in ['medium']:
        print(f"\nBenchmarking {model_size.upper()} model...")
        model = load_model(model_size, device)
        params = count_parameters(model)
        cfg = MODEL_CONFIGS[model_size]
        print(f"Model parameters: {params:,}")

        mode_results = {}
        for mode in ['parallel', 'recurrent']:
            print(f"\nMode: {mode.upper()}")
            print("  Latency (1000 runs)...")
            lat = benchmark_latency(model, device, mode=mode)

            print("  Throughput across batch sizes...")
            thr = benchmark_throughput(model, device, mode=mode)

            mem = {}
            if device.type == 'cuda':
                print("  Memory usage across batch sizes...")
                mem = benchmark_memory_usage(model, device, mode=mode)

            mode_results[mode] = {'latency': lat, 'throughput': thr, 'memory': mem}

        results['models'][model_size] = {
            'parameters': params,
            'config': cfg,
            'parallel': mode_results['parallel'],
            'recurrent': mode_results['recurrent'],
        }

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"benchmark_results_retnet_bimode_{ts}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")
    return results

if __name__ == "__main__":
    _ = run_full_benchmark()
