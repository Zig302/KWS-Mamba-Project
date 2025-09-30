"""
Dataset wrapper(s) for KWS-Mamba using HuggingFace Speech Commands v0.02.

Exports:
- SpeechCommands: wraps an HF split (train/valid/test) -> (features[T, F], label)
- compute_dataset_stats: dataset-level mean/std over features
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .audio import WaveToSpec, Augment, pad_or_trim_waveform


class SpeechCommands(Dataset):
    """
    HuggingFace split -> (feature, label), with fixed-length waveform preprocess.

    Behavior:
      • Pads/crops raw waveform to `wav_len` (default: 16_000).
      • Applies optional waveform `Augment`.
      • Extracts features via `WaveToSpec`.
      • Normalizes features using dataset mean/std (pass 0/1 to disable).
      • Returns features as [T, F] (transpose from [1, F, T]) for sequence models.

    Args:
        hf_split: HuggingFace dataset split (e.g., ds["train"]).
        aug: optional waveform-level augmentation callable.
        frontend: WaveToSpec instance (train: masks on; eval: masks off).
        wav_len: target length (samples) for pad/crop.
        mean, std: dataset-wide normalization constants.
    """

    def __init__(
        self,
        hf_split,
        aug: Optional[Augment],
        frontend: WaveToSpec,
        wav_len: int = 16_000,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        self.ds = hf_split
        self.aug = aug
        self.front = frontend
        self.wav_len = wav_len
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        sample = self.ds[idx]
        wav = torch.from_numpy(sample["audio"]["array"]).float()  # [T]

        # Ensure fixed 1s audio (matches training code)
        wav = pad_or_trim_waveform(wav, self.wav_len)

        if self.aug is not None:
            wav = self.aug(wav)

        # Front-end features: [1, F, T] -> normalize -> [T, F]
        feats = self.front(wav)  # [1, F, T]
        feats = (feats - self.mean) / (self.std + 1e-6)
        feats = feats.squeeze(0).transpose(0, 1)  # [T, F]

        return feats, sample["label"]


@torch.no_grad()
def compute_dataset_stats(
    hf_split,
    frontend: WaveToSpec,
    wav_len: int = 16_000,
) -> Tuple[float, float]:
    """
    Compute dataset-level mean/std across all time frames and frequency bins.

    Args:
        hf_split: HF dataset split with {"audio": {"array"}, "label"} items.
        frontend: WaveToSpec instance (use eval version: masks off).
        wav_len: number of audio samples to pad/trim.

    Returns:
        (mean, std): floats over concatenated features.
    """
    feats_all = []
    for sample in hf_split:
        wav = torch.from_numpy(sample["audio"]["array"]).float()
        wav = pad_or_trim_waveform(wav, wav_len)
        x = frontend(wav).squeeze(0).transpose(0, 1)  # [T, F]
        feats_all.append(x)

    if not feats_all:
        return 0.0, 1.0

    stacked = torch.cat(feats_all, dim=0)  # [sum_T, F]
    return stacked.mean().item(), stacked.std().item()
