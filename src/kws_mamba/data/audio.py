"""
Audio utilities for KWS-Mamba:
- WaveToSpec: waveform -> Mel (dB) or MFCC features with optional SpecAugment
- Augment: waveform-level time shift / tempo stretch / Gaussian noise
- Collate helpers: sequence-style (B, T, F) and image-style (B, 1, F, T)

This module mirrors the front-ends and augmentations used in the original notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Sequence

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


# ---------------------------
# Waveform-level augmentation
# ---------------------------
@dataclass
class Augment:
    """
    Waveform augmentation with random time-shift, optional tempo stretch (sox),
    and Gaussian noise.

    Args:
        stretch: (min, max) tempo factor. Use (1.0, 1.0) to disable.
        shift_ms: +/- max shift in milliseconds.
        noise: (min_sigma, max_sigma) white noise std range added to waveform.
        sr: sample rate.
    """
    stretch: Tuple[float, float] = (1.0, 1.0)
    shift_ms: int = 100
    noise: Tuple[float, float] = (0.0, 0.05)
    sr: int = 16_000

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        if self.shift_ms <= 0:
            return x
        max_shift = int(self.shift_ms * self.sr / 1000)
        if max_shift == 0:
            return x
        s = int(torch.randint(-max_shift, max_shift + 1, ()).item())
        if s == 0:
            return x
        return (F.pad(x, (s, 0))[:, :-s] if s > 0 else F.pad(x, (0, -s))[:, -s:])

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """Accepts [T] or [1, T]; returns same shape."""
        squeezed = False
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            squeezed = True

        # Optional tempo stretch (sox "tempo")
        if self.stretch != (1.0, 1.0):
            factor = float(torch.empty(()).uniform_(*self.stretch))
            if abs(factor - 1.0) > 1e-3:
                wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                    wav, self.sr, [["tempo", f"{factor:.6f}"]]
                )

        # Random circular-like shift via pad+crop
        wav = self._shift(wav)

        # Additive white noise
        if self.noise[1] > 0:
            sigma = float(torch.empty(()).uniform_(*self.noise))
            if sigma > 0:
                wav = wav + sigma * torch.randn_like(wav)

        return wav.squeeze(0) if squeezed else wav


# ---------------------------
# Waveform -> Spectrogram/MFCC
# ---------------------------
class WaveToSpec:
    """
    Waveform -> log-Mel **or** MFCC features with optional SpecAugment.

    Args:
        feature_type: "mel" or "mfcc"
        sample_rate, n_fft, hop_length, n_mels, n_mfcc: frontend params
        top_db: dynamic range for AmplitudeToDB (Mel only)
        apply_mask: enable SpecAugment masks
        freq_mask_param, time_mask_param: mask extents
        mask_on_mfcc: also apply masks on MFCC features (common in your MFCC runs)
    """

    def __init__(
        self,
        feature_type: str = "mel",
        sample_rate: int = 16_000,
        n_fft: int = 2048,
        hop_length: int = 256,
        n_mels: int = 128,
        n_mfcc: int = 40,
        top_db: Optional[int] = 80,
        apply_mask: bool = True,
        freq_mask_param: int = 15,
        time_mask_param: int = 10,
        mask_on_mfcc: bool = True,
    ):
        ft = feature_type.lower()
        assert ft in {"mel", "mfcc"}, "feature_type must be 'mel' or 'mfcc'"
        self.feature_type = ft
        self.apply_mask = apply_mask
        self.mask_on_mfcc = mask_on_mfcc

        if ft == "mel":
            self.spec = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
            )
            self.to_db = T.AmplitudeToDB(stype="power", top_db=top_db)
            # Masks used only if apply_mask is True
            self.freq_mask = T.FrequencyMasking(freq_mask_param) if apply_mask else None
            self.time_mask = T.TimeMasking(time_mask_param) if apply_mask else None
            self.n_out = n_mels
        else:
            self.spec = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs=dict(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels),
            )
            self.to_db = None
            self.freq_mask = T.FrequencyMasking(freq_mask_param) if (apply_mask and mask_on_mfcc) else None
            self.time_mask = T.TimeMasking(time_mask_param) if (apply_mask and mask_on_mfcc) else None
            self.n_out = n_mfcc

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: [T] or [1, T] float tensor
        Returns:
            [1, F, T_frames] tensor
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        feats = self.spec(wav)  # Mel: [1, M, T]; MFCC: [1, C, T]

        if self.feature_type == "mel":
            # Convert to log-dB, then optionally apply masks
            feats = self.to_db(feats.clamp(min=1e-10))
            if self.apply_mask and self.freq_mask is not None:
                feats = self.freq_mask(feats)
                feats = self.time_mask(feats)
        else:
            # MFCC: optionally apply masks too (some of your runs do this)
            if self.apply_mask and self.freq_mask is not None:
                feats = self.freq_mask(feats)
                feats = self.freq_mask(feats)
                feats = self.time_mask(feats)
                feats = self.time_mask(feats)

        return feats  # [1, F, T]


# ---------------------------
# Small utilities
# ---------------------------
def pad_or_trim_waveform(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Pad with zeros or trim the waveform to exactly `target_len` samples.

    Accepts [T] or [1, T]; returns same rank as input.
    """
    squeezed = False
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
        squeezed = True

    n = wav.size(-1)
    if n < target_len:
        wav = F.pad(wav, (0, target_len - n))
    elif n > target_len:
        wav = wav[..., :target_len]

    return wav.squeeze(0) if squeezed else wav


# ---------------------------
# Collate helpers
# ---------------------------
def collate_seq(batch: Sequence[tuple[torch.Tensor, int]]):
    """
    For sequence models (Mamba/RetNet): pads along time.
    Input items: (feats[T, F], label)
    Returns:
        feats_padded: (B, T_max, F)
        labels: (B,)
        lengths: (B,)
    """
    feats, labels = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in feats], dtype=torch.long)
    # pad to (B, T_max, F)
    Fdim = feats[0].size(1)
    T_max = max(x.size(0) for x in feats)
    out = feats[0].new_zeros(len(feats), T_max, Fdim)
    for i, x in enumerate(feats):
        out[i, : x.size(0)] = x
    return out, torch.tensor(labels, dtype=torch.long), lengths


def collate_image(batch: Sequence[tuple[torch.Tensor, int]]):
    """
    For CNN-like models: outputs image-style tensors.
    Input items: (feats[T, F], label)  OR (feats[1, F, T], label)
    Returns:
        feats_padded: (B, 1, F, T_max)
        labels: (B,)
    """
    feats, labels = zip(*batch)
    # Convert [T, F] -> [1, F, T]
    feats = [x.unsqueeze(0).transpose(1, 2) if x.dim() == 2 else x for x in feats]
    B, Fbins = len(feats), feats[0].size(1)
    T_max = max(x.size(-1) for x in feats)
    out = feats[0].new_zeros(B, 1, Fbins, T_max)
    for i, x in enumerate(feats):
        Tlen = x.size(-1)
        out[i, :, :, :Tlen] = x
    return out, torch.tensor(labels, dtype=torch.long)
