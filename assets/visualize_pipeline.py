# examples/visualize_pipeline.py
#!/usr/bin/env python3
"""
Visualize the full audio→features pipeline for Google Speech Commands:
- raw waveform & fixed-length waveform
- mel filter banks and frame windows
- mel spectrogram (linear & dB)
- MFCC

Saves: assets/audio_pipeline_visualization.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Pipeline module
# ---------------------------------------------------------------------
class AudioProcessingPipeline(torch.nn.Module):
    """PyTorch Module-based audio processing pipeline."""

    def __init__(
        self,
        target_length: float = 1.0,
        sample_rate: int = 16_000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 256,
        normalize: Tuple[float, float] | None = None,
        output_mfcc: bool = False,
        n_mfcc: int = 40,
    ):
        super().__init__()
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.output_mfcc = output_mfcc

        self.mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.amp_to_db = AmplitudeToDB()

        self.mfcc_transform = (
            torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "n_mels": n_mels,
                },
            )
            if output_mfcc
            else None
        )

    def fix_length(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Pad or truncate to target_length seconds."""
        length = int(self.target_length * sample_rate)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        n = waveform.size(-1)
        if n < length:
            return torch.nn.functional.pad(waveform, (0, length - n))
        if n > length:
            return waveform[..., :length]
        return waveform

    def forward(self, data: dict):
        """
        Args:
            data: {'samples', 'sample_rate', 'label'}
        Returns:
            (features, label)
        """
        samples = data["samples"]
        sample_rate = data["sample_rate"]

        waveform = (
            torch.tensor(samples, dtype=torch.float32)
            if not isinstance(samples, torch.Tensor)
            else samples
        )
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = self.fix_length(waveform, sample_rate)

        if self.output_mfcc:
            feats = self.mfcc_transform(waveform)
        else:
            mel = self.mel_spec(waveform)
            feats = self.amp_to_db(mel)

        if self.normalize is not None:
            mean, std = self.normalize
            feats = (feats - mean) / (std + 1e-6)

        return feats, data["label"]


# ---------------------------------------------------------------------
# HF dataset wrapper
# ---------------------------------------------------------------------
class HuggingFaceSpeechCommandsDataset(Dataset):
    """Google Speech Commands dataset wrapper for HuggingFace dataset."""

    def __init__(self, hf_dataset, processor: AudioProcessingPipeline | None = None):
        self.dataset = hf_dataset
        self.processor = processor
        self.label_names = self.dataset.features["label"].names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        data = {
            "samples": np.array(item["audio"]["array"]).astype(np.float32),
            "sample_rate": item["audio"]["sampling_rate"],
            "label": item["label"],
        }
        if self.processor:
            features, label = self.processor(data)
            return features, label
        return data


def load_speech_commands(
    batch_size: int = 4,
    num_workers: int = 2,
    output_mfcc: bool = False,
    n_mfcc: int = 40,
):
    """Load Google Speech Commands via HF and create dataloaders."""
    hf = load_dataset("google/speech_commands", "v0.02")

    processor = AudioProcessingPipeline(
        target_length=1.0,
        sample_rate=16_000,
        n_mels=128,
        n_fft=2048,
        hop_length=256,
        output_mfcc=output_mfcc,
        n_mfcc=n_mfcc,
    )

    train_ds = HuggingFaceSpeechCommandsDataset(hf["train"], processor=processor)
    val_ds = HuggingFaceSpeechCommandsDataset(hf["validation"], processor=processor)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    num_classes = len(hf["train"].features["label"].names)
    return train_loader, val_loader, num_classes


# ---------------------------------------------------------------------
# Visualization 
# ---------------------------------------------------------------------
def visualize_pipeline(processor: AudioProcessingPipeline, data: dict, max_frames_to_show: int | None = 32):
    """
    Draws: waveform (raw & fixed), mel filters, frame windows,
           mel spectrogram (linear & dB), MFCC.
    """
    # Prepare waveforms
    samples = data["samples"]
    sample_rate = data["sample_rate"]

    waveform = torch.tensor(samples, dtype=torch.float32) if not isinstance(samples, torch.Tensor) else samples
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]
    fixed_waveform = processor.fix_length(waveform, sample_rate)

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(4, 2)

    # 1) Original waveform
    ax1 = fig.add_subplot(gs[0, 0])
    t_raw = np.arange(waveform.size(-1)) / sample_rate
    ax1.plot(t_raw, waveform[0].numpy())
    ax1.set_title("Original Waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")

    # 2) Fixed-length waveform
    ax2 = fig.add_subplot(gs[0, 1])
    t_fix = np.arange(fixed_waveform.size(-1)) / sample_rate
    ax2.plot(t_fix, fixed_waveform[0].numpy())
    ax2.set_title(f"Fixed Length Waveform ({processor.target_length}s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")

    # 3) Mel filter banks (schematic)
    ax3 = fig.add_subplot(gs[1, 0])
    n_mels = processor.mel_spec.n_mels
    sr = processor.sample_rate

    def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)

    f_min, f_max = 0.0, sr / 2
    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    freq_bins = np.linspace(0, sr / 2, 1000)

    for i in range(n_mels):
        f_l, f_c, f_r = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        resp = np.zeros_like(freq_bins)
        mask = (freq_bins >= f_l) & (freq_bins < f_c)
        resp[mask] = (freq_bins[mask] - f_l) / (f_c - f_l)
        mask = (freq_bins >= f_c) & (freq_bins < f_r)
        resp[mask] = (f_r - freq_bins[mask]) / (f_r - f_c)
        ax3.plot(freq_bins, resp, linewidth=1.0)

    ax3.set_title(f"Mel Filter Banks ({n_mels} filters)")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Response")
    ax3.set_xlim(0, sr / 2)

    # 4) Time frames overlay
    ax4 = fig.add_subplot(gs[1, 1])
    hop = processor.mel_spec.hop_length
    n_fft = processor.mel_spec.n_fft
    n_frames = max(0, (fixed_waveform.size(1) - n_fft) // hop + 1)
    ax4.plot(fixed_waveform[0].numpy(), alpha=0.6)

    frame_indices = np.arange(0, n_frames) * hop
    use_indices = frame_indices if max_frames_to_show is None else frame_indices[: max(0, min(len(frame_indices), max_frames_to_show))]
    for idx in use_indices:
        rect = plt.Rectangle((idx, -1.0), n_fft, 2.0, alpha=0.15)
        ax4.add_patch(rect)

    ax4.set_title(f"Time Frames (hop={hop}, n_fft={n_fft}) — total frames: {n_frames}")
    ax4.set_xlabel("Sample Index")
    ax4.set_ylabel("Amplitude")
    ax4.set_ylim(-1.1, 1.1)

    # 5) Mel spectrogram (linear)
    mel_spec = processor.mel_spec(fixed_waveform)  # [1, n_mels, T]
    ax5 = fig.add_subplot(gs[2, 0])
    im5 = ax5.imshow(mel_spec[0].numpy(), aspect="auto", origin="lower", interpolation="none")
    ax5.set_title(f"Mel Spectrogram ({n_mels} mels)")
    ax5.set_xlabel("Time Frame")
    ax5.set_ylabel("Mel Bin")
    fig.colorbar(im5, ax=ax5, label="Energy")

    # 6) Mel spectrogram (dB)
    mel_spec_db = processor.amp_to_db(mel_spec)
    ax6 = fig.add_subplot(gs[2, 1])
    im6 = ax6.imshow(mel_spec_db[0].numpy(), aspect="auto", origin="lower", interpolation="none")
    ax6.set_title("Mel Spectrogram (dB)")
    ax6.set_xlabel("Time Frame")
    ax6.set_ylabel("Mel Bin")
    fig.colorbar(im6, ax=ax6, label="dB")

    # 7) MFCC (linear)
    if hasattr(processor, "mfcc_transform") and processor.mfcc_transform is not None:
        mfcc_tx = processor.mfcc_transform
    else:
        mfcc_tx = torchaudio.transforms.MFCC(
            sample_rate=processor.sample_rate,
            n_mfcc=40,
            melkwargs={
                "n_fft": processor.mel_spec.n_fft,
                "hop_length": processor.mel_spec.hop_length,
                "n_mels": processor.mel_spec.n_mels,
            },
        )
    mfcc = mfcc_tx(fixed_waveform)  # [1, n_mfcc, T]
    ax7 = fig.add_subplot(gs[3, 0])
    im7 = ax7.imshow(mfcc[0].numpy(), aspect="auto", origin="lower", interpolation="none")
    ax7.set_title(f"MFCC (n_mfcc={mfcc.size(1)})")
    ax7.set_xlabel("Time Frame")
    ax7.set_ylabel("MFCC Coef")
    fig.colorbar(im7, ax=ax7, label="Coefficient")

    plt.tight_layout()
    return fig


def main():
    # Load a batch
    train_loader, val_loader, num_classes = load_speech_commands(batch_size=4)
    print(f"Dataset loaded with {num_classes} classes")
    print(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
    for inputs, labels in train_loader:
        print(f"Input batch shape: {inputs.shape} | Labels shape: {labels.shape}")
        break

    # Visualize pipeline on a single "yes" sample
    hf = load_dataset("google/speech_commands", "v0.02")
    processor = AudioProcessingPipeline(
        target_length=1.0, sample_rate=16_000, n_mels=128, n_fft=2048, hop_length=256
    )

    sample = None
    for item in hf["train"]:
        if item["label"] == "yes":
            sample = item
            break
    if sample is None:
        sample = hf["train"][0]

    data = {
        "samples": np.array(sample["audio"]["array"]).astype(np.float32),
        "sample_rate": sample["audio"]["sampling_rate"],
        "label": sample["label"],
    }

    fig = visualize_pipeline(processor, data)
    Path("assets").mkdir(parents=True, exist_ok=True)
    out_path = Path("assets/audio_pipeline_visualization.png")
    fig.savefig(out_path, dpi=150)
    print(f"Visualization saved to '{out_path}'")

    # MFCC variant quick check
    train_loader_mfcc, _, _ = load_speech_commands(output_mfcc=True)
    for inputs, labels in train_loader_mfcc:
        print(f"With MFCC, input batch shape: {inputs.shape}")
        break


if __name__ == "__main__":
    main()
