# src/kws_mamba/train.py
from __future__ import annotations

import math
from pathlib import Path
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from kws_mamba.audio import Augment, WaveToSpec
from kws_mamba.dataset import SpeechCommands, collate_fn, compute_dataset_stats
from kws_mamba.models.mamba_mel import build_medium as build_mamba_mel_medium
from kws_mamba.models.mamba_mel import build_small as build_mamba_mel_small, build_large as build_mamba_mel_large
from kws_mamba.models.mamba_mfcc import build_small as build_mamba_mfcc_small, build_medium as build_mamba_mfcc_medium, build_large as build_mamba_mfcc_large
from kws_mamba.models.retnet import RetNetKWS
from kws_mamba.models.cnn import build_mobilenet_v2

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Tuple[float, float]:
    """Length-aware evaluation."""
    model.eval()
    tot = correct = loss_sum = 0
    for xb, yb, lb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        lb = lb.to(device)
        logits = model(xb, lengths=lb)
        loss = criterion(logits, yb)
        loss_sum += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        tot += xb.size(0)
    return loss_sum / tot, 100.0 * correct / tot

# -------------------------
# Train (Medium Mel-Mamba)
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--base_lr", type=float, default=5e-4)
    parser.add_argument("--warmup_frac", type=float, default=0.12)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")  # local, instead of Colab/Drive
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    # ---- dataset
    ds = load_dataset("google/speech_commands", "v0.02")
    n_classes = len(ds["train"].features["label"].names)

    # log-mel + SpecAugment (train only) — Medium variant settings
    feature_type = "mel"  # "mel"/"mfcc"
    frontend_train = WaveToSpec(
        feature_type=feature_type,
        n_mfcc=40, n_mels=128,
        apply_mask=True,          # SpecAugment on train
        freq_mask_param=12,
        time_mask_param=20,
    )
    frontend_eval = WaveToSpec(
        feature_type=feature_type,
        n_mfcc=40, n_mels=128,
        apply_mask=False
    )
    frontend_stats = WaveToSpec(
        feature_type=feature_type,
        n_mfcc=40, n_mels=128,
        apply_mask=False
    )

    # Waveform augs (Medium): shift=120 ms, light noise
    aug = Augment(shift_ms=120, noise=(0., 0.005))

    # Normalization stats from train split using eval frontend (no masks)
    train_mean, train_std = compute_dataset_stats(ds["train"], frontend_stats)

    # Datasets
    train_ds = SpeechCommands(ds["train"], aug,  frontend_train, mean=train_mean, std=train_std)
    val_ds   = SpeechCommands(ds["validation"], None, frontend_eval,  mean=train_mean, std=train_std)
    test_ds  = SpeechCommands(ds["test"], None, frontend_eval,  mean=train_mean, std=train_std)

    # Loaders
    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_dl   = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    # Model (Mel-Mamba Medium: d_model=128, n_layers=10, d_state=16)
    model = build_mamba_mel_medium(num_classes=n_classes).to(device)

    # Loss/opt/sched (per-batch schedule; linear warmup → cosine with floor 0.005)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.07)
    opt = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1.8e-4, betas=(0.9, 0.999))

    steps_per_epoch = len(train_dl)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = int(total_steps * args.warmup_frac)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.005, 0.5 * (1.0 + math.cos(math.pi * progress)))  # one-cycle style

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Checkpoints (local)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    BEST_PATH = save_dir / "best_kws.pt"
    LAST_PATH = save_dir / "last_kws.pt"

    best_val_acc = 0.0
    prev_val_acc = 0.0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch:02d}")
        for xb, yb, lb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=use_amp and torch.cuda.is_bf16_supported()):
                if torch.isnan(xb).any():
                    xb = torch.nan_to_num(xb, nan=0.0)
                logits = model(xb, lengths=lb)
                loss = criterion(logits, yb)
                if not torch.isfinite(loss):
                    continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            scaler.step(opt)
            scaler.update()
            sched.step()  # per-batch
            global_step += 1

            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            running_loss += loss.item() * yb.size(0)

            pbar.set_postfix(
                train_loss=f"{running_loss / max(1,total):.3f}",
                train_acc=f"{100.0 * correct / max(1,total):.2f}%",
                lr=f"{opt.param_groups[0]['lr']:.2e}"
            )

        tr_acc = 100.0 * correct / max(1, total)
        val_loss, val_acc = evaluate(model, val_dl, device, criterion)
        print(f"Epoch {epoch:02d} ➜ train {tr_acc:.2f}% | val {val_acc:.2f}% (loss {val_loss:.3f}) | lr {opt.param_groups[0]['lr']:.2e}")

        # Collapse guard: big drop → restore best + shrink LR ×5
        if epoch > 1 and prev_val_acc > 50.0 and val_acc < 0.5 * prev_val_acc:
            print(f"WARNING: accuracy collapse ({prev_val_acc:.2f}% → {val_acc:.2f}%). Restoring best and reducing LR ×5.")
            if BEST_PATH.exists():
                model.load_state_dict(torch.load(BEST_PATH, map_location=device))
            for g in opt.param_groups:
                g['lr'] = max(g['lr'] / 5.0, 1e-6)
            print(f"New LR: {opt.param_groups[0]['lr']:.2e}")

        # Best-by-accuracy checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_PATH)
            print(f"** Saved new best model params ** @ {best_val_acc:.2f}%")

        prev_val_acc = val_acc

    # Save LAST
    torch.save(model.state_dict(), LAST_PATH)
    print(f"Saved LAST model to {LAST_PATH}")

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------
# Other training for different models/setups
# -------------------------------------------------------------------

# # Mamba-Mel Small
# model = build_mamba_mel_small(num_classes=n_classes).to(device)
# criterion = nn.CrossEntropyLoss(label_smoothing=0.07)
# opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1.8e-4, betas=(0.9,0.999))
# # SpecAugment: freq_mask=12, time_mask=20 (or 14/24 in some runs); Augment(shift_ms=100, noise=(0., 0.005))
# # Other parts identical (epochs=100, warmup_frac=0.12, sched per-batch, grad_clip=0.3, collapse guard)

# # Mamba-Mel Large
# model = build_mamba_mel_large(num_classes=n_classes).to(device)
# criterion = nn.CrossEntropyLoss(label_smoothing=0.07)
# opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4, betas=(0.9,0.999))
# # SpecAugment: freq_mask=15, time_mask=25; Augment(shift_ms=100, noise=(0., 0.01))
# # All other training mechanics unchanged (per-batch warmup+cosine with floor 0.005, etc.)

# # Mamba-MFCC (Small/Medium/Large)
# # Swap frontend feature_type="mfcc" and use build_mamba_mfcc_* variants.
# # Weight decay: 1.8e-4 for S/M, 5e-4 for L (LR, epochs, warmup same).

# # RetNet
# # from kws_mamba.models.retnet import RetNetKWS
# # model = RetNetKWS(num_classes=n_classes, d_model=128, n_layers=8, d_state=16).to(device)
# # criterion = nn.CrossEntropyLoss(label_smoothing=0.07); opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1.8e-4)

# # CNN (MobileNetV2 baseline)
# # from kws_mamba.models.cnn import build_mobilenet_v2
# # model = build_mobilenet_v2(n_classes=n_classes, alpha=0.75, pretrained=False).to(device)
# # criterion = nn.CrossEntropyLoss(label_smoothing=0.05); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-5)
