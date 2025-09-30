#!/usr/bin/env python3
"""
Training Script for Mamba-based Keyword Spotting

This script handles the complete training pipeline including data loading,
model initialization, training loop, and checkpointing.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mamba_kws.models import MambaKWS
from mamba_kws.data import SpeechCommands, WaveToSpec, Augment, collate_fn


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Mamba-KWS model")
    
    # Model arguments
    parser.add_argument("--model-size", choices=["small", "medium", "large"], 
                       default="medium", help="Model size variant")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=10, help="Number of layers")
    parser.add_argument("--d-state", type=int, default=16, help="State dimension")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1.8e-4, help="Weight decay")
    parser.add_argument("--warmup-frac", type=float, default=0.12, help="Warmup fraction")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, help="Path to dataset")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./outputs", 
                       help="Output directory")
    parser.add_argument("--experiment-name", type=str, default="mamba_kws",
                       help="Experiment name")
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(args) -> nn.Module:
    """Create model based on arguments"""
    
    # Model size presets
    size_configs = {
        "small": {"d_model": 64, "n_layers": 8},
        "medium": {"d_model": 128, "n_layers": 10}, 
        "large": {"d_model": 192, "n_layers": 12},
    }
    
    if args.model_size in size_configs:
        config = size_configs[args.model_size]
        d_model = config["d_model"]
        n_layers = config["n_layers"]
    else:
        d_model = args.d_model
        n_layers = args.n_layers
    
    model = MambaKWS(
        n_classes=35,  # Google Speech Commands V2-35
        d_model=d_model,
        n_layers=n_layers,
        d_state=args.d_state,
    )
    
    return model


def create_dataloaders(args):
    """Create training and validation dataloaders"""
    
    # TODO: Paste your data loading code here
    # This is a placeholder - replace with your actual data loading
    
    # Frontend preprocessing
    frontend_train = WaveToSpec(n_mels=40, apply_mask=True)
    frontend_eval = WaveToSpec(n_mels=40, apply_mask=False)
    
    # Augmentation
    augment = Augment()
    
    # TODO: Load actual dataset
    # For now, create dummy datasets
    train_dataset = None  # Replace with actual SpeechCommands dataset
    val_dataset = None    # Replace with actual SpeechCommands dataset
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    ) if train_dataset else None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    ) if val_dataset else None
    
    return train_loader, val_loader


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    
    # TODO: Paste your training loop implementation here
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets, lengths) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device) 
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
        optimizer.step()
        scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    
    # TODO: Paste your validation implementation here
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy


def main():
    """Main training function"""
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Update args with config values
        for key, value in config.items():
            setattr(args, key.replace('-', '_'), value)
    
    # Setup output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(args)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)
    
    if train_loader is None:
        print("ERROR: No training data loaded. Please implement data loading!")
        return
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=0.07)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_frac * total_steps)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.005, 0.5 * (1.0 + torch.cos(torch.pi * torch.tensor(progress))))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Setup tensorboard
    writer = SummaryWriter(output_dir / "tensorboard")
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        
        # Validate
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = 0.0, 0.0
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"New best model saved! Accuracy: {best_acc:.2f}%")
    
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")
    writer.close()


if __name__ == "__main__":
    main()