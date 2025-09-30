"""
Utilities for reproducibility and training setup

Contains functions for setting random seeds, AMP helpers, and other training utilities.
"""

import random
import torch
import numpy as np
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AMPHelper:
    """
    Automatic Mixed Precision helper class
    
    Simplifies usage of torch.amp for training with mixed precision.
    """
    
    def __init__(self, enabled: bool = True, dtype: torch.dtype = torch.bfloat16):
        self.enabled = enabled
        self.dtype = dtype
        
        if enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def autocast(self):
        """Get autocast context manager"""
        if self.enabled:
            return torch.cuda.amp.autocast(dtype=self.dtype)
        else:
            return torch.cuda.amp.autocast(enabled=False)
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass"""
        if self.enabled and self.scaler:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Step optimizer with gradient scaling"""
        if self.enabled and self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with optional scaling"""
        if self.enabled and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
        'total_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    total_steps: int = 1000,
    warmup_steps: int = 100,
    min_lr_ratio: float = 0.01,
    **kwargs
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('cosine', 'linear', 'exponential')
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum learning rate as ratio of base LR
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler
    """
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        
        if scheduler_type == 'cosine':
            # Cosine annealing
            return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
        elif scheduler_type == 'linear':
            # Linear decay
            return max(min_lr_ratio, 1.0 - progress)
        else:
            # Constant after warmup
            return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    metrics: dict,
    filepath: str,
    **kwargs
) -> None:
    """
    Save training checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        **kwargs
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: str = 'cpu'
) -> dict:
    """
    Load training checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load tensors to
        
    Returns:
        Dictionary with checkpoint data
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint