"""
Utility functions for metrics calculation

Contains helpers for accuracy, F1 score, confusion matrix, and per-class metrics.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate classification accuracy
    
    Args:
        predictions: Model predictions (B, n_classes) or (B,)
        targets: Ground truth labels (B,)
        
    Returns:
        accuracy: Accuracy as float between 0 and 1
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    
    return correct / total


def calculate_f1_scores(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate F1 scores
    
    Args:
        predictions: Model predictions (B, n_classes) or (B,)
        targets: Ground truth labels (B,)
        average: Averaging strategy ('micro', 'macro', 'weighted', None)
        
    Returns:
        Dictionary with F1 scores
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    results = {}
    
    if average in ['micro', 'macro', 'weighted']:
        results[f'f1_{average}'] = f1_score(targets, predictions, average=average)
    
    # Per-class F1 scores
    if average is None or 'per_class' in average:
        per_class_f1 = f1_score(targets, predictions, average=None)
        results['f1_per_class'] = per_class_f1.tolist()
    
    return results


def calculate_confusion_matrix(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Calculate confusion matrix
    
    Args:
        predictions: Model predictions (B, n_classes) or (B,)
        targets: Ground truth labels (B,)
        class_names: Optional list of class names
        
    Returns:
        Dictionary with confusion matrix and statistics
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    cm = confusion_matrix(targets, predictions)
    
    results = {
        'confusion_matrix': cm.tolist(),
        'accuracy': accuracy_score(targets, predictions)
    }
    
    if class_names:
        results['class_names'] = class_names
    
    return results


def get_classification_report(
    predictions: torch.Tensor,
    targets: torch.Tensor, 
    class_names: Optional[List[str]] = None
) -> str:
    """
    Generate detailed classification report
    
    Args:
        predictions: Model predictions (B, n_classes) or (B,)
        targets: Ground truth labels (B,)
        class_names: Optional list of class names
        
    Returns:
        Classification report as string
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    return classification_report(
        targets, 
        predictions, 
        target_names=class_names,
        digits=4
    )


def calculate_per_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int
) -> Dict[int, float]:
    """
    Calculate per-class accuracy
    
    Args:
        predictions: Model predictions (B, n_classes) or (B,)
        targets: Ground truth labels (B,)
        n_classes: Number of classes
        
    Returns:
        Dictionary mapping class index to accuracy
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    per_class_acc = {}
    
    for class_idx in range(n_classes):
        class_mask = targets == class_idx
        if class_mask.sum() > 0:
            class_predictions = predictions[class_mask]
            class_targets = targets[class_mask]
            per_class_acc[class_idx] = calculate_accuracy(class_predictions, class_targets)
        else:
            per_class_acc[class_idx] = 0.0
    
    return per_class_acc


def top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy
    
    Args:
        predictions: Model predictions (B, n_classes)
        targets: Ground truth labels (B,)
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy as float
    """
    if predictions.dim() == 1:
        # If predictions are already argmax'd, can't compute top-k
        return calculate_accuracy(predictions, targets)
    
    # Get top-k predictions
    _, top_k_preds = predictions.topk(k, dim=1)
    
    # Check if target is in top-k predictions
    targets_expanded = targets.unsqueeze(1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=1).sum().item()
    
    return correct / targets.size(0)