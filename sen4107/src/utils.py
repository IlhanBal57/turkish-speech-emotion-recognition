"""
Utility Functions for Turkish Speech Emotion Recognition

This module provides various helper functions for:
- Loading configurations
- Plotting training curves
- Creating confusion matrices
- Saving/loading checkpoints
- Setting random seeds for reproducibility

Author: SEN4107 Course Project
"""

import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For even more reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = 'cuda') -> torch.device:
    """
    Get PyTorch device for computation.
    
    Args:
        device_name: 'cuda', 'cpu', or 'mps' (for Apple Silicon)
        
    Returns:
        PyTorch device object
    """
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif device_name == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    save_path: str,
    is_best: bool = False
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        val_accuracy: Validation accuracy
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
    }
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")
    
    # If best model, save separately
    if is_best:
        best_path = os.path.join(
            os.path.dirname(save_path),
            'best_model.pth'
        )
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        optimizer: PyTorch optimizer (optional)
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint onto
        
    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Val Accuracy: {checkpoint['val_accuracy']:.2f}%")
    
    return checkpoint


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str
):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalize: bool = False,
    title: str = 'Confusion Matrix'
):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix as numpy array
        class_names: List of class names for axes
        save_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
        title: Plot title
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def save_metrics(
    metrics: Dict[str, Any],
    save_path: str
):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save JSON file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    print(f"Metrics saved: {save_path}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """
    Print summary of model architecture and parameters.
    
    Args:
        model: PyTorch model
        model_name: Name of the model for display
    """
    print("\n" + "=" * 70)
    print(f"{model_name} Summary")
    print("=" * 70)
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Calculate model size in MB
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    print(f"Model size: {size_mb:.2f} MB")
    
    print("=" * 70 + "\n")


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics.
        
        Args:
            val: New value
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric value
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


if __name__ == "__main__":
    """
    Demo: Utility functions
    """
    print("Utility Functions - Demo")
    print("=" * 70)
    
    print("\n1. CONFIGURATION LOADING")
    print("-" * 70)
    print("Example: config = load_config('config/baseline_config.yaml')")
    
    print("\n2. REPRODUCIBILITY")
    print("-" * 70)
    print("Example: set_seed(42)")
    set_seed(42)
    print("Random seed set for reproducibility!")
    
    print("\n3. DEVICE SELECTION")
    print("-" * 70)
    device = get_device('cuda')
    
    print("\n4. EARLY STOPPING")
    print("-" * 70)
    early_stop = EarlyStopping(patience=5, mode='max')
    print("Early stopping initialized with patience=5")
    
    print("\n5. METRICS TRACKING")
    print("-" * 70)
    meter = AverageMeter()
    meter.update(0.85)
    meter.update(0.90)
    print(f"Average metric: {meter.avg:.4f}")
    
    print("\n" + "=" * 70)
    print("Utilities ready for use!")
