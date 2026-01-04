"""
Training Script for Turkish Speech Emotion Recognition

This script handles the complete training pipeline:
1. Load configuration and dataset
2. Initialize model, optimizer, and loss function
3. Train model with validation
4. Save checkpoints and metrics
5. Generate training curves

Usage:
    python src/train.py --model baseline
    python src/train.py --model comparison --config config/comparison_config.yaml

Author: SEN4107 Course Project
"""

import os
import sys
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from models.baseline_model import create_baseline_cnn
from models.comparison_model import create_cnn_bilstm
from datasets import load_turkish_ser_dataset, split_dataset, create_data_loaders
from utils import (
    load_config, set_seed, get_device, save_checkpoint,
    plot_training_curves, AverageMeter, EarlyStopping,
    print_model_summary
)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> tuple:
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()
    
    # Metrics trackers
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    print(f"\nEpoch {epoch} - Training")
    print("-" * 70)
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        # Move data to device
        features = features.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, probs = model(features)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(probs, dim=1)
        accuracy = (predictions == labels).float().mean() * 100
        
        # Update meters
        loss_meter.update(loss.item(), features.size(0))
        acc_meter.update(accuracy.item(), features.size(0))
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] | "
                  f"Loss: {loss_meter.avg:.4f} | Acc: {acc_meter.avg:.2f}%")
    
    print(f"\nEpoch {epoch} Training Summary:")
    print(f"  Average Loss: {loss_meter.avg:.4f}")
    print(f"  Average Accuracy: {acc_meter.avg:.2f}%")
    
    return loss_meter.avg, acc_meter.avg


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple:
    """
    Validate model on validation set.
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, average_accuracy, macro_f1)
    """
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    print(f"\nEpoch {epoch} - Validation")
    print("-" * 70)
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, probs = model(features)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            predictions = torch.argmax(probs, dim=1)
            accuracy = (predictions == labels).float().mean() * 100
            
            # Update meters
            loss_meter.update(loss.item(), features.size(0))
            acc_meter.update(accuracy.item(), features.size(0))
            
            # Store predictions for F1 score
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate macro F1 score
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    
    print(f"\nEpoch {epoch} Validation Summary:")
    print(f"  Average Loss: {loss_meter.avg:.4f}")
    print(f"  Average Accuracy: {acc_meter.avg:.2f}%")
    print(f"  Macro F1 Score: {macro_f1:.2f}%")
    
    return loss_meter.avg, acc_meter.avg, macro_f1


def train_model(config: dict, model_type: str):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
        model_type: 'baseline' or 'comparison'
    """
    print("\n" + "=" * 70)
    print(f"Turkish Speech Emotion Recognition - {model_type.upper()} Model Training")
    print("=" * 70)
    
    # Set random seed for reproducibility
    set_seed(config['seed'])
    
    # Get device
    device = get_device(config['hardware']['device'])
    
    # ========================================================================
    # LOAD DATASET
    # ========================================================================
    print("\n1. Loading Dataset...")
    print("-" * 70)
    
    # Load audio paths and labels
    audio_paths, labels = load_turkish_ser_dataset(
        config['data']['data_dir'],
        config['data'].get('emotion_map')
    )
    
    if len(audio_paths) == 0:
        print("ERROR: No data loaded. Please provide a Turkish SER dataset.")
        print("Using dummy data for demonstration purposes...")
        # Create dummy data for testing
        audio_paths = [f"dummy_{i}.wav" for i in range(100)]
        labels = [i % config['model']['num_classes'] for i in range(100)]
    
    # Split dataset
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_dataset(
        audio_paths,
        labels,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        random_seed=config['seed'],
        stratify=config['data']['stratify']
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        batch_size=config['training']['batch_size'],
        feature_type=config['features']['type'],
        sr=config['features']['sr'],
        target_length=config['features']['target_length'],
        num_workers=config['hardware']['num_workers']
    )
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    print("\n2. Creating Model...")
    print("-" * 70)
    
    if model_type == 'baseline':
        model = create_baseline_cnn(
            num_classes=config['model']['num_classes'],
            dropout_rate=config['model']['dropout_rate']
        )
    else:  # comparison
        model = create_cnn_bilstm(
            num_classes=config['model']['num_classes'],
            lstm_hidden_size=config['model']['lstm']['hidden_size'],
            lstm_num_layers=config['model']['lstm']['num_layers'],
            dropout_rate=config['model']['cnn']['dropout_rate']
        )
    
    model = model.to(device)
    print_model_summary(model, f"{model_type.upper()} Model")
    
    # ========================================================================
    # SETUP TRAINING
    # ========================================================================
    print("\n3. Setting Up Training...")
    print("-" * 70)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config['training']['loss'].get('label_smoothing', 0.0)
    )
    
    # Optimizer
    if config['training']['optimizer']['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=config['training']['optimizer']['betas'],
            eps=config['training']['optimizer']['eps']
        )
    elif config['training']['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']['type']}")
    
    # Learning rate scheduler
    if config['training']['scheduler']['type'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config['training']['scheduler']['mode'],
            factor=config['training']['scheduler']['factor'],
            patience=config['training']['scheduler']['patience'],
            min_lr=config['training']['scheduler']['min_lr']
        )
    
    # Early stopping
    early_stopping = None
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
            mode='max'  # For accuracy
        )
    
    # Create directories
    checkpoint_dir = config['checkpoints']['save_dir']
    log_dir = config['logging']['log_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir) if config['logging']['tensorboard'] else None
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("\n4. Starting Training...")
    print("=" * 70)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_f1s = []
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        epoch_start = time.time()
        
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('F1/val', val_f1, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        if epoch % config['checkpoints']['save_every'] == 0 or is_best:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'checkpoint_epoch_{epoch}.pth'
            )
            save_checkpoint(
                model, optimizer, epoch,
                train_loss, val_loss, val_acc,
                checkpoint_path, is_best
            )
        
        # Early stopping
        if early_stopping:
            if early_stopping(val_acc):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
        print("=" * 70)
    
    # ========================================================================
    # TRAINING COMPLETED
    # ========================================================================
    total_time = time.time() - start_time
    print(f"\n Training Completed!")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save training curves
    if config['logging']['save_plots']:
        curves_path = os.path.join(log_dir, 'training_curves.png')
        plot_training_curves(
            train_losses, val_losses,
            train_accs, val_accs,
            curves_path
        )
    
    # Close TensorBoard writer
    if writer:
        writer.close()
    
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")
    print("=" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Train Turkish Speech Emotion Recognition Model'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['baseline', 'comparison'],
        help='Model to train: baseline or comparison'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (optional)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config is None:
        if args.model == 'baseline':
            config_path = 'config/baseline_config.yaml'
        else:
            config_path = 'config/comparison_config.yaml'
    else:
        config_path = args.config
    
    config = load_config(config_path)
    
    # Train model
    train_model(config, args.model)


if __name__ == "__main__":
    main()
