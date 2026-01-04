"""
Evaluation Script for Turkish Speech Emotion Recognition

This script evaluates trained models on the test set and generates:
1. Test accuracy and metrics
2. Confusion matrix
3. Per-class accuracy
4. Classification report

Usage:
    python src/eval.py --model baseline --checkpoint checkpoints/baseline_cnn/best_model.pth
    python src/eval.py --model comparison --checkpoint checkpoints/cnn_bilstm/best_model.pth

Author: SEN4107 Course Project
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from models.baseline_model import create_baseline_cnn
from models.comparison_model import create_cnn_bilstm
from datasets import load_turkish_ser_dataset, split_dataset, create_data_loaders
from utils import (
    load_config, set_seed, get_device, load_checkpoint,
    plot_confusion_matrix, save_metrics
)


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list
) -> dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained neural network model
        test_loader: Test data loader
        device: Device to evaluate on
        class_names: List of emotion class names
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating on test set...")
    print("-" * 70)
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, probs = model(features)
            
            # Get predictions
            predictions = torch.argmax(probs, dim=1)
            
            # Store results
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================
    
    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    macro_precision = precision_score(all_labels, all_preds, average='macro') * 100
    macro_recall = recall_score(all_labels, all_preds, average='macro') * 100
    
    # Per-class metrics
    per_class_acc = []
    for i in range(len(class_names)):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == i).sum() / mask.sum() * 100
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    print("\nTEST SET EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Macro F1 Score: {macro_f1:.2f}%")
    print(f"  Weighted F1 Score: {weighted_f1:.2f}%")
    print(f"  Macro Precision: {macro_precision:.2f}%")
    print(f"  Macro Recall: {macro_recall:.2f}%")
    
    print(f"\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {per_class_acc[i]:.2f}%")
    
    print(f"\nClassification Report:")
    print(report)
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # ========================================================================
    # PREPARE METRICS DICTIONARY
    # ========================================================================
    
    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'per_class_accuracy': {
            class_names[i]: float(per_class_acc[i])
            for i in range(len(class_names))
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    return metrics, cm


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate Turkish Speech Emotion Recognition Model'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['baseline', 'comparison'],
        help='Model to evaluate: baseline or comparison'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save evaluation results'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print(f"Turkish Speech Emotion Recognition - {args.model.upper()} Model Evaluation")
    print("=" * 70)
    
    # Load configuration
    if args.config is None:
        if args.model == 'baseline':
            config_path = 'config/baseline_config.yaml'
        else:
            config_path = 'config/comparison_config.yaml'
    else:
        config_path = args.config
    
    config = load_config(config_path)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Get device
    device = get_device(config['hardware']['device'])
    
    # ========================================================================
    # LOAD DATASET
    # ========================================================================
    print("\n1. Loading Dataset...")
    print("-" * 70)
    
    audio_paths, labels = load_turkish_ser_dataset(
        config['data']['data_dir'],
        config['data'].get('emotion_map')
    )
    
    if len(audio_paths) == 0:
        print("ERROR: No data loaded. Using dummy data for demonstration...")
        audio_paths = [f"dummy_{i}.wav" for i in range(100)]
        labels = [i % config['model']['num_classes'] for i in range(100)]
    
    # Split dataset (use same split as training)
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_dataset(
        audio_paths,
        labels,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        random_seed=config['seed'],
        stratify=config['data']['stratify']
    )
    
    print(f"Test samples: {len(test_paths)}")
    
    # Create test data loader
    _, _, test_loader = create_data_loaders(
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        batch_size=config['training']['batch_size'],
        feature_type=config['features']['type'],
        sr=config['features']['sr'],
        target_length=config['features']['target_length'],
        num_workers=config['hardware']['num_workers']
    )
    
    # Get class names
    emotion_map = config['data'].get('emotion_map', {
        'mutlu': 0, 'uzgun': 1, 'kizgin': 2, 'notr': 3,
        'korku': 4, 'saskin': 5, 'igrenme': 6
    })
    class_names = [k for k, v in sorted(emotion_map.items(), key=lambda x: x[1])]
    
    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    print("\n2. Loading Model...")
    print("-" * 70)
    
    if args.model == 'baseline':
        model = create_baseline_cnn(
            num_classes=config['model']['num_classes'],
            dropout_rate=config['model']['dropout_rate']
        )
    else:
        model = create_cnn_bilstm(
            num_classes=config['model']['num_classes'],
            lstm_hidden_size=config['model']['lstm']['hidden_size'],
            lstm_num_layers=config['model']['lstm']['num_layers'],
            dropout_rate=config['model']['cnn']['dropout_rate']
        )
    
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, None, args.checkpoint, device)
    
    # ========================================================================
    # EVALUATE
    # ========================================================================
    print("\n3. Running Evaluation...")
    print("-" * 70)
    
    metrics, cm = evaluate_model(model, test_loader, device, class_names)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n4. Saving Results...")
    print("-" * 70)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f'{args.model}_test_metrics.json')
    save_metrics(metrics, metrics_path)
    
    # Save confusion matrix plot
    cm_path = os.path.join(output_dir, f'{args.model}_confusion_matrix.png')
    plot_confusion_matrix(
        cm,
        class_names,
        cm_path,
        normalize=False,
        title=f'{args.model.upper()} - Confusion Matrix (Test Set)'
    )
    
    # Save normalized confusion matrix
    cm_norm_path = os.path.join(output_dir, f'{args.model}_confusion_matrix_normalized.png')
    plot_confusion_matrix(
        cm,
        class_names,
        cm_norm_path,
        normalize=True,
        title=f'{args.model.upper()} - Normalized Confusion Matrix (Test Set)'
    )
    
    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
