"""
Baseline CNN Model for Turkish Speech Emotion Recognition

This module implements a Convolutional Neural Network (CNN) for emotion
classification from log-mel spectrograms or MFCCs. This serves as the
baseline model (Model 1) in our project.

Architecture:
- 3 convolutional blocks with increasing filter depth
- Batch normalization after each conv layer
- ReLU activation and Dropout for regularization
- Max pooling for dimensionality reduction
- Fully connected layers for classification

Key Design Principles:
1. Increasing channel depth (16 -> 32 -> 64) to capture hierarchical features
2. Small 3x3 kernels inspired by VGGNet for detailed feature extraction
3. Batch normalization for stable training and faster convergence
4. Dropout to prevent overfitting

Author: SEN4107 Course Project
Inspired by: IliaZenkov's CNN blocks and marcogdepinto's emotion-classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BaselineCNN(nn.Module):
    """
    Baseline Convolutional Neural Network for Speech Emotion Recognition.
    
    This CNN treats mel spectrograms/MFCCs as grayscale images and applies
    2D convolutions to extract spatial features. The architecture follows
    proven patterns from image classification adapted for audio spectrograms.
    
    Architecture Flow:
        Input: (batch, 1, n_mels, time_steps)
        ↓
        Conv Block 1: 1 -> 16 channels
        ↓
        Conv Block 2: 16 -> 32 channels
        ↓
        Conv Block 3: 32 -> 64 channels
        ↓
        Flatten + Fully Connected
        ↓
        Output: (batch, num_classes)
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        input_channels: int = 1,
        dropout_rate: float = 0.3
    ):
        """
        Initialize the Baseline CNN model.
        
        Args:
            num_classes: Number of emotion classes to predict
            input_channels: Number of input channels (1 for grayscale spectrogram)
            dropout_rate: Dropout probability for regularization (default 0.3)
        """
        super(BaselineCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        
        # ============================================================
        # CONVOLUTIONAL BLOCK 1: Extract low-level features
        # ============================================================
        # Input: (batch, 1, 128, 256) - example for 128 mel bands, 256 time steps
        # Output: (batch, 16, 64, 128) after conv and maxpool
        self.conv_block1 = nn.Sequential(
            # Conv2D: 1 input channel -> 16 output channels
            # Kernel size 3x3 is standard for capturing local patterns
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1  # 'same' padding to maintain spatial dimensions
            ),
            # Batch Normalization: normalize activations for stable training
            # Helps with gradient flow and allows higher learning rates
            nn.BatchNorm2d(16),
            # ReLU: non-saturating activation, faster convergence than sigmoid/tanh
            nn.ReLU(inplace=True),
            # Max Pooling: reduce spatial dimensions by 2x, keep dominant features
            # Provides translation invariance and reduces computation
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Dropout: randomly zero out 30% of activations during training
            # Prevents co-adaptation of neurons, improves generalization
            nn.Dropout2d(p=dropout_rate)
        )
        
        # ============================================================
        # CONVOLUTIONAL BLOCK 2: Extract mid-level features
        # ============================================================
        # Input: (batch, 16, 64, 128)
        # Output: (batch, 32, 16, 32) after conv and maxpool
        self.conv_block2 = nn.Sequential(
            # Increase filter depth: 16 -> 32
            # Deeper layers capture more complex, abstract features
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Larger pooling kernel to aggressively reduce dimensions
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout2d(p=dropout_rate)
        )
        
        # ============================================================
        # CONVOLUTIONAL BLOCK 3: Extract high-level features
        # ============================================================
        # Input: (batch, 32, 16, 32)
        # Output: (batch, 64, 4, 8) after conv and maxpool
        self.conv_block3 = nn.Sequential(
            # Further increase depth: 32 -> 64
            # These filters capture emotion-specific patterns
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Final spatial reduction
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout2d(p=dropout_rate)
        )
        
        # ============================================================
        # FULLY CONNECTED LAYERS: Classification head
        # ============================================================
        # After 3 conv blocks with pooling, spatial dimensions are greatly reduced
        # For input (1, 128, 256):
        #   After block1: (16, 64, 128)
        #   After block2: (32, 16, 32)
        #   After block3: (64, 4, 8)
        #   Flattened: 64 * 4 * 8 = 2048
        
        # Calculate flattened dimension (this is dataset-dependent)
        # For safety, we use adaptive pooling before FC layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 8))
        
        # Flattened size: 64 * 4 * 8 = 2048
        self.fc1 = nn.Linear(64 * 4 * 8, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc1_dropout = nn.Dropout(p=0.5)  # Higher dropout in FC layers
        
        # Output layer: map to num_classes
        self.fc2 = nn.Linear(128, num_classes)
        
        # Initialize weights using He initialization (good for ReLU)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization.
        
        He initialization is optimal for ReLU activations - sets initial weights
        based on the number of input units to maintain variance across layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                # Initialize BN parameters
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # He initialization for FC layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_steps)
               Example: (32, 1, 128, 256) for batch size 32
        
        Returns:
            Tuple of (logits, probabilities)
            - logits: Raw output scores, shape (batch, num_classes)
            - probabilities: Softmax probabilities, shape (batch, num_classes)
        """
        # Pass through convolutional blocks
        # x shape: (batch, 1, 128, 256) -> (batch, 16, 64, 128)
        x = self.conv_block1(x)
        
        # x shape: (batch, 16, 64, 128) -> (batch, 32, 16, 32)
        x = self.conv_block2(x)
        
        # x shape: (batch, 32, 16, 32) -> (batch, 64, 4, 8)
        x = self.conv_block3(x)
        
        # Adaptive pooling to ensure consistent size
        # x shape: (batch, 64, 4, 8)
        x = self.adaptive_pool(x)
        
        # Flatten: (batch, 64, 4, 8) -> (batch, 2048)
        x = x.view(x.size(0), -1)
        
        # First fully connected layer with batch norm and dropout
        x = self.fc1(x)  # (batch, 2048) -> (batch, 128)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc1_dropout(x)
        
        # Output layer: (batch, 128) -> (batch, num_classes)
        logits = self.fc2(x)
        
        # Compute softmax probabilities for predictions
        probabilities = F.softmax(logits, dim=1)
        
        return logits, probabilities
    
    def get_num_params(self) -> int:
        """
        Calculate total number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_baseline_cnn(
    num_classes: int = 7,
    dropout_rate: float = 0.3
) -> BaselineCNN:
    """
    Factory function to create a baseline CNN model.
    
    This is a convenience function for creating the model with
    standard hyperparameters.
    
    Args:
        num_classes: Number of emotion classes
        dropout_rate: Dropout probability
        
    Returns:
        Initialized BaselineCNN model
        
    Example:
        >>> model = create_baseline_cnn(num_classes=7, dropout_rate=0.3)
        >>> print(f"Model has {model.get_num_params():,} parameters")
    """
    model = BaselineCNN(
        num_classes=num_classes,
        input_channels=1,
        dropout_rate=dropout_rate
    )
    return model


if __name__ == "__main__":
    """
    Demo: Baseline CNN model architecture and forward pass
    """
    print("Baseline CNN Model - Architecture Demo")
    print("=" * 70)
    
    # Create model
    model = create_baseline_cnn(num_classes=7, dropout_rate=0.3)
    
    print(f"\n1. MODEL SUMMARY")
    print("-" * 70)
    print(f"Total trainable parameters: {model.get_num_params():,}")
    print(f"Number of classes: {model.num_classes}")
    print(f"Dropout rate: {model.dropout_rate}")
    
    print(f"\n2. ARCHITECTURE OVERVIEW")
    print("-" * 70)
    print(model)
    
    print(f"\n3. FORWARD PASS TEST")
    print("-" * 70)
    # Create dummy input: batch_size=4, channels=1, height=128, width=256
    dummy_input = torch.randn(4, 1, 128, 256)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits, probs = model(dummy_input)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Output probabilities shape: {probs.shape}")
    print(f"\nSample logits (first sample): {logits[0].numpy()}")
    print(f"Sample probabilities (first sample): {probs[0].numpy()}")
    print(f"Sum of probabilities: {probs[0].sum():.4f} (should be ~1.0)")
    print(f"Predicted class: {torch.argmax(probs[0]).item()}")
    
    print(f"\n4. KEY ARCHITECTURAL FEATURES")
    print("-" * 70)
    print("✓ 3 convolutional blocks with increasing depth (16->32->64)")
    print("✓ Batch normalization for stable training")
    print("✓ ReLU activations for non-linearity")
    print("✓ Max pooling for spatial reduction")
    print("✓ Dropout for regularization (prevent overfitting)")
    print("✓ Fully connected layers for classification")
    print("✓ Adaptive pooling for flexible input sizes")
    
    print("\n" + "=" * 70)
    print("Model ready for training!")
