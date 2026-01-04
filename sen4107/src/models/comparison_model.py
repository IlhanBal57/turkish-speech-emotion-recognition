"""
Comparison Model: CNN + BiLSTM for Turkish Speech Emotion Recognition

This module implements a hybrid CNN-BiLSTM architecture that combines:
1. CNN layers for spatial feature extraction from spectrograms
2. Bidirectional LSTM for temporal sequence modeling

This serves as Model 2 (comparison model) in our project.

Architecture Motivation:
- CNN: Extract spatial patterns from spectrograms (frequency relationships)
- BiLSTM: Model temporal dependencies in the extracted features
- Hybrid: Combine spatial and temporal modeling strengths

Why BiLSTM after CNN?
1. CNN reduces spatial dimensions while preserving temporal structure
2. BiLSTM processes the time sequence bidirectionally
3. Captures both past and future context for each time step
4. Better than unidirectional LSTM for emotion recognition

Author: SEN4107 Course Project  
Inspired by: Hybrid architectures from recent SER literature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNBiLSTM(nn.Module):
    """
    Hybrid CNN-BiLSTM model for Speech Emotion Recognition.
    
    Architecture:
    1. Two convolutional blocks extract spatial features
    2. Features are reshaped into temporal sequences
    3. Bidirectional LSTM processes sequences
    4. Fully connected layers for classification
    
    Key Differences from Baseline CNN:
    - Uses BiLSTM to model temporal dynamics
    - Fewer conv layers (2 vs 3) to preserve temporal resolution
    - BiLSTM hidden states capture sequence context
    - More parameters due to LSTM layers
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        input_channels: int = 1,
        cnn_channels: Tuple[int, int] = (32, 64),
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout_rate: float = 0.3
    ):
        """
        Initialize the CNN-BiLSTM model.
        
        Args:
            num_classes: Number of emotion classes
            input_channels: Number of input channels (1 for spectrogram)
            cnn_channels: Tuple of output channels for each CNN block
            lstm_hidden_size: Hidden size of LSTM layers
            lstm_num_layers: Number of stacked LSTM layers
            dropout_rate: Dropout probability for regularization
        """
        super(CNNBiLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cnn_channels = cnn_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout_rate = dropout_rate
        
        # ============================================================
        # CNN BLOCK 1: Extract low-level spatial features
        # ============================================================
        # Input: (batch, 1, 128, 256)
        # Output: (batch, 32, 64, 128)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=cnn_channels[0],  # 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            # Use smaller pooling to preserve more temporal information
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate)
        )
        
        # ============================================================
        # CNN BLOCK 2: Extract high-level spatial features
        # ============================================================
        # Input: (batch, 32, 64, 128)
        # Output: (batch, 64, 32, 64)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=cnn_channels[0],  # 32
                out_channels=cnn_channels[1],  # 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(inplace=True),
            # Moderate pooling to preserve temporal resolution for LSTM
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate)
        )
        
        # After CNN blocks, we have: (batch, 64, 32, 64)
        # We'll reshape this to: (batch, 64, 32*64) for LSTM
        # where 64 time steps, each with 32*64=2048 features
        
        # ============================================================
        # BIDIRECTIONAL LSTM LAYERS: Temporal sequence modeling
        # ============================================================
        # Why BiLSTM?
        # - Processes sequence in both forward and backward directions
        # - Captures past and future context for each time step
        # - Better than unidirectional LSTM for non-causal tasks
        # - Hidden state at each time step has information from entire sequence
        
        # After CNN, frequency dimension is reduced to 32
        # We'll treat this as feature dimension for LSTM
        self.lstm_input_size = 32  # Frequency dimension after CNNs
        
        # BiLSTM expects input of shape: (sequence_length, batch, input_size)
        # We'll permute our CNN output accordingly
        
        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,  # Input shape: (batch, seq, feature)
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=True  # Process sequence in both directions
        )
        
        # BiLSTM output size is 2 * hidden_size (forward + backward)
        self.lstm_output_size = 2 * lstm_hidden_size
        
        # ============================================================
        # FULLY CONNECTED LAYERS: Classification head
        # ============================================================
        # We'll use the final hidden state from BiLSTM for classification
        # Alternatively, could use attention mechanism over all time steps
        
        self.fc1 = nn.Linear(self.lstm_output_size, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc1_dropout = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                # Initialize LSTM weights
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CNN-BiLSTM network.
        
        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_steps)
        
        Returns:
            Tuple of (logits, probabilities)
        """
        batch_size = x.size(0)
        
        # ============================================================
        # CNN Feature Extraction
        # ============================================================
        # Input: (batch, 1, 128, 256)
        x = self.conv_block1(x)  # -> (batch, 32, 64, 128)
        x = self.conv_block2(x)  # -> (batch, 64, 32, 64)
        
        # Current shape: (batch, channels=64, freq=32, time=64)
        
        # ============================================================
        # Reshape for LSTM: Treat frequency as feature dimension
        # ============================================================
        # We want: (batch, time_steps, features)
        # Current: (batch, 64, 32, 64) where last dim is time
        
        # Permute: (batch, channels, freq, time) -> (batch, time, freq, channels)
        x = x.permute(0, 3, 2, 1)  # -> (batch, 64, 32, 64)
        
        # Flatten frequency and channel dimensions
        # (batch, time, freq, channels) -> (batch, time, freq*channels)
        time_steps = x.size(1)
        freq_dim = x.size(2)
        channel_dim = x.size(3)
        
        # Reshape: (batch, 64, 32, 64) -> (batch, 64, 32*64=2048)
        x = x.contiguous().view(batch_size, time_steps, freq_dim * channel_dim)
        
        # For simplicity, we'll reduce feature dimension with a linear layer
        # This avoids having huge LSTM input size
        if not hasattr(self, 'feature_reduction'):
            self.feature_reduction = nn.Linear(
                freq_dim * channel_dim,
                self.lstm_input_size
            ).to(x.device)
        
        # Reduce features: (batch, time, 2048) -> (batch, time, 32)
        x = self.feature_reduction(x)
        
        # ============================================================
        # BiLSTM Sequence Processing
        # ============================================================
        # Input: (batch, sequence_length, input_size)
        # Output: (batch, sequence_length, 2*hidden_size)
        
        # BiLSTM forward pass
        # lstm_out: (batch, seq_len, 2*hidden_size)
        # hidden: tuple of (h_n, c_n) where each is (num_layers*2, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.bilstm(x)
        
        # We use the final hidden state for classification
        # h_n shape: (num_layers*2, batch, hidden_size)
        # We want the last layer's forward and backward hidden states
        
        # Extract last layer's hidden states
        # Forward direction: h_n[-2, :, :]
        # Backward direction: h_n[-1, :, :]
        h_forward = h_n[-2, :, :]  # (batch, hidden_size)
        h_backward = h_n[-1, :, :]  # (batch, hidden_size)
        
        # Concatenate forward and backward hidden states
        # (batch, hidden_size) + (batch, hidden_size) -> (batch, 2*hidden_size)
        final_hidden = torch.cat([h_forward, h_backward], dim=1)
        
        # ============================================================
        # Classification Head
        # ============================================================
        # Input: (batch, 2*hidden_size)
        x = self.fc1(final_hidden)  # -> (batch, 128)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc1_dropout(x)
        
        # Output layer
        logits = self.fc2(x)  # -> (batch, num_classes)
        probabilities = F.softmax(logits, dim=1)
        
        return logits, probabilities
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cnn_bilstm(
    num_classes: int = 7,
    lstm_hidden_size: int = 128,
    lstm_num_layers: int = 2,
    dropout_rate: float = 0.3
) -> CNNBiLSTM:
    """
    Factory function to create CNN-BiLSTM model.
    
    Args:
        num_classes: Number of emotion classes
        lstm_hidden_size: LSTM hidden dimension
        lstm_num_layers: Number of stacked LSTM layers
        dropout_rate: Dropout probability
        
    Returns:
        Initialized CNNBiLSTM model
    """
    model = CNNBiLSTM(
        num_classes=num_classes,
        input_channels=1,
        cnn_channels=(32, 64),
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        dropout_rate=dropout_rate
    )
    return model


if __name__ == "__main__":
    """
    Demo: CNN-BiLSTM model architecture
    """
    print("CNN-BiLSTM Comparison Model - Architecture Demo")
    print("=" * 70)
    
    # Create model
    model = create_cnn_bilstm(
        num_classes=7,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        dropout_rate=0.3
    )
    
    print(f"\n1. MODEL SUMMARY")
    print("-" * 70)
    print(f"Total trainable parameters: {model.get_num_params():,}")
    print(f"Number of classes: {model.num_classes}")
    print(f"LSTM hidden size: {model.lstm_hidden_size}")
    print(f"LSTM layers: {model.lstm_num_layers}")
    print(f"Dropout rate: {model.dropout_rate}")
    
    print(f"\n2. KEY ARCHITECTURAL DIFFERENCES FROM BASELINE")
    print("-" * 70)
    print("✓ Hybrid CNN-RNN architecture (vs pure CNN)")
    print("✓ Bidirectional LSTM for temporal modeling")
    print("✓ Fewer conv layers (2 vs 3) to preserve temporal resolution")
    print("✓ LSTM processes sequence of CNN features")
    print("✓ Captures both spatial (CNN) and temporal (BiLSTM) patterns")
    print("✓ More parameters due to LSTM layers")
    
    print(f"\n3. FORWARD PASS TEST")
    print("-" * 70)
    dummy_input = torch.randn(4, 1, 128, 256)
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        logits, probs = model(dummy_input)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Output probabilities shape: {probs.shape}")
    print(f"\nSample probabilities (first sample): {probs[0].numpy()}")
    print(f"Predicted class: {torch.argmax(probs[0]).item()}")
    
    print(f"\n4. WHY CNN + BiLSTM?")
    print("-" * 70)
    print("CNN Advantages:")
    print("  • Extracts spatial features from spectrograms")
    print("  • Captures frequency relationships")
    print("  • Translation invariant via pooling")
    print("  • Reduces dimensionality efficiently")
    print("\nBiLSTM Advantages:")
    print("  • Models temporal dependencies")
    print("  • Bidirectional context (past + future)")
    print("  • Captures emotion evolution over time")
    print("  • Better than unidirectional LSTM")
    print("\nHybrid Benefits:")
    print("  • Combines spatial and temporal modeling")
    print("  • CNN reduces dimensions for efficient LSTM processing")
    print("  • BiLSTM adds sequence context to CNN features")
    print("  • Potentially higher accuracy than pure CNN")
    
    print("\n" + "=" * 70)
    print("Model ready for training!")
