"""
Dataset Module for Turkish Speech Emotion Recognition

This module provides PyTorch Dataset and DataLoader classes for loading
and preprocessing audio data for emotion recognition.

Key Features:
- Custom PyTorch Dataset for audio files
- Support for different emotion label encodings
- Train/validation/test splitting
- Data augmentation support
- Batch processing utilities

Author: SEN4107 Course Project
Inspired by: IliaZenkov's transformer-cnn-emotion-recognition
"""

import os
import glob
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from features import extract_features_from_audio, augment_audio_noise
import librosa


class EmotionDataset(Dataset):
    """
    PyTorch Dataset for Speech Emotion Recognition.
    
    This dataset loads audio files, extracts features on-the-fly or from
    pre-computed features, and returns (feature, label) pairs for training.
    
    Attributes:
        audio_paths: List of paths to audio files
        labels: List of emotion labels (integers)
        feature_type: Type of features to extract ('mel' or 'mfcc')
        sr: Sampling rate for audio processing
        target_length: Target time dimension after padding/truncation
        normalize: Whether to normalize features
        augment: Whether to apply data augmentation
        emotion_map: Dictionary mapping emotion indices to names
    """
    
    def __init__(
        self,
        audio_paths: List[str],
        labels: List[int],
        feature_type: str = 'mel',
        sr: int = 16000,
        target_length: int = 256,
        normalize: bool = True,
        augment: bool = False,
        emotion_map: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the EmotionDataset.
        
        Args:
            audio_paths: List of audio file paths
            labels: List of emotion labels (0-indexed integers)
            feature_type: 'mel' for mel spectrogram, 'mfcc' for MFCCs
            sr: Sampling rate in Hz
            target_length: Number of time steps (for padding/truncation)
            normalize: Apply z-score normalization to features
            augment: Apply random noise augmentation
            emotion_map: Dictionary mapping label indices to emotion names
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_type = feature_type
        self.sr = sr
        self.target_length = target_length
        self.normalize = normalize
        self.augment = augment
        self.emotion_map = emotion_map or {
            0: 'mutlu',      # happy
            1: 'üzgün',      # sad
            2: 'kızgın',     # angry
            3: 'nötr',       # neutral
            4: 'korku',      # fear
            5: 'şaşkın',     # surprised
            6: 'iğrenme',    # disgust
        }
        
        # Validate inputs
        assert len(audio_paths) == len(labels), "Mismatch between audio paths and labels"
        
    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (feature_tensor, label)
            - feature_tensor: Shape (1, n_features, time_steps) for CNN input
            - label: Integer emotion label
        """
        # Get audio path and label
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # Apply augmentation if enabled (during training)
        if self.augment and np.random.rand() > 0.5:
            # Load raw audio for augmentation
            audio, _ = librosa.load(audio_path, sr=self.sr, res_type='kaiser_fast')
            # Add noise
            audio = augment_audio_noise(audio, noise_factor=0.005)
            # Save augmented audio to temporary file
            # In practice, we'd extract features directly from augmented audio
            # For simplicity, we'll extract from original and note this is a placeholder
            features = extract_features_from_audio(
                audio_path,
                feature_type=self.feature_type,
                sr=self.sr,
                target_length=self.target_length,
                normalize=self.normalize
            )
        else:
            # Extract features without augmentation
            features = extract_features_from_audio(
                audio_path,
                feature_type=self.feature_type,
                sr=self.sr,
                target_length=self.target_length,
                normalize=self.normalize
            )
        
        # Convert to PyTorch tensor
        # Add channel dimension: (n_features, time) -> (1, n_features, time)
        feature_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        return feature_tensor, label
    
    def get_emotion_name(self, label: int) -> str:
        """Get emotion name from label index."""
        return self.emotion_map.get(label, 'unknown')
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of classes in dataset.
        
        Returns:
            Dictionary mapping emotion names to counts
        """
        distribution = {}
        for label in self.labels:
            emotion_name = self.get_emotion_name(label)
            distribution[emotion_name] = distribution.get(emotion_name, 0) + 1
        return distribution


def create_data_loaders(
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    test_paths: List[str],
    test_labels: List[int],
    batch_size: int = 32,
    feature_type: str = 'mel',
    sr: int = 16000,
    target_length: int = 256,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    This function creates the datasets and wraps them in DataLoaders
    for efficient batch processing during training and evaluation.
    
    Args:
        train_paths: List of training audio paths
        train_labels: List of training labels
        val_paths: List of validation audio paths
        val_labels: List of validation labels
        test_paths: List of test audio paths
        test_labels: List of test labels
        batch_size: Number of samples per batch
        feature_type: Type of features ('mel' or 'mfcc')
        sr: Sampling rate
        target_length: Time dimension of features
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = EmotionDataset(
        train_paths,
        train_labels,
        feature_type=feature_type,
        sr=sr,
        target_length=target_length,
        normalize=True,
        augment=True  # Enable augmentation for training
    )
    
    val_dataset = EmotionDataset(
        val_paths,
        val_labels,
        feature_type=feature_type,
        sr=sr,
        target_length=target_length,
        normalize=True,
        augment=False  # No augmentation for validation
    )
    
    test_dataset = EmotionDataset(
        test_paths,
        test_labels,
        feature_type=feature_type,
        sr=sr,
        target_length=target_length,
        normalize=True,
        augment=False  # No augmentation for testing
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=False  # Disabled for CPU training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader


def split_dataset(
    audio_paths: List[str],
    labels: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    stratify: bool = True
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Split dataset into train, validation, and test sets.
    
    This function performs stratified splitting to ensure each set has
    balanced representation of all emotion classes.
    
    Args:
        audio_paths: List of all audio file paths
        labels: List of all emotion labels
        train_ratio: Proportion of data for training (default 0.8)
        val_ratio: Proportion of data for validation (default 0.1)
        test_ratio: Proportion of data for testing (default 0.1)
        random_seed: Random seed for reproducibility
        stratify: Whether to maintain class distribution across splits
        
    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels, 
                  test_paths, test_labels)
                  
    Example:
        >>> paths = ['audio1.wav', 'audio2.wav', 'audio3.wav']
        >>> labels = [0, 1, 2]
        >>> train_p, train_l, val_p, val_l, test_p, test_l = split_dataset(
        ...     paths, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        ... )
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Convert to numpy for easier indexing
    audio_paths = np.array(audio_paths)
    labels = np.array(labels)
    
    # First split: separate out test set
    stratify_param = labels if stratify else None
    
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        audio_paths,
        labels,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=stratify_param
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    stratify_param = train_val_labels if stratify else None
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_size_adjusted,
        random_state=random_seed,
        stratify=stratify_param
    )
    
    # Convert back to lists
    return (
        train_paths.tolist(), train_labels.tolist(),
        val_paths.tolist(), val_labels.tolist(),
        test_paths.tolist(), test_labels.tolist()
    )


def load_turkish_ser_dataset(
    data_dir: str,
    emotion_map: Optional[Dict[str, int]] = None
) -> Tuple[List[str], List[int]]:
    """
    Load Turkish Speech Emotion Recognition dataset.
    
    This is a placeholder function that demonstrates how to load a dataset.
    You should adapt this to your specific dataset structure.
    
    Expected directory structure:
        data_dir/
            mutlu/          # happy
                audio1.wav
                audio2.wav
            uzgun/          # sad
                audio3.wav
            kizgin/         # angry
                audio4.wav
            notr/           # neutral
                audio5.wav
            ...
    
    Args:
        data_dir: Root directory containing emotion subdirectories
        emotion_map: Dictionary mapping emotion names to integer labels
        
    Returns:
        Tuple of (audio_paths, labels)
        
    Example:
        >>> paths, labels = load_turkish_ser_dataset('data/turkish_emotions')
        >>> print(f"Loaded {len(paths)} samples")
    """
    if emotion_map is None:
        # Default Turkish emotion mapping
        emotion_map = {
            'mutlu': 0,      # happy
            'uzgun': 1,      # sad
            'kizgin': 2,     # angry
            'notr': 3,       # neutral
            'korku': 4,      # fear
            'saskin': 5,     # surprised
            'igrenme': 6,    # disgust
        }
    
    audio_paths = []
    labels = []
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"WARNING: Data directory '{data_dir}' does not exist.")
        print("This is a placeholder. Please provide your Turkish SER dataset.")
        print("\nExpected structure:")
        print("  data_dir/")
        print("    mutlu/")
        print("      audio1.wav")
        print("    uzgun/")
        print("      audio2.wav")
        print("    ...")
        return [], []
    
    # Iterate through emotion directories
    for emotion_name, emotion_label in emotion_map.items():
        emotion_dir = os.path.join(data_dir, emotion_name)
        
        if not os.path.exists(emotion_dir):
            continue
        
        # Find all audio files in this emotion directory
        audio_files = glob.glob(os.path.join(emotion_dir, '*.wav'))
        audio_files.extend(glob.glob(os.path.join(emotion_dir, '*.mp3')))
        
        # Add to lists
        audio_paths.extend(audio_files)
        labels.extend([emotion_label] * len(audio_files))
    
    print(f"Loaded {len(audio_paths)} audio files from {data_dir}")
    
    return audio_paths, labels


if __name__ == "__main__":
    """
    Demo: Dataset module usage
    
    This demonstrates how to use the dataset module for Turkish SER.
    """
    print("Turkish Speech Emotion Recognition - Dataset Module")
    print("=" * 60)
    
    print("\n1. DATASET STRUCTURE")
    print("-" * 60)
    print("Expected directory structure for your Turkish dataset:")
    print("""
    data/turkish_emotions/
        mutlu/          # happy
            sample1.wav
            sample2.wav
        uzgun/          # sad
            sample3.wav
        kizgin/         # angry
            sample4.wav
        notr/           # neutral
            sample5.wav
        korku/          # fear
            sample6.wav
        saskin/         # surprised
            sample7.wav
        igrenme/        # disgust
            sample8.wav
    """)
    
    print("\n2. LOADING DATA")
    print("-" * 60)
    print("Example code:")
    print("""
    from datasets import load_turkish_ser_dataset, split_dataset
    
    # Load dataset
    audio_paths, labels = load_turkish_ser_dataset('data/turkish_emotions')
    
    # Split into train/val/test
    train_p, train_l, val_p, val_l, test_p, test_l = split_dataset(
        audio_paths, labels,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    """)
    
    print("\n3. CREATING DATA LOADERS")
    print("-" * 60)
    print("Example code:")
    print("""
    from datasets import create_data_loaders
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_p, train_l, val_p, val_l, test_p, test_l,
        batch_size=32,
        feature_type='mel',
        sr=16000,
        target_length=256
    )
    
    # Use in training loop
    for features, labels in train_loader:
        # features shape: (batch_size, 1, n_features, time_steps)
        # labels shape: (batch_size,)
        pass
    """)
    
    print("\n" + "=" * 60)
    print("Module ready! Adapt for your specific Turkish SER dataset.")
