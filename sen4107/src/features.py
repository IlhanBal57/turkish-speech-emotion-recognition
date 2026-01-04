"""
Feature Extraction Module for Turkish Speech Emotion Recognition

This module provides functions to extract acoustic features from audio files:
- Log-Mel Spectrograms: Time-frequency representation with perceptually motivated mel scale
- MFCCs (Mel-Frequency Cepstral Coefficients): Compact spectral envelope representation

Both features are widely used in speech emotion recognition and capture complementary information:
- Mel spectrograms: preserve detailed time-frequency structure
- MFCCs: provide compact representation of spectral envelope

Author: SEN4107 Course Project
Inspired by: IliaZenkov's transformer-cnn-emotion-recognition
"""

import librosa
import librosa.display
import numpy as np
import torch
from typing import Tuple, Optional


def extract_mel_spectrogram(
    audio_path: str,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    duration: Optional[float] = None,
    offset: float = 0.0
) -> np.ndarray:
    """
    Extract log-mel spectrogram from audio file.
    
    Mel spectrograms convert audio signals into a time-frequency representation
    using perceptually-motivated mel frequency scale. Log scaling makes the 
    features more suitable for neural network training.
    
    Args:
        audio_path: Path to audio file (supports .wav, .mp3, etc.)
        sr: Target sampling rate in Hz (default 22050)
        n_mels: Number of mel frequency bands (default 128)
        n_fft: FFT window size (default 2048)
        hop_length: Number of samples between successive frames (default 512)
        duration: Duration to load in seconds (None = entire file)
        offset: Start reading after this time in seconds
        
    Returns:
        Log-mel spectrogram as numpy array of shape (n_mels, time_steps)
        
    Example:
        >>> mel_spec = extract_mel_spectrogram('audio.wav', sr=16000, n_mels=64)
        >>> print(mel_spec.shape)  # (64, time_steps)
    """
    # Load audio file with librosa
    # res_type='kaiser_fast' provides good quality with faster processing
    audio, _ = librosa.load(
        audio_path,
        sr=sr,
        duration=duration,
        offset=offset,
        res_type='kaiser_fast'
    )
    
    # Compute mel spectrogram
    # This applies: STFT -> Power spectrum -> Mel filterbank
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convert to log scale (dB) for better neural network training
    # Add small epsilon to avoid log(0)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec


def extract_mfcc(
    audio_path: str,
    sr: int = 22050,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    duration: Optional[float] = None,
    offset: float = 0.0
) -> np.ndarray:
    """
    Extract MFCCs (Mel-Frequency Cepstral Coefficients) from audio file.
    
    MFCCs provide a compact representation of the spectral envelope and are
    widely used in speech processing. They are computed by taking DCT of 
    log-mel spectrogram, capturing energy distribution across mel bands.
    
    Args:
        audio_path: Path to audio file
        sr: Target sampling rate in Hz
        n_mfcc: Number of MFCCs to extract (typical: 13, 20, 40)
        n_fft: FFT window size
        hop_length: Hop length between frames
        n_mels: Number of mel bands (used in intermediate computation)
        duration: Duration to load in seconds
        offset: Start time in seconds
        
    Returns:
        MFCC features as numpy array of shape (n_mfcc, time_steps)
        
    Example:
        >>> mfccs = extract_mfcc('audio.wav', sr=16000, n_mfcc=20)
        >>> print(mfccs.shape)  # (20, time_steps)
    """
    # Load audio file
    audio, _ = librosa.load(
        audio_path,
        sr=sr,
        duration=duration,
        offset=offset,
        res_type='kaiser_fast'
    )
    
    # Extract MFCCs
    # This applies: STFT -> Power spectrum -> Mel filterbank -> Log -> DCT
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    return mfccs


def pad_or_truncate(
    feature: np.ndarray,
    target_length: int,
    axis: int = 1
) -> np.ndarray:
    """
    Pad or truncate feature array to target length along specified axis.
    
    This ensures all samples have the same temporal dimension, which is
    required for batch processing in neural networks.
    
    Args:
        feature: Input feature array (e.g., mel spectrogram or MFCC)
        target_length: Desired length along the time axis
        axis: Axis along which to pad/truncate (default 1 for time)
        
    Returns:
        Feature array with shape[axis] == target_length
        
    Example:
        >>> mel = np.random.randn(128, 100)  # 100 time steps
        >>> mel_fixed = pad_or_truncate(mel, target_length=200)
        >>> print(mel_fixed.shape)  # (128, 200)
    """
    current_length = feature.shape[axis]
    
    if current_length > target_length:
        # Truncate: take first target_length frames
        if axis == 1:
            return feature[:, :target_length]
        elif axis == 0:
            return feature[:target_length, :]
        else:
            raise ValueError(f"Unsupported axis: {axis}")
            
    elif current_length < target_length:
        # Pad: add zeros at the end
        pad_width = [(0, 0)] * feature.ndim
        pad_width[axis] = (0, target_length - current_length)
        return np.pad(feature, pad_width, mode='constant', constant_values=0)
    
    else:
        # Already correct length
        return feature


def normalize_features(
    feature: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
    """
    Normalize feature array for better neural network training.
    
    Normalization helps:
    - Speed up convergence during training
    - Prevent gradient vanishing/explosion
    - Make features scale-invariant
    
    Args:
        feature: Input feature array
        method: Normalization method
            - 'standard': zero mean, unit variance (z-score)
            - 'minmax': scale to [0, 1] range
            - 'l2': L2 normalization per sample
            
    Returns:
        Normalized feature array with same shape
        
    Example:
        >>> mel = np.random.randn(128, 200)
        >>> mel_norm = normalize_features(mel, method='standard')
        >>> print(np.mean(mel_norm), np.std(mel_norm))  # ~0.0, ~1.0
    """
    if method == 'standard':
        # Standardize: (x - mean) / std
        mean = np.mean(feature)
        std = np.std(feature)
        return (feature - mean) / (std + 1e-8)  # epsilon to avoid division by zero
        
    elif method == 'minmax':
        # Min-max scaling: (x - min) / (max - min)
        min_val = np.min(feature)
        max_val = np.max(feature)
        return (feature - min_val) / (max_val - min_val + 1e-8)
        
    elif method == 'l2':
        # L2 normalization
        norm = np.linalg.norm(feature)
        return feature / (norm + 1e-8)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def augment_audio_noise(
    audio: np.ndarray,
    noise_factor: float = 0.005
) -> np.ndarray:
    """
    Add white Gaussian noise to audio for data augmentation.
    
    Data augmentation helps:
    - Increase dataset size without collecting more data
    - Improve model generalization
    - Make model robust to noisy inputs
    
    Args:
        audio: Input audio waveform as numpy array
        noise_factor: Standard deviation of Gaussian noise (default 0.005)
        
    Returns:
        Augmented audio waveform
        
    Example:
        >>> audio = np.random.randn(16000)  # 1 second at 16kHz
        >>> audio_noisy = augment_audio_noise(audio, noise_factor=0.01)
    """
    # Generate white Gaussian noise with same shape as audio
    noise = np.random.normal(0, noise_factor, audio.shape)
    
    # Add noise to original audio
    augmented = audio + noise
    
    # Clip to valid range [-1, 1] to prevent overflow
    augmented = np.clip(augmented, -1.0, 1.0)
    
    return augmented


def extract_features_from_audio(
    audio_path: str,
    feature_type: str = 'mel',
    sr: int = 22050,
    target_length: int = 200,
    normalize: bool = True
) -> np.ndarray:
    """
    High-level function to extract and preprocess features from audio file.
    
    This is the main function to use for feature extraction. It handles:
    - Loading audio
    - Feature extraction (mel spectrogram or MFCC)
    - Padding/truncation to fixed length
    - Optional normalization
    
    Args:
        audio_path: Path to audio file
        feature_type: 'mel' for mel spectrogram, 'mfcc' for MFCCs
        sr: Sampling rate
        target_length: Target number of time steps (for padding/truncation)
        normalize: Whether to apply z-score normalization
        
    Returns:
        Preprocessed feature array ready for model input
        Shape: (n_features, target_length)
        
    Example:
        >>> features = extract_features_from_audio(
        ...     'emotion.wav',
        ...     feature_type='mel',
        ...     sr=16000,
        ...     target_length=256
        ... )
        >>> print(features.shape)  # (128, 256) for mel spectrogram
    """
    # Extract raw features
    if feature_type == 'mel':
        features = extract_mel_spectrogram(audio_path, sr=sr)
    elif feature_type == 'mfcc':
        features = extract_mfcc(audio_path, sr=sr)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    # Pad or truncate to target length
    features = pad_or_truncate(features, target_length=target_length, axis=1)
    
    # Normalize if requested
    if normalize:
        features = normalize_features(features, method='standard')
    
    return features


if __name__ == "__main__":
    """
    Demo: how to use the feature extraction functions
    
    This shows typical usage patterns for extracting features from audio files.
    """
    print("Feature Extraction Module - Demo")
    print("=" * 50)
    
    # Example: Extract mel spectrogram
    print("\n1. Mel Spectrogram Extraction:")
    print("   - Time-frequency representation")
    print("   - Perceptually motivated mel scale")
    print("   - Usage: extract_mel_spectrogram('audio.wav', sr=16000, n_mels=128)")
    
    # Example: Extract MFCCs
    print("\n2. MFCC Extraction:")
    print("   - Compact spectral envelope")
    print("   - Widely used in speech processing")
    print("   - Usage: extract_mfcc('audio.wav', sr=16000, n_mfcc=40)")
    
    # Example: Complete pipeline
    print("\n3. Complete Feature Extraction Pipeline:")
    print("   - Usage: extract_features_from_audio(")
    print("       'audio.wav',")
    print("       feature_type='mel',")
    print("       sr=16000,")
    print("       target_length=256,")
    print("       normalize=True")
    print("   )")
    
    print("\n" + "=" * 50)
    print("Module ready for use in training pipeline!")
