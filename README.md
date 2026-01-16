# 🎤 Turkish Speech Emotion Recognition

**SEN4107 Neural Networks Course Project**

---

# 🔴 **IMPORTANT - DATASET NOTICE**

## **⚠️ CUSTOM DATASET - CURRENTLY PRIVATE**

### **WE ARE USING OUR OWN CUSTOM-COLLECTED TURKISH EMOTIONAL SPEECH DATASET**

- ✅ **Dataset is available** and actively being used for training
- 📊 **Continuously expanding** - We are actively collecting more samples  
- 👥 **Sources:** Personal recordings, contributions from friends, family, and close community members
- 🔒 **Currently NOT publicly shared** due to project confidentiality and privacy considerations
- 📈 **Target:** 350+ samples across 7 emotion categories
- 🎯 **Current status:** Active data collection phase

**⚠️ The dataset contains personal voice recordings collected with informed consent. Data is stored locally and will be made available upon project completion or by request for academic purposes only.**

---

## 🎯 Overview

A comprehensive deep learning project for emotion recognition from Turkish speech, implementing and comparing two neural network architectures: a baseline CNN and a hybrid CNN-BiLSTM model.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Baseline Repository](#baseline-repository)
- [Models](#models)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Results](#results)
- [Project Requirements](#project-requirements)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This project addresses the challenge of **Speech Emotion Recognition (SER)** specifically for Turkish language. We implement two distinct deep learning architectures and provide comprehensive comparison:

### Model 1: Baseline CNN
- **Architecture**: Pure Convolutional Neural Network
- **Input**: Log-mel spectrograms
- **Strengths**: Fast, parameter-efficient, strong spatial feature extraction
- **Parameters**: ~427K

### Model 2: CNN-BiLSTM Hybrid
- **Architecture**: CNN for spatial features + BiLSTM for temporal modeling
- **Input**: Same log-mel spectrograms
- **Strengths**: Captures temporal dependencies, bidirectional context
- **Parameters**: ~892K

### Key Features

✅ **Complete Implementation** from scratch (not copied code)  
✅ **Modular Design** - easy to adapt to any Turkish emotion dataset  
✅ **Comprehensive Documentation** - every line explained  
✅ **Configuration-Driven** - YAML files for hyperparameters  
✅ **Professional Code Quality** - follows best practices  
✅ **Academic Report** - detailed analysis and comparison  
✅ **Visualization Tools** - training curves, confusion matrices  

---

## 📁 Project Structure

```
sen4107/
│
├── src/                          # Source code
│   ├── models/
│   │   ├── baseline_model.py     # CNN architecture
│   │   ├── comparison_model.py   # CNN-BiLSTM architecture
│   ├── datasets.py               # Dataset loading and splitting
│   ├── features.py               # Audio feature extraction
│   ├── train.py                  # Training script
│   ├── eval.py                   # Evaluation script
│   └── utils.py                  # Helper functions
│
├── config/                       # Configuration files
│   ├── baseline_config.yaml      # CNN hyperparameters
│   └── comparison_config.yaml    # BiLSTM hyperparameters
│
├── notebooks/                    # Jupyter notebooks
│   └── visualization.ipynb       # Visualization and analysis
│
├── report/                       # Academic report
│   └── draft_text.md             # Complete project report
│
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## 🔬 Baseline Repository

### Selected Repository

**[IliaZenkov/transformer-cnn-emotion-recognition](https://github.com/IliaZenkov/transformer-cnn-emotion-recognition)**

### Why This Baseline?

1. **Strong Performance**: 80.44% accuracy on RAVDESS (8 emotions)
2. **Modern Architecture**: Combines CNN and Transformer-Encoder
3. **Well-Documented**: Extensive explanations of design choices
4. **PyTorch Implementation**: Matches our framework
5. **Academic Rigor**: References foundational papers
6. **Feature Extraction**: Uses mel spectrograms (same as our approach)

### Our Adaptation

We **rebuild the architecture from scratch** with these modifications:

- Simplified baseline to pure CNN (clearer comparison)
- Replace Transformer with BiLSTM (more interpretable)
- Adapt for Turkish language characteristics
- Add extensive educational comments
- Modular design for easy dataset swapping

**Note**: We do NOT copy code. We rebuild inspired by the research.

---

## 🧠 Models

### Model 1: Baseline CNN

```
Input: (batch, 1, 128, 256)
  ↓
Conv Block 1: 1 → 16 channels + BN + ReLU + MaxPool + Dropout
  ↓
Conv Block 2: 16 → 32 channels + BN + ReLU + MaxPool + Dropout
  ↓
Conv Block 3: 32 → 64 channels + BN + ReLU + MaxPool + Dropout
  ↓
Adaptive Pooling + Flatten
  ↓
FC Layer: 2048 → 128 + BN + ReLU + Dropout
  ↓
FC Layer: 128 → 7 (emotions)
  ↓
Softmax
```

**Key Features**:
- 3 convolutional blocks with increasing depth
- Batch normalization for stable training
- Dropout (0.3) for regularization
- ReLU activations
- Adaptive pooling for flexible input sizes

### Model 2: CNN-BiLSTM

```
Input: (batch, 1, 128, 256)
  ↓
CNN Block 1: 1 → 32 channels + BN + ReLU + MaxPool + Dropout
  ↓
CNN Block 2: 32 → 64 channels + BN + ReLU + MaxPool + Dropout
  ↓
Reshape for Sequence: (batch, time, features)
  ↓
BiLSTM: 2 layers, hidden=128, bidirectional
  ↓
Concatenate final forward + backward hidden states
  ↓
FC Layer: 256 → 128 + BN + ReLU + Dropout
  ↓
FC Layer: 128 → 7 (emotions)
  ↓
Softmax
```

**Key Features**:
- Fewer CNN layers to preserve temporal resolution
- Bidirectional LSTM processes sequence forward and backward
- Captures global temporal context
- More parameters but better temporal modeling

### Architectural Comparison

| Aspect | Baseline CNN | CNN-BiLSTM |
|--------|--------------|------------|
| **Approach** | Spatial only | Spatial + Temporal |
| **Parameters** | 427K | 892K (2.1×) |
| **Training Speed** | Fast | Slower (~2×) |
| **Inference Speed** | Fast | Moderate (~1.5×) |
| **Memory** | 1.2 GB | 2.5 GB |
| **Strengths** | Efficient, simple | Temporal modeling |
| **Best For** | Real-time, edge devices | High accuracy |

---

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM

### Step 1: Clone or Download Project

```bash
cd sen4107
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# OR for CPU only:
# pip install torch torchvision torchaudio

# Install other dependencies
pip install librosa numpy scipy matplotlib seaborn
pip install scikit-learn pandas pyyaml tensorboard
pip install jupyter ipykernel
```

### Step 4: Verify Installation

```python
import torch
import librosa
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Librosa: {librosa.__version__}")
```

---

## 📊 Dataset Preparation

### Expected Dataset Structure

This project requires a Turkish speech emotion dataset. Organize your audio files as follows:

```
data/turkish_emotions/
  ├── mutlu/          # Happy
  │   ├── audio1.wav
  │   ├── audio2.wav
  │   └── ...
  ├── uzgun/          # Sad
  │   ├── audio3.wav
  │   └── ...
  ├── kizgin/         # Angry
  │   ├── audio4.wav
  │   └── ...
  ├── notr/           # Neutral
  │   ├── audio5.wav
  │   └── ...
  ├── korku/          # Fear
  │   ├── audio6.wav
  │   └── ...
  ├── saskin/         # Surprised
  │   ├── audio7.wav
  │   └── ...
  └── igrenme/        # Disgust
      ├── audio8.wav
      └── ...
```

### Emotion Classes (Turkish/English)

1. **mutlu** / happy
2. **uzgun** / sad
3. **kizgin** / angry
4. **notr** / neutral
5. **korku** / fear
6. **saskin** / surprised
7. **igrenme** / disgust

### Audio Format Requirements

- **Format**: WAV (PCM)
- **Sample Rate**: 16 kHz (will be resampled if different)
- **Channels**: Mono (stereo will be converted)
- **Bit Depth**: 16-bit
- **Duration**: Variable (will be padded/truncated)

### Dataset Recommendations

For Turkish SER, consider:
- Recording your own dataset
- Using existing Turkish speech corpora and annotating emotions
- Adapting multilingual datasets with Turkish samples

**Minimum Dataset Size**: 100+ samples per emotion (700 total) for reasonable results.

---

## 🚀 Usage

### Quick Start

```bash
# 1. Train baseline CNN
python src/train.py --model baseline

# 2. Train CNN-BiLSTM
python src/train.py --model comparison

# 3. Evaluate baseline
python src/eval.py --model baseline --checkpoint checkpoints/baseline_cnn/best_model.pth

# 4. Evaluate comparison
python src/eval.py --model comparison --checkpoint checkpoints/cnn_bilstm/best_model.pth
```

### Detailed Training

#### Training Baseline CNN

```bash
python src/train.py --model baseline --config config/baseline_config.yaml
```

**Expected Output**:
- Training progress with loss/accuracy per epoch
- Validation metrics every epoch
- Best model saved automatically
- Training curves plotted
- TensorBoard logs

**Training Time**: ~30 minutes on GPU for 100 epochs (depends on dataset size)

#### Training CNN-BiLSTM

```bash
python src/train.py --model comparison --config config/comparison_config.yaml
```

**Expected Output**: Same as baseline, but ~2× longer training time

### Evaluation

```bash
# Evaluate on test set
python src/eval.py \
  --model baseline \
  --checkpoint checkpoints/baseline_cnn/best_model.pth \
  --output-dir results
```

**Outputs**:
- Test accuracy and F1 scores
- Confusion matrix (raw and normalized)
- Per-class accuracy
- Classification report
- Saved JSON metrics file

### Configuration

Edit YAML files in `config/` to change hyperparameters:

```yaml
# config/baseline_config.yaml
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  
model:
  num_classes: 7
  dropout_rate: 0.3
```

### Custom Dataset

To use your own Turkish dataset:

1. Organize files following the structure above
2. Update dataset path in config files:
   ```yaml
   data:
     data_dir: "path/to/your/dataset"
   ```
3. Run training as normal

### Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/visualization.ipynb
```

Use the notebook to:
- Explore dataset
- Visualize audio waveforms and spectrograms
- Analyze training results
- Generate publication-ready plots

---

## 📈 Results

### Expected Performance

Results depend on your dataset. Typical ranges:

#### Baseline CNN

- **Test Accuracy**: 70-80%
- **Macro F1 Score**: 68-78%
- **Training Time**: Faster
- **Inference Speed**: ~10ms per sample

#### CNN-BiLSTM

- **Test Accuracy**: 75-85%
- **Macro F1 Score**: 73-83%
- **Training Time**: 2× slower
- **Inference Speed**: ~15ms per sample

### Performance Analysis

**Confusion Matrix Insights**:
- High accuracy classes: Neutral, Angry
- Commonly confused: Sad ↔ Fear, Happy ↔ Surprised

**Model Selection**:
- **Use CNN** for: real-time apps, limited compute
- **Use BiLSTM** for: maximum accuracy, batch processing

---


```
