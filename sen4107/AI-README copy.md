# AI Agent Context - Turkish Speech Emotion Recognition Project

## 🤖 For AI Agents Working on This Project

This document provides comprehensive technical context for AI agents to understand and continue development on this Turkish Speech Emotion Recognition (SER) system.

---

## 📋 Project Overview

**Type:** Deep Learning - Speech Emotion Recognition  
**Language:** Turkish  
**Framework:** PyTorch 2.9.1 + Flask 3.0  
**Architecture:** Web-based ML system with real-time training monitoring  
**Status:** Active development - Data collection phase  
**Dataset:** Custom collected, currently 21-24 samples, expanding to 350+

---

## 🏗️ System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                     WEB INTERFACE (Flask)                    │
│  Dashboard | Record | Train | History | Dataset Browser     │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐            ┌────────▼─────────┐
│  Flask Backend │◄──────────►│  Socket.IO       │
│  (app.py)      │            │  (Real-time)     │
└───────┬────────┘            └────────┬─────────┘
        │                               │
        ├───────────────┬───────────────┤
        │               │               │
┌───────▼─────┐  ┌─────▼──────┐  ┌────▼────────┐
│  Database   │  │  Training   │  │  Models     │
│  (SQLite)   │  │  Manager    │  │  (PyTorch)  │
└─────────────┘  └─────┬──────┘  └─────┬────────┘
                       │                │
                ┌──────▼────────────────▼──────┐
                │    Training Script            │
                │    (src/train.py)             │
                └───────────────────────────────┘
```

---

## 📁 Critical Files & Their Roles

### Core Application Files

#### 1. **app.py** (487 lines)
**Purpose:** Main Flask web server with API endpoints and WebSocket support  
**Key Components:**
- Flask app initialization with SocketIO
- 5 HTML page routes: `/`, `/record`, `/train`, `/history`, `/dataset`
- API endpoints:
  - `/api/dataset/*` - Dataset management (upload, list, delete)
  - `/api/training/*` - Training control (start, stop, status, current)
  - `/api/stats` - Dashboard statistics
- WebSocket handlers:
  - `connect/disconnect` - Client connection management
  - `ping/pong` - Keepalive mechanism
  - `stop_training` - Emergency training stop
- Audio upload handling with WebM to WAV conversion (librosa)
- Training curves image serving
- Database integration for history

**Important Variables:**
```python
db = Database('data/app.db')  # SQLite database
training_manager = get_training_manager(socketio, db)  # Singleton training manager
DATA_DIR = Path('data/turkish_emotions')  # Dataset root
```

**Port:** 5001 (changed from 5000 to avoid conflicts)

#### 2. **training_manager.py** (246 lines)
**Purpose:** Background training execution and progress monitoring  
**Key Features:**
- Runs training in subprocess (isolated from web server)
- Real-time log parsing with regex patterns
- WebSocket emission of progress updates
- Training state management (singleton pattern)

**Critical Implementation Details:**
- Uses `venv/Scripts/python.exe` explicitly (not `sys.executable`)
- Python run with `-u` flag for unbuffered output
- Regex patterns for parsing train.py output:
  ```python
  epoch_pattern = r'Epoch (\d+)'
  acc_pattern = r'Average Accuracy: ([\d.]+)%'
  loss_pattern = r'Average Loss: ([\d.]+)'
  epoch_completed_pattern = r'Epoch (\d+) completed'
  ```
- Distinguishes training vs validation phase by "Validation" keyword in logs
- Emits `training_progress` event after each epoch completion

**Training Flow:**
1. Create database record with status='running'
2. Build command: `venv/Scripts/python.exe -u src/train.py --model <type> --config <path>`
3. Start subprocess with PIPE stdout
4. Parse each line in real-time
5. Emit `training_log` for console display
6. Emit `training_progress` when epoch completes
7. Update database with best_val_acc and final_epoch
8. Emit `training_completed` or `training_failed`

#### 3. **database.py** (319 lines)
**Purpose:** SQLite ORM wrapper for training history and dataset stats  
**Tables:**

**trainings:**
```sql
id INTEGER PRIMARY KEY
model_type TEXT (baseline/comparison)
start_time TEXT (ISO format)
end_time TEXT
status TEXT (running/completed/failed/stopped)
best_val_acc REAL
best_val_f1 REAL
final_epoch INTEGER
total_epochs INTEGER
training_time_minutes REAL
checkpoint_path TEXT
config JSON
created_at TEXT
```

**dataset_stats:**
```sql
id INTEGER PRIMARY KEY
total_files INTEGER
emotion_counts JSON
recorded_at TEXT
```

**Key Methods:**
- `add_training()` - Create new training record, returns ID
- `update_training()` - Update status, metrics, etc.
- `get_training(id)` - Get single training by ID
- `get_all_trainings()` - Get all trainings ordered by created_at DESC
- `add_dataset_stats()` - Snapshot dataset size
- `get_latest_stats()` - Get most recent dataset stats

---

### Deep Learning Files

#### 4. **src/train.py** (460 lines)
**Purpose:** Main training script for both models  
**Command Line Interface:**
```bash
python src/train.py --model baseline --config config/baseline_config.yaml
python src/train.py --model comparison --config config/comparison_config.yaml
```

**Training Pipeline:**
1. Load config from YAML
2. Set random seeds for reproducibility
3. Load dataset from `data/turkish_emotions/`
4. Split: train/val/test (ratios from config)
5. Create DataLoaders with `num_workers=0` (Windows compatibility)
6. Initialize model (baseline CNN or CNN-BiLSTM)
7. Setup optimizer (Adam), criterion (CrossEntropyLoss), scheduler (ReduceLROnPlateau)
8. Training loop:
   - `train_one_epoch()` - Forward pass, backward pass, optimizer step
   - `validate()` - Validation metrics (accuracy, F1, loss)
   - Early stopping check (patience from config)
   - Save best model and checkpoints
9. Final test evaluation
10. Save training curves PNG
11. Generate classification report

**Output Structure:**
```
checkpoints/baseline_cnn/
  ├── best_model.pth
  ├── checkpoint_epoch_1.pth
  ├── checkpoint_epoch_10.pth
  └── ...
logs/baseline_cnn/
  ├── training_curves.png
  └── training_log.txt
```

**Critical Notes:**
- Expects 7 emotion folders in `data/turkish_emotions/`
- Uses mel-spectrogram features (128 mel bins)
- Target length: 256 time frames
- No stratification (stratify=false) due to small dataset
- Early stopping patience: 10 epochs (default)

#### 5. **src/models/baseline_model.py** (124 lines)
**Architecture:** Pure CNN  
**Parameters:** 286,951  
**Structure:**
```python
Conv2d(1, 16, 3x3) → BN → ReLU → MaxPool(2x2) → Dropout(0.3)
Conv2d(16, 32, 3x3) → BN → ReLU → MaxPool(2x2) → Dropout(0.3)
Conv2d(32, 64, 3x3) → BN → ReLU → MaxPool(2x2) → Dropout(0.3)
AdaptiveAvgPool2d(4x4)
Flatten → Linear(1024, 128) → BN → ReLU → Dropout(0.3)
Linear(128, 7)
```

**Input:** (batch, 1, 128, 256) - mel-spectrogram  
**Output:** (batch, 7) - logits for 7 emotions

#### 6. **src/models/comparison_model.py** (180 lines)
**Architecture:** CNN + BiLSTM  
**Parameters:** 892,000  
**Structure:**
```python
Conv2d(1, 32, 3x3) → BN → ReLU → MaxPool(2x2) → Dropout(0.3)
Conv2d(32, 64, 3x3) → BN → ReLU → MaxPool(2x2) → Dropout(0.3)
Reshape to (batch, time, features)
BiLSTM(2 layers, hidden=128, bidirectional)
Take final hidden states (forward + backward)
Linear(256, 128) → BN → ReLU → Dropout(0.3)
Linear(128, 7)
```

**Key Difference:** BiLSTM processes time dimension to capture temporal dependencies

#### 7. **src/data_loader.py** (245 lines)
**Purpose:** Dataset loading, splitting, and DataLoader creation  
**Functions:**
- `load_turkish_ser_dataset()` - Scans emotion folders, returns paths and labels
- `split_dataset()` - Train/val/test split with optional stratification
- `create_data_loaders()` - Creates PyTorch DataLoaders

**Emotion Mapping:**
```python
emotion_map = {
    'mutlu': 0,    # happy
    'uzgun': 1,    # sad
    'kizgin': 2,   # angry
    'notr': 3,     # neutral
    'korku': 4,    # fear
    'saskin': 5,   # surprise
    'igrenme': 6   # disgust
}
```

#### 8. **src/features.py** (189 lines)
**Purpose:** Audio feature extraction  
**Key Function:** `extract_mel_spectrogram()`

**Process:**
1. Load audio with librosa (sr=16000)
2. Convert to mono if stereo
3. Compute mel-spectrogram (n_mels=128, n_fft=2048, hop_length=512)
4. Convert to log scale (log(S + 1e-9))
5. Pad/truncate to target_length frames (256)
6. Normalize to [0, 1]

**Output Shape:** (1, 128, 256) - single channel image-like tensor

---

### Configuration Files

#### 9. **config/baseline_config.yaml**
**Key Settings:**
```yaml
data:
  data_dir: "data/turkish_emotions"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  stratify: false  # Due to small dataset

features:
  type: "mel_spectrogram"
  sr: 16000
  n_mels: 128
  target_length: 256

model:
  type: "baseline"
  num_classes: 7
  dropout_rate: 0.3

training:
  batch_size: 8  # Small due to dataset size
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
  scheduler_patience: 5
  scheduler_factor: 0.5

hardware:
  num_workers: 0  # CRITICAL: Must be 0 on Windows
  device: "cpu"
```

#### 10. **config/comparison_config.yaml**
Similar structure, but:
```yaml
model:
  type: "comparison"
  lstm_hidden_size: 128
  lstm_num_layers: 2
  bidirectional: true
```

---

### Web Interface Files

#### 11. **web/templates/train.html** (364 lines)
**Purpose:** Training control page with real-time monitoring  
**Key Features:**
- Model selection (baseline vs comparison)
- Start/Stop training buttons
- Real-time progress bar (epoch/total_epochs)
- Live metrics display (train/val acc, loss)
- Console output with real-time logs
- Elapsed timer
- WebSocket connection with ping/pong keepalive

**Socket.IO Events Handled:**
```javascript
'connect' → checkTrainingStatus()
'disconnect' → show reconnecting message
'reconnect' → restore connection, check status
'training_log' → append to console output
'training_progress' → update progress bar, metrics
'training_completed' → show completion message
'training_failed' → show error message
'training_stopped' → show stopped message
'ping' → keep connection alive (every 25 seconds)
```

**Important Functions:**
- `checkTrainingStatus()` - Called on page load, fetches `/api/training/current`
- `showTrainingInProgress()` - Reconnects to ongoing training
- `updateProgress()` - Updates UI with latest metrics
- `addConsoleLog()` - Adds timestamped log to console output
- `startTimer()` - Updates elapsed time every second

#### 12. **web/templates/training_detail.html** (290 lines)
**Purpose:** Detailed results page for completed training  
**Route:** `/history/<training_id>`  
**Displays:**
- Training summary (status, accuracy, epochs, time)
- Performance analysis (automatic evaluation based on accuracy)
- Training curves graph (PNG from logs/)
- Recommendations (dataset size, model selection, etc.)
- Full configuration JSON

**JavaScript Analysis Logic:**
```javascript
if (accuracy >= 70%) → "Excellent Performance" 🎉
else if (accuracy >= 50%) → "Good Performance" 👍
else → "Needs Improvement" ⚠️

Always recommend:
1. Collect more data (350+ samples)
2. Data augmentation
3. Balance dataset
4. Try advanced model if using baseline
5. Record in quiet environment
6. Longer audio duration (3-5 seconds)
```

#### 13. **web/templates/history.html** (229 lines)
**Purpose:** Training history list with filtering  
**Features:**
- Cards for each training session (clickable → detail page)
- Filter by model type and status
- Shows: ID, model, date, accuracy, F1, epochs, duration
- Color-coded status badges
- Hover effect for cards

#### 14. **web/templates/dataset.html** (295 lines)
**Purpose:** Dataset browser and management  
**Features:**
- Tabs for each emotion (7 emotions)
- File list with play button (HTML5 audio)
- Delete button with confirmation
- Total count per emotion
- Audio player with waveform visualization

**API Calls:**
- `GET /api/dataset/files` - Get all files grouped by emotion
- `DELETE /api/dataset/file/<emotion>/<filename>` - Delete file
- `POST /api/dataset/upload` - Upload new recording

#### 15. **web/templates/record.html** (380 lines)
**Purpose:** Audio recording interface  
**Features:**
- Browser-based audio recording (MediaRecorder API)
- Emotion selection dropdown
- Visual waveform during recording
- Playback before saving
- Upload to server with automatic WAV conversion

**Recording Flow:**
1. Request microphone permission
2. Create MediaStream and MediaRecorder
3. Record audio chunks (WebM format)
4. Stop recording → create Blob
5. Play preview
6. Upload to `/api/dataset/upload`
7. Backend converts WebM → WAV using librosa
8. Save to `data/turkish_emotions/<emotion>/`

**Known Issue:** WebM recordings sometimes corrupt, conversion added to fix

#### 16. **web/static/css/style.css** (450 lines)
**Purpose:** Dark mode UI styling  
**Theme Colors:**
```css
--bg-primary: #0a0a0a;
--bg-secondary: #121212;
--bg-tertiary: #1a1a1a;
--text-primary: #ffffff;
--text-secondary: #b0b0b0;
--accent-primary: #3b82f6;  /* blue */
--accent-secondary: #8b5cf6; /* purple */
--accent-success: #10b981;   /* green */
--accent-warning: #f59e0b;   /* orange */
--accent-error: #ef4444;     /* red */
```

---

## 🔄 Data Flow

### Training Request Flow

```
User clicks "Start Training" on /train
    ↓
JavaScript POST to /api/training/start
    ↓
app.py creates database record (status=running)
    ↓
training_manager.start_training() spawns subprocess
    ↓
subprocess runs: venv/Scripts/python.exe -u src/train.py
    ↓
training_manager monitors stdout line by line
    ↓
Regex parsing → emit 'training_log' and 'training_progress'
    ↓
train.html receives WebSocket events → updates UI
    ↓
Training completes → database updated (status=completed)
    ↓
training_manager emits 'training_completed'
    ↓
UI shows completion message
```

### Audio Upload Flow

```
User records in browser
    ↓
MediaRecorder produces WebM blob
    ↓
JavaScript FormData POST to /api/dataset/upload
    ↓
app.py receives file + emotion
    ↓
Save temp file
    ↓
librosa loads audio → converts to WAV
    ↓
Save to data/turkish_emotions/<emotion>/<timestamp>.wav
    ↓
Return success
    ↓
JavaScript updates UI (shows new file count)
```

---

## 🐛 Known Issues & Solutions

### Issue 1: Port 5000 Already in Use
**Symptom:** `OSError [WinError 10048]`  
**Solution:** Changed port to 5001 in app.py line 472

### Issue 2: PyTorch DataLoader Multiprocessing on Windows
**Symptom:** `PermissionError [WinError 5]` during training  
**Cause:** Windows doesn't handle multiprocessing well  
**Solution:** Set `num_workers: 0` in both config YAMLs

### Issue 3: Training subprocess uses wrong Python
**Symptom:** `ModuleNotFoundError: No module named 'torch'`  
**Cause:** `sys.executable` pointed to system Python, not venv  
**Solution:** Explicitly use `venv/Scripts/python.exe` in training_manager.py

### Issue 4: Training progress not showing in UI
**Symptom:** Epoch stays at 0, no updates  
**Cause:** Regex patterns didn't match training script output format  
**Solution:** Updated patterns to detect "Epoch X completed" and distinguish training/validation phases

### Issue 5: WebSocket disconnects during long training
**Symptom:** Client repeatedly connects/disconnects  
**Cause:** No keepalive, connection times out  
**Solution:** Added ping/pong every 25 seconds in train.html

### Issue 6: Buffered output delays logs
**Symptom:** Logs appear in batches, not real-time  
**Solution:** Add `-u` flag to python command for unbuffered output

### Issue 7: Elapsed time resets on page refresh
**Symptom:** Timer starts from 0:00:00 when page reloads  
**Solution:** Store start_time in database, retrieve via `/api/training/current`

### Issue 8: WebM recordings unplayable/corrupt
**Symptom:** `LibsndfileError: Format not recognised`  
**Solution:** Backend converts WebM to WAV using librosa in upload endpoint

---

## 📊 Dataset Structure

### Current State
```
data/turkish_emotions/
├── mutlu/      (3 files)
├── uzgun/      (3 files)
├── kizgin/     (3 files)
├── notr/       (3 files)
├── korku/      (3 files)
├── saskin/     (3 files)
└── igrenme/    (3 files)
Total: 21 files
```

### Target State
```
Each emotion: 50-100 samples
Total: 350-700 samples
```

### File Format Requirements
- Format: WAV (PCM)
- Sample rate: 16 kHz
- Channels: Mono
- Bit depth: 16-bit
- Naming: `<emotion>_<timestamp>.wav`

---

## 🔧 Configuration Variables

### Critical Environment-Specific Settings

**app.py:**
- `port=5001` - Changed from 5000
- `host='0.0.0.0'` - Listen on all interfaces
- `debug=False` - Production mode
- `allow_unsafe_werkzeug=True` - Allow SocketIO with Werkzeug

**training_manager.py:**
- `venv_python = Path(__file__).parent / 'venv' / 'Scripts' / 'python.exe'`
- Must use Windows path with `Scripts` (not `bin`)

**Config YAMLs:**
- `num_workers: 0` - MUST be 0 on Windows
- `batch_size: 8` - Small due to dataset size
- `stratify: false` - Can't stratify with 3 samples per class

---

## 🚀 Startup Sequence

### Manual Start (User)
```bash
cd C:\Users\IlhanBal\Desktop\İlhan\sen4107
.\venv\Scripts\Activate
python app.py
```

### What Happens:
1. Flask app initializes
2. SocketIO attached
3. Database initialized (creates tables if not exist)
4. TrainingManager singleton created
5. Routes registered
6. Server starts on http://localhost:5001
7. Web interface accessible

### Dependencies Check:
```python
import torch  # 2.9.1
import librosa  # 0.11.0
import flask  # 3.0.0
import flask_socketio  # 5.3.5
import numpy  # 2.x
import sklearn  # 1.8.0
import soundfile
import pyyaml
import tqdm
```

---

## 📈 Performance Metrics

### Current Training Results (21 samples)
- Best validation accuracy: **33.33%**
- Training time: **~0.5 minutes**
- Early stopping: **Epoch 21**
- Reason: Dataset too small, no improvement detected

### Expected Performance (350+ samples)
- Target accuracy: **70-85%**
- Training time: **20-30 minutes**
- Epochs: **50-100**

---

## 🎯 Development Priorities

### High Priority
1. **Collect more data** - Need 350+ samples
2. **Data augmentation** - Time stretching, pitch shifting
3. **Balance dataset** - Equal samples per emotion
4. **Test with larger dataset** - Validate model performance

### Medium Priority
1. **Add inference endpoint** - `/api/predict` for real-time prediction
2. **Export models** - ONNX format for deployment
3. **Add visualization** - Attention maps, feature importance
4. **Improve error handling** - Better user feedback

### Low Priority
1. **Multi-GPU support** - For faster training
2. **Hyperparameter tuning** - Grid search
3. **Model ensemble** - Combine predictions
4. **Docker deployment** - Containerization

---

## 🧪 Testing Status

### What's Tested
✅ Training completes without errors  
✅ Real-time progress updates work  
✅ Database records training history  
✅ Web interface loads all pages  
✅ Audio upload and conversion works  
✅ Dataset browser shows files  
✅ Training can be stopped mid-execution  

### What Needs Testing
⚠️ Performance with large dataset (100+ samples)  
⚠️ Multiple concurrent training sessions  
⚠️ Edge cases (empty dataset, corrupt audio)  
⚠️ Long-running training (hours)  
⚠️ Inference on new audio samples  

---

## 🔐 Security Considerations

### Current State (Development)
- No authentication
- Local access only (localhost:5001)
- File uploads unrestricted
- Database is SQLite file (no password)

### Production Requirements (If Deploying)
- Add user authentication
- Restrict file upload size/type
- Rate limiting on API endpoints
- HTTPS/SSL
- Environment variables for secrets
- Input validation/sanitization

---

## 📚 Code Patterns & Conventions

### Python Style
- PEP 8 compliant
- Type hints used sparingly
- Docstrings for all functions
- Comments explain "why", not "what"

### JavaScript Style
- ES6+ features (const, arrow functions)
- Async/await for API calls
- Event-driven with Socket.IO
- Functional where possible

### File Organization
- `src/` - Core ML code
- `web/` - Frontend assets
- `config/` - Configuration files
- `data/` - Dataset storage
- `checkpoints/` - Model weights
- `logs/` - Training outputs

---

## 🔄 State Management

### Application State
- **Training Manager:** Singleton, tracks current training
- **Database:** Persistent storage for history
- **SocketIO:** Real-time state synchronization

### Training State Machine
```
IDLE → RUNNING → COMPLETED
  ↓       ↓          ↑
  ↓    FAILED ───────┘
  ↓       ↓
  └→ STOPPED
```

### WebSocket Connection State
```
DISCONNECTED → CONNECTED → AUTHENTICATED
      ↑            ↓
      └────────────┘
         RECONNECT
```

---

## 🆘 Debugging Guide

### Enable Verbose Logging
```python
# app.py - add at top
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Training Process
```bash
# Windows
Get-Process python | Where-Object {$_.Path -like "*venv*"}
```

### View Database Contents
```python
import sqlite3
conn = sqlite3.connect('data/app.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM trainings")
print(cursor.fetchall())
```

### Test Audio Processing
```python
import librosa
audio, sr = librosa.load('data/turkish_emotions/mutlu/file.wav', sr=16000)
print(f"Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}")
```

### Monitor WebSocket Events
```javascript
// In browser console
socket.on('training_log', (data) => console.log('LOG:', data));
socket.on('training_progress', (data) => console.log('PROGRESS:', data));
```

---

## 📖 Academic Context

### Baseline Paper
**Speech Emotion Recognition Using Deep 1D & 2D CNN LSTM Networks**  
Zhao, Jianfeng, et al. (2019)  
Biomedical Signal Processing and Control, Elsevier  
Citations: 1328+

### Course Information
- Course: SEN4107 Neural Networks
- Semester: Fall 2025
- Team: İlhan Bal, Ali Yangın
- Institution: [University Name]

### Project Requirements Met
✅ Two model architectures (CNN, CNN-BiLSTM)  
✅ PyTorch implementation from scratch  
✅ Configuration-driven training  
✅ Comprehensive documentation  
✅ Web interface for demonstration  
✅ Training history and visualization  
✅ Academic report structure  

---

## 🔮 Future Enhancements

### Planned Features
1. **Real-time Inference API** - Predict emotion from uploaded/recorded audio
2. **Data Augmentation Pipeline** - Generate synthetic samples
3. **Model Comparison Dashboard** - Side-by-side metrics
4. **Attention Visualization** - Show which audio segments are important
5. **Multi-language Support** - Extend to other languages
6. **Transfer Learning** - Fine-tune pre-trained models
7. **Mobile App** - React Native or Flutter client
8. **Cloud Deployment** - AWS/Azure/GCP hosting

---

## 🤝 Collaboration Notes

### Git Workflow (When Set Up)
```bash
# Clone
git clone https://github.com/ilhanbal/turkish-speech-emotion-recognition.git

# Branch strategy
main - stable releases
dev - active development
feature/* - new features
fix/* - bug fixes

# Commit message format
[TYPE] Short description
- TYPE: feat, fix, docs, style, refactor, test, chore
```

### Code Review Checklist
- [ ] Follows existing code style
- [ ] Includes docstrings
- [ ] Updates README if needed
- [ ] Tests pass (when implemented)
- [ ] No hardcoded paths
- [ ] Config-driven where possible

---

## 💾 Backup & Recovery

### Critical Files to Backup
1. `data/app.db` - Training history
2. `data/turkish_emotions/` - Audio dataset
3. `checkpoints/` - Trained models
4. `config/` - Training configurations
5. `src/` - Source code

### Recovery Procedure
1. Restore files to project directory
2. Verify venv exists: `.\venv\Scripts\Activate`
3. Check dependencies: `pip list`
4. Test database: `python -c "from database import Database; db = Database()"`
5. Start server: `python app.py`

---

## 🎓 Learning Resources

### Understanding the Code
1. Start with `src/train.py` - main training loop
2. Read `src/models/baseline_model.py` - simple CNN
3. Study `training_manager.py` - subprocess management
4. Explore `app.py` - Flask routing and APIs
5. Examine `web/templates/train.html` - real-time UI

### PyTorch Concepts Used
- nn.Module subclassing
- DataLoader and Dataset
- Optimizer (Adam) and Scheduler (ReduceLROnPlateau)
- CrossEntropyLoss
- Model checkpointing
- Device management (CPU/GPU)

### Audio Processing Concepts
- Mel-spectrogram extraction
- Sample rate conversion
- Audio normalization
- Padding/truncation

---

## ⚡ Quick Commands Reference

```bash
# Activate environment
.\venv\Scripts\Activate

# Start server
python app.py

# Train baseline model
python src/train.py --model baseline --config config/baseline_config.yaml

# Train comparison model  
python src/train.py --model comparison --config config/comparison_config.yaml

# Check dataset size
Get-ChildItem -Recurse data\turkish_emotions\*.wav | Measure-Object

# Stop all Python processes
Get-Process python | Stop-Process -Force

# View logs
Get-Content logs\baseline_cnn\training_log.txt -Tail 50

# Check database
sqlite3 data\app.db "SELECT * FROM trainings ORDER BY id DESC LIMIT 5"
```

---

## 🎯 Critical Success Factors

1. **Dataset Quality** - Clean, diverse, balanced samples
2. **Training Stability** - No crashes, proper error handling
3. **Real-time Monitoring** - Live progress updates working
4. **Model Performance** - >70% accuracy on test set
5. **User Experience** - Intuitive web interface
6. **Code Quality** - Well-documented, maintainable
7. **Reproducibility** - Consistent results with same config

---

## 🔍 Troubleshooting Checklist

**Server Won't Start:**
- [ ] Virtual environment activated?
- [ ] All dependencies installed? (`pip list`)
- [ ] Port 5001 available? (`netstat -ano | findstr :5001`)
- [ ] Database file accessible? (check `data/` exists)

**Training Fails:**
- [ ] Dataset exists? (`data/turkish_emotions/`)
- [ ] Audio files are WAV format?
- [ ] Config file valid YAML?
- [ ] GPU/CUDA issues? (try `device: cpu` in config)
- [ ] Enough disk space? (checkpoints can be large)

**UI Not Updating:**
- [ ] WebSocket connected? (check browser console)
- [ ] Training process running? (`Get-Process python`)
- [ ] Firewall blocking WebSocket?
- [ ] Browser supports WebSocket? (modern browsers only)

**Audio Upload Issues:**
- [ ] Microphone permission granted?
- [ ] Audio format supported by browser?
- [ ] File size reasonable? (<10MB per file)
- [ ] Directory writable? (`data/turkish_emotions/`)

---

## 📞 Contact & Support

**For AI Agents:**
- This document should be sufficient to understand and continue development
- Check code comments for inline documentation
- Refer to `README.md` for user-facing documentation
- Database schema is self-documenting (see database.py)

**For Humans:**
- GitHub: @ilhanbal
- Project: turkish-speech-emotion-recognition
- Course: SEN4107

---

**Document Version:** 1.0  
**Last Updated:** December 12, 2025  
**Maintained By:** AI Agent Context System  
**Next Review:** Upon major architectural changes

---

## 🤖 AI Agent Handoff Checklist

When transferring this project to another AI agent:

✅ Read this entire AI-README.md  
✅ Review project structure in `README.md`  
✅ Understand training flow (app.py → training_manager.py → train.py)  
✅ Check current dataset size (`data/turkish_emotions/`)  
✅ Review recent training history (database or `/history` page)  
✅ Test server startup (`python app.py`)  
✅ Verify dependencies installed (`pip list`)  
✅ Check for pending issues (Known Issues section above)  
✅ Review Git status if repository exists  
✅ Understand user's current goals (data collection? model improvement?)

**Ready to Continue Development!** 🚀

