# AI Agent Context - Turkish Speech Emotion Recognition Project

## 🤖 For AI Agents Working on This Project

Bu doküman, Türkçe Konuşma Duygu Tanıma (Speech Emotion Recognition - SER) sistemi üzerinde çalışacak AI ajanları için kapsamlı teknik bağlam sağlar.

---

## 📋 Proje Özeti

| Özellik | Değer |
|---------|-------|
| **Tip** | Deep Learning - Speech Emotion Recognition |
| **Dil** | Türkçe |
| **Framework** | PyTorch 2.9.1 + Flask 3.0 |
| **Mimari** | Real-time training monitoring ile web tabanlı ML sistemi |
| **Durum** | Aktif geliştirme - Veri toplama aşaması |
| **Dataset** | Özel toplanan, 7 duygu kategorisi |
| **Port** | 5001 (localhost) |

---

## 🚀 Hızlı Başlangıç

### Projeyi Çalıştırma

```powershell
# 1. Proje dizinine git
cd c:\Users\Hunter\Desktop\sen4107

# 2. Sunucuyu başlat
python app.py

# 3. Tarayıcıda aç
# http://localhost:5001
```

### Bağımlılıkları Yükleme (İlk kurulum)

```powershell
# Flask ve web bağımlılıkları
python -m pip install flask flask-socketio flask-cors pyyaml eventlet

# PyTorch ve ML bağımlılıkları
python -m pip install torch torchaudio librosa pandas numpy scikit-learn seaborn matplotlib tensorboard pydub resampy
```

### Sunucuyu Durdurma ve Yeniden Başlatma

```powershell
# Tüm Python işlemlerini durdur
Get-Process python | Stop-Process -Force

# Yeniden başlat
python app.py
```

---

## 🏗️ Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WEB INTERFACE (Flask)                         │
│     Dashboard  │  Record  │  Train  │  History  │  Dataset          │
│     (index)    │  (kayıt) │ (eğitim)│ (geçmiş)  │  (veri seti)      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼─────────┐         ┌────────▼─────────┐
     │   Flask Backend  │◄───────►│    Socket.IO     │
     │     (app.py)     │         │   (Real-time)    │
     └────────┬─────────┘         └────────┬─────────┘
              │                             │
    ┌─────────┼─────────────────────────────┤
    │         │                             │
┌───▼───┐ ┌───▼────────────┐ ┌─────────────▼───────────┐
│SQLite │ │TrainingManager │ │    PyTorch Models       │
│  DB   │ │(Background)    │ │ (CNN / CNN-BiLSTM)      │
└───────┘ └───────┬────────┘ └─────────────────────────┘
                  │
          ┌───────▼───────┐
          │  train.py     │
          │ (subprocess)  │
          └───────────────┘
```

---

## 📁 Dosya Yapısı

```
sen4107/
├── app.py                      # Ana Flask sunucusu (601 satır)
├── training_manager.py         # Arka plan training yöneticisi (272 satır)
├── database.py                 # SQLite ORM wrapper (374 satır)
├── requirements_web.txt        # Web bağımlılıkları
├── AI-README.md               # Bu dosya
├── README.md                  # Proje README
│
├── config/
│   ├── baseline_config.yaml   # Baseline CNN konfigürasyonu
│   └── comparison_config.yaml # CNN-BiLSTM konfigürasyonu
│
├── src/
│   ├── train.py               # Ana training scripti (460 satır)
│   ├── datasets.py            # Dataset ve DataLoader yönetimi
│   ├── features.py            # Mel-spectrogram özellik çıkarımı
│   ├── eval.py                # Değerlendirme metrikleri
│   ├── utils.py               # Yardımcı fonksiyonlar
│   └── models/
│       ├── baseline_model.py  # Saf CNN modeli (287K parametre)
│       └── comparison_model.py # CNN-BiLSTM modeli (892K parametre)
│
├── web/
│   ├── templates/
│   │   ├── index.html         # Dashboard sayfası
│   │   ├── record.html        # Ses kayıt sayfası
│   │   ├── train.html         # Training sayfası (canlı izleme)
│   │   ├── history.html       # Training geçmişi
│   │   ├── training_detail.html # Training detay sayfası
│   │   └── dataset.html       # Dataset tarayıcısı
│   └── static/
│       └── css/
│           └── style.css      # Modern dark theme CSS
│
├── data/
│   ├── app.db                 # SQLite veritabanı
│   └── turkish_emotions/      # Dataset klasörü
│       ├── mutlu/             # 😊 Mutlu ses kayıtları
│       ├── uzgun/             # 😢 Üzgün ses kayıtları
│       ├── kizgin/            # 😠 Kızgın ses kayıtları
│       ├── notr/              # 😐 Nötr ses kayıtları
│       ├── korku/             # 😨 Korku ses kayıtları
│       ├── saskin/            # 😲 Şaşkın ses kayıtları
│       └── igrenme/           # 🤢 İğrenme ses kayıtları
│
├── checkpoints/               # Eğitilmiş model kayıtları
│   ├── baseline_cnn/
│   │   ├── best_model.pth
│   │   └── checkpoint_epoch_*.pth
│   └── cnn_bilstm/
│       ├── best_model.pth
│       └── checkpoint_epoch_*.pth
│
└── logs/                      # TensorBoard logları
    ├── baseline_cnn/
    └── cnn_bilstm/
```

---

## 🔌 API Endpoints

### Sayfa Rotaları

| Route | Sayfa | Açıklama |
|-------|-------|----------|
| `/` | index.html | Dashboard - genel bakış |
| `/record` | record.html | Ses kayıt arayüzü |
| `/train` | train.html | Model eğitimi |
| `/history` | history.html | Training geçmişi listesi |
| `/history/<id>` | training_detail.html | Belirli training detayı |
| `/dataset` | dataset.html | Dataset tarayıcısı |

### REST API Endpoints

#### Dataset Yönetimi
```
GET  /api/dataset/files          # Tüm ses dosyalarını listele (emotion bazlı)
GET  /api/dataset/audio/<e>/<f>  # Belirli ses dosyasını oynat
POST /api/dataset/upload         # Yeni ses kaydı yükle (FormData: file, emotion)
POST /api/dataset/delete         # Ses dosyası sil (JSON: emotion, filename)
GET  /api/stats                  # Dashboard istatistikleri
```

#### Training Yönetimi
```
POST /api/training/start         # Training başlat (JSON: model_type)
GET  /api/training/current       # Aktif training durumu
GET  /api/training/history       # Training geçmişi (query: limit)
GET  /api/training/<id>          # Belirli training detayı
DELETE /api/training/<id>        # Training kaydını sil
DELETE /api/training/all         # Tüm training kayıtlarını sil
```

### WebSocket Events

#### Server → Client
```javascript
'training_started'    // Training başladı
'training_log'        // Console log satırı
'training_progress'   // Epoch metrikleri (acc, loss, epoch)
'training_completed'  // Training tamamlandı (training_id ile)
'training_failed'     // Training başarısız
'training_stopped'    // Training durduruldu
```

#### Client → Server
```javascript
'connect'            // Bağlantı kuruldu
'disconnect'         // Bağlantı kesildi
'stop_training'      // Training durdur
'ping'               // Keepalive (her 25 saniye)
```

---

## 🧠 Model Mimarileri

### 1. Baseline CNN (287K parametre)

```
Input: (batch, 1, 128, 256) - Mel-spectrogram

Conv2d(1→16, 3x3) → BatchNorm → ReLU → MaxPool(2x2) → Dropout(0.3)
Conv2d(16→32, 3x3) → BatchNorm → ReLU → MaxPool(2x2) → Dropout(0.3)
Conv2d(32→64, 3x3) → BatchNorm → ReLU → MaxPool(2x2) → Dropout(0.3)
AdaptiveAvgPool2d(4x4)
Flatten → Linear(1024→128) → BatchNorm → ReLU → Dropout(0.3)
Linear(128→7)

Output: (batch, 7) - 7 duygu için logits
```

### 2. CNN-BiLSTM (892K parametre)

```
Input: (batch, 1, 128, 256) - Mel-spectrogram

Conv2d(1→32, 3x3) → BatchNorm → ReLU → MaxPool(2x2) → Dropout(0.3)
Conv2d(32→64, 3x3) → BatchNorm → ReLU → MaxPool(2x2) → Dropout(0.3)
Reshape → (batch, time_steps, features)
BiLSTM(2 layers, hidden=128, bidirectional=True)
Take final hidden states (forward + backward)
Linear(256→128) → BatchNorm → ReLU → Dropout(0.3)
Linear(128→7)

Output: (batch, 7) - 7 duygu için logits
```

---

## 🎯 Duygu Kategorileri

```python
EMOTIONS = {
    'mutlu':   {'id': 0, 'icon': '😊', 'name': 'Mutlu',    'color': '#10b981'},
    'uzgun':   {'id': 1, 'icon': '😢', 'name': 'Üzgün',    'color': '#3b82f6'},
    'kizgin':  {'id': 2, 'icon': '😠', 'name': 'Kızgın',   'color': '#ef4444'},
    'notr':    {'id': 3, 'icon': '😐', 'name': 'Nötr',     'color': '#6b7280'},
    'korku':   {'id': 4, 'icon': '😨', 'name': 'Korku',    'color': '#8b5cf6'},
    'saskin':  {'id': 5, 'icon': '😲', 'name': 'Şaşkın',   'color': '#f59e0b'},
    'igrenme': {'id': 6, 'icon': '🤢', 'name': 'İğrenme',  'color': '#14b8a6'}
}
```

---

## 🔄 Önemli Akışlar

### Training Akışı

```
1. Kullanıcı /train sayfasında model seçer
2. "Start Training" → Modal onay
3. POST /api/training/start (model_type: 'baseline' veya 'comparison')
4. app.py → training_manager.start_training()
5. Yeni thread'de subprocess başlatılır:
   python -u src/train.py --model <type> --config <path>
6. training_manager stdout'u satır satır okur
7. Regex ile epoch, accuracy, loss parse edilir
8. Socket.IO ile 'training_progress' emit edilir
9. train.html canlı güncellenir (epoch sayısı, grafikler)
10. Training bitince 'training_completed' emit edilir
11. 2 saniye sonra /history/<training_id> sayfasına yönlendirilir
```

### Ses Kayıt Akışı

```
1. Kullanıcı /record sayfasında duygu seçer
2. "Start Recording" → Mikrofon izni istenir
3. Web Audio API ile ses kaydedilir (WAV format, 16kHz, mono)
4. ScriptProcessorNode ile PCM data toplanır
5. "Stop" → WAV Blob oluşturulur
6. Önizleme ile dinleme
7. "Save" → FormData ile POST /api/dataset/upload
8. Sunucu dosyayı data/turkish_emotions/<emotion>/ klasörüne kaydeder
9. Başarı bildirimi gösterilir
```

### Epoch ve Early Stopping

```
- Training 100 epoch için planlanır
- Her epoch'ta train ve validation yapılır
- Validation accuracy iyileşmezse patience sayacı artar
- 20 epoch boyunca iyileşme olmazsa training erken durur
- Best model her iyileşmede kaydedilir
- Örnek: 25/100 epoch = 25 epoch sonunda erken durdu
```

---

## 💾 Veritabanı Şeması

### trainings Tablosu

```sql
CREATE TABLE trainings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,           -- 'baseline' veya 'comparison'
    start_time TEXT NOT NULL,           -- ISO format (Istanbul UTC+3)
    end_time TEXT,                      -- ISO format
    status TEXT DEFAULT 'running',      -- running/completed/failed/stopped
    best_val_acc REAL,                  -- En iyi validation accuracy (0-1)
    best_val_f1 REAL,                   -- En iyi validation F1 score (0-1)
    final_epoch INTEGER,                -- Son epoch numarası
    total_epochs INTEGER,               -- Planlanan toplam epoch
    training_time_minutes REAL,         -- Eğitim süresi (dakika)
    checkpoint_path TEXT,               -- Model dosya yolu
    config JSON,                        -- Training konfigürasyonu
    created_at TEXT                     -- Oluşturulma zamanı (Istanbul UTC+3)
);
```

### dataset_stats Tablosu

```sql
CREATE TABLE dataset_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_files INTEGER,                -- Toplam ses dosyası sayısı
    emotion_counts JSON,                -- Duygu bazlı sayılar
    recorded_at TEXT                    -- Kayıt zamanı
);
```

---

## ⚙️ Konfigürasyon Parametreleri

### config/baseline_config.yaml

```yaml
data:
  data_dir: "data/turkish_emotions"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  stratify: false           # Küçük dataset için false

features:
  type: "mel_spectrogram"
  sr: 16000                 # Sample rate
  n_mels: 128               # Mel band sayısı
  n_fft: 2048               # FFT pencere boyutu
  hop_length: 512           # Hop uzunluğu
  target_length: 256        # Hedef frame sayısı

model:
  type: "baseline"
  num_classes: 7
  dropout_rate: 0.3

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 20   # 20 epoch sabır
  scheduler_patience: 5
  scheduler_factor: 0.5

hardware:
  num_workers: 0            # Windows için 0 olmalı!
  device: "cpu"             # veya "cuda" GPU için
```

---

## 🎨 Frontend Teknolojileri

### Kullanılan Kütüphaneler

| Kütüphane | Versiyon | Kullanım |
|-----------|----------|----------|
| Chart.js | 4.4.0 | Canlı training grafikleri |
| WaveSurfer.js | 7.x | Ses dalga formu görselleştirme |
| Socket.IO Client | 4.5.4 | Real-time WebSocket iletişimi |
| Inter Font | - | UI fontu |
| JetBrains Mono | - | Console/kod fontu |

### CSS Tema Değişkenleri

```css
:root {
    --bg-primary: #0a0a0a;
    --bg-secondary: #121212;
    --bg-tertiary: #1a1a1a;
    --bg-card: rgba(26, 26, 26, 0.8);
    --text-primary: #ffffff;
    --text-secondary: #b0b0b0;
    --text-tertiary: #666666;
    --accent-primary: #3b82f6;    /* Mavi */
    --accent-secondary: #8b5cf6;  /* Mor */
    --accent-success: #10b981;    /* Yeşil */
    --accent-warning: #f59e0b;    /* Turuncu */
    --accent-error: #ef4444;      /* Kırmızı */
}
```

---

## 🐛 Bilinen Sorunlar ve Çözümler

### 1. Port 5000 Kullanımda
**Belirti:** `OSError [WinError 10048]`  
**Çözüm:** Port 5001 kullanılıyor (app.py son satırları)

### 2. Windows'ta DataLoader Multiprocessing
**Belirti:** `PermissionError [WinError 5]`  
**Çözüm:** `num_workers: 0` config dosyalarında

### 3. pip Komutu Tanınmıyor
**Belirti:** `pip: The term 'pip' is not recognized`  
**Çözüm:** `python -m pip install ...` kullan

### 4. Training Çok Hızlı Bitiyor
**Sebep:** Early stopping + küçük dataset  
**Açıklama:** 20 epoch iyileşme olmazsa durur, normal davranış

### 5. F1 Score / Duration Boş Görünüyor
**Çözüm:** Veritabanını sil ve yeniden training yap:
```powershell
Remove-Item "data/app.db" -Force
python app.py
```

### 6. Ses Kaydı Çalışmıyor
**Sebep:** Eski WebM format  
**Çözüm:** WAV formatına geçildi (Web Audio API)

---

## 📊 Training Çıktı Formatı

train.py şu formatta çıktı üretir (training_manager bu formatı parse eder):

```
🚀 Starting training...
Model: baseline
Config: config/baseline_config.yaml

Epoch 1 Training:
  Batch 1/3 - Loss: 1.9456, Acc: 14.29%
  Average Loss: 1.8234
  Average Accuracy: 28.57%

Epoch 1 Validation Summary:
  Average Loss: 1.7891
  Average Accuracy: 33.33%
  Macro F1 Score: 25.45%

Epoch 1 completed in 2.34s
✅ Best model saved (val_acc: 33.33%)

... (epoch 2-N)

🎉 Training completed!
Best Validation Accuracy: 45.67%
Total Training Time: 1.5 minutes
```

---

## 🔧 Geliştirme Notları

### Yeni Özellik Eklerken

1. **Backend değişikliği:** app.py veya training_manager.py düzenle
2. **Frontend değişikliği:** web/templates/*.html düzenle
3. **Model değişikliği:** src/models/*.py düzenle
4. **Config değişikliği:** config/*.yaml düzenle

### Sunucuyu Yeniden Başlatma Gerektiren Değişiklikler

- Python dosyalarındaki herhangi bir değişiklik
- Config dosyalarındaki değişiklikler (training için)

### Sunucuyu Yeniden Başlatma Gerektirmeyen Değişiklikler

- HTML template değişiklikleri (sayfa yenilemesi yeterli)
- CSS değişiklikleri (sayfa yenilemesi yeterli)
- JavaScript değişiklikleri (sayfa yenilemesi yeterli)

---

## 🎯 Proje Hedefleri

1. **Veri Toplama:** Her duygu için 50+ ses kaydı (toplam 350+)
2. **Model Eğitimi:** %70+ validation accuracy
3. **Real-time Inference:** Kaydedilen sesi anında sınıflandırma
4. **Karşılaştırma:** Baseline CNN vs CNN-BiLSTM performans analizi

---

## 📝 Son Güncelleme

**Tarih:** 12 Aralık 2025  
**Değişiklikler:**
- Ses kaydı WebM'den WAV formatına geçirildi
- Training tamamlandığında otomatik detail sayfasına yönlendirme
- F1 Score ve Duration veritabanına kaydediliyor
- Istanbul timezone (UTC+3) eklendi
- Dataset sayfası modernleştirildi (scroll özellikli kartlar)
- Progress bar kaldırıldı, sadece epoch sayısı gösteriliyor

---

## 🤖 AI Agent İçin Öneriler

Bu proje üzerinde çalışırken:

1. **Değişiklik yapmadan önce** ilgili dosyayı oku ve anla
2. **Python değişikliklerinden sonra** sunucuyu yeniden başlat
3. **Veritabanı sorunlarında** `data/app.db` dosyasını sil
4. **Training test etmek için** küçük bir ses kaydı yap ve baseline model ile dene
5. **Frontend değişikliklerini** tarayıcıda F5 ile kontrol et
6. **Hata ayıklama için** terminal çıktısını kontrol et

---

*Bu doküman, projenin mevcut durumunu yansıtmaktadır. Önemli değişikliklerden sonra güncellenmesi önerilir.*

