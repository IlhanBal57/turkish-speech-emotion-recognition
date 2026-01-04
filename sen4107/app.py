"""
Flask Web Application - Turkish Speech Emotion Recognition
Main application file with API endpoints and WebSocket support
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from pathlib import Path
import json
import os
import time
from datetime import datetime
from database import Database
from training_manager import get_training_manager

# Initialize Flask app
app = Flask(__name__, 
            template_folder='web/templates',
            static_folder='web/static')
app.config['SECRET_KEY'] = 'turkish-ser-secret-key-2025'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Enable CORS
CORS(app)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize Database
db = Database('data/app.db')

# Global training manager
training_manager = None


# ============================================================================
# ROUTES - WEB PAGES
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/record')
def record():
    """Audio recording page."""
    return render_template('record.html')


@app.route('/train')
def train_page():
    """Training page."""
    return render_template('train.html')


@app.route('/dataset')
def dataset_page():
    """Dataset browser page."""
    return render_template('dataset.html')


@app.route('/history')
def history():
    """Training history page."""
    return render_template('history.html')


@app.route('/history/<int:training_id>')
def training_detail(training_id):
    """Detailed training results page."""
    training = db.get_training(training_id)
    if not training:
        return "Training not found", 404
    return render_template('training_detail.html', training=training)


@app.route('/api/training/<int:training_id>/curves')
def get_training_curves(training_id):
    """Serve training curves image."""
    training = db.get_training(training_id)
    if not training:
        return "Training not found", 404
    
    # Find curves image
    model_type = training['model_type']
    if model_type == 'baseline':
        curves_path = Path('logs/baseline_cnn/training_curves.png')
    else:
        curves_path = Path('logs/cnn_bilstm/training_curves.png')
    
    if curves_path.exists():
        return send_from_directory(curves_path.parent, curves_path.name)
    else:
        return "Curves not found", 404


# ============================================================================
# API ENDPOINTS - DATASET
# ============================================================================

@app.route('/api/dataset/files', methods=['GET'])
def get_dataset_files():
    """Get all audio files in dataset by emotion."""
    try:
        data_dir = Path('data/turkish_emotions')
        emotions = ['mutlu', 'uzgun', 'kizgin', 'notr', 'korku', 'saskin', 'igrenme']
        
        result = {}
        for emotion in emotions:
            emotion_dir = data_dir / emotion
            if emotion_dir.exists():
                files = [f.name for f in emotion_dir.glob('*.wav')]
                result[emotion] = sorted(files)
            else:
                result[emotion] = []
        
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dataset/audio/<emotion>/<filename>')
def serve_audio(emotion, filename):
    """Serve audio file for playback."""
    try:
        audio_path = Path('data/turkish_emotions') / emotion
        return send_from_directory(audio_path, filename)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404


@app.route('/api/dataset/delete', methods=['POST'])
def delete_audio():
    """Delete audio file from dataset."""
    try:
        data = request.get_json()
        emotion = data.get('emotion')
        filename = data.get('filename')
        
        filepath = Path('data/turkish_emotions') / emotion / filename
        if filepath.exists():
            filepath.unlink()
            print(f"🗑️ Deleted: {filepath}")
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dataset/stats', methods=['GET'])
def get_dataset_stats():
    """
    Get current dataset statistics.
    
    Returns:
        JSON with emotion counts and total files
    """
    try:
        data_dir = Path('data/turkish_emotions')
        
        if not data_dir.exists():
            return jsonify({
                'success': True,
                'data': {
                    'total_files': 0,
                    'emotions': {},
                    'message': 'Dataset directory not found'
                }
            })
        
        emotions = ['mutlu', 'uzgun', 'kizgin', 'notr', 'korku', 'saskin', 'igrenme']
        emotion_counts = {}
        total_files = 0
        
        for emotion in emotions:
            emotion_dir = data_dir / emotion
            if emotion_dir.exists():
                wav_files = list(emotion_dir.glob('*.wav'))
                count = len(wav_files)
                emotion_counts[emotion] = count
                total_files += count
            else:
                emotion_counts[emotion] = 0
        
        # Save to database
        if total_files > 0:
            db.save_dataset_stats(total_files, emotion_counts)
        
        return jsonify({
            'success': True,
            'data': {
                'total_files': total_files,
                'emotions': emotion_counts,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/dataset/upload', methods=['POST'])
def upload_audio():
    """
    Upload recorded audio file to dataset.
    
    Expected form data:
        - audio: Audio file (WAV from browser)
        - emotion: Emotion label (mutlu, uzgun, etc.)
    """
    try:
        import soundfile as sf
        import numpy as np
        
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400
        
        audio_file = request.files['audio']
        emotion = request.form.get('emotion', 'notr')
        
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        # Create emotion directory if not exists
        emotion_dir = Path('data/turkish_emotions') / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{emotion}_{timestamp}.wav"
        filepath = emotion_dir / filename
        
        # Check if it's already a WAV file
        if audio_file.filename.lower().endswith('.wav'):
            # Read the WAV data and resample to 16kHz if needed
            try:
                import io
                audio_bytes = audio_file.read()
                
                # Read with soundfile
                audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Resample to 16kHz if needed
                if sr != 16000:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                    sr = 16000
                
                # Save as 16-bit PCM WAV
                sf.write(str(filepath), audio_data, sr, subtype='PCM_16')
                
                print(f"✅ Audio saved: {filepath}")
                
                return jsonify({
                    'success': True,
                    'data': {
                        'filename': filename,
                        'emotion': emotion,
                        'path': str(filepath)
                    }
                })
            except Exception as e:
                print(f"❌ Error processing WAV: {e}")
                # Fallback: just save as-is
                audio_file.seek(0)
                audio_file.save(str(filepath))
                print(f"✅ Audio saved directly: {filepath}")
                return jsonify({
                    'success': True,
                    'data': {
                        'filename': filename,
                        'emotion': emotion,
                        'path': str(filepath)
                    }
                })
        else:
            # For non-WAV files, try to convert using librosa
            import librosa
            
            # Save temp file
            temp_path = Path('data/temp') / audio_file.filename
            temp_path.parent.mkdir(exist_ok=True)
            audio_file.save(str(temp_path))
            
            try:
                audio_data, sr = librosa.load(str(temp_path), sr=16000)
                sf.write(str(filepath), audio_data, sr, subtype='PCM_16')
                temp_path.unlink()
                
                print(f"✅ Audio converted and saved: {filepath}")
                
                return jsonify({
                    'success': True,
                    'data': {
                        'filename': filename,
                        'emotion': emotion,
                        'path': str(filepath)
                    }
                })
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                print(f"❌ Error converting audio: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Error converting audio: {str(e)}'
                }), 500
    
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API ENDPOINTS - TRAINING
# ============================================================================

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """
    Start model training in background.
    
    Expected JSON:
        {
            "model_type": "baseline" or "comparison"
        }
    """
    try:
        # Check if training already running
        status = training_manager.get_status()
        if status['is_training']:
            return jsonify({
                'success': False,
                'error': 'Training already in progress'
            }), 400
        
        data = request.get_json()
        model_type = data.get('model_type', 'baseline')
        
        if model_type not in ['baseline', 'comparison']:
            return jsonify({
                'success': False,
                'error': 'Invalid model type'
            }), 400
        
        # Config path
        config_path = f'config/{model_type}_config.yaml'
        if not Path(config_path).exists():
            return jsonify({
                'success': False,
                'error': f'Config file not found: {config_path}'
            }), 404
        
        # Start training in background
        training_id = training_manager.start_training(model_type, config_path)
        
        print(f"🚀 Training started: ID={training_id}, Model={model_type}")
        
        return jsonify({
            'success': True,
            'data': {
                'training_id': training_id,
                'model_type': model_type,
                'status': 'running'
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/training/current', methods=['GET'])
def get_current_training():
    """Get current training status."""
    try:
        status = training_manager.get_status()
        
        # If training is running, get start time from database
        if status['is_training'] and status['training_id']:
            training = db.get_training(status['training_id'])
            if training:
                status['start_time'] = training['start_time']
        
        return jsonify({
            'success': True,
            'data': status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/training/status/<int:training_id>', methods=['GET'])
def get_training_status(training_id):
    """
    Get current training status.
    
    Args:
        training_id: Training session ID
    """
    try:
        training = db.get_training(training_id)
        
        if not training:
            return jsonify({
                'success': False,
                'error': 'Training not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': training
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/training/history', methods=['GET'])
def get_training_history():
    """Get recent training sessions."""
    try:
        limit = request.args.get('limit', 10, type=int)
        trainings = db.get_all_trainings(limit=limit)
        
        return jsonify({
            'success': True,
            'data': trainings
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/training/<int:training_id>', methods=['DELETE'])
def delete_training(training_id):
    """Delete a training session."""
    try:
        deleted = db.delete_training(training_id)
        
        if deleted:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Training not found'}), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/training/all', methods=['DELETE'])
def delete_all_trainings():
    """Delete all training sessions."""
    try:
        count = db.delete_all_trainings()
        
        return jsonify({
            'success': True,
            'deleted_count': count
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# WEBSOCKET EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Client connected to WebSocket."""
    print(f'✅ Client connected: {request.sid}')
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected from WebSocket."""
    print(f'❌ Client disconnected: {request.sid}')


@socketio.on('ping')
def handle_ping():
    """Handle ping to keep connection alive."""
    emit('pong', {'time': time.time()})


@socketio.on('request_training_update')
def handle_training_update_request():
    """Client requests current training status."""
    status = training_manager.get_status()
    emit('training_update', status)


@socketio.on('stop_training')
def handle_stop_training():
    """Client requests to stop training."""
    success = training_manager.stop_training()
    emit('training_stopped', {'success': success})


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def emit_training_progress(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc):
    """
    Emit training progress to all connected clients.
    
    This will be called from training script in future integration.
    """
    socketio.emit('training_progress', {
        'epoch': epoch,
        'total_epochs': total_epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Initialize training manager after socketio is ready
    training_manager = get_training_manager(socketio, db)
    
    print("\n" + "="*70)
    print("🚀 Turkish Speech Emotion Recognition - Web App")
    print("="*70)
    print(f"📊 Dashboard: http://localhost:5001")
    print(f"🎤 Record: http://localhost:5001/record")
    print(f"🚀 Train: http://localhost:5001/train")
    print(f"📈 History: http://localhost:5001/history")
    print("="*70 + "\n")
    
    # Run with SocketIO
    socketio.run(app, debug=False, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)
