"""
Training Manager for Web Integration
Runs training in background thread and emits progress via WebSocket
"""

import os
import sys
import subprocess
import threading
import time
import re
from datetime import datetime
from pathlib import Path
from database import Database


class TrainingManager:
    """
    Manages training processes and progress tracking.
    Runs training scripts in background and monitors progress.
    """
    
    def __init__(self, socketio, db: Database):
        """
        Initialize training manager.
        
        Args:
            socketio: Flask-SocketIO instance for real-time updates
            db: Database instance
        """
        self.socketio = socketio
        self.db = db
        self.current_process = None
        self.current_training_id = None
        self.is_training = False
        self.training_thread = None
    
    def start_training(self, model_type: str, config_path: str) -> int:
        """
        Start training in background thread.
        
        Args:
            model_type: 'baseline' or 'comparison'
            config_path: Path to config file
            
        Returns:
            Training ID
        """
        if self.is_training:
            raise RuntimeError("Training already in progress")
        
        # Create training record
        config = {'model': {'type': model_type}, 'training': {'num_epochs': 100}}
        training_id = self.db.add_training(model_type, config)
        
        self.current_training_id = training_id
        self.is_training = True
        
        # Start training thread
        self.training_thread = threading.Thread(
            target=self._run_training,
            args=(training_id, model_type, config_path),
            daemon=True
        )
        self.training_thread.start()
        
        return training_id
    
    def _run_training(self, training_id: int, model_type: str, config_path: str):
        """
        Run training process and monitor progress.
        
        Args:
            training_id: Training session ID
            model_type: Model type
            config_path: Config file path
        """
        try:
            # Emit start event
            self.socketio.emit('training_started', {
                'training_id': training_id,
                'model_type': model_type,
                'status': 'running'
            })
            
            # Build command - use current python executable
            # Always use the same Python that's running this app
            python_exe = sys.executable
            script_path = Path(__file__).parent / 'src' / 'train.py'
            
            cmd = [
                str(python_exe),
                '-u',  # Unbuffered output for real-time logs
                str(script_path),
                '--model', model_type,
                '--config', config_path
            ]
            
            print(f"🚀 Starting training: {' '.join(cmd)}")
            
            # Start process
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor output
            epoch_pattern = re.compile(r'Epoch (\d+)')
            acc_pattern = re.compile(r'Average Accuracy: ([\d.]+)%')
            loss_pattern = re.compile(r'Average Loss: ([\d.]+)')
            epoch_completed_pattern = re.compile(r'Epoch (\d+) completed')
            
            current_epoch = 0
            train_acc = 0
            train_loss = 0
            val_acc = 0
            val_loss = 0
            best_val_acc = 0
            best_val_f1 = 0
            in_validation = False
            training_start_time = time.time()
            
            for line in self.current_process.stdout:
                print(line, end='')  # Print to console
                
                # Emit console log to web UI
                self.socketio.emit('training_log', {
                    'training_id': training_id,
                    'log': line.strip()
                })
                
                # Parse epoch start
                epoch_match = epoch_pattern.search(line)
                if epoch_match and 'Training' in line:
                    current_epoch = int(epoch_match.group(1))
                    in_validation = False
                
                # Check if we're in validation phase
                if 'Validation' in line:
                    in_validation = True
                
                # Parse accuracy
                acc_match = acc_pattern.search(line)
                if acc_match:
                    if in_validation:
                        val_acc = float(acc_match.group(1)) / 100
                    else:
                        train_acc = float(acc_match.group(1)) / 100
                
                # Parse loss
                loss_match = loss_pattern.search(line)
                if loss_match:
                    if in_validation:
                        val_loss = float(loss_match.group(1))
                    else:
                        train_loss = float(loss_match.group(1))
                
                # Parse F1 score from "Macro F1 Score: XX.XX%"
                f1_match = re.search(r'Macro F1 Score:\s*([\d.]+)%', line)
                if f1_match:
                    parsed_f1 = float(f1_match.group(1)) / 100  # Convert from percentage
                    if parsed_f1 > best_val_f1:
                        best_val_f1 = parsed_f1
                
                # Emit progress when epoch completes
                epoch_completed_match = epoch_completed_pattern.search(line)
                if epoch_completed_match:
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                    
                    self.socketio.emit('training_progress', {
                        'training_id': training_id,
                        'epoch': current_epoch,
                        'total_epochs': 100,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    })
                    
                    # Update database
                    self.db.update_training(
                        training_id,
                        best_val_acc=best_val_acc,
                        final_epoch=current_epoch
                    )
            
            # Wait for process to complete
            return_code = self.current_process.wait()
            
            # Calculate training duration
            training_duration_minutes = (time.time() - training_start_time) / 60
            
            if return_code == 0:
                # Training completed successfully
                self.db.update_training(
                    training_id, 
                    status='completed',
                    best_val_acc=best_val_acc,
                    best_val_f1=best_val_f1,
                    training_time_minutes=training_duration_minutes,
                    final_epoch=current_epoch
                )
                self.socketio.emit('training_completed', {
                    'training_id': training_id,
                    'status': 'completed',
                    'best_val_acc': best_val_acc,
                    'best_val_f1': best_val_f1,
                    'duration_minutes': training_duration_minutes
                })
                print(f"✅ Training {training_id} completed successfully in {training_duration_minutes:.1f} minutes")
            else:
                # Training failed
                self.db.update_training(training_id, status='failed')
                self.socketio.emit('training_failed', {
                    'training_id': training_id,
                    'error': f'Process exited with code {return_code}'
                })
                print(f"❌ Training {training_id} failed with code {return_code}")
        
        except Exception as e:
            # Handle errors
            self.db.update_training(training_id, status='failed')
            self.socketio.emit('training_failed', {
                'training_id': training_id,
                'error': str(e)
            })
            print(f"❌ Training {training_id} error: {e}")
        
        finally:
            # Cleanup
            self.is_training = False
            self.current_process = None
            self.current_training_id = None
    
    def stop_training(self):
        """Stop current training process."""
        if self.current_process and self.is_training:
            print(f"⏸️ Stopping training {self.current_training_id}")
            self.current_process.terminate()
            self.db.update_training(self.current_training_id, status='stopped')
            self.is_training = False
            
            self.socketio.emit('training_stopped', {
                'training_id': self.current_training_id
            })
            
            return True
        return False
    
    def get_status(self):
        """Get current training status."""
        return {
            'is_training': self.is_training,
            'training_id': self.current_training_id
        }


# Singleton instance
_training_manager = None

def get_training_manager(socketio=None, db=None):
    """Get or create training manager singleton."""
    global _training_manager
    if _training_manager is None and socketio and db:
        _training_manager = TrainingManager(socketio, db)
    return _training_manager
