"""
Database Module - SQLite Helper
Turkish Speech Emotion Recognition Web App
"""

import sqlite3
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Istanbul timezone (UTC+3)
ISTANBUL_TZ = timezone(timedelta(hours=3))

def get_istanbul_time():
    """Get current time in Istanbul timezone (UTC+3)."""
    return datetime.now(ISTANBUL_TZ)


class Database:
    """
    Simple SQLite database for storing training history and dataset stats.
    
    Tables:
    - trainings: Training session records
    - dataset_stats: Dataset statistics snapshots
    """
    
    def __init__(self, db_path: str = "data/app.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create data directory if not exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Training sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trainings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT DEFAULT 'running',
                best_val_acc REAL,
                best_val_f1 REAL,
                final_epoch INTEGER,
                total_epochs INTEGER,
                training_time_minutes REAL,
                checkpoint_path TEXT,
                config JSON,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Dataset statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_files INTEGER,
                emotion_counts JSON,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database initialized: {self.db_path}")
    
    def add_training(self, model_type: str, config: Dict) -> int:
        """
        Start a new training session.
        
        Args:
            model_type: 'baseline' or 'comparison'
            config: Training configuration dictionary
            
        Returns:
            Training session ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trainings (
                model_type, start_time, status, total_epochs, config
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            model_type,
            get_istanbul_time().isoformat(),
            'running',
            config.get('training', {}).get('num_epochs', 100),
            json.dumps(config)
        ))
        
        training_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Training session created: ID={training_id}")
        return training_id
    
    def update_training(
        self,
        training_id: int,
        status: Optional[str] = None,
        best_val_acc: Optional[float] = None,
        best_val_f1: Optional[float] = None,
        final_epoch: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        training_time_minutes: Optional[float] = None
    ):
        """
        Update training session with progress/results.
        
        Args:
            training_id: Training session ID
            status: 'running', 'completed', 'stopped', 'failed'
            best_val_acc: Best validation accuracy
            best_val_f1: Best validation F1 score
            final_epoch: Final epoch number
            checkpoint_path: Path to saved model
            training_time_minutes: Training duration in minutes
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        values = []
        
        if status:
            updates.append("status = ?")
            values.append(status)
        
        if best_val_acc is not None:
            updates.append("best_val_acc = ?")
            values.append(best_val_acc)
        
        if best_val_f1 is not None:
            updates.append("best_val_f1 = ?")
            values.append(best_val_f1)
        
        if final_epoch is not None:
            updates.append("final_epoch = ?")
            values.append(final_epoch)
        
        if checkpoint_path:
            updates.append("checkpoint_path = ?")
            values.append(checkpoint_path)
        
        if training_time_minutes is not None:
            updates.append("training_time_minutes = ?")
            values.append(training_time_minutes)
        
        if status in ['completed', 'stopped', 'failed']:
            updates.append("end_time = ?")
            values.append(get_istanbul_time().isoformat())
        
        if updates:
            values.append(training_id)
            query = f"UPDATE trainings SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, values)
            conn.commit()
        
        conn.close()
    
    def get_training(self, training_id: int) -> Optional[Dict]:
        """
        Get training session details.
        
        Args:
            training_id: Training session ID
            
        Returns:
            Training session dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM trainings WHERE id = ?", (training_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            data = dict(row)
            # Parse JSON fields
            if data['config']:
                data['config'] = json.loads(data['config'])
            return data
        return None
    
    def get_all_trainings(self, limit: int = 10) -> List[Dict]:
        """
        Get recent training sessions.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            List of training session dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM trainings 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        trainings = []
        for row in rows:
            data = dict(row)
            if data['config']:
                data['config'] = json.loads(data['config'])
            trainings.append(data)
        
        return trainings
    
    def save_dataset_stats(self, total_files: int, emotion_counts: Dict[str, int]):
        """
        Save current dataset statistics snapshot.
        
        Args:
            total_files: Total number of audio files
            emotion_counts: Dictionary mapping emotion to count
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO dataset_stats (total_files, emotion_counts)
            VALUES (?, ?)
        """, (total_files, json.dumps(emotion_counts)))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Dataset stats saved: {total_files} files")
    
    def get_latest_dataset_stats(self) -> Optional[Dict]:
        """
        Get most recent dataset statistics.
        
        Returns:
            Dataset stats dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM dataset_stats 
            ORDER BY recorded_at DESC 
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            data = dict(row)
            data['emotion_counts'] = json.loads(data['emotion_counts'])
            return data
        return None
    
    def delete_training(self, training_id: int) -> bool:
        """
        Delete a training session.
        
        Args:
            training_id: Training session ID
            
        Returns:
            True if deleted, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM trainings WHERE id = ?", (training_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        if deleted:
            print(f"🗑️ Training {training_id} deleted")
        return deleted
    
    def delete_all_trainings(self) -> int:
        """
        Delete all training sessions.
        
        Returns:
            Number of deleted records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM trainings")
        count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"🗑️ Deleted {count} training records")
        return count
    
    def close(self):
        """Close database connection (not needed for SQLite, but good practice)."""
        pass


# Test database
if __name__ == "__main__":
    print("=== Testing Database ===")
    
    db = Database("data/test_app.db")
    
    # Test training record
    config = {
        'training': {'num_epochs': 100},
        'model': {'type': 'baseline'}
    }
    
    training_id = db.add_training('baseline', config)
    print(f"Created training: {training_id}")
    
    db.update_training(
        training_id,
        best_val_acc=0.75,
        best_val_f1=0.73,
        final_epoch=50,
        status='completed'
    )
    
    training = db.get_training(training_id)
    print(f"Training details: {training}")
    
    # Test dataset stats
    emotion_counts = {
        'mutlu': 20,
        'uzgun': 18,
        'kizgin': 22,
        'notr': 19,
        'korku': 15,
        'saskin': 21,
        'igrenme': 17
    }
    
    db.save_dataset_stats(132, emotion_counts)
    stats = db.get_latest_dataset_stats()
    print(f"Dataset stats: {stats}")
    
    print("\n✅ Database tests passed!")
