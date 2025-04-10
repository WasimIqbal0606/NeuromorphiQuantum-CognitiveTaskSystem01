"""
Persistence manager for the Neuromorphic Quantum-Cognitive Task System
"""

import os
import json
import threading
from datetime import datetime
import time
from typing import Dict, List, Any, Optional


class PersistenceManager:
    """Manages persistence of tasks and system state"""
    
    def __init__(self, storage_dir='./data'):
        self.storage_dir = storage_dir
        self.tasks_dir = os.path.join(storage_dir, 'tasks')
        self.snapshots_dir = os.path.join(storage_dir, 'snapshots')
        self.lock = threading.RLock()
        
        # Create directories if they don't exist
        self._ensure_directories()
        
        # Periodic save timer
        self.last_snapshot_time = time.time()
        self.snapshot_interval = 900  # 15 minutes
        
        # Keep track of pending writes
        self.pending_writes = set()
        self.write_thread = None
        self.stop_flag = False
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.tasks_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
    
    def start_write_thread(self):
        """Start a background thread for handling writes"""
        if self.write_thread is None or not self.write_thread.is_alive():
            self.stop_flag = False
            self.write_thread = threading.Thread(target=self._write_worker, daemon=True)
            self.write_thread.start()
    
    def stop_write_thread(self):
        """Stop the background write thread"""
        self.stop_flag = True
        if self.write_thread and self.write_thread.is_alive():
            self.write_thread.join(timeout=2.0)
    
    def _write_worker(self):
        """Background worker for handling file writes"""
        while not self.stop_flag:
            # Process any pending writes
            with self.lock:
                pending = list(self.pending_writes)
                self.pending_writes.clear()
            
            for task_id in pending:
                try:
                    self._write_task_to_disk(task_id)
                except Exception as e:
                    print(f"Error writing task {task_id} to disk: {e}")
                    # Add back to pending writes
                    with self.lock:
                        self.pending_writes.add(task_id)
            
            # Check if we need to take a snapshot
            current_time = time.time()
            if current_time - self.last_snapshot_time >= self.snapshot_interval:
                try:
                    self._create_snapshot()
                    self.last_snapshot_time = current_time
                except Exception as e:
                    print(f"Error creating snapshot: {e}")
            
            # Sleep for a bit
            time.sleep(1.0)
    
    def _write_task_to_disk(self, task_id):
        """Write a task to disk from the in-memory copy"""
        # This method must be called by _write_worker
        # or externally with proper locking
        
        # Get the task from the main multiverse
        task_data = self.task_multiverse.get_task(task_id)
        
        if not task_data:
            # Task doesn't exist, maybe it was deleted
            # Remove the file if it exists
            task_file = os.path.join(self.tasks_dir, f"{task_id}.json")
            if os.path.exists(task_file):
                os.remove(task_file)
            return
        
        # Convert task to JSON and write to file
        task_file = os.path.join(self.tasks_dir, f"{task_id}.json")
        with open(task_file, 'w') as f:
            json.dump(task_data.to_dict(), f, indent=2)
    
    def _create_snapshot(self):
        """Create a snapshot of the current state"""
        # Get a snapshot from the multiverse
        snapshot = self.task_multiverse.get_state_snapshot()
        
        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = os.path.join(self.snapshots_dir, f"snapshot_{timestamp}.json")
        
        # Write snapshot to disk
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        # Clean up old snapshots (keep last 10)
        self._cleanup_old_snapshots()
    
    def _cleanup_old_snapshots(self):
        """Remove older snapshots, keeping only the most recent ones"""
        snapshot_files = []
        for filename in os.listdir(self.snapshots_dir):
            if filename.startswith("snapshot_") and filename.endswith(".json"):
                full_path = os.path.join(self.snapshots_dir, filename)
                snapshot_files.append((full_path, os.path.getmtime(full_path)))
        
        # Sort by modification time, newest first
        snapshot_files.sort(key=lambda x: x[1], reverse=True)
        
        # Keep the 10 most recent snapshots
        for file_path, _ in snapshot_files[10:]:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing old snapshot {file_path}: {e}")
    
    def set_task_multiverse(self, task_multiverse):
        """Set the task multiverse reference"""
        self.task_multiverse = task_multiverse
        
        # Start the write thread
        self.start_write_thread()
    
    def save_task(self, task):
        """Queue a task for saving to disk"""
        with self.lock:
            self.pending_writes.add(task.id)
    
    def delete_task(self, task_id):
        """Delete a task from disk"""
        task_file = os.path.join(self.tasks_dir, f"{task_id}.json")
        try:
            if os.path.exists(task_file):
                os.remove(task_file)
        except Exception as e:
            print(f"Error deleting task file {task_id}: {e}")
    
    def load_all_tasks(self):
        """Load all tasks from disk and return as a dictionary"""
        tasks = {}
        
        try:
            for filename in os.listdir(self.tasks_dir):
                if filename.endswith(".json"):
                    task_id = filename[:-5]  # Remove .json extension
                    task_file = os.path.join(self.tasks_dir, filename)
                    
                    try:
                        with open(task_file, 'r') as f:
                            task_data = json.load(f)
                            tasks[task_id] = task_data
                    except Exception as e:
                        print(f"Error loading task {task_id}: {e}")
        except Exception as e:
            print(f"Error reading tasks directory: {e}")
        
        return tasks
    
    def load_latest_snapshot(self):
        """Load the most recent snapshot if available"""
        try:
            snapshot_files = []
            for filename in os.listdir(self.snapshots_dir):
                if filename.startswith("snapshot_") and filename.endswith(".json"):
                    full_path = os.path.join(self.snapshots_dir, filename)
                    snapshot_files.append((full_path, os.path.getmtime(full_path)))
            
            if not snapshot_files:
                return None
            
            # Sort by modification time, newest first
            snapshot_files.sort(key=lambda x: x[1], reverse=True)
            
            # Load the most recent snapshot
            with open(snapshot_files[0][0], 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading latest snapshot: {e}")
            return None
    
    def get_snapshot_history(self, limit=5):
        """Get a list of available snapshots"""
        try:
            snapshot_files = []
            for filename in os.listdir(self.snapshots_dir):
                if filename.startswith("snapshot_") and filename.endswith(".json"):
                    full_path = os.path.join(self.snapshots_dir, filename)
                    mtime = os.path.getmtime(full_path)
                    snapshot_files.append({
                        "file": filename,
                        "path": full_path,
                        "timestamp": datetime.fromtimestamp(mtime).isoformat(),
                        "mtime": mtime
                    })
            
            # Sort by modification time, newest first
            snapshot_files.sort(key=lambda x: x["mtime"], reverse=True)
            
            # Return limited number of snapshots
            return snapshot_files[:limit]
        except Exception as e:
            print(f"Error getting snapshot history: {e}")
            return []
    
    def force_snapshot(self):
        """Force creation of a snapshot immediately"""
        try:
            self._create_snapshot()
            self.last_snapshot_time = time.time()
            return True
        except Exception as e:
            print(f"Error creating forced snapshot: {e}")
            return False
