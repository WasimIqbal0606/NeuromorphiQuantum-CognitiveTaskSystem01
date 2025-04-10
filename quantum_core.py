"""
Core logic for the Neuromorphic Quantum-Cognitive Task System
"""

import threading
import uuid
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

from models import Task, TaskState, EntanglementInfo


class TaskMultiverse:
    """Manages the multiverse of quantum tasks with vector embeddings"""
    def __init__(self, embedding_engine=None, persistence_manager=None):
        self.tasks = {}  # id -> Task
        self.lock = threading.RLock()  # Thread-safe operations
        self.audit_log = []  # For tracking state changes
        self.embedding_engine = embedding_engine
        self.persistence_manager = persistence_manager
        self.task_change_callbacks = []  # Callbacks for task changes
        self.entropy_measurements = []  # Track entropy over time
        
        # Initialize entropy measurement history
        self._measure_entropy()

    def register_change_callback(self, callback):
        """Register callback for task changes"""
        if callback not in self.task_change_callbacks:
            self.task_change_callbacks.append(callback)
        
    def unregister_change_callback(self, callback):
        """Unregister callback for task changes"""
        if callback in self.task_change_callbacks:
            self.task_change_callbacks.remove(callback)
            
    def _notify_change(self, change_type, task_id=None, data=None):
        """Notify all registered callbacks of a change"""
        for callback in self.task_change_callbacks:
            try:
                callback(change_type, task_id, data)
            except Exception as e:
                print(f"Error in change callback: {e}")
    
    def _measure_entropy(self):
        """Measure and record current entropy state"""
        total_entropy = sum(task.entropy for task in self.tasks.values()) if self.tasks else 0
        self.entropy_measurements.append({
            "timestamp": datetime.now().isoformat(),
            "total_entropy": total_entropy,
            "task_count": len(self.tasks),
            "state_counts": {
                TaskState.PENDING: sum(1 for t in self.tasks.values() if t.state == TaskState.PENDING),
                TaskState.ENTANGLED: sum(1 for t in self.tasks.values() if t.state == TaskState.ENTANGLED),
                TaskState.RESOLVED: sum(1 for t in self.tasks.values() if t.state == TaskState.RESOLVED),
                TaskState.DEFERRED: sum(1 for t in self.tasks.values() if t.state == TaskState.DEFERRED),
                TaskState.CANCELLED: sum(1 for t in self.tasks.values() if t.state == TaskState.CANCELLED)
            }
        })
        
        # Keep only last 100 measurements
        if len(self.entropy_measurements) > 100:
            self.entropy_measurements = self.entropy_measurements[-100:]
            
    def add_task(self, task):
        """Add a new task to the multiverse with thread safety"""
        with self.lock:
            # Add embedding if engine is available
            if self.embedding_engine:
                self.embedding_engine.add_task_embedding(task)
            
            # Add to the tasks dictionary
            self.tasks[task.id] = task
            
            # Log the action
            self.audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "TASK_CREATED",
                "task_id": task.id,
                "description": task.description[:50]
            })
            
            # Persist if persistence manager is available
            if self.persistence_manager:
                self.persistence_manager.save_task(task)
                
            # Update entropy measurements
            self._measure_entropy()
            
            # Notify about change
            self._notify_change("task_added", task.id, task.to_dict())
            
            return task
    
    def get_task(self, task_id):
        """Get task by ID with thread safety"""
        with self.lock:
            return self.tasks.get(task_id)
    
    def update_task(self, task_id, updates):
        """Update task attributes with thread safety"""
        with self.lock:
            if task_id not in self.tasks:
                return None
                
            task = self.tasks[task_id]
            task.update(**updates)
            
            # Update embedding if description changed and engine is available
            if 'description' in updates and updates['description'] and self.embedding_engine:
                self.embedding_engine.update_task_embedding(task)
                
            # Log the update
            self.audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "TASK_UPDATED",
                "task_id": task_id,
                "updates": str(updates)
            })
            
            # Persist if persistence manager is available
            if self.persistence_manager:
                self.persistence_manager.save_task(task)
                
            # Update entropy measurements
            self._measure_entropy()
            
            # Notify about change
            self._notify_change("task_updated", task_id, task.to_dict())
            
            return task
    
    def update_task_state(self, task_id, new_state):
        """Update a task's state with quantum semantics and thread safety"""
        with self.lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found in multiverse")
            
            task = self.tasks[task_id]
            old_state = task.state
            
            # Update the task state
            task.update_state(new_state)
            
            # Update task in embedding engine if available
            if self.embedding_engine:
                self.embedding_engine.update_task_metadata(task)
            
            # For resolved tasks, reduce entropy
            if new_state == TaskState.RESOLVED and task.entropy > 0.2:
                task.entropy = 0.2
            
            # For pending tasks that were previously entangled, increase entropy
            if new_state == TaskState.PENDING and old_state == TaskState.ENTANGLED:
                task.entropy = min(1.0, task.entropy + 0.3)
            
            # Persist if persistence manager is available
            if self.persistence_manager:
                self.persistence_manager.save_task(task)
            
            # Update entropy measurements
            self._measure_entropy()
            
            # Notify about change
            self._notify_change("state_changed", task_id, {
                "old_state": old_state,
                "new_state": new_state,
                "task": task.to_dict()
            })
            
            return task
    
    def delete_task(self, task_id):
        """Delete a task from the multiverse with thread safety"""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            # Get the task
            task = self.tasks[task_id]
            
            # Remove from embedding engine if available
            if self.embedding_engine:
                self.embedding_engine.remove_task_embedding(task_id)
            
            # Remove entanglements from other tasks
            for other_id, other_task in self.tasks.items():
                if task_id in other_task.entangled_with:
                    other_task.remove_entanglement(task_id)
            
            # Remove the task
            del self.tasks[task_id]
            
            # Log the deletion
            self.audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "TASK_DELETED",
                "task_id": task_id
            })
            
            # Delete from persistence manager if available
            if self.persistence_manager:
                self.persistence_manager.delete_task(task_id)
            
            # Update entropy measurements
            self._measure_entropy()
            
            # Notify about change
            self._notify_change("task_deleted", task_id, None)
            
            return True
    
    def find_related_tasks(self, task_id, threshold=0.7, max_results=5):
        """Find tasks related to the given task using vector similarity or fallback method"""
        with self.lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found in multiverse")
            
            task = self.tasks[task_id]
            
            # Use embedding engine if available
            if self.embedding_engine:
                related_tasks = self.embedding_engine.find_similar_tasks(
                    task, threshold=threshold, max_results=max_results
                )
                return related_tasks
            
            # Fallback to simple text matching
            related_task_ids = []
            task_words = set(task.description.lower().split())
            for other_id, other_task in self.tasks.items():
                if other_id != task_id:  # Skip self
                    other_words = set(other_task.description.lower().split())
                    # Simple Jaccard similarity
                    if task_words and other_words:
                        similarity = len(task_words.intersection(other_words)) / len(task_words.union(other_words))
                        if similarity >= threshold:
                            related_task_ids.append(other_id)
            
            # Limit to max_results
            related_task_ids = related_task_ids[:max_results]
            
            return related_task_ids
    
    def entangle_tasks(self, task_id1, task_id2):
        """Create bidirectional entanglement between two tasks"""
        with self.lock:
            if task_id1 not in self.tasks or task_id2 not in self.tasks:
                return False
            
            task1 = self.tasks[task_id1]
            task2 = self.tasks[task_id2]
            
            # Create mutual entanglement
            task1.add_entanglement(task_id2)
            task2.add_entanglement(task_id1)
            
            # Log the entanglement
            self.audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "TASKS_ENTANGLED",
                "task_id1": task_id1,
                "task_id2": task_id2
            })
            
            # Persist if persistence manager is available
            if self.persistence_manager:
                self.persistence_manager.save_task(task1)
                self.persistence_manager.save_task(task2)
            
            # Update entropy measurements
            self._measure_entropy()
            
            # Notify about changes
            self._notify_change("entanglement_created", task_id1, {
                "task1": task_id1,
                "task2": task_id2
            })
            
            return True
    
    def break_entanglement(self, task_id1, task_id2):
        """Break bidirectional entanglement between two tasks"""
        with self.lock:
            if task_id1 not in self.tasks or task_id2 not in self.tasks:
                return False
            
            task1 = self.tasks[task_id1]
            task2 = self.tasks[task_id2]
            
            # Remove mutual entanglement
            task1.remove_entanglement(task_id2)
            task2.remove_entanglement(task_id1)
            
            # Log the breaking of entanglement
            self.audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "ENTANGLEMENT_BROKEN",
                "task_id1": task_id1,
                "task_id2": task_id2
            })
            
            # Persist if persistence manager is available
            if self.persistence_manager:
                self.persistence_manager.save_task(task1)
                self.persistence_manager.save_task(task2)
            
            # Update entropy measurements
            self._measure_entropy()
            
            # Notify about changes
            self._notify_change("entanglement_broken", task_id1, {
                "task1": task_id1,
                "task2": task_id2
            })
            
            return True
    
    def get_all_tasks(self, filters=None):
        """Get all tasks, optionally filtered by criteria"""
        with self.lock:
            tasks = list(self.tasks.values())
            
            if filters:
                filtered_tasks = []
                for task in tasks:
                    include = True
                    
                    # Filter by state
                    if filters.state and task.state != filters.state:
                        include = False
                    
                    # Filter by assignee
                    if filters.assignee and task.assignee != filters.assignee:
                        include = False
                    
                    # Filter by priority range
                    if filters.priority_min is not None and task.priority < filters.priority_min:
                        include = False
                    if filters.priority_max is not None and task.priority > filters.priority_max:
                        include = False
                    
                    # Filter by search text
                    if filters.search_text:
                        search_lower = filters.search_text.lower()
                        if search_lower not in task.description.lower():
                            include = False
                    
                    # Filter by tags
                    if filters.tags:
                        if not any(tag in task.tags for tag in filters.tags):
                            include = False
                    
                    # Filter by deadline
                    if filters.deadline_before and task.deadline and task.deadline > filters.deadline_before:
                        include = False
                    if filters.deadline_after and task.deadline and task.deadline < filters.deadline_after:
                        include = False
                    
                    if include:
                        filtered_tasks.append(task)
                
                tasks = filtered_tasks
            
            # Sort tasks if needed
            if filters and filters.sort_by:
                reverse = filters.sort_order.lower() == "desc"
                
                if filters.sort_by == "priority":
                    tasks.sort(key=lambda t: t.priority, reverse=reverse)
                elif filters.sort_by == "created_at":
                    tasks.sort(key=lambda t: t.created_at, reverse=reverse)
                elif filters.sort_by == "deadline":
                    # Sort tasks with no deadline at the end
                    tasks.sort(
                        key=lambda t: (t.deadline is None, t.deadline or datetime.max),
                        reverse=reverse
                    )
                elif filters.sort_by == "entropy":
                    tasks.sort(key=lambda t: t.entropy, reverse=reverse)
                else:  # Default to updated_at
                    tasks.sort(key=lambda t: t.updated_at, reverse=reverse)
            
            # Apply pagination if needed
            if filters:
                start = filters.offset
                end = start + filters.limit
                tasks = tasks[start:end]
            
            return tasks
    
    def calculate_entropy_map(self):
        """Calculate the entropy distribution across all tasks with thread safety"""
        with self.lock:
            # Calculate entropy by state
            entropy_by_state = {
                TaskState.PENDING: sum(t.entropy for t in self.tasks.values() if t.state == TaskState.PENDING),
                TaskState.ENTANGLED: sum(t.entropy for t in self.tasks.values() if t.state == TaskState.ENTANGLED),
                TaskState.RESOLVED: sum(t.entropy for t in self.tasks.values() if t.state == TaskState.RESOLVED),
                TaskState.DEFERRED: sum(t.entropy for t in self.tasks.values() if t.state == TaskState.DEFERRED),
                TaskState.CANCELLED: sum(t.entropy for t in self.tasks.values() if t.state == TaskState.CANCELLED)
            }
            
            # Calculate total entropy
            total_entropy = sum(entropy_by_state.values())
            
            # Get task entropies
            task_entropies = {t.id: t.entropy for t in self.tasks.values()}
            
            # Identify overloaded zones
            overloaded_zones = self._identify_overloaded_zones()
            
            # Get entropy trend from measurements
            entropy_trend = [
                {
                    "timestamp": m["timestamp"],
                    "total_entropy": m["total_entropy"]
                }
                for m in self.entropy_measurements
            ]
            
            return {
                "total_entropy": total_entropy,
                "entropy_by_state": entropy_by_state,
                "task_entropies": task_entropies,
                "overloaded_zones": overloaded_zones,
                "entropy_trend": entropy_trend
            }
    
    def _identify_overloaded_zones(self):
        """Find zones (assignees) with too many tasks or high entropy"""
        assignee_tasks = {}
        assignee_entropy = {}
        
        for task in self.tasks.values():
            if task.assignee:
                if task.assignee not in assignee_tasks:
                    assignee_tasks[task.assignee] = []
                    assignee_entropy[task.assignee] = 0
                
                assignee_tasks[task.assignee].append(task.id)
                assignee_entropy[task.assignee] += task.entropy
        
        # Identify overloaded assignees (> 3 tasks or high total entropy)
        overloaded = []
        for assignee, tasks in assignee_tasks.items():
            if len(tasks) > 3 or assignee_entropy[assignee] > 2.0:
                overloaded.append({
                    "assignee": assignee,
                    "task_count": len(tasks),
                    "total_entropy": assignee_entropy[assignee],
                    "reason": "high_task_count" if len(tasks) > 3 else "high_entropy",
                    "task_ids": tasks
                })
        
        return overloaded
    
    def get_entanglement_network(self):
        """Get network data for visualization of entanglements"""
        with self.lock:
            nodes = []
            edges = []
            seen_edges = set()  # To avoid duplicate edges
            
            # Create nodes
            for task in self.tasks.values():
                nodes.append({
                    "id": task.id,
                    "label": task.description[:30] + ("..." if len(task.description) > 30 else ""),
                    "state": task.state,
                    "entropy": task.entropy,
                    "priority": task.priority
                })
                
                # Create edges from entanglements
                for target_id in task.entangled_with:
                    # Create edge ID (smaller ID always first for uniqueness)
                    edge_id = f"{min(task.id, target_id)}-{max(task.id, target_id)}"
                    
                    if edge_id not in seen_edges:
                        edges.append({
                            "id": edge_id,
                            "source": task.id,
                            "target": target_id,
                            "type": "entanglement"
                        })
                        seen_edges.add(edge_id)
            
            return {"nodes": nodes, "edges": edges}
    
    def get_state_snapshot(self):
        """Get a complete snapshot of the current multiverse state"""
        with self.lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "task_count": len(self.tasks),
                "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
                "entropy_map": self.calculate_entropy_map(),
                "entanglement_network": self.get_entanglement_network(),
                "recent_activity": self.audit_log[-30:] if self.audit_log else []
            }
    
    def auto_suggest_entanglements(self, threshold=0.65):
        """Automatically suggest entanglements based on similarity"""
        with self.lock:
            if not self.embedding_engine or len(self.tasks) < 2:
                return []
            
            suggestions = []
            
            task_ids = list(self.tasks.keys())
            
            # For each task, find related tasks
            for i, task_id in enumerate(task_ids):
                task = self.tasks[task_id]
                
                # Skip tasks that are already resolved or cancelled
                if task.state in [TaskState.RESOLVED, TaskState.CANCELLED]:
                    continue
                
                # Find related tasks
                related_ids = self.embedding_engine.find_similar_tasks(
                    task, threshold=threshold, max_results=3
                )
                
                for related_id in related_ids:
                    # Skip if already entangled
                    if related_id in task.entangled_with:
                        continue
                    
                    related_task = self.tasks.get(related_id)
                    
                    # Skip if related task doesn't exist or is resolved/cancelled
                    if not related_task or related_task.state in [TaskState.RESOLVED, TaskState.CANCELLED]:
                        continue
                    
                    # Calculate similarity
                    similarity = self.embedding_engine.calculate_similarity(task, related_task)
                    
                    suggestions.append({
                        "task1": task.id,
                        "task2": related_id,
                        "similarity": similarity,
                        "task1_description": task.description,
                        "task2_description": related_task.description
                    })
            
            # Sort by similarity score
            suggestions.sort(key=lambda x: x["similarity"], reverse=True)
            
            return suggestions
    
    def optimize_task_distribution(self):
        """Optimize task distribution among assignees"""
        with self.lock:
            # Get overloaded assignees
            overloaded_zones = self._identify_overloaded_zones()
            
            if not overloaded_zones:
                return []
            
            # Get all assignees
            all_assignees = set(task.assignee for task in self.tasks.values() if task.assignee)
            
            # Find least loaded assignees
            assignee_load = {}
            for assignee in all_assignees:
                assigned_tasks = [t for t in self.tasks.values() if t.assignee == assignee]
                assignee_load[assignee] = {
                    "task_count": len(assigned_tasks),
                    "total_entropy": sum(t.entropy for t in assigned_tasks)
                }
            
            # Sort assignees by load
            sorted_assignees = sorted(
                assignee_load.items(),
                key=lambda x: (x[1]["task_count"], x[1]["total_entropy"])
            )
            
            suggestions = []
            
            # For each overloaded assignee, suggest redistributions
            for overloaded in overloaded_zones:
                overloaded_assignee = overloaded["assignee"]
                overloaded_tasks = [self.tasks[tid] for tid in overloaded["task_ids"]]
                
                # Sort tasks by priority (lowest first)
                overloaded_tasks.sort(key=lambda t: t.priority)
                
                # Take the least important tasks
                redistribution_candidates = overloaded_tasks[:2]
                
                for task in redistribution_candidates:
                    # Find suitable assignees with lowest load
                    for assignee, load in sorted_assignees:
                        # Skip the overloaded assignee
                        if assignee == overloaded_assignee:
                            continue
                        
                        # Skip if already at capacity
                        if load["task_count"] >= 3:
                            continue
                        
                        suggestions.append({
                            "task_id": task.id,
                            "task_description": task.description,
                            "current_assignee": overloaded_assignee,
                            "suggested_assignee": assignee,
                            "reason": f"Redistribute from overloaded assignee ({overloaded['task_count']} tasks)"
                        })
                        
                        # Update the load count for this assignee
                        load["task_count"] += 1
                        break
            
            return suggestions
    
    def adjust_entropy_decay(self):
        """Adjust entropy of tasks over time - simulate quantum decoherence"""
        with self.lock:
            changes = []
            
            for task_id, task in self.tasks.items():
                # Skip resolved or cancelled tasks
                if task.state in [TaskState.RESOLVED, TaskState.CANCELLED]:
                    continue
                
                # Calculate entropy decay based on time since last update
                time_since_update = (datetime.now() - task.updated_at).total_seconds() / 86400  # days
                
                old_entropy = task.entropy
                
                if time_since_update > 7:  # More than a week
                    # Increase entropy for forgotten tasks
                    new_entropy = min(1.0, task.entropy + 0.1)
                    if new_entropy != old_entropy:
                        task.entropy = new_entropy
                        changes.append({
                            "task_id": task_id,
                            "description": task.description,
                            "old_entropy": old_entropy,
                            "new_entropy": new_entropy,
                            "reason": "Task neglected for more than a week"
                        })
                elif task.state == TaskState.ENTANGLED:
                    # Entangled tasks slowly decrease entropy
                    new_entropy = max(0.4, task.entropy - 0.05)
                    if new_entropy != old_entropy:
                        task.entropy = new_entropy
                        changes.append({
                            "task_id": task_id,
                            "description": task.description,
                            "old_entropy": old_entropy,
                            "new_entropy": new_entropy,
                            "reason": "Entropy decay due to entanglement"
                        })
            
            if changes and self.persistence_manager:
                # Persist tasks that changed
                for change in changes:
                    self.persistence_manager.save_task(self.tasks[change["task_id"]])
            
            # Update entropy measurements
            self._measure_entropy()
            
            return changes
    
    def collapse_superposition(self, task_id):
        """Collapse task superposition by selecting one resolution path"""
        with self.lock:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            
            # Need at least one multiverse path
            if not task.multiverse_paths:
                return None
            
            # Pick a path based on weighted probability (influenced by entropy)
            chosen_path = random.choice(task.multiverse_paths)
            
            # Clear other paths
            task.multiverse_paths = [chosen_path]
            
            # Reduce entropy
            task.entropy = 0.3
            
            # Update state to indicate superposition collapse
            self.update_task_state(task_id, TaskState.ENTANGLED if task.entangled_with else TaskState.PENDING)
            
            # Log the action
            self.audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "SUPERPOSITION_COLLAPSED",
                "task_id": task_id,
                "chosen_path": chosen_path
            })
            
            # Persist if persistence manager is available
            if self.persistence_manager:
                self.persistence_manager.save_task(task)
            
            # Update entropy measurements
            self._measure_entropy()
            
            # Notify about change
            self._notify_change("superposition_collapsed", task_id, {
                "task": task.to_dict(),
                "chosen_path": chosen_path
            })
            
            return task
