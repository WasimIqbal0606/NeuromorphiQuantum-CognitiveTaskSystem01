"""
API server for the Neuromorphic Quantum-Cognitive Task System
"""

import os
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body, Depends, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from models import TaskState, Task, TaskCreate, TaskUpdate, TaskResponse, QueryParams, EntanglementInfo, EntropyReport
from quantum_core import TaskMultiverse
from embedding_engine import EmbeddingEngine
from persistence import PersistenceManager
import utils


app = FastAPI(
    title="Neuromorphic Quantum-Cognitive Task System API",
    description="API for managing quantum-inspired tasks with advanced embeddings and entanglement",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the system components
embedding_engine = EmbeddingEngine()
persistence_manager = PersistenceManager()
task_multiverse = TaskMultiverse(embedding_engine, persistence_manager)

# Set reference to task_multiverse in persistence_manager
persistence_manager.set_task_multiverse(task_multiverse)

# Background task for entropy decay and optimization
@app.on_event("startup")
async def startup_event():
    # Load saved tasks if any
    saved_tasks = persistence_manager.load_all_tasks()
    for task_id, task_data in saved_tasks.items():
        try:
            # Create task object from saved data
            task = Task(
                description=task_data.get("description", ""),
                priority=task_data.get("priority", 0.5),
                deadline=datetime.fromisoformat(task_data.get("deadline")) if task_data.get("deadline") else None,
                assignee=task_data.get("assignee"),
                tags=task_data.get("tags", [])
            )
            
            # Restore other properties
            task.id = task_id
            task.state = task_data.get("state", TaskState.PENDING)
            task.created_at = datetime.fromisoformat(task_data.get("created_at"))
            task.updated_at = datetime.fromisoformat(task_data.get("updated_at"))
            task.entangled_with = task_data.get("entangled_with", [])
            task.entropy = task_data.get("entropy", 1.0)
            task.multiverse_paths = task_data.get("multiverse_paths", [])
            if "history" in task_data:
                task.history = task_data["history"]
            
            # Add to multiverse
            task_multiverse.tasks[task_id] = task
            
            # Add embedding
            if embedding_engine.initialized:
                embedding_engine.add_task_embedding(task)
        except Exception as e:
            print(f"Error loading task {task_id}: {e}")
    
    print(f"Loaded {len(saved_tasks)} tasks from storage")
    
    # Start background tasks
    asyncio.create_task(periodic_maintenance())

async def periodic_maintenance():
    """Run periodic maintenance tasks in the background"""
    try:
        while True:
            # Run entropy decay
            task_multiverse.adjust_entropy_decay()
            
            # Suggest new entanglements periodically
            if random.random() < 0.2:  # 20% chance each cycle
                task_multiverse.auto_suggest_entanglements()
            
            # Wait before next cycle
            await asyncio.sleep(300)  # 5 minutes
    except Exception as e:
        print(f"Error in periodic maintenance: {e}")


# API Endpoints

@app.get("/", tags=["System"])
async def root():
    """System health check and information"""
    return {
        "name": "Neuromorphic Quantum-Cognitive Task System",
        "status": "operational",
        "version": "1.0.0",
        "task_count": len(task_multiverse.tasks),
        "embedding_engine": embedding_engine.initialized,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/tasks/", response_model=TaskResponse, tags=["Tasks"])
async def create_task(task_data: TaskCreate):
    """Create a new task in the quantum multiverse"""
    # Generate suggested tags if none provided
    if not task_data.tags and task_data.description:
        suggested_tags = utils.get_suggested_tags(task_data.description)
        if suggested_tags:
            task_data.tags = suggested_tags
    
    # Create and add the task
    task = Task(
        description=task_data.description,
        priority=task_data.priority,
        deadline=task_data.deadline,
        assignee=task_data.assignee,
        tags=task_data.tags
    )
    
    task_multiverse.add_task(task)
    
    # Find and suggest related tasks
    related_tasks = task_multiverse.find_related_tasks(task.id, threshold=0.65)
    
    response_data = task.to_response_model().dict()
    response_data["suggested_entanglements"] = related_tasks
    
    return response_data

@app.get("/tasks/", tags=["Tasks"])
async def get_all_tasks(
    state: Optional[str] = None,
    assignee: Optional[str] = None,
    search: Optional[str] = None,
    priority_min: Optional[float] = None,
    priority_max: Optional[float] = None,
    tags: Optional[str] = None,
    sort_by: str = "updated_at",
    sort_order: str = "desc",
    limit: int = 50,
    offset: int = 0
):
    """Get all tasks, with optional filtering"""
    # Process tag filter
    tag_list = tags.split(",") if tags else None
    
    # Construct query params
    filters = QueryParams(
        state=state,
        assignee=assignee,
        priority_min=priority_min,
        priority_max=priority_max,
        search_text=search,
        tags=tag_list,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset
    )
    
    # Get filtered tasks
    tasks = task_multiverse.get_all_tasks(filters)
    
    # Convert to response models
    return {
        "total": len(tasks),
        "offset": offset,
        "limit": limit,
        "tasks": [task.to_response_model().dict() for task in tasks]
    }

@app.get("/tasks/{task_id}", response_model=TaskResponse, tags=["Tasks"])
async def get_task(task_id: str):
    """Get a specific task by ID"""
    task = task_multiverse.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
    
    return task.to_response_model()

@app.put("/tasks/{task_id}", response_model=TaskResponse, tags=["Tasks"])
async def update_task(task_id: str, task_update: TaskUpdate):
    """Update an existing task"""
    # Create update dictionary with only non-None values
    updates = {k: v for k, v in task_update.dict().items() if v is not None}
    
    updated_task = task_multiverse.update_task(task_id, updates)
    if not updated_task:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
    
    return updated_task.to_response_model()

@app.put("/tasks/{task_id}/state/{new_state}", response_model=TaskResponse, tags=["Tasks"])
async def update_task_state(task_id: str, new_state: str):
    """Update a task's state"""
    # Validate the state
    valid_states = [TaskState.PENDING, TaskState.ENTANGLED, TaskState.RESOLVED, 
                    TaskState.DEFERRED, TaskState.CANCELLED]
    
    if new_state not in valid_states:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid state. Must be one of: {', '.join(valid_states)}"
        )
    
    try:
        updated_task = task_multiverse.update_task_state(task_id, new_state)
        return updated_task.to_response_model()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/tasks/{task_id}", tags=["Tasks"])
async def delete_task(task_id: str):
    """Delete a task"""
    success = task_multiverse.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
    
    return {"status": "success", "message": f"Task {task_id} deleted"}

@app.get("/tasks/{task_id}/related", tags=["Relationships"])
async def get_related_tasks(
    task_id: str, 
    threshold: float = 0.7,
    max_results: int = 5
):
    """Find tasks related to the given task based on semantic similarity"""
    try:
        related_task_ids = task_multiverse.find_related_tasks(
            task_id, threshold=threshold, max_results=max_results
        )
        
        related_tasks = []
        for related_id in related_task_ids:
            task = task_multiverse.get_task(related_id)
            if task:
                # Calculate similarity between tasks
                similarity = embedding_engine.calculate_similarity(
                    task_multiverse.get_task(task_id), task
                )
                
                related_tasks.append({
                    "task": task.to_response_model().dict(),
                    "similarity": similarity
                })
        
        return {
            "source_task_id": task_id,
            "threshold": threshold,
            "related_tasks": related_tasks
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/tasks/{task_id1}/entangle/{task_id2}", tags=["Relationships"])
async def entangle_tasks(task_id1: str, task_id2: str):
    """Create bidirectional entanglement between two tasks"""
    success = task_multiverse.entangle_tasks(task_id1, task_id2)
    if not success:
        raise HTTPException(
            status_code=404, 
            detail=f"One or both tasks not found: {task_id1}, {task_id2}"
        )
    
    return {
        "status": "success", 
        "message": f"Tasks {task_id1} and {task_id2} are now entangled"
    }

@app.post("/tasks/{task_id1}/break-entanglement/{task_id2}", tags=["Relationships"])
async def break_entanglement(task_id1: str, task_id2: str):
    """Break bidirectional entanglement between two tasks"""
    success = task_multiverse.break_entanglement(task_id1, task_id2)
    if not success:
        raise HTTPException(
            status_code=404, 
            detail=f"One or both tasks not found: {task_id1}, {task_id2}"
        )
    
    return {
        "status": "success", 
        "message": f"Entanglement between tasks {task_id1} and {task_id2} has been broken"
    }

@app.get("/entropy", response_model=EntropyReport, tags=["Analytics"])
async def get_entropy_map():
    """Get the current entropy distribution across the task multiverse"""
    entropy_map = task_multiverse.calculate_entropy_map()
    return entropy_map

@app.get("/network", tags=["Analytics"])
async def get_entanglement_network():
    """Get the task entanglement network for visualization"""
    network = task_multiverse.get_entanglement_network()
    return network

@app.get("/suggestions/entanglements", tags=["Suggestions"])
async def suggest_entanglements(threshold: float = 0.65):
    """Suggest potential task entanglements based on similarity"""
    suggestions = task_multiverse.auto_suggest_entanglements(threshold=threshold)
    return {
        "suggestions": suggestions, 
        "threshold": threshold,
        "count": len(suggestions)
    }

@app.get("/suggestions/optimization", tags=["Suggestions"])
async def suggest_task_optimization():
    """Suggest optimization of task distribution"""
    suggestions = task_multiverse.optimize_task_distribution()
    return {
        "suggestions": suggestions,
        "count": len(suggestions)
    }

@app.post("/tasks/{task_id}/collapse", response_model=TaskResponse, tags=["Quantum Operations"])
async def collapse_task_superposition(task_id: str):
    """Collapse a task's superposition by selecting one resolution path"""
    task = task_multiverse.collapse_superposition(task_id)
    if not task:
        raise HTTPException(
            status_code=400, 
            detail=f"Unable to collapse task {task_id}. Task not found or no multiple paths available."
        )
    
    return task.to_response_model()

@app.get("/tasks/{task_id}/history", tags=["Tasks"])
async def get_task_history(task_id: str):
    """Get the change history for a task"""
    task = task_multiverse.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
    
    return {
        "task_id": task_id,
        "history": task.history
    }

@app.get("/system/snapshot", tags=["System"])
async def get_system_snapshot():
    """Get a snapshot of the current system state"""
    snapshot = task_multiverse.get_state_snapshot()
    return snapshot

@app.post("/system/backup", tags=["System"])
async def create_system_backup():
    """Force creation of a system backup/snapshot"""
    success = persistence_manager.force_snapshot()
    return {
        "status": "success" if success else "error",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/system/embedding-stats", tags=["System"])
async def get_embedding_statistics():
    """Get statistics about the embedding engine"""
    stats = embedding_engine.get_embedding_statistics() if embedding_engine else {
        "status": "not_available"
    }
    return stats


# Run the server if this file is executed directly
if __name__ == "__main__":
    import random
    random = __import__("random")
    
    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
