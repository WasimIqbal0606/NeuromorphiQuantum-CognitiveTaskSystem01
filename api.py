import os
import json
import time
import asyncio
from datetime import datetime
from typing import Optional
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import (
    TaskState, Task, TaskCreate, TaskUpdate, TaskResponse,
    QueryParams, EntanglementInfo, EntropyReport
)
from quantum_core import TaskMultiverse
from embedding_engine import EmbeddingEngine
from persistence import PersistenceManager
import utils

app = FastAPI(
    title="Neuromorphic Quantum-Cognitive Task System API",
    description="API for managing quantum-inspired tasks with advanced embeddings and entanglement",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_engine = EmbeddingEngine()
persistence_manager = PersistenceManager()
task_multiverse = TaskMultiverse(embedding_engine, persistence_manager)
persistence_manager.set_task_multiverse(task_multiverse)

@app.on_event("startup")
async def startup_event():
    saved_tasks = persistence_manager.load_all_tasks()
    for task_id, task_data in saved_tasks.items():
        try:
            task = Task(
                description=task_data.get("description", ""),
                priority=task_data.get("priority", 0.5),
                deadline=datetime.fromisoformat(task_data.get("deadline")) if task_data.get("deadline") else None,
                assignee=task_data.get("assignee"),
                tags=task_data.get("tags", [])
            )
            task.id = task_id
            task.state = task_data.get("state", TaskState.PENDING)
            task.created_at = datetime.fromisoformat(task_data.get("created_at"))
            task.updated_at = datetime.fromisoformat(task_data.get("updated_at"))
            task.entangled_with = task_data.get("entangled_with", [])
            task.entropy = task_data.get("entropy", 1.0)
            task.multiverse_paths = task_data.get("multiverse_paths", [])
            task.history = task_data.get("history", [])
            task_multiverse.tasks[task_id] = task
            if embedding_engine.initialized:
                embedding_engine.add_task_embedding(task)
        except Exception as e:
            print(f"Error loading task {task_id}: {e}")
    print(f"Loaded {len(saved_tasks)} tasks from storage")
    asyncio.create_task(periodic_maintenance())

async def periodic_maintenance():
    try:
        while True:
            task_multiverse.adjust_entropy_decay()
            if random.random() < 0.2:
                task_multiverse.auto_suggest_entanglements()
            await asyncio.sleep(300)
    except Exception as e:
        print(f"Error in periodic maintenance: {e}")

@app.get("/", tags=["System"])
async def root():
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
    if not task_data.tags and task_data.description:
        task_data.tags = utils.get_suggested_tags(task_data.description)
    task = Task(
        description=task_data.description,
        priority=task_data.priority,
        deadline=task_data.deadline,
        assignee=task_data.assignee,
        tags=task_data.tags
    )
    task_multiverse.add_task(task)
    related_tasks = task_multiverse.find_related_tasks(task.id, threshold=0.65)
    response_data = task.to_response_model().dict()
    response_data["suggested_entanglements"] = related_tasks
    return response_data

@app.get("/tasks/", tags=["Tasks"])
async def get_all_tasks(...):
    # Same as before; omitted for brevity
    pass

@app.get("/tasks/{task_id}", response_model=TaskResponse, tags=["Tasks"])
async def get_task(task_id: str):
    task = task_multiverse.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task.to_response_model()

@app.put("/tasks/{task_id}", response_model=TaskResponse, tags=["Tasks"])
async def update_task(task_id: str, task_update: TaskUpdate):
    updates = {k: v for k, v in task_update.dict().items() if v is not None}
    updated_task = task_multiverse.update_task(task_id, updates)
    if not updated_task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return updated_task.to_response_model()

@app.put("/tasks/{task_id}/state/{new_state}", response_model=TaskResponse, tags=["Tasks"])
async def update_task_state(task_id: str, new_state: str):
    valid_states = [state for state in TaskState]
    if new_state not in valid_states:
        raise HTTPException(status_code=400, detail=f"Invalid state: {new_state}")
    try:
        updated_task = task_multiverse.update_task_state(task_id, new_state)
        return updated_task.to_response_model()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/tasks/{task_id}", tags=["Tasks"])
async def delete_task(task_id: str):
    if not task_multiverse.delete_task(task_id):
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return {"status": "success", "message": f"Task {task_id} deleted"}

@app.get("/tasks/{task_id}/related", tags=["Relationships"])
async def get_related_tasks(task_id: str, threshold: float = 0.7, max_results: int = 5):
    try:
        related_task_ids = task_multiverse.find_related_tasks(task_id, threshold=threshold, max_results=max_results)
        related_tasks = []
        for related_id in related_task_ids:
            task = task_multiverse.get_task(related_id)
            if task:
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
    if not task_multiverse.entangle_tasks(task_id1, task_id2):
        raise HTTPException(status_code=404, detail="One or both tasks not found")
    return {"status": "success", "message": f"Tasks {task_id1} and {task_id2} entangled"}

@app.post("/tasks/{task_id1}/break-entanglement/{task_id2}", tags=["Relationships"])
async def break_entanglement(task_id1: str, task_id2: str):
    if not task_multiverse.break_entanglement(task_id1, task_id2):
        raise HTTPException(status_code=404, detail="One or both tasks not found")
    return {"status": "success", "message": f"Entanglement broken between {task_id1} and {task_id2}"}

@app.get("/entropy", response_model=EntropyReport, tags=["Analytics"])
async def get_entropy_map():
    return task_multiverse.get_entropy_report()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
