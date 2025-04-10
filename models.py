"""
Data models for the Neuromorphic Quantum-Cognitive Task System
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class TaskState:
    """Quantum-inspired task states"""
    PENDING = "PENDING"     # Superposition state
    ENTANGLED = "ENTANGLED" # Connected to other tasks
    RESOLVED = "RESOLVED"   # Wave function collapsed
    DEFERRED = "DEFERRED"   # Temporarily put aside
    CANCELLED = "CANCELLED" # No longer relevant


class TaskBase(BaseModel):
    """Base Pydantic model for task validation"""
    description: str
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    deadline: Optional[datetime] = None
    assignee: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class TaskCreate(TaskBase):
    """Model for creating a new task"""
    pass


class TaskUpdate(BaseModel):
    """Model for updating an existing task"""
    description: Optional[str] = None
    priority: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    deadline: Optional[datetime] = None
    assignee: Optional[str] = None
    state: Optional[str] = None
    tags: Optional[List[str]] = None
    entropy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    

class TaskResponse(TaskBase):
    """Model for API responses"""
    id: str
    state: str
    created_at: datetime
    updated_at: datetime
    entangled_with: List[str] = []
    entropy: float
    multiverse_paths: List[str] = []

    class Config:
        orm_mode = True


class Task:
    """Quantum task representation with metadata"""
    def __init__(self, description, priority=0.5, deadline=None, assignee=None, tags=None):
        self.id = str(uuid.uuid4())
        self.description = description
        self.state = TaskState.PENDING
        self.priority = priority  # 0.0 to 1.0
        self.deadline = deadline
        self.assignee = assignee
        self.tags = tags or []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.entangled_with = []  # IDs of related tasks
        self.entropy = 1.0  # Higher means more chaotic/uncertain
        self.embedding = None  # Will store vector representation
        self.multiverse_paths = []  # Possible resolution paths
        self.history = [{
            "timestamp": self.created_at.isoformat(),
            "action": "CREATED",
            "state": self.state
        }]
        
    def update_state(self, new_state):
        """Update task state and record the transition"""
        old_state = self.state
        self.state = new_state
        self.updated_at = datetime.now()
        self.history.append({
            "timestamp": self.updated_at.isoformat(),
            "action": "STATE_CHANGE",
            "old_state": old_state,
            "new_state": new_state
        })

    def update(self, **kwargs):
        """Update task attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                old_value = getattr(self, key)
                setattr(self, key, value)
                self.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "ATTRIBUTE_UPDATE",
                    "attribute": key,
                    "old_value": str(old_value),
                    "new_value": str(value)
                })
        self.updated_at = datetime.now()

    def add_entanglement(self, task_id):
        """Entangle this task with another task"""
        if task_id not in self.entangled_with and task_id != self.id:
            self.entangled_with.append(task_id)
            self.updated_at = datetime.now()
            if self.state == TaskState.PENDING:
                self.update_state(TaskState.ENTANGLED)
            self.history.append({
                "timestamp": self.updated_at.isoformat(),
                "action": "ENTANGLEMENT_ADDED",
                "entangled_with": task_id
            })

    def remove_entanglement(self, task_id):
        """Remove entanglement with another task"""
        if task_id in self.entangled_with:
            self.entangled_with.remove(task_id)
            self.updated_at = datetime.now()
            if self.state == TaskState.ENTANGLED and not self.entangled_with:
                self.update_state(TaskState.PENDING)
            self.history.append({
                "timestamp": self.updated_at.isoformat(),
                "action": "ENTANGLEMENT_REMOVED",
                "removed_task": task_id
            })

    def add_multiverse_path(self, path_description):
        """Add a possible resolution path to the task"""
        if path_description not in self.multiverse_paths:
            self.multiverse_paths.append(path_description)
            self.updated_at = datetime.now()
            self.history.append({
                "timestamp": self.updated_at.isoformat(),
                "action": "PATH_ADDED",
                "path": path_description
            })

    def calculate_priority_score(self):
        """Calculate a combined priority score based on deadline and priority"""
        base_score = self.priority * 100
        
        if self.deadline:
            # Calculate urgency based on days remaining
            days_remaining = (self.deadline - datetime.now()).days
            if days_remaining <= 0:
                # Past deadline, highest urgency
                urgency_factor = 2.0
            elif days_remaining <= 1:
                # Due within a day
                urgency_factor = 1.5
            elif days_remaining <= 3:
                # Due within 3 days
                urgency_factor = 1.25
            elif days_remaining <= 7:
                # Due within a week
                urgency_factor = 1.1
            else:
                urgency_factor = 1.0
            
            # Apply urgency factor to base score
            return base_score * urgency_factor
        
        return base_score

    def to_dict(self):
        """Convert task to dictionary representation"""
        return {
            "id": self.id,
            "description": self.description,
            "state": self.state,
            "priority": self.priority,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "assignee": self.assignee,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "entangled_with": self.entangled_with,
            "entropy": self.entropy,
            "multiverse_paths": self.multiverse_paths,
            "history": self.history
        }
    
    def to_response_model(self):
        """Convert to Pydantic response model"""
        return TaskResponse(
            id=self.id,
            description=self.description,
            state=self.state,
            priority=self.priority,
            deadline=self.deadline,
            assignee=self.assignee,
            tags=self.tags,
            created_at=self.created_at,
            updated_at=self.updated_at,
            entangled_with=self.entangled_with,
            entropy=self.entropy,
            multiverse_paths=self.multiverse_paths
        )


class QueryParams(BaseModel):
    """Search and filter parameters"""
    state: Optional[str] = None
    assignee: Optional[str] = None
    priority_min: Optional[float] = None
    priority_max: Optional[float] = None
    search_text: Optional[str] = None
    tags: Optional[List[str]] = None
    deadline_before: Optional[datetime] = None
    deadline_after: Optional[datetime] = None
    sort_by: Optional[str] = "updated_at"
    sort_order: Optional[str] = "desc"
    limit: int = 50
    offset: int = 0


class EntanglementInfo(BaseModel):
    """Information about task entanglements"""
    source_id: str
    target_id: str
    similarity_score: float
    relationship_type: str = "semantic"


class EntropyReport(BaseModel):
    """Report on task entropy distribution"""
    total_entropy: float
    entropy_by_state: Dict[str, float]
    task_entropies: Dict[str, float]
    overloaded_zones: List[Dict[str, Any]]
    entropy_trend: List[Dict[str, Any]]
