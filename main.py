print("✅ Starting Streamlit app")

import streamlit as st
import os
# Configure page must be the first streamlit command
st.set_page_config(
    page_title="Quantum Task Manager",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
# Set port to match deployment configuration
PORT = 8501
HEALTH_PORT = PORT

"""
Neuromorphic Quantum-Cognitive Task Management System UI
A Streamlit-based interface for managing quantum-inspired tasks with advanced visualizations
"""

import json
import time
import asyncio
import base64
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import io
import threading
import random
from PIL import Image
import functools

# API endpoint - Updated for local development
API_URL = "http://localhost:5000"

# Setup flag to use mock data by default, with an option to switch to mock
if 'use_mock_data' not in st.session_state:
    st.session_state.use_mock_data = False

# Try to directly initialize the embedding engine
try:
    from embedding_engine import EmbeddingEngine
    embedding_engine = EmbeddingEngine()
    print("✅ Embedding engine initialized directly")
    direct_embedding_available = True
except Exception as e:
    print(f"⚠️ Direct embedding engine not available: {e}")
    embedding_engine = None
    direct_embedding_available = False

# Session state initialization for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'

if 'task_details_id' not in st.session_state:
    st.session_state.task_details_id = None

if 'edit_task_id' not in st.session_state:
    st.session_state.edit_task_id = None

if 'task_filter_state' not in st.session_state:
    st.session_state.task_filter_state = None

if 'task_filter_assignee' not in st.session_state:
    st.session_state.task_filter_assignee = None

if 'task_filter_search' not in st.session_state:
    st.session_state.task_filter_search = ""

if 'refresh_trigger' not in st.session_state:
    st.session_state.refresh_trigger = 0

if 'show_completed' not in st.session_state:
    st.session_state.show_completed = False

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# Navigation function
def navigate_to(page, **kwargs):
    """Navigate to a specific page and optionally set additional state"""
    st.session_state.page = page
    for key, value in kwargs.items():
        if key in st.session_state:
            st.session_state[key] = value
    st.rerun()

# Trigger refresh
def trigger_refresh():
    """Trigger a refresh of the UI"""
    st.session_state.refresh_trigger += 1
    st.rerun()

# Date formatting utility
def format_date(date_str):
    """Format an ISO date string to a readable format"""
    if not date_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        now = datetime.now()
        diff = now - dt
        
        if diff.days == 0:
            if diff.seconds < 60:
                return "Just now"
            elif diff.seconds < 3600:
                return f"{diff.seconds // 60} minutes ago"
            else:
                return f"{diff.seconds // 3600} hours ago"
        elif diff.days == 1:
            return "Yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        else:
            return dt.strftime("%Y-%m-%d")
    except Exception:
        return date_str

# Deadline formatting utility
def format_deadline(deadline_str):
    """Format a deadline string into a human-readable format with urgency indicators"""
    if not deadline_str:
        return "", "normal"
    
    try:
        deadline = datetime.fromisoformat(deadline_str.replace('Z', '+00:00'))
        now = datetime.now()
        diff = deadline - now
        
        if diff.days < 0:
            return f"⚠️ Overdue by {abs(diff.days)} days", "overdue"
        elif diff.days == 0:
            remaining_hours = diff.seconds // 3600
            if remaining_hours < 1:
                return f"⚠️ Due in {diff.seconds // 60} minutes", "urgent"
            else:
                return f"⚠️ Due in {remaining_hours} hours", "urgent"
        elif diff.days == 1:
            return "Due tomorrow", "soon"
        elif diff.days <= 3:
            return f"Due in {diff.days} days", "soon"
        else:
            return deadline.strftime("%Y-%m-%d"), "normal"
    except Exception:
        return deadline_str, "normal"

# Priority formatting
def format_priority(priority):
    """Format priority as text and return appropriate color"""
    if priority >= 0.8:
        return "Critical", "red"
    elif priority >= 0.6:
        return "High", "orange"
    elif priority >= 0.4:
        return "Medium", "blue"
    elif priority >= 0.2:
        return "Low", "green"
    else:
        return "Minimal", "gray"

# API interaction functions with improved error handling
def api_request(endpoint, method="GET", data=None, params=None):
    """Make an API request to the backend with improved error handling"""
    url = f"{API_URL}{endpoint}"
    
    # If mock data is enabled, return simulated responses
    if st.session_state.use_mock_data:
        return generate_mock_response(endpoint, method, data, params)
    
    # Try direct access for embedding stats if available
    if endpoint == "/system/embedding-stats" and direct_embedding_available:
        try:
            return embedding_engine.get_embedding_statistics()
        except Exception as e:
            print(f"⚠️ Error getting direct embedding statistics: {e}")
            # Continue to try API request as fallback
    
    # Try to connect to API server
    try:
        print(f"Making {method} request to {url}")  # Debug logging
        
        if method == "GET":
            response = requests.get(url, params=params, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, timeout=10)
        else:
            st.error(f"Unsupported method: {method}")
            return None
        
        if response.status_code >= 400:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None
        
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Connection error: Could not connect to {url}. Is the backend running?")
        # Fall back to mock data if API connection fails
        if not st.session_state.use_mock_data:
            print("Falling back to mock data due to connection error")
            return generate_mock_response(endpoint, method, data, params)
        return None
    except requests.exceptions.Timeout:
        st.error(f"Timeout: Request to {url} timed out after 10 seconds")
        return None
    except Exception as e:
        st.error(f"API Request Error: {type(e).__name__}: {e}")
        return None

def generate_mock_response(endpoint, method, data, params):
    """Generate mock responses for development when API server is not running"""
    import uuid
    from datetime import datetime, timedelta
    
    # Root endpoint
    if endpoint == "/":
        return {
            "name": "Neuromorphic Quantum-Cognitive Task System",
            "status": "operational",
            "version": "1.0.0",
            "task_count": 4,
            "embedding_engine": True,
            "timestamp": datetime.now().isoformat()
        }
    
    # Mock tasks list
    if endpoint == "/tasks/" and method == "GET":
        # Create some example tasks
        mock_tasks = [
            {
                "id": "task1",
                "description": "Implement quantum entanglement visualization for related tasks",
                "priority": 0.8,
                "deadline": (datetime.now() + timedelta(days=2)).isoformat(),
                "assignee": "Alice",
                "state": "PENDING",
                "created_at": (datetime.now() - timedelta(days=5)).isoformat(),
                "updated_at": (datetime.now() - timedelta(hours=8)).isoformat(),
                "entangled_with": ["task2"],
                "entropy": 0.75,
                "tags": ["feature", "design", "urgent"],
                "multiverse_paths": [
                    "Use graph-based visualization with D3.js",
                    "Implement 3D visualization with Three.js",
                ]
            },
            {
                "id": "task2",
                "description": "Enhance neural embeddings with adaptive similarity thresholds",
                "priority": 0.6,
                "deadline": (datetime.now() + timedelta(days=7)).isoformat(),
                "assignee": "Bob",
                "state": "ENTANGLED",
                "created_at": (datetime.now() - timedelta(days=10)).isoformat(),
                "updated_at": (datetime.now() - timedelta(days=1)).isoformat(),
                "entangled_with": ["task1", "task3"],
                "entropy": 0.5,
                "tags": ["enhancement", "research"],
                "multiverse_paths": [
                    "Implement adaptive threshold based on task distribution",
                    "Use Bayesian optimization for threshold tuning"
                ]
            },
            {
                "id": "task3",
                "description": "Fix entropy decay calculation bug in quantum core",
                "priority": 0.9,
                "deadline": (datetime.now() - timedelta(days=1)).isoformat(),
                "assignee": "Charlie",
                "state": "PENDING",
                "created_at": (datetime.now() - timedelta(days=2)).isoformat(),
                "updated_at": (datetime.now() - timedelta(hours=4)).isoformat(),
                "entangled_with": ["task2"],
                "entropy": 0.95,
                "tags": ["bug", "urgent"],
                "multiverse_paths": [
                    "Recalibrate decay function parameters",
                    "Rewrite decay algorithm using tensor operations"
                ]
            },
            {
                "id": "task4",
                "description": "Document system architecture and quantum principles",
                "priority": 0.4,
                "deadline": (datetime.now() + timedelta(days=14)).isoformat(),
                "assignee": "Diana",
                "state": "RESOLVED",
                "created_at": (datetime.now() - timedelta(days=20)).isoformat(),
                "updated_at": (datetime.now() - timedelta(days=2)).isoformat(),
                "entangled_with": [],
                "entropy": 0.2,
                "tags": ["documentation"],
                "multiverse_paths": [
                    "Create technical whitepaper",
                    "Write user-friendly documentation"
                ]
            }
        ]
        
        return {
            "total": len(mock_tasks),
            "offset": 0,
            "limit": 100,
            "tasks": mock_tasks
        }
    
    # Create task
    if endpoint == "/tasks/" and method == "POST":
        task_data = data
        new_task = {
            "id": f"task_{uuid.uuid4().hex[:8]}",
            "description": task_data.get("description", "New Task"),
            "priority": task_data.get("priority", 0.5),
            "deadline": task_data.get("deadline"),
            "assignee": task_data.get("assignee"),
            "state": "PENDING",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "entangled_with": [],
            "entropy": 1.0,
            "tags": task_data.get("tags", []),
            "multiverse_paths": ["Initial resolution path"]
        }
        
        return {**new_task, "suggested_entanglements": []}
        
    # Get task by ID
    if endpoint.startswith("/tasks/") and endpoint.count("/") == 2 and method == "GET":
        task_id = endpoint.split("/")[-1]
        
        # Return a mock task for any ID
        return {
            "id": task_id,
            "description": f"Example task {task_id}",
            "priority": 0.7,
            "deadline": (datetime.now() + timedelta(days=3)).isoformat(),
            "assignee": "Alice",
            "state": "PENDING",
            "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
            "updated_at": datetime.now().isoformat(),
            "entangled_with": [],
            "entropy": 0.8,
            "tags": ["example", "mock"],
            "multiverse_paths": ["Path 1", "Path 2"]
        }

    # Get related tasks
    if endpoint.endswith("/related") and method == "GET":
        task_id = endpoint.split("/")[-2]
        
        return {
            "source_task_id": task_id,
            "threshold": params.get("threshold", 0.7),
            "related_tasks": [
                {
                    "task": {
                        "id": f"related_{uuid.uuid4().hex[:8]}",
                        "description": "A related task with similar content",
                        "priority": 0.6,
                        "deadline": (datetime.now() + timedelta(days=5)).isoformat(),
                        "assignee": "Bob",
                        "state": "PENDING",
                        "created_at": (datetime.now() - timedelta(days=3)).isoformat(),
                        "updated_at": (datetime.now() - timedelta(hours=6)).isoformat(),
                        "entangled_with": [],
                        "entropy": 0.7,
                        "tags": ["example", "related"],
                        "multiverse_paths": ["Path A", "Path B"]
                    },
                    "similarity": 0.85
                }
            ]
        }
    
    # Get entropy map
    if endpoint == "/entropy":
        return {
            "total_entropy": 3.5,
            "entropy_by_state": {
                "PENDING": 2.2,
                "ENTANGLED": 0.8,
                "RESOLVED": 0.3,
                "DEFERRED": 0.2,
                "CANCELLED": 0.0
            },
            "task_entropies": {
                "task1": 0.75,
                "task2": 0.5,
                "task3": 0.95,
                "task4": 0.2
            },
            "overloaded_zones": [
                {"assignee": "Charlie", "task_count": 3, "total_entropy": 2.1, "alert_level": "high"}
            ],
            "entropy_trend": [
                {"timestamp": (datetime.now() - timedelta(days=7)).isoformat(), "total_entropy": 4.2},
                {"timestamp": (datetime.now() - timedelta(days=6)).isoformat(), "total_entropy": 4.0},
                {"timestamp": (datetime.now() - timedelta(days=5)).isoformat(), "total_entropy": 3.8},
                {"timestamp": (datetime.now() - timedelta(days=4)).isoformat(), "total_entropy": 3.9},
                {"timestamp": (datetime.now() - timedelta(days=3)).isoformat(), "total_entropy": 3.7},
                {"timestamp": (datetime.now() - timedelta(days=2)).isoformat(), "total_entropy": 3.6},
                {"timestamp": (datetime.now() - timedelta(days=1)).isoformat(), "total_entropy": 3.5},
                {"timestamp": datetime.now().isoformat(), "total_entropy": 3.5}
            ]
        }
    
    # Get task entanglement network
    if endpoint == "/network":
        return {
            "nodes": [
                {"id": "task1", "label": "Task 1", "entropy": 0.75, "state": "PENDING"},
                {"id": "task2", "label": "Task 2", "entropy": 0.5, "state": "ENTANGLED"},
                {"id": "task3", "label": "Task 3", "entropy": 0.95, "state": "PENDING"},
                {"id": "task4", "label": "Task 4", "entropy": 0.2, "state": "RESOLVED"}
            ],
            "links": [
                {"source": "task1", "target": "task2", "similarity": 0.82, "type": "entangled"},
                {"source": "task2", "target": "task3", "similarity": 0.68, "type": "entangled"},
                {"source": "task1", "target": "task3", "similarity": 0.55, "type": "suggested"}
            ]
        }
        
    # Entanglement suggestions
    if endpoint == "/suggestions/entanglements":
        return {
            "suggestions": [
                {
                    "task1": {
                        "id": "task1",
                        "description": "Implement quantum entanglement visualization"
                    },
                    "task2": {
                        "id": "task5",
                        "description": "Research visualization libraries for network graphs"
                    },
                    "similarity": 0.78,
                    "reason": "Both tasks involve visualization technology"
                },
                {
                    "task1": {
                        "id": "task3",
                        "description": "Fix entropy decay calculation bug"
                    },
                    "task2": {
                        "id": "task6",
                        "description": "Optimize performance of core quantum algorithms"
                    },
                    "similarity": 0.72,
                    "reason": "Both involve quantum core optimizations"
                }
            ],
            "threshold": params.get("threshold", 0.65),
            "count": 2
        }
        
    # Task optimization suggestions
    if endpoint == "/suggestions/optimization":
        return {
            "suggestions": [
                {
                    "task_id": "task3",
                    "task_description": "Fix entropy decay calculation bug in quantum core",
                    "current_assignee": "Charlie",
                    "suggested_assignee": "Alice",
                    "reason": "Alice has lower workload and expertise in quantum algorithms"
                },
                {
                    "task_id": "task7",
                    "task_description": "Update embedding model to latest version",
                    "current_assignee": "Charlie",
                    "suggested_assignee": "Bob",
                    "reason": "Bob has completed similar tasks successfully"
                }
            ],
            "count": 2
        }
    
    # Get task history
    if endpoint.endswith("/history"):
        task_id = endpoint.split("/")[-2]
        
        return {
            "task_id": task_id,
            "history": [
                {
                    "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
                    "event_type": "CREATED",
                    "details": {"description": f"Example task {task_id}", "priority": 0.5}
                },
                {
                    "timestamp": (datetime.now() - timedelta(days=3)).isoformat(),
                    "event_type": "UPDATED",
                    "details": {"priority": 0.5, "priority": 0.7}
                },
                {
                    "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
                    "event_type": "STATE_CHANGED",
                    "details": {"from": "PENDING", "to": "ENTANGLED"}
                },
                {
                    "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                    "event_type": "ENTANGLEMENT_ADDED",
                    "details": {"entangled_with": "task2"}
                }
            ]
        }
    
    # System snapshot
    if endpoint == "/system/snapshot":
        return {
            "timestamp": datetime.now().isoformat(),
            "task_count": 4,
            "entropy_level": 3.5,
            "embedding_engine_status": "operational",
            "task_states": {
                "PENDING": 2,
                "ENTANGLED": 1,
                "RESOLVED": 1,
                "DEFERRED": 0, 
                "CANCELLED": 0
            }
        }
        
    # Create system backup
    if endpoint == "/system/backup" and method == "POST":
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    
    # Embedding engine statistics - Updated with better mock response
    if endpoint == "/system/embedding-stats":
        return {
            "status": "initialized",
            "model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "indexed_tasks": 4,
            "similarity_threshold": {
                "current": 0.72,
                "history": [
                    {"timestamp": (datetime.now() - timedelta(hours=5)).isoformat(), "threshold": 0.71, "samples": 10},
                    {"timestamp": (datetime.now() - timedelta(hours=3)).isoformat(), "threshold": 0.72, "samples": 15},
                    {"timestamp": datetime.now().isoformat(), "threshold": 0.72, "samples": 20}
                ]
            },
            "similarity_distribution": {
                "mean": 0.68,
                "median": 0.70,
                "min": 0.45,
                "max": 0.92,
                "samples": 20
            }
        }

    # Default response for unhandled endpoints
    return {
        "status": "mock_response",
        "endpoint": endpoint,
        "method": method,
        "message": "This is a mock response as API server is not running"
    }

@st.cache_data(ttl=10, max_entries=100)
def get_all_tasks(state=None, assignee=None, search=None, priority_min=None, priority_max=None, 
                  tags=None, sort_by="updated_at", sort_order="desc"):
    """Get all tasks with filtering - cached for improved performance"""
    params = {
        "sort_by": sort_by,
        "sort_order": sort_order,
        "limit": 100
    }
    
    if state:
        params["state"] = state
    
    if assignee:
        params["assignee"] = assignee
    
    if search:
        params["search"] = search
    
    if priority_min is not None:
        params["priority_min"] = priority_min
    
    if priority_max is not None:
        params["priority_max"] = priority_max
    
    if tags:
        params["tags"] = tags
    
    # Add cache key using refresh trigger to allow invalidation    
    refresh_key = st.session_state.refresh_trigger
    
    return api_request("/tasks/", params=params)

@st.cache_data(ttl=10, max_entries=100)
def get_task(task_id):
    """Get a single task by ID - cached for improved performance"""
    # Add cache key using refresh trigger to allow invalidation
    refresh_key = st.session_state.refresh_trigger
    
    return api_request(f"/tasks/{task_id}")

def create_task(task_data):
    """Create a new task"""
    return api_request("/tasks/", method="POST", data=task_data)

def update_task(task_id, task_data):
    """Update an existing task"""
    return api_request(f"/tasks/{task_id}", method="PUT", data=task_data)

def update_task_state(task_id, new_state):
    """Update a task's state"""
    return api_request(f"/tasks/{task_id}/state/{new_state}", method="PUT")

def delete_task(task_id):
    """Delete a task"""
    return api_request(f"/tasks/{task_id}", method="DELETE")

def get_related_tasks(task_id, threshold=0.7, max_results=5):
    """Get tasks related to the given task"""
    params = {
        "threshold": threshold,
        "max_results": max_results
    }
    return api_request(f"/tasks/{task_id}/related", params=params)

def entangle_tasks(task_id1, task_id2):
    """Create bidirectional entanglement between two tasks"""
    return api_request(f"/tasks/{task_id1}/entangle/{task_id2}", method="POST")

def break_entanglement(task_id1, task_id2):
    """Break bidirectional entanglement between two tasks"""
    return api_request(f"/tasks/{task_id1}/break-entanglement/{task_id2}", method="POST")

@st.cache_data(ttl=10, max_entries=10)
def get_entropy_map():
    """Get the current entropy distribution - cached for performance"""
    # Add cache key using refresh trigger to allow invalidation
    refresh_key = st.session_state.refresh_trigger
    return api_request("/entropy")

@st.cache_data(ttl=10, max_entries=10)
def get_entanglement_network():
    """Get the task entanglement network - cached for performance"""
    # Add cache key using refresh trigger to allow invalidation
    refresh_key = st.session_state.refresh_trigger
    return api_request("/network")

@st.cache_data(ttl=10, max_entries=10)
def suggest_entanglements(threshold=0.65):
    """Get suggestions for task entanglements - cached for performance"""
    params = {"threshold": threshold}
    # Add cache key using refresh trigger to allow invalidation
    refresh_key = st.session_state.refresh_trigger
    return api_request("/suggestions/entanglements", params=params)

@st.cache_data(ttl=10, max_entries=10)
def suggest_optimization():
    """Get suggestions for task optimization - cached for performance"""
    # Add cache key using refresh trigger to allow invalidation
    refresh_key = st.session_state.refresh_trigger
    return api_request("/suggestions/optimization")

def collapse_task_superposition(task_id):
    """Collapse a task's superposition"""
    return api_request(f"/tasks/{task_id}/collapse", method="POST")

def get_task_history(task_id):
    """Get a task's history"""
    return api_request(f"/tasks/{task_id}/history")

@st.cache_data(ttl=30, max_entries=5)
def get_system_snapshot():
    """Get a snapshot of the system state - cached for performance"""
    # Add cache key using refresh trigger to allow invalidation
    refresh_key = st.session_state.refresh_trigger
    return api_request("/system/snapshot")

def create_system_backup():
    """Create a system backup"""
    return api_request("/system/backup", method="POST")

@st.cache_data(ttl=30, max_entries=5)
def get_embedding_statistics():
    """Get statistics about the embedding engine - cached for performance"""
    # Add cache key using refresh trigger to allow invalidation
    refresh_key = st.session_state.refresh_trigger
    
    # Try direct access if available
    if not st.session_state.use_mock_data and direct_embedding_available:
        try:
            return embedding_engine.get_embedding_statistics()
        except Exception as e:
            print(f"Error getting direct embedding statistics: {e}")
    
    # Fall back to API request
    return api_request("/system/embedding-stats")

# UI Components

@st.cache_data(ttl=10)
def get_system_info():
    """Get basic system information for the header - cached for performance"""
    # Add cache key using refresh trigger to allow invalidation
    refresh_key = st.session_state.refresh_trigger
    return api_request("/")

def render_header():
    """Render the application header"""
    col1, col2 = st.columns([1, 5])
    
    with col1:
        st.image("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e0.svg", width=80)
    
    with col2:
        st.title("Neuromorphic Quantum-Cognitive Task System")
    
    # System status (using cached function)
    system_info = get_system_info()
    if system_info:
        status_cols = st.columns(5)  # Added a column for API mode
        
        with status_cols[0]:
            st.metric("System Status", "Operational" if system_info.get("status") == "operational" else "Error")
        
        with status_cols[1]:
            st.metric("Task Count", system_info.get("task_count", 0))
        
        with status_cols[2]:
            embedding_status = "Active" if system_info.get("embedding_engine", False) else "Inactive"
            st.metric("Embedding Engine", embedding_status)
        
        with status_cols[3]:
            st.metric("Version", system_info.get("version", "Unknown"))
            
        with status_cols[4]:
            api_mode = "Mock" if st.session_state.use_mock_data else "Live"
            st.metric("API Mode", api_mode)

def render_sidebar():
    """Render the sidebar navigation"""
    st.sidebar.title("Navigation")
    
    # Navigation buttons
    if st.sidebar.button("📊 Dashboard", use_container_width=True):
        navigate_to('dashboard')
    
    if st.sidebar.button("📝 Task Management", use_container_width=True):
        navigate_to('tasks')
    
    if st.sidebar.button("🔄 Task Relationships", use_container_width=True):
        navigate_to('relationships')
    
    if st.sidebar.button("📈 Entropy Analytics", use_container_width=True):
        navigate_to('entropy')
    
    if st.sidebar.button("💡 Suggestions", use_container_width=True):
        navigate_to('suggestions')
    
    if st.sidebar.button("⚙️ System", use_container_width=True):
        navigate_to('system')
    
    # Create new task button (always visible)
    st.sidebar.divider()
    if st.sidebar.button("➕ New Task", type="primary", use_container_width=True):
        navigate_to('new_task')
    
    # Settings / utilities
    st.sidebar.divider()
    st.sidebar.caption("Settings")
    dark_mode = st.sidebar.toggle("Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    # Add Mock/Live toggle
    use_mock = st.sidebar.toggle("Use Mock Data", value=st.session_state.use_mock_data)
    if use_mock != st.session_
