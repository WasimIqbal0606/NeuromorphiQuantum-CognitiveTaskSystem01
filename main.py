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

# API endpoint
# For development, we need to make sure this matches where the API server is running
# Currently, we're setting up to use a direct connection without separate API server for simplicity
API_URL = "http://0.0.0.0:8000"

# Setup flag to use mock data since API server is not running
USE_MOCK_DATA = True

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

# API interaction functions
def api_request(endpoint, method="GET", data=None, params=None):
    """Make an API request to the backend"""
    url = f"{API_URL}{endpoint}"
    
    # If mock data is enabled, return simulated responses
    if USE_MOCK_DATA:
        return generate_mock_response(endpoint, method, data, params)
    
    # Else try to connect to API server
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            st.error(f"Unsupported method: {method}")
            return None
        
        if response.status_code >= 400:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None
        
        return response.json()
    except Exception as e:
        st.error(f"API Request Error: {e}")
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
        status_cols = st.columns(4)
        
        with status_cols[0]:
            st.metric("System Status", "Operational" if system_info.get("status") == "operational" else "Error")
        
        with status_cols[1]:
            st.metric("Task Count", system_info.get("task_count", 0))
        
        with status_cols[2]:
            embedding_status = "Active" if system_info.get("embedding_engine", False) else "Inactive"
            st.metric("Embedding Engine", embedding_status)
        
        with status_cols[3]:
            st.metric("Version", system_info.get("version", "Unknown"))

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
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        trigger_refresh()
    
    # System information in footer
    st.sidebar.divider()
    st.sidebar.caption(f"System v1.0.0")
    st.sidebar.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

def render_task_card(task, is_detail=False, show_actions=True):
    """Render a single task card"""
    card = st.container()
    
    with card:
        # Create columns for the card header
        header_cols = st.columns([4, 1, 1])
        
        with header_cols[0]:
            # Task title with priority indicator
            priority_text, priority_color = format_priority(task.get("priority", 0.5))
            st.markdown(f"### {task.get('description', 'Untitled Task')}")
            
        with header_cols[1]:
            # Priority badge
            st.markdown(f"<span style='background-color:{priority_color};padding:3px 8px;border-radius:10px;color:white;font-size:0.8em'>{priority_text}</span>", unsafe_allow_html=True)
        
        with header_cols[2]:
            # State badge
            state = task.get("state", "PENDING")
            state_color = {
                "PENDING": "orange",
                "ENTANGLED": "blue",
                "RESOLVED": "green",
                "DEFERRED": "gray",
                "CANCELLED": "red"
            }.get(state, "gray")
            
            st.markdown(f"<span style='background-color:{state_color};padding:3px 8px;border-radius:10px;color:white;font-size:0.8em'>{state}</span>", unsafe_allow_html=True)
        
        # Task details
        details_cols = st.columns([1, 1])
        
        with details_cols[0]:
            # Dates and ID
            st.caption(f"Created: {format_date(task.get('created_at'))}")
            st.caption(f"Updated: {format_date(task.get('updated_at'))}")
            
            # Tags
            if task.get("tags") and len(task.get("tags")) > 0:
                tags_html = " ".join([f"<span style='background-color:#f0f0f0;color:#333;padding:2px 6px;border-radius:10px;font-size:0.7em;margin-right:4px'>{tag}</span>" for tag in task.get("tags")])
                st.markdown(f"Tags: {tags_html}", unsafe_allow_html=True)
            
            # Task ID (smaller text)
            st.caption(f"ID: {task.get('id', 'Unknown')}")
        
        with details_cols[1]:
            # Assignee
            assignee = task.get("assignee") or "Unassigned"
            st.caption(f"Assignee: {assignee}")
            
            # Deadline with formatting
            deadline_text, deadline_status = format_deadline(task.get("deadline"))
            deadline_color = {
                "normal": "black",
                "soon": "orange",
                "urgent": "red",
                "overdue": "crimson"
            }.get(deadline_status, "black")
            
            st.markdown(f"<span style='color:{deadline_color}'>{deadline_text}</span>", unsafe_allow_html=True)
            
            # Entropy
            entropy = task.get("entropy", 1.0)
            st.progress(entropy, f"Entropy: {entropy:.2f}")
        
        # Entanglements
        if task.get("entangled_with") and len(task.get("entangled_with")) > 0:
            st.markdown("**Entangled with:**")
            entangled_cols = st.columns(min(3, len(task.get("entangled_with"))))
            
            for i, entangled_id in enumerate(task.get("entangled_with")):
                with entangled_cols[i % 3]:
                    # For each entangled task, we'll show a mini reference
                    st.markdown(f"🔄 `{entangled_id[:8]}...`")
                    
                    # Add a button to view this task
                    if st.button(f"View", key=f"view_entangled_{entangled_id}", use_container_width=True):
                        navigate_to('task_details', task_details_id=entangled_id)
        
        # Multiverse paths (resolution options)
        if task.get("multiverse_paths") and len(task.get("multiverse_paths")) > 0:
            st.markdown("**Possible resolution paths:**")
            for i, path in enumerate(task.get("multiverse_paths")):
                st.markdown(f"{i+1}. {path}")
            
            # Add collapse button if there are multiple paths
            if len(task.get("multiverse_paths")) > 1 and show_actions:
                if st.button("🌀 Collapse Superposition", key=f"collapse_{task.get('id')}", use_container_width=True):
                    result = collapse_task_superposition(task.get("id"))
                    if result:
                        st.success("Superposition collapsed successfully")
                        time.sleep(1)
                        trigger_refresh()
        
        # Action buttons (conditionally shown)
        if show_actions:
            action_cols = st.columns(5)
            
            with action_cols[0]:
                if st.button("📋 Details", key=f"details_{task.get('id')}", use_container_width=True):
                    navigate_to('task_details', task_details_id=task.get('id'))
            
            with action_cols[1]:
                if st.button("✏️ Edit", key=f"edit_{task.get('id')}", use_container_width=True):
                    navigate_to('edit_task', edit_task_id=task.get('id'))
            
            with action_cols[2]:
                # Change state button based on current state
                if task.get("state") == "PENDING":
                    if st.button("✅ Resolve", key=f"resolve_{task.get('id')}", use_container_width=True):
                        result = update_task_state(task.get('id'), "RESOLVED")
                        if result:
                            st.success("Task resolved successfully")
                            time.sleep(1)
                            trigger_refresh()
                elif task.get("state") == "RESOLVED":
                    if st.button("🔄 Reopen", key=f"reopen_{task.get('id')}", use_container_width=True):
                        result = update_task_state(task.get('id'), "PENDING")
                        if result:
                            st.success("Task reopened successfully")
                            time.sleep(1)
                            trigger_refresh()
                else:
                    if st.button("🔄 Set Pending", key=f"pending_{task.get('id')}", use_container_width=True):
                        result = update_task_state(task.get('id'), "PENDING")
                        if result:
                            st.success("Task set to pending")
                            time.sleep(1)
                            trigger_refresh()
            
            with action_cols[3]:
                if st.button("🔗 Find Related", key=f"related_{task.get('id')}", use_container_width=True):
                    navigate_to('find_related', task_details_id=task.get('id'))
            
            with action_cols[4]:
                if st.button("🗑️ Delete", key=f"delete_{task.get('id')}", use_container_width=True):
                    st.warning("Are you sure you want to delete this task?")
                    confirm_cols = st.columns(2)
                    with confirm_cols[0]:
                        if st.button("Yes, delete", key=f"confirm_delete_{task.get('id')}", use_container_width=True):
                            result = delete_task(task.get('id'))
                            if result:
                                st.success("Task deleted successfully")
                                time.sleep(1)
                                trigger_refresh()
                    with confirm_cols[1]:
                        if st.button("Cancel", key=f"cancel_delete_{task.get('id')}", use_container_width=True):
                            trigger_refresh()
        
        st.divider()

def render_task_list(tasks_data, show_filter=True):
    """Render a list of tasks with optional filtering"""
    if tasks_data is None or "tasks" not in tasks_data:
        st.error("Failed to load tasks")
        return
    
    tasks = tasks_data.get("tasks", [])
    
    if len(tasks) == 0:
        st.info("No tasks found")
        return
    
    if show_filter:
        # Filter options
        filter_cols = st.columns(4)
        
        with filter_cols[0]:
            # State filter
            states = ["All States", "PENDING", "ENTANGLED", "RESOLVED", "DEFERRED", "CANCELLED"]
            selected_state = st.selectbox(
                "Filter by State", 
                states, 
                index=states.index(st.session_state.task_filter_state) if st.session_state.task_filter_state in states else 0
            )
            
            if selected_state != st.session_state.task_filter_state:
                st.session_state.task_filter_state = selected_state if selected_state != "All States" else None
                trigger_refresh()
        
        with filter_cols[1]:
            # Get unique assignees from data
            all_assignees = ["All Assignees", "Unassigned"] + list(set([
                task.get("assignee") for task in tasks 
                if task.get("assignee") and task.get("assignee") != "Unassigned"
            ]))
            
            selected_assignee = st.selectbox(
                "Filter by Assignee", 
                all_assignees,
                index=all_assignees.index(st.session_state.task_filter_assignee) if st.session_state.task_filter_assignee in all_assignees else 0
            )
            
            if selected_assignee != st.session_state.task_filter_assignee:
                if selected_assignee == "Unassigned":
                    st.session_state.task_filter_assignee = ""
                else:
                    st.session_state.task_filter_assignee = selected_assignee if selected_assignee != "All Assignees" else None
                trigger_refresh()
        
        with filter_cols[2]:
            # Search box
            search_term = st.text_input("Search Tasks", value=st.session_state.task_filter_search)
            if search_term != st.session_state.task_filter_search:
                st.session_state.task_filter_search = search_term
                trigger_refresh()
        
        with filter_cols[3]:
            # Show completed toggle
            show_completed = st.toggle("Show Completed", value=st.session_state.show_completed)
            if show_completed != st.session_state.show_completed:
                st.session_state.show_completed = show_completed
                trigger_refresh()
    
    # If not showing completed and no specific state filter is set, filter out RESOLVED tasks
    if not st.session_state.show_completed and not st.session_state.task_filter_state:
        tasks = [task for task in tasks if task.get("state") != "RESOLVED"]
    
    # Render each task
    for task in tasks:
        render_task_card(task)

def render_new_task_form():
    """Render the form for creating a new task"""
    st.header("Create New Task")

    with st.form("new_task_form"):
        # Task description
        description = st.text_area("Task Description", placeholder="Describe the task...")

        # Columns for other fields
        col1, col2 = st.columns(2)

        with col1:
            # Priority slider
            priority = st.slider("Priority", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

            # Assignee
            assignee = st.text_input("Assignee", placeholder="Who should work on this?")

        with col2:
            # Deadline
            deadline = st.date_input("Deadline (optional)", value=None)

            # Tags
            tags_input = st.text_input("Tags (comma-separated)", placeholder="tag1, tag2, tag3")

        # Suggested tags based on description (computed when description changes)
        if description:
            import re
            # Simple tag extraction based on common patterns
            potential_tags = []

            # Look for common categories
            categories = {
                "bug": ["bug", "issue", "fix", "error", "crash", "broken"],
                "feature": ["feature", "enhancement", "new", "implement", "add"],
                "documentation": ["doc", "documentation", "explain", "clarify"],
                "refactor": ["refactor", "clean", "optimize", "restructure"],
                "research": ["research", "investigate", "explore", "analyze"],
                "design": ["design", "UI", "UX", "interface", "visual"],
                "testing": ["test", "QA", "verify", "validation"],
                "urgent": ["urgent", "ASAP", "immediately", "critical"]
            }

            # Check for category matches
            description_lower = description.lower()
            for category, keywords in categories.items():
                if any(keyword in description_lower for keyword in keywords):
                    potential_tags.append(category)

            # Look for hashtags in the description
            hashtags = re.findall(r'#(\w+)', description)
            if hashtags:
                potential_tags.extend(hashtags)

            # Display suggested tags if found
            if potential_tags:
                st.caption("Suggested tags: " + ", ".join(potential_tags))

        # Submit button
        submitted = st.form_submit_button("Create Task", type="primary")

    # Note: These buttons are now OUTSIDE the form
    if submitted:
        # Process the tags
        tags = []
        if tags_input:
            tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]

        # Prepare the task data
        task_data = {
            "description": description,
            "priority": priority,
            "assignee": assignee if assignee else None,
            "tags": tags
        }

        # Add deadline if set
        if deadline:
            task_data["deadline"] = datetime.combine(deadline, datetime.min.time()).isoformat()

        # Create the task
        result = create_task(task_data)

        if result:
            st.success("Task created successfully!")

            # Show suggested entanglements if any
            if "suggested_entanglements" in result and result["suggested_entanglements"]:
                st.subheader("Suggested Related Tasks")
                st.info("We found some potentially related tasks. Would you like to create entanglements?")

                for related_id in result["suggested_entanglements"]:
                    related_task = get_task(related_id)
                    if related_task:
                        st.markdown(f"**{related_task.get('description', 'Unknown Task')}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Create Entanglement with {related_id[:8]}", key=f"entangle_{related_id}", use_container_width=True):
                                entangle_result = entangle_tasks(result["id"], related_id)
                                if entangle_result:
                                    st.success("Entanglement created successfully")
                        with col2:
                            if st.button(f"Skip", key=f"skip_{related_id}", use_container_width=True):
                                pass
                        st.divider()

            # Option to go to task list or create another
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View Task List", use_container_width=True):
                    navigate_to('tasks')
            with col2:
                if st.button("Create Another Task", use_container_width=True):
                    navigate_to('new_task')
        else:
            st.error("Failed to create task. Please try again.")

def render_edit_task_form(task_id):
    """Render the form for editing an existing task"""
    task = get_task(task_id)

    if not task:
        st.error(f"Failed to load task {task_id}")
        return

    st.header(f"Edit Task: {task.get('description', 'Unknown')[:50]}...")

    with st.form("edit_task_form"):
        # Task description
        description = st.text_area("Task Description", value=task.get("description", ""))

        # Columns for other fields
        col1, col2 = st.columns(2)

        with col1:
            # Priority slider
            priority = st.slider("Priority", min_value=0.0, max_value=1.0, value=task.get("priority", 0.5), step=0.1)

            # Assignee
            assignee = st.text_input("Assignee", value=task.get("assignee", ""))

            # State selection
            states = ["PENDING", "ENTANGLED", "RESOLVED", "DEFERRED", "CANCELLED"]
            state = st.selectbox("State", states, index=states.index(task.get("state", "PENDING")) if task.get("state") in states else 0)

        with col2:
            # Deadline
            current_deadline = None
            if task.get("deadline"):
                try:
                    current_deadline = datetime.fromisoformat(task.get("deadline").replace('Z', '+00:00')).date()
                except:
                    pass

            deadline = st.date_input("Deadline (optional)", value=current_deadline)

            # Tags
            current_tags = ", ".join(task.get("tags", []))
            tags_input = st.text_input("Tags (comma-separated)", value=current_tags)

            # Entropy (advanced)
            show_advanced = st.toggle("Show Advanced Options")

            if show_advanced:
                entropy = st.slider("Entropy", min_value=0.0, max_value=1.0, value=task.get("entropy", 1.0), step=0.05)

        # Submit button
        submitted = st.form_submit_button("Update Task", type="primary")

    # Note: These buttons are now OUTSIDE the form
    if submitted:
        # Process the tags
        tags = []
        if tags_input:
            tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]

        # Prepare the task data
        task_data = {
            "description": description,
            "priority": priority,
            "assignee": assignee if assignee else None,
            "state": state,
            "tags": tags
        }

        # Add entropy if changed
        if show_advanced:
            task_data["entropy"] = entropy

        # Add deadline if set
        if deadline:
            task_data["deadline"] = datetime.combine(deadline, datetime.min.time()).isoformat()

        # Update the task
        result = update_task(task_id, task_data)

        if result:
            st.success("Task updated successfully!")

            # Option to go back
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View Task Details", use_container_width=True):
                    navigate_to('task_details', task_details_id=task_id)
            with col2:
                if st.button("Return to Task List", use_container_width=True):
                    navigate_to('tasks')
        else:
            st.error("Failed to update task. Please try again.")

def render_task_details(task_id):
    """Render detailed view of a single task"""
    task = get_task(task_id)
    
    if not task:
        st.error(f"Failed to load task {task_id}")
        return
    
    # Task header
    st.header(f"Task Details: {task.get('description', 'Unknown Task')[:50]}...")
    
    # Render the task card without actions
    render_task_card(task, is_detail=True, show_actions=False)
    
    # Actions tab and more details
    tab1, tab2, tab3, tab4 = st.tabs(["Actions", "History", "Related Tasks", "Advanced"])
    
    with tab1:
        # Actions
        action_cols = st.columns(4)
        
        with action_cols[0]:
            if st.button("✏️ Edit Task", use_container_width=True):
                navigate_to('edit_task', edit_task_id=task_id)
        
        with action_cols[1]:
            if task.get("state") == "PENDING":
                if st.button("✅ Mark as Resolved", use_container_width=True):
                    result = update_task_state(task_id, "RESOLVED")
                    if result:
                        st.success("Task resolved successfully")
                        time.sleep(1)
                        trigger_refresh()
            elif task.get("state") == "RESOLVED":
                if st.button("🔄 Mark as Pending", use_container_width=True):
                    result = update_task_state(task_id, "PENDING")
                    if result:
                        st.success("Task set to pending")
                        time.sleep(1)
                        trigger_refresh()
            else:
                state_options = ["PENDING", "ENTANGLED", "RESOLVED", "DEFERRED", "CANCELLED"]
                new_state = st.selectbox("Change State", state_options, index=state_options.index(task.get("state")) if task.get("state") in state_options else 0)
                if new_state != task.get("state"):
                    if st.button("Update State", use_container_width=True):
                        result = update_task_state(task_id, new_state)
                        if result:
                            st.success(f"Task state updated to {new_state}")
                            time.sleep(1)
                            trigger_refresh()
        
        with action_cols[2]:
            if st.button("🔄 Find Related Tasks", use_container_width=True):
                navigate_to('find_related', task_details_id=task_id)
        
        with action_cols[3]:
            if st.button("🗑️ Delete Task", use_container_width=True, type="secondary"):
                st.warning("Are you sure you want to delete this task?")
                confirm_cols = st.columns(2)
                with confirm_cols[0]:
                    if st.button("Yes, delete", key=f"confirm_delete_details", use_container_width=True):
                        result = delete_task(task_id)
                        if result:
                            st.success("Task deleted successfully")
                            time.sleep(1)
                            navigate_to('tasks')
                with confirm_cols[1]:
                    if st.button("Cancel", key=f"cancel_delete_details", use_container_width=True):
                        trigger_refresh()
                        
        # Entanglement management
        st.subheader("Manage Entanglements")
        
        if task.get("entangled_with") and len(task.get("entangled_with")) > 0:
            st.markdown("**Current Entanglements:**")
            
            for entangled_id in task.get("entangled_with"):
                related_task = get_task(entangled_id)
                if related_task:
                    ent_cols = st.columns([3, 1, 1])
                    with ent_cols[0]:
                        st.markdown(f"🔄 **{related_task.get('description', 'Unknown')[:50]}...**")
                    with ent_cols[1]:
                        if st.button(f"View", key=f"view_rel_{entangled_id}", use_container_width=True):
                            navigate_to('task_details', task_details_id=entangled_id)
                    with ent_cols[2]:
                        if st.button(f"Break", key=f"break_{entangled_id}", use_container_width=True):
                            result = break_entanglement(task_id, entangled_id)
                            if result:
                                st.success("Entanglement broken successfully")
                                time.sleep(1)
                                trigger_refresh()
                    st.divider()
        else:
            st.info("This task has no entanglements yet.")
            
            # Suggest finding related tasks
            if st.button("Find Tasks to Entangle", use_container_width=True):
                navigate_to('find_related', task_details_id=task_id)
    
    with tab2:
        # Task history
        history = get_task_history(task_id)
        
        if history and "history" in history:
            events = history.get("history", [])
            
            if len(events) > 0:
                # Create a dataframe for the history
                history_data = []
                for event in events:
                    event_type = event.get("action", "UNKNOWN")
                    timestamp = format_date(event.get("timestamp", ""))
                    details = ""
                    
                    if event_type == "STATE_CHANGE":
                        details = f"From {event.get('old_state')} to {event.get('new_state')}"
                    elif event_type == "ATTRIBUTE_UPDATE":
                        details = f"Changed {event.get('attribute')} from '{event.get('old_value')}' to '{event.get('new_value')}'"
                    elif event_type == "ENTANGLEMENT_ADDED":
                        details = f"Added entanglement with {event.get('entangled_with')}"
                    elif event_type == "ENTANGLEMENT_REMOVED":
                        details = f"Removed entanglement with {event.get('removed_task')}"
                    elif event_type == "PATH_ADDED":
                        details = f"Added resolution path: {event.get('path')}"
                    
                    history_data.append({
                        "Timestamp": timestamp,
                        "Event": event_type,
                        "Details": details
                    })
                
                # Create dataframe and display
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("No history events recorded for this task.")
        else:
            st.error("Failed to load task history")
    
    with tab3:
        # Related tasks
        related = get_related_tasks(task_id)
        
        if related and "related_tasks" in related:
            related_tasks = related.get("related_tasks", [])
            
            if len(related_tasks) > 0:
                st.subheader("Similar Tasks")
                
                for related_item in related_tasks:
                    related_task = related_item.get("task")
                    similarity = related_item.get("similarity", 0)
                    
                    rel_cols = st.columns([3, 1, 1, 1])
                    with rel_cols[0]:
                        st.markdown(f"**{related_task.get('description', 'Unknown')[:50]}...**")
                        st.progress(similarity, f"Similarity: {similarity:.2f}")
                    
                    with rel_cols[1]:
                        if st.button(f"View", key=f"sim_view_{related_task.get('id')}", use_container_width=True):
                            navigate_to('task_details', task_details_id=related_task.get('id'))
                    
                    with rel_cols[2]:
                        is_entangled = related_task.get('id') in task.get('entangled_with', [])
                        
                        if is_entangled:
                            if st.button(f"Break Entanglement", key=f"sim_break_{related_task.get('id')}", use_container_width=True):
                                result = break_entanglement(task_id, related_task.get('id'))
                                if result:
                                    st.success("Entanglement broken successfully")
                                    time.sleep(1)
                                    trigger_refresh()
                        else:
                            if st.button(f"Entangle", key=f"sim_entangle_{related_task.get('id')}", use_container_width=True):
                                result = entangle_tasks(task_id, related_task.get('id'))
                                if result:
                                    st.success("Entanglement created successfully")
                                    time.sleep(1)
                                    trigger_refresh()
                    
                    st.divider()
            else:
                st.info("No related tasks found.")
        else:
            st.error("Failed to load related tasks")
    
    with tab4:
        # Advanced options
        st.subheader("Advanced Options")
        
        # Superposition collapse (if multiple paths)
        if task.get("multiverse_paths") and len(task.get("multiverse_paths")) > 1:
            st.markdown("**Multiverse Paths:**")
            for i, path in enumerate(task.get("multiverse_paths")):
                st.markdown(f"{i+1}. {path}")
            
            if st.button("🌀 Collapse Superposition", use_container_width=True):
                result = collapse_task_superposition(task_id)
                if result:
                    st.success("Superposition collapsed successfully")
                    time.sleep(1)
                    trigger_refresh()
        
        # Manual entropy adjustment
        st.markdown("**Entropy Adjustment:**")
        current_entropy = task.get("entropy", 1.0)
        new_entropy = st.slider("Task Entropy", min_value=0.0, max_value=1.0, value=current_entropy, step=0.05)
        
        if new_entropy != current_entropy:
            if st.button("Update Entropy", use_container_width=True):
                result = update_task(task_id, {"entropy": new_entropy})
                if result:
                    st.success("Entropy updated successfully")
                    time.sleep(1)
                    trigger_refresh()
        
        # Add resolution path
        st.markdown("**Add Resolution Path:**")
        new_path = st.text_area("Description of possible resolution")
        if new_path:
            if st.button("Add Path", use_container_width=True):
                # Get current paths and add new one
                current_paths = task.get("multiverse_paths", [])
                if new_path not in current_paths:
                    current_paths.append(new_path)
                    
                    result = update_task(task_id, {"multiverse_paths": current_paths})
                    if result:
                        st.success("Resolution path added successfully")
                        time.sleep(1)
                        trigger_refresh()
                else:
                    st.warning("This path already exists")
    
    # Back button
    if st.button("← Back to Task List", use_container_width=True):
        navigate_to('tasks')

def render_find_related(task_id):
    """Render interface for finding and managing related tasks"""
    task = get_task(task_id)
    
    if not task:
        st.error(f"Failed to load task {task_id}")
        return
    
    st.header(f"Find Related Tasks for: {task.get('description', 'Unknown Task')[:50]}...")
    
    # Show the current task
    st.subheader("Current Task")
    render_task_card(task, is_detail=True, show_actions=False)
    
    # Similarity threshold slider
    threshold = st.slider("Similarity Threshold", min_value=0.4, max_value=0.9, value=0.65, step=0.05)
    max_results = st.slider("Maximum Results", min_value=1, max_value=20, value=5, step=1)
    
    # Search button
    if st.button("🔍 Find Related Tasks", type="primary", use_container_width=True):
        with st.spinner("Searching for related tasks..."):
            related = get_related_tasks(task_id, threshold=threshold, max_results=max_results)
            
            if related and "related_tasks" in related:
                related_tasks = related.get("related_tasks", [])
                
                if len(related_tasks) > 0:
                    st.success(f"Found {len(related_tasks)} related tasks")
                    
                    for related_item in related_tasks:
                        related_task = related_item.get("task")
                        similarity = related_item.get("similarity", 0)
                        
                        # Create a card-like display
                        with st.container():
                            rel_cols = st.columns([3, 1])
                            
                            with rel_cols[0]:
                                st.markdown(f"### {related_task.get('description', 'Unknown')}")
                                st.progress(similarity, f"Similarity: {similarity:.2f}")
                                
                                # Show state and other details
                                detail_cols = st.columns(3)
                                with detail_cols[0]:
                                    state = related_task.get("state", "PENDING")
                                    state_color = {
                                        "PENDING": "orange",
                                        "ENTANGLED": "blue",
                                        "RESOLVED": "green",
                                        "DEFERRED": "gray",
                                        "CANCELLED": "red"
                                    }.get(state, "gray")
                                    
                                    st.markdown(f"<span style='background-color:{state_color};padding:3px 8px;border-radius:10px;color:white;font-size:0.8em'>{state}</span>", unsafe_allow_html=True)
                                
                                with detail_cols[1]:
                                    st.caption(f"Assignee: {related_task.get('assignee') or 'Unassigned'}")
                                
                                with detail_cols[2]:
                                    deadline_text, _ = format_deadline(related_task.get("deadline"))
                                    st.caption(f"Deadline: {deadline_text}")
                            
                            with rel_cols[1]:
                                is_entangled = related_task.get('id') in task.get('entangled_with', [])
                                
                                if is_entangled:
                                    st.info("Already entangled")
                                    if st.button(f"Break Entanglement", key=f"break_rel_{related_task.get('id')}", use_container_width=True):
                                        result = break_entanglement(task_id, related_task.get('id'))
                                        if result:
                                            st.success("Entanglement broken successfully")
                                            time.sleep(1)
                                            trigger_refresh()
                                else:
                                    if st.button(f"Create Entanglement", key=f"create_rel_{related_task.get('id')}", use_container_width=True):
                                        result = entangle_tasks(task_id, related_task.get('id'))
                                        if result:
                                            st.success("Entanglement created successfully")
                                            time.sleep(1)
                                            trigger_refresh()
                                
                                if st.button(f"View Details", key=f"view_rel_details_{related_task.get('id')}", use_container_width=True):
                                    navigate_to('task_details', task_details_id=related_task.get('id'))
                            
                            st.divider()
                else:
                    st.info("No related tasks found with the current threshold. Try lowering the threshold or adding more task description details.")
            else:
                st.error("Failed to load related tasks")
    
    # Back button
    if st.button("← Back to Task Details", use_container_width=True):
        navigate_to('task_details', task_details_id=task_id)

def render_dashboard():
    """Render the main dashboard with statistics and visualizations"""
    st.header("Quantum-Cognitive Task Dashboard")
    
    # Get system snapshot
    snapshot = get_system_snapshot()
    
    if not snapshot:
        st.error("Failed to load system data")
        return
    
    # Get entropy data
    entropy_data = get_entropy_map()
    
    if not entropy_data:
        st.error("Failed to load entropy data")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_tasks = snapshot.get("task_count", 0)
        st.metric("Total Tasks", total_tasks)
    
    with col2:
        total_entropy = entropy_data.get("total_entropy", 0)
        st.metric("System Entropy", f"{total_entropy:.2f}")
    
    with col3:
        # Count by state
        state_counts = {
            "PENDING": 0,
            "ENTANGLED": 0,
            "RESOLVED": 0,
            "DEFERRED": 0,
            "CANCELLED": 0
        }
        
        for task in snapshot.get("tasks", {}).values():
            state = task.get("state", "PENDING")
            if state in state_counts:
                state_counts[state] += 1
        
        pending_count = state_counts.get("PENDING", 0)
        st.metric("Pending Tasks", pending_count)
    
    with col4:
        entangled_count = state_counts.get("ENTANGLED", 0)
        st.metric("Entangled Tasks", entangled_count)
    
    # Charts
    st.subheader("Task Analytics")
    
    chart_cols = st.columns(2)
    
    with chart_cols[0]:
        # Create state distribution chart
        data = []
        for state, count in state_counts.items():
            if count > 0:
                data.append({"State": state, "Count": count})
        
        if data:
            df = pd.DataFrame(data)
            st.bar_chart(df.set_index("State"))
        else:
            st.info("No task data available for charting")
    
    with chart_cols[1]:
        # Entropy by state chart
        entropy_by_state = entropy_data.get("entropy_by_state", {})
        
        if entropy_by_state:
            entropy_data = []
            for state, entropy in entropy_by_state.items():
                if entropy > 0:
                    entropy_data.append({"State": state, "Entropy": entropy})
            
            if entropy_data:
                df = pd.DataFrame(entropy_data)
                st.bar_chart(df.set_index("State"))
            else:
                st.info("No entropy data available")
        else:
            st.info("No entropy data available")
    
    # Task urgency section
    st.subheader("Task Urgency Overview")
    
    # Calculate urgency for pending tasks
    urgent_tasks = []
    for task_id, task in snapshot.get("tasks", {}).items():
        if task.get("state") in ["PENDING", "ENTANGLED"]:
            urgency = 0
            
            # Base urgency on priority
            urgency += task.get("priority", 0.5) * 50
            
            # Add urgency for deadline
            if task.get("deadline"):
                try:
                    deadline = datetime.fromisoformat(task.get("deadline").replace('Z', '+00:00'))
                    days_remaining = (deadline - datetime.now()).days
                    
                    if days_remaining <= 0:
                        urgency += 50  # Overdue
                    elif days_remaining <= 1:
                        urgency += 40  # Due within a day
                    elif days_remaining <= 3:
                        urgency += 30  # Due within 3 days
                    elif days_remaining <= 7:
                        urgency += 20  # Due within a week
                except:
                    pass
            
            # Add urgency for entropy
            urgency += task.get("entropy", 0.5) * 10
            
            urgent_tasks.append({
                "id": task_id,
                "description": task.get("description", "Unknown"),
                "urgency": urgency,
                "state": task.get("state", "PENDING"),
                "priority": task.get("priority", 0.5),
                "deadline": task.get("deadline"),
                "assignee": task.get("assignee", "Unassigned")
            })
    
    # Sort by urgency
    urgent_tasks.sort(key=lambda x: x["urgency"], reverse=True)
    
    # Display top urgent tasks
    if urgent_tasks:
        top_urgent = urgent_tasks[:5]
        
        # Create urgency table
        urgency_data = []
        
        for task in top_urgent:
            deadline_text, _ = format_deadline(task.get("deadline"))
            priority_text, _ = format_priority(task.get("priority", 0.5))
            
            urgency_data.append({
                "Task": task.get("description", "")[:50] + ("..." if len(task.get("description", "")) > 50 else ""),
                "Urgency": int(task.get("urgency", 0)),
                "Priority": priority_text,
                "Deadline": deadline_text,
                "Assignee": task.get("assignee", "Unassigned")
            })
        
        urgency_df = pd.DataFrame(urgency_data)
        st.dataframe(urgency_df, use_container_width=True)
        
        # Button to view all tasks
        if st.button("View All Tasks", use_container_width=True):
            navigate_to('tasks')
    else:
        st.info("No pending tasks in the system")
    
    # Network visualization
    st.subheader("Task Entanglement Network")
    
    # Get network data
    network_data = get_entanglement_network()
    
    if network_data and "nodes" in network_data and "edges" in network_data:
        nodes = network_data.get("nodes", [])
        edges = network_data.get("edges", [])
        
        if nodes and edges:
            # Create a simplified network visualization
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Create a simple layout (in a real implementation, use networkx or similar)
            # For now, just place nodes in a circle
            node_positions = {}
            num_nodes = len(nodes)
            
            radius = 5
            for i, node in enumerate(nodes):
                angle = 2 * np.pi * i / num_nodes
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                node_positions[node["id"]] = (x, y)
            
            # Map states to colors
            state_colors = {
                'PENDING': '#fcba03',
                'ENTANGLED': '#0373fc',
                'RESOLVED': '#27ae60',
                'DEFERRED': '#95a5a6',
                'CANCELLED': '#e74c3c'
            }
            
            # Draw edges first so they appear behind nodes
            for edge in edges:
                source_id = edge["source"]
                target_id = edge["target"]
                
                if source_id in node_positions and target_id in node_positions:
                    source_pos = node_positions[source_id]
                    target_pos = node_positions[target_id]
                    
                    ax.plot([source_pos[0], target_pos[0]], 
                           [source_pos[1], target_pos[1]], 
                           'k-', alpha=0.5, zorder=1)
            
            # Draw nodes
            for node in nodes:
                node_id = node["id"]
                if node_id in node_positions:
                    pos = node_positions[node_id]
                    state = node.get("state", "PENDING")
                    color = state_colors.get(state, '#7f8c8d')
                    
                    # Size based on priority
                    size = 100 + (node.get("priority", 0.5) * 100)
                    
                    # Draw node
                    ax.scatter(pos[0], pos[1], s=size, c=color, alpha=0.8, edgecolors='black', zorder=2)
                    
                    # Add label
                    ax.annotate(node.get("label", node_id[:6]), 
                               xy=pos, xytext=(0, -10),
                               textcoords="offset points",
                               ha='center', va='center',
                               fontsize=8)
            
            # Remove axes
            ax.set_axis_off()
            
            # Set title
            ax.set_title('Task Entanglement Network')
            
            # Add legend for states
            state_patches = [plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=color, markersize=10, label=state)
                            for state, color in state_colors.items()]
            ax.legend(handles=state_patches, loc='upper right')
            
            # Show in Streamlit
            st.pyplot(fig)
            
            # Add a button to view detailed relationships
            if st.button("View Relationships in Detail", use_container_width=True):
                navigate_to('relationships')
        else:
            st.info("No entanglement data available for visualization")
    else:
        st.info("No entanglement network available")

def render_task_management():
    """Render the task management interface"""
    st.header("Task Management")
    
    # Tab for different views
    tab1, tab2, tab3 = st.tabs(["All Tasks", "By Assignee", "By State"])
    
    with tab1:
        # Get tasks with filters from session state
        tasks_data = get_all_tasks(
            state=st.session_state.task_filter_state,
            assignee=st.session_state.task_filter_assignee,
            search=st.session_state.task_filter_search
        )
        
        # Render task list
        render_task_list(tasks_data)
    
    with tab2:
        # Get all tasks
        all_tasks_data = get_all_tasks()
        
        if all_tasks_data and "tasks" in all_tasks_data:
            # Group by assignee
            assignees = {}
            
            for task in all_tasks_data.get("tasks", []):
                assignee = task.get("assignee") or "Unassigned"
                
                if assignee not in assignees:
                    assignees[assignee] = []
                
                assignees[assignee].append(task)
            
            # Display tasks by assignee
            if assignees:
                for assignee, tasks in assignees.items():
                    with st.expander(f"{assignee} ({len(tasks)})", expanded=False):
                        for task in tasks:
                            render_task_card(task)
            else:
                st.info("No tasks found")
        else:
            st.error("Failed to load tasks")
    
    with tab3:
        # Get all tasks
        all_tasks_data = get_all_tasks()
        
        if all_tasks_data and "tasks" in all_tasks_data:
            # Group by state
            states = {
                "PENDING": [],
                "ENTANGLED": [],
                "RESOLVED": [],
                "DEFERRED": [],
                "CANCELLED": []
            }
            
            for task in all_tasks_data.get("tasks", []):
                state = task.get("state", "PENDING")
                
                if state in states:
                    states[state].append(task)
                else:
                    states[state] = [task]
            
            # Display tasks by state
            for state, tasks in states.items():
                with st.expander(f"{state} ({len(tasks)})", expanded=(state == "PENDING" or state == "ENTANGLED")):
                    if tasks:
                        for task in tasks:
                            render_task_card(task)
                    else:
                        st.info(f"No tasks in {state} state")
        else:
            st.error("Failed to load tasks")

def render_relationship_view():
    """Render the task relationship visualization and management"""
    st.header("Task Relationships and Entanglements")
    
    # Import visualization functions
    from visualization import create_entanglement_network_visualization, create_interactive_network_data
    
    # Get network data
    network_data = get_entanglement_network()
    
    # Get all tasks for reference
    all_tasks_data = get_all_tasks()
    
    if not network_data:
        st.error("Failed to load network data")
        # Check if using either "links" or "edges" as the key
        if network_data and "nodes" in network_data:
            if "links" not in network_data and "edges" not in network_data:
                st.info("Network data is missing connections information.")
        return
    
    if not all_tasks_data or "tasks" not in all_tasks_data:
        st.error("Failed to load task data")
        return
    
    # Ensure nodes is a list
    nodes = network_data.get("nodes", [])
    
    # Check for both potential keys for connections
    links = network_data.get("links", [])
    if not links:
        links = network_data.get("edges", [])
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Network Visualization", "Entanglement Management", "Auto-suggestions"])
    
    with tab1:
        # Display enhanced visualization title
        st.subheader("Quantum Task Entanglement Network")
        
        if not nodes:
            st.warning("No tasks found in the system. Create some tasks to see them visualized here.")
            return
            
        if len(nodes) == 1:
            st.info("Only one task exists. Create more tasks and connect them to visualize the network.")
            
        # Generate the interactive data
        node_data, link_data, static_viz = create_interactive_network_data(network_data)
        
        # Show statistics 
        stats_cols = st.columns(3)
        with stats_cols[0]:
            st.metric("Total Tasks", len(nodes))
        with stats_cols[1]:
            st.metric("Connections", len(links))
        with stats_cols[2]:
            # Calculate average connections per node
            if nodes:
                avg_connections = len(links) / len(nodes) if len(nodes) > 0 else 0
                st.metric("Avg. Connections", f"{avg_connections:.1f}")
        
        # Show visualization options
        viz_options = st.radio("Visualization Type:", 
                           ["Interactive Tables", "Network Graph"], 
                           horizontal=True)
        
        if viz_options == "Network Graph":
            # Show the enhanced static visualization
            if static_viz:
                st.image(f"data:image/png;base64,{static_viz}", use_column_width=True)
            else:
                st.warning("Unable to generate network visualization.")
        else:
            # Show interactive tables
            if node_data:
                st.subheader("Task Nodes")
                # Create sortable dataframe for nodes
                import pandas as pd
                node_df = pd.DataFrame(node_data)
                st.dataframe(node_df, use_container_width=True)
            
            if link_data:
                st.subheader("Task Connections")
                link_df = pd.DataFrame(link_data)
                st.dataframe(link_df, use_container_width=True)
                
            # Display color legend
            st.markdown("### Task State Color Legend")
            legend_cols = st.columns(5)
            
            state_colors = {
                'PENDING': '#fcba03',    # Yellow 
                'ENTANGLED': '#0373fc',  # Blue
                'RESOLVED': '#27ae60',   # Green
                'DEFERRED': '#95a5a6',   # Gray
                'CANCELLED': '#e74c3c'   # Red
            }
            
            for i, (state, color) in enumerate(state_colors.items()):
                with legend_cols[i]:
                    st.markdown(
                        f"<div style='background-color:{color};padding:10px;border-radius:5px;color:white;text-align:center'>"
                        f"{state}</div>", 
                        unsafe_allow_html=True
                    )
    
    with tab2:
        # Entanglement management
        st.subheader("Manage Task Entanglements")
        
        # Create a mapping of task_id to task for quick lookup
        task_map = {}
        for task in all_tasks_data.get("tasks", []):
            task_map[task.get("id")] = task
        
        # Show existing entanglements
        if links:
            st.markdown("**Current Entanglements:**")
            
            # Use enumeration for unique button keys
            for i, link in enumerate(links):
                source_id = link["source"]
                target_id = link["target"]
                
                source_task = task_map.get(source_id, {"description": "Unknown Task"})
                target_task = task_map.get(target_id, {"description": "Unknown Task"})
                
                with st.container():
                    ent_cols = st.columns([3, 3, 1])
                    
                    with ent_cols[0]:
                        source_desc = source_task.get("description", "Unknown")[:30]
                        st.markdown(f"**{source_desc}...**")
                        st.caption(f"ID: {source_id[:8]}...")
                        
                        if st.button(f"View", key=f"view_src_{i}_{source_id}", use_container_width=True):
                            navigate_to('task_details', task_details_id=source_id)
                    
                    with ent_cols[1]:
                        target_desc = target_task.get("description", "Unknown")[:30]
                        st.markdown(f"**{target_desc}...**")
                        st.caption(f"ID: {target_id[:8]}...")
                        
                        if st.button(f"View", key=f"view_tgt_{i}_{target_id}", use_container_width=True):
                            navigate_to('task_details', task_details_id=target_id)
                    
                    with ent_cols[2]:
                        if st.button(f"Break", key=f"break_ent_{i}_{source_id}_{target_id}", use_container_width=True):
                            result = break_entanglement(source_id, target_id)
                            if result:
                                st.success("Entanglement broken successfully")
                                time.sleep(1)
                                trigger_refresh()
                    
                    st.divider()
        else:
            st.info("No entanglements have been created yet.")
        
        # Interface to create new entanglements
        st.subheader("Create New Entanglement")
        
        # Task selection
        all_tasks = [{"id": t.get("id"), "description": t.get("description")} for t in all_tasks_data.get("tasks", [])]
        
        task1_options = ["Select first task"] + [f"{t['description'][:40]}... ({t['id'][:8]})" for t in all_tasks]
        task1_selection = st.selectbox("Select First Task", task1_options)
        
        if task1_selection != "Select first task":
            selected_task1_id = all_tasks[task1_options.index(task1_selection) - 1]["id"]
            
            # Filter out the selected task from the second dropdown
            remaining_tasks = [t for t in all_tasks if t["id"] != selected_task1_id]
            task2_options = ["Select second task"] + [f"{t['description'][:40]}... ({t['id'][:8]})" for t in remaining_tasks]
            task2_selection = st.selectbox("Select Second Task", task2_options)
            
            if task2_selection != "Select second task":
                task_index = task2_options.index(task2_selection) - 1
                selected_task2_id = remaining_tasks[task_index]["id"]
                
                if st.button("Create Entanglement", type="primary", use_container_width=True):
                    result = entangle_tasks(selected_task1_id, selected_task2_id)
                    if result:
                        st.success("Entanglement created successfully")
                        time.sleep(1)
                        trigger_refresh()
    
    with tab3:
        # Automatic entanglement suggestions
        st.subheader("Suggested Entanglements")
        
        # Get suggestions with a threshold
        threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.9, value=0.65, step=0.05)
        
        if st.button("Find Suggested Entanglements", type="primary", use_container_width=True):
            with st.spinner("Analyzing task relationships..."):
                suggestions = suggest_entanglements(threshold=threshold)
                
                if suggestions and "suggestions" in suggestions:
                    suggestion_list = suggestions.get("suggestions", [])
                    
                    if suggestion_list:
                        st.success(f"Found {len(suggestion_list)} potential entanglements")
                        
                        for i, suggestion in enumerate(suggestion_list):
                            task1_id = suggestion.get("task1")
                            task2_id = suggestion.get("task2")
                            similarity = suggestion.get("similarity", 0)
                            
                            task1_desc = suggestion.get("task1_description", "Unknown")
                            task2_desc = suggestion.get("task2_description", "Unknown")
                            
                            with st.container():
                                sugg_cols = st.columns([3, 3, 1])
                                
                                with sugg_cols[0]:
                                    st.markdown(f"**{task1_desc[:40]}...**")
                                    st.caption(f"ID: {task1_id[:8]}...")
                                    
                                    if st.button(f"View", key=f"view_sugg1_{i}_{task1_id}", use_container_width=True):
                                        navigate_to('task_details', task_details_id=task1_id)
                                
                                with sugg_cols[1]:
                                    st.markdown(f"**{task2_desc[:40]}...**")
                                    st.caption(f"ID: {task2_id[:8]}...")
                                    
                                    if st.button(f"View", key=f"view_sugg2_{i}_{task2_id}", use_container_width=True):
                                        navigate_to('task_details', task_details_id=task2_id)
                                
                                with sugg_cols[2]:
                                    if st.button(f"Entangle", key=f"create_sugg_{i}_{task1_id}_{task2_id}", use_container_width=True):
                                        result = entangle_tasks(task1_id, task2_id)
                                        if result:
                                            st.success("Entanglement created successfully")
                                            time.sleep(1)
                                            trigger_refresh()
                                
                                # Similarity bar
                                st.progress(similarity, f"Similarity: {similarity:.2f}")
                                st.divider()
                    else:
                        st.info("No suggested entanglements found with the current threshold.")
                else:
                    st.error("Failed to get entanglement suggestions")

def render_entropy_analytics():
    """Render entropy analytics and visualizations"""
    st.header("Quantum Entropy Analytics")
    
    # Get entropy data
    entropy_data = get_entropy_map()
    
    if not entropy_data:
        st.error("Failed to load entropy data")
        return
    
    # Display overview metrics
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        total_entropy = entropy_data.get("total_entropy", 0)
        st.metric("Total System Entropy", f"{total_entropy:.2f}")
    
    with metric_cols[1]:
        # Get max entropy state
        entropy_by_state = entropy_data.get("entropy_by_state", {})
        max_state = max(entropy_by_state.items(), key=lambda x: x[1]) if entropy_by_state else ("None", 0)
        st.metric("Highest Entropy State", f"{max_state[0]}: {max_state[1]:.2f}")
    
    with metric_cols[2]:
        # Count overloaded zones
        overloaded = entropy_data.get("overloaded_zones", [])
        st.metric("Overloaded Zones", len(overloaded))
    
    with metric_cols[3]:
        # Get entropy trend
        trend = entropy_data.get("entropy_trend", [])
        if trend and len(trend) >= 2:
            current = trend[-1].get("total_entropy", 0)
            previous = trend[-2].get("total_entropy", 0)
            delta = current - previous
            st.metric("Entropy Trend", f"{current:.2f}", delta=f"{delta:.2f}")
        else:
            st.metric("Entropy Trend", f"{total_entropy:.2f}")
    
    # Tabs for different entropy visualizations
    tab1, tab2, tab3 = st.tabs(["Entropy Distribution", "Entropy Over Time", "Overloaded Zones"])
    
    with tab1:
        # Entropy distribution by state
        st.subheader("Entropy Distribution by State")
        
        entropy_by_state = entropy_data.get("entropy_by_state", {})
        
        if entropy_by_state:
            # Prepare data for chart
            states = []
            values = []
            
            for state, value in entropy_by_state.items():
                if value > 0:
                    states.append(state)
                    values.append(value)
            
            if states:
                # Create dataframe
                df = pd.DataFrame({"State": states, "Entropy": values})
                
                # Plot
                st.bar_chart(df.set_index("State"))
                
                # Add description
                st.markdown("""
                **Entropy Distribution:**
                - Higher values indicate more uncertainty/volatility in that state
                - PENDING and ENTANGLED states typically have higher entropy
                - Low entropy in RESOLVED tasks is expected and healthy
                """)
            else:
                st.info("No entropy data to display")
        else:
            st.info("No entropy data available")
    
    with tab2:
        # Entropy over time
        st.subheader("System Entropy Over Time")
        
        trend = entropy_data.get("entropy_trend", [])
        
        if trend:
            # Extract timestamps and values
            times = []
            values = []
            
            for point in trend:
                try:
                    timestamp = datetime.fromisoformat(point.get("timestamp").replace('Z', '+00:00'))
                    times.append(timestamp)
                    values.append(point.get("total_entropy", 0))
                except:
                    continue
            
            if times:
                # Create dataframe
                df = pd.DataFrame({"Time": times, "Entropy": values})
                
                # Plot
                st.line_chart(df.set_index("Time"))
                
                # Add description
                st.markdown("""
                **Entropy Trend Analysis:**
                - Rising entropy indicates increasing system complexity or uncertainty
                - Declining entropy suggests tasks are being resolved or well-managed
                - Sudden spikes may indicate new batches of complex tasks
                """)
                
                # Calculate trend
                if len(values) >= 2:
                    first_value = values[0]
                    last_value = values[-1]
                    
                    if last_value > first_value:
                        st.warning(f"⚠️ System entropy has increased by {last_value - first_value:.2f} over this period")
                    elif last_value < first_value:
                        st.success(f"✅ System entropy has decreased by {first_value - last_value:.2f} over this period")
            else:
                st.info("No trend data to display")
        else:
            st.info("No entropy trend data available")
    
    with tab3:
        # Overloaded zones (assignees)
        st.subheader("Overloaded Zones Analysis")
        
        overloaded = entropy_data.get("overloaded_zones", [])
        
        if overloaded:
            st.warning(f"⚠️ Detected {len(overloaded)} overloaded zones that may need attention")
            
            # Create dataframe
            data = []
            for zone in overloaded:
                data.append({
                    "Assignee": zone.get("assignee", "Unknown"),
                    "Task Count": zone.get("task_count", 0),
                    "Total Entropy": zone.get("total_entropy", 0),
                    "Reason": zone.get("reason", "Unknown")
                })
            
            # Display as table
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            
            # Show details for each overloaded zone
            for i, zone in enumerate(overloaded):
                with st.expander(f"Details for {zone.get('assignee', 'Unknown')}", expanded=False):
                    st.markdown(f"**Assignee:** {zone.get('assignee', 'Unknown')}")
                    st.markdown(f"**Task Count:** {zone.get('task_count', 0)}")
                    st.markdown(f"**Total Entropy:** {zone.get('total_entropy', 0):.2f}")
                    st.markdown(f"**Reason:** {zone.get('reason', 'Unknown')}")
                    
                    # Show tasks
                    if 'task_ids' in zone and zone['task_ids']:
                        st.markdown("**Tasks:**")
                        
                        for task_id in zone['task_ids']:
                            task = get_task(task_id)
                            if task:
                                task_cols = st.columns([3, 1])
                                
                                with task_cols[0]:
                                    st.markdown(f"• {task.get('description', 'Unknown')[:50]}...")
                                
                                with task_cols[1]:
                                    if st.button(f"View", key=f"view_overload_{i}_{task_id}", use_container_width=True):
                                        navigate_to('task_details', task_details_id=task_id)
            
            # Suggestions section
            st.subheader("Optimization Suggestions")
            
            if st.button("Generate Task Redistribution Suggestions", type="primary", use_container_width=True):
                with st.spinner("Analyzing workload distribution..."):
                    suggestions = suggest_optimization()
                    
                    if suggestions and "suggestions" in suggestions:
                        suggestion_list = suggestions.get("suggestions", [])
                        
                        if suggestion_list:
                            st.success(f"Generated {len(suggestion_list)} redistribution suggestions")
                            
                            for suggestion in suggestion_list:
                                with st.container():
                                    st.markdown(f"**Task:** {suggestion.get('task_description', 'Unknown')[:50]}...")
                                    st.markdown(f"**Currently Assigned To:** {suggestion.get('current_assignee', 'Unknown')}")
                                    st.markdown(f"**Suggest Reassigning To:** {suggestion.get('suggested_assignee', 'Unknown')}")
                                    st.markdown(f"**Reason:** {suggestion.get('reason', 'Balance workload')}")
                                    
                                    # Add button to apply the suggestion
                                    if st.button("Apply Suggestion", key=f"apply_sugg_{suggestion.get('task_id')}", use_container_width=True):
                                        result = update_task(suggestion.get('task_id'), {
                                            "assignee": suggestion.get('suggested_assignee')
                                        })
                                        
                                        if result:
                                            st.success("Task reassigned successfully")
                                            time.sleep(1)
                                            trigger_refresh()
                                    
                                    st.divider()
                        else:
                            st.info("No redistribution suggestions available. The current distribution appears optimal.")
                    else:
                        st.error("Failed to generate optimization suggestions")
        else:
            st.success("✅ No overloaded zones detected. The system appears to be well-balanced.")

def render_suggestions():
    """Render system suggestions and optimization recommendations"""
    st.header("Quantum System Suggestions")
    
    tabs = st.tabs(["Task Entanglement Suggestions", "Workload Optimization", "Resolution Paths"])
    
    with tabs[0]:
        # Entanglement suggestions
        st.subheader("Suggested Task Entanglements")
        
        threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.9, value=0.65, step=0.05)
        
        if st.button("Find Potential Entanglements", type="primary", use_container_width=True):
            with st.spinner("Analyzing task similarities..."):
                suggestions = suggest_entanglements(threshold=threshold)
                
                if suggestions and "suggestions" in suggestions:
                    suggestion_list = suggestions.get("suggestions", [])
                    
                    if suggestion_list:
                        st.success(f"Found {len(suggestion_list)} potential entanglements")
                        
                        for i, suggestion in enumerate(suggestion_list):
                            task1_id = suggestion.get("task1")
                            task2_id = suggestion.get("task2")
                            similarity = suggestion.get("similarity", 0)
                            
                            task1_desc = suggestion.get("task1_description", "Unknown")
                            task2_desc = suggestion.get("task2_description", "Unknown")
                            
                            with st.container():
                                st.markdown(f"**Similarity Score:** {similarity:.2f}")
                                
                                sugg_cols = st.columns(2)
                                
                                with sugg_cols[0]:
                                    st.markdown(f"**Task 1:** {task1_desc[:40]}...")
                                    st.caption(f"ID: {task1_id[:8]}...")
                                    
                                    if st.button(f"View Task 1", key=f"view_sugg1_rec_{i}_{task1_id}", use_container_width=True):
                                        navigate_to('task_details', task_details_id=task1_id)
                                
                                with sugg_cols[1]:
                                    st.markdown(f"**Task 2:** {task2_desc[:40]}...")
                                    st.caption(f"ID: {task2_id[:8]}...")
                                    
                                    if st.button(f"View Task 2", key=f"view_sugg2_rec_{i}_{task2_id}", use_container_width=True):
                                        navigate_to('task_details', task_details_id=task2_id)
                                
                                action_cols = st.columns(2)
                                
                                with action_cols[0]:
                                    if st.button(f"Create Entanglement", key=f"create_sugg_rec_{i}_{task1_id}_{task2_id}", use_container_width=True):
                                        result = entangle_tasks(task1_id, task2_id)
                                        if result:
                                            st.success("Entanglement created successfully")
                                            time.sleep(1)
                                            trigger_refresh()
                                
                                with action_cols[1]:
                                    if st.button(f"Skip Suggestion", key=f"skip_sugg_{i}_{task1_id}_{task2_id}", use_container_width=True):
                                        pass
                                
                                st.divider()
                    else:
                        st.info("No suggested entanglements found with the current threshold. Try lowering the threshold for more suggestions.")
                else:
                    st.error("Failed to get entanglement suggestions")
    
    with tabs[1]:
        # Workload optimization
        st.subheader("Workload Redistribution Suggestions")
        
        if st.button("Analyze Workload Distribution", type="primary", use_container_width=True):
            with st.spinner("Analyzing workload and generating suggestions..."):
                suggestions = suggest_optimization()
                
                if suggestions and "suggestions" in suggestions:
                    suggestion_list = suggestions.get("suggestions", [])
                    
                    if suggestion_list:
                        st.success(f"Generated {len(suggestion_list)} workload optimization suggestions")
                        
                        for i, suggestion in enumerate(suggestion_list):
                            task_id = suggestion.get("task_id")
                            
                            with st.container():
                                st.markdown(f"**Task:** {suggestion.get('task_description', 'Unknown')[:50]}...")
                                
                                cols = st.columns(2)
                                
                                with cols[0]:
                                    st.markdown(f"**Currently Assigned To:** {suggestion.get('current_assignee', 'Unknown')}")
                                
                                with cols[1]:
                                    st.markdown(f"**Suggest Reassigning To:** {suggestion.get('suggested_assignee', 'Unknown')}")
                                
                                st.markdown(f"**Reason:** {suggestion.get('reason', 'Balance workload')}")
                                
                                action_cols = st.columns(3)
                                
                                with action_cols[0]:
                                    if st.button(f"Apply Suggestion", key=f"apply_opt_{i}_{task_id}", use_container_width=True):
                                        result = update_task(task_id, {
                                            "assignee": suggestion.get('suggested_assignee')
                                        })
                                        
                                        if result:
                                            st.success("Task reassigned successfully")
                                            time.sleep(1)
                                            trigger_refresh()
                                
                                with action_cols[1]:
                                    if st.button(f"View Task", key=f"view_opt_{i}_{task_id}", use_container_width=True):
                                        navigate_to('task_details', task_details_id=task_id)
                                
                                with action_cols[2]:
                                    if st.button(f"Skip", key=f"skip_opt_{i}_{task_id}", use_container_width=True):
                                        pass
                                
                                st.divider()
                    else:
                        st.info("No workload optimization suggestions available. The current distribution appears optimal.")
                else:
                    st.error("Failed to generate optimization suggestions")
    
    with tabs[2]:
        # Resolution paths
        st.subheader("Task Resolution Path Analysis")
        
        # Get all tasks
        tasks_data = get_all_tasks(state="PENDING")
        
        if tasks_data and "tasks" in tasks_data:
            tasks = tasks_data.get("tasks", [])
            
            # Filter to tasks without resolution paths
            tasks_without_paths = [task for task in tasks if not task.get("multiverse_paths")]
            
            if tasks_without_paths:
                st.info(f"Found {len(tasks_without_paths)} tasks without resolution paths")
                
                # Select a task to generate paths for
                task_options = ["Select a task"] + [f"{t.get('description', 'Unknown')[:40]}... ({t.get('id', '')[:8]})" for t in tasks_without_paths]
                selected_task = st.selectbox("Select Task to Generate Resolution Paths", task_options)
                
                if selected_task != "Select a task":
                    task_index = task_options.index(selected_task) - 1
                    selected_task_id = tasks_without_paths[task_index].get("id")
                    selected_task_description = tasks_without_paths[task_index].get("description")
                    
                    st.markdown(f"**Task:** {selected_task_description}")
                    
                    # Generate resolution paths
                    if st.button("Generate Resolution Paths", type="primary", use_container_width=True):
                        with st.spinner("Analyzing task and generating resolution paths..."):
                            # Simulate path generation (in a real system, this would use LLM)
                            time.sleep(2)
                            
                            # Generate some possible paths
                            paths = [
                                f"Complete the task by following the standard procedure for {selected_task_description[:20]}...",
                                f"Consult with team members to develop a collaborative approach for {selected_task_description[:15]}...",
                                f"Break down the task into smaller sub-tasks to manage complexity"
                            ]
                            
                            # Update the task with paths
                            result = update_task(selected_task_id, {
                                "multiverse_paths": paths
                            })
                            
                            if result:
                                st.success("Resolution paths generated successfully")
                                
                                for i, path in enumerate(paths):
                                    st.markdown(f"{i+1}. {path}")
                                
                                # Option to collapse superposition
                                if st.button("Collapse Superposition (Choose Optimal Path)", use_container_width=True):
                                    collapse_result = collapse_task_superposition(selected_task_id)
                                    if collapse_result:
                                        st.success("Superposition collapsed successfully")
                                        time.sleep(1)
                                        trigger_refresh()
                            else:
                                st.error("Failed to update task with resolution paths")
            else:
                st.success("All tasks have resolution paths defined")
        else:
            st.error("Failed to load tasks")

def render_system_page():
    """Render system management and statistics"""
    st.header("Quantum System Management")
    
    # System tabs
    tab1, tab2, tab3 = st.tabs(["System Overview", "Embedding Engine", "Backup & Restore"])
    
    with tab1:
        # System overview and status
        st.subheader("System Status")
        
        # Get system info
        system_info = api_request("/")
        
        if system_info:
            status_cols = st.columns(3)
            
            with status_cols[0]:
                st.metric("System Status", "Operational" if system_info.get("status") == "operational" else "Error")
            
            with status_cols[1]:
                st.metric("API Version", system_info.get("version", "Unknown"))
            
            with status_cols[2]:
                st.metric("Task Count", system_info.get("task_count", 0))
        else:
            st.error("Failed to load system information")
        
        # System snapshot
        st.subheader("System Snapshot")
        snapshot = get_system_snapshot()
        
        if snapshot:
            # Basic statistics
            snapshot_cols = st.columns(3)
            
            with snapshot_cols[0]:
                timestamp = snapshot.get("timestamp", "Unknown")
                if timestamp != "Unknown":
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                
                st.markdown(f"**Snapshot Time:** {timestamp}")
                st.markdown(f"**Task Count:** {snapshot.get('task_count', 0)}")
            
            with snapshot_cols[1]:
                # Count tasks by state
                states = {}
                for task in snapshot.get("tasks", {}).values():
                    state = task.get("state", "PENDING")
                    if state in states:
                        states[state] += 1
                    else:
                        states[state] = 1
                
                for state, count in states.items():
                    st.markdown(f"**{state}:** {count}")
            
            with snapshot_cols[2]:
                # Entropy info
                entropy_map = snapshot.get("entropy_map", {})
                st.markdown(f"**Total Entropy:** {entropy_map.get('total_entropy', 0):.2f}")
                st.markdown(f"**Overloaded Zones:** {len(entropy_map.get('overloaded_zones', []))}")
            
            # Recent activity
            st.subheader("Recent System Activity")
            
            recent_activity = snapshot.get("recent_activity", [])
            
            if recent_activity:
                data = []
                for activity in recent_activity:
                    timestamp = format_date(activity.get("timestamp", ""))
                    action = activity.get("action", "UNKNOWN")
                    details = ""
                    
                    if "task_id" in activity:
                        details += f"Task: {activity['task_id'][:8]}..."
                    
                    if "old_state" in activity and "new_state" in activity:
                        details += f" | {activity['old_state']} → {activity['new_state']}"
                    
                    data.append({
                        "Time": timestamp,
                        "Action": action,
                        "Details": details
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No recent activity recorded")
        else:
            st.error("Failed to load system snapshot")
    
    with tab2:
        # Embedding engine statistics
        st.subheader("Embedding Engine Statistics")
        
        embedding_stats = get_embedding_statistics()
        
        if embedding_stats:
            status = embedding_stats.get("status", "not_initialized")
            
            if status == "initialized":
                st.success("✅ Embedding engine is active and operational")
                
                stat_cols = st.columns(3)
                
                with stat_cols[0]:
                    st.markdown(f"**Model:** {embedding_stats.get('model', 'Unknown')}")
                    st.markdown(f"**Embedding Dimension:** {embedding_stats.get('embedding_dimension', 0)}")
                
                with stat_cols[1]:
                    st.markdown(f"**Indexed Tasks:** {embedding_stats.get('indexed_tasks', 0)}")
                    
                    # Get current similarity threshold
                    threshold_info = embedding_stats.get("similarity_threshold", {})
                    current_threshold = threshold_info.get("current", 0.7)
                    
                    st.markdown(f"**Current Similarity Threshold:** {current_threshold:.2f}")
                
                with stat_cols[2]:
                    # Show distribution info
                    dist = embedding_stats.get("similarity_distribution", {})
                    if "mean" in dist and dist["mean"] is not None:
                        st.markdown(f"**Mean Similarity:** {dist['mean']:.3f}")
                    if "median" in dist and dist["median"] is not None:
                        st.markdown(f"**Median Similarity:** {dist['median']:.3f}")
                    if "samples" in dist:
                        st.markdown(f"**Samples:** {dist['samples']}")
                
                # Threshold history
                st.subheader("Threshold Adaptation History")
                
                threshold_history = threshold_info.get("history", [])
                
                if threshold_history:
                    data = []
                    for entry in threshold_history:
                        try:
                            timestamp = datetime.fromisoformat(entry.get("timestamp", "").replace('Z', '+00:00'))
                            threshold = entry.get("threshold", 0)
                            samples = entry.get("samples", 0)
                            
                            data.append({
                                "Time": timestamp,
                                "Threshold": threshold,
                                "Samples": samples
                            })
                        except:
                            continue
                    
                    if data:
                        df = pd.DataFrame(data)
                        
                        # Plot threshold over time
                        st.line_chart(df.set_index("Time")["Threshold"])
                        
                        st.caption("The system automatically adapts the similarity threshold based on the distribution of task similarities")
                    else:
                        st.info("No threshold history data available")
                else:
                    st.info("No threshold adaptation history available")
            else:
                st.warning("⚠️ Embedding engine is not initialized")
                
                fallback = embedding_stats.get("fallback_method", "none")
                reason = embedding_stats.get("reason", "Unknown reason")
                
                st.markdown(f"**Fallback Method:** {fallback}")
                st.markdown(f"**Reason:** {reason}")
                
                st.info("The system is using text-based similarity as a fallback instead of neural embeddings")
        else:
            st.error("Failed to load embedding engine statistics")
    
    with tab3:
        # Backup and restore
        st.subheader("System Backup & Restore")
        
        # Create backup button
        if st.button("Create System Backup", type="primary", use_container_width=True):
            result = create_system_backup()
            
            if result and result.get("status") == "success":
                st.success("System backup created successfully")
            else:
                st.error("Failed to create system backup")

# Main application
def main():
    """Main application entry point"""
    # Add header with system info
    render_header()
    
    # Add sidebar navigation
    render_sidebar()
    
    # Main content area - conditional rendering based on navigation state
    if st.session_state.page == 'dashboard':
        render_dashboard()
    
    elif st.session_state.page == 'tasks':
        render_task_management()
    
    elif st.session_state.page == 'relationships':
        render_relationship_view()
    
    elif st.session_state.page == 'entropy':
        render_entropy_analytics()
    
    elif st.session_state.page == 'suggestions':
        render_suggestions()
    
    elif st.session_state.page == 'system':
        render_system_page()
    
    elif st.session_state.page == 'new_task':
        render_new_task_form()
    
    elif st.session_state.page == 'edit_task':
        render_edit_task_form(st.session_state.edit_task_id)
    
    elif st.session_state.page == 'task_details':
        render_task_details(st.session_state.task_details_id)
    
    elif st.session_state.page == 'find_related':
        render_find_related(st.session_state.task_details_id)
    
    else:
        st.error(f"Unknown page: {st.session_state.page}")

if __name__ == "__main__":
    main()
