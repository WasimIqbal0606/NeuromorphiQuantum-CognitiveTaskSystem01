"""
Utility functions for the Neuromorphic Quantum-Cognitive Task System
"""

import random
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Constants for the system
DEFAULT_SIMILARITY_THRESHOLD = 0.7
MAX_TASK_RESULTS = 100
ENTROPY_DECAY_RATE = 0.05
ENTANGLEMENT_STRENGTH_FACTOR = 0.2
DEFAULT_PERSISTENCE_INTERVAL = 300  # seconds


def generate_id():
    """Generate a unique ID for a task"""
    return str(uuid.uuid4())


def format_datetime(dt):
    """Format a datetime object to a readable string"""
    if not dt:
        return "N/A"
    
    now = datetime.now()
    diff = now - dt
    
    if diff.days < 0:
        # Future date
        if diff.days > -7:
            # Within a week
            return f"In {abs(diff.days)} days"
        else:
            return dt.strftime("%Y-%m-%d")
    elif diff.days == 0:
        # Today
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


def format_time_remaining(deadline):
    """Format time remaining until deadline in a readable format"""
    if not deadline:
        return "No deadline"
    
    now = datetime.now()
    diff = deadline - now
    
    if diff.days < 0:
        # Past deadline
        if diff.days == -1:
            return "Overdue by 1 day"
        else:
            return f"Overdue by {abs(diff.days)} days"
    elif diff.days == 0:
        # Due today
        if diff.seconds < 3600:
            return f"Due in {diff.seconds // 60} minutes"
        else:
            return f"Due in {diff.seconds // 3600} hours"
    elif diff.days == 1:
        return "Due tomorrow"
    elif diff.days < 7:
        return f"Due in {diff.days} days"
    elif diff.days < 30:
        return f"Due in {diff.days // 7} weeks"
    else:
        return deadline.strftime("%Y-%m-%d")


def calculate_priority_class(priority):
    """Map a numeric priority to a descriptive class"""
    if priority >= 0.8:
        return "Critical", "danger"
    elif priority >= 0.6:
        return "High", "warning"
    elif priority >= 0.4:
        return "Medium", "primary"
    elif priority >= 0.2:
        return "Low", "info"
    else:
        return "Minimal", "secondary"


def get_state_badge_color(state):
    """Map a task state to an appropriate color for display"""
    state_colors = {
        "PENDING": "warning",
        "ENTANGLED": "primary",
        "RESOLVED": "success",
        "DEFERRED": "secondary",
        "CANCELLED": "danger"
    }
    return state_colors.get(state, "secondary")


def compute_task_urgency(task):
    """Compute a task's urgency score based on priority and deadline"""
    base_score = task.priority * 100
    
    if task.deadline:
        # Calculate days remaining
        days_remaining = (task.deadline - datetime.now()).days
        
        if days_remaining <= 0:
            # Past deadline
            urgency_factor = 2.0
        elif days_remaining <= 1:
            # Due within a day
            urgency_factor = 1.75
        elif days_remaining <= 3:
            # Due within 3 days
            urgency_factor = 1.5
        elif days_remaining <= 7:
            # Due within a week
            urgency_factor = 1.25
        else:
            urgency_factor = 1.0
        
        # Apply urgency factor to base score
        return base_score * urgency_factor
    
    return base_score


def get_suggested_tags(description):
    """Extract suggested tags from a task description using simple heuristics"""
    tags = []
    
    # Common categories
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
    
    # Check if any words in description match categories
    description_lower = description.lower()
    for category, keywords in categories.items():
        if any(keyword in description_lower for keyword in keywords):
            tags.append(category)
    
    return tags


def truncate_text(text, max_length=50):
    """Truncate text to a maximum length and add ellipsis if needed"""
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def sanitize_html(text):
    """Sanitize text for safe HTML display"""
    if not text:
        return ""
    
    # Simple sanitization - replace problematic characters
    sanitized = text.replace("<", "&lt;").replace(">", "&gt;")
    return sanitized
