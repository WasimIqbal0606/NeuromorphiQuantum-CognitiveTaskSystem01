"""
Visualization utilities for the Neuromorphic Quantum-Cognitive Task System
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import io
import base64
from typing import List, Dict, Any, Tuple, Optional


def create_entropy_chart(entropy_data):
    """Create a chart of entropy over time"""
    timestamps = []
    entropy_values = []
    
    # Extract data from entropy measurements
    for measurement in entropy_data["entropy_trend"]:
        try:
            timestamps.append(datetime.fromisoformat(measurement["timestamp"]))
            entropy_values.append(measurement["total_entropy"])
        except (ValueError, KeyError) as e:
            # Skip invalid entries
            continue
    
    if not timestamps:
        # Return empty chart if no data
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, entropy_values, 'b-', marker='o', markersize=4)
    
    # Set labels and title
    ax.set_title('Quantum System Entropy Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Entropy')
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(timestamps) > 1:
        z = np.polyfit(np.array([(t - timestamps[0]).total_seconds() for t in timestamps]), 
                       entropy_values, 1)
        p = np.poly1d(z)
        x_trend = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        ax.plot(timestamps, p(x_trend), "r--", alpha=0.8)
    
    # Convert plot to base64 encoded string
    buffer = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def create_state_distribution_chart(tasks):
    """Create a pie chart of task states"""
    # Count tasks by state
    state_counts = {}
    for task in tasks:
        state = task.state
        if state in state_counts:
            state_counts[state] += 1
        else:
            state_counts[state] = 1
    
    if not state_counts:
        # Return empty chart if no data
        return None
    
    # Create color mapping for states
    state_colors = {
        'PENDING': '#fcba03',     # Yellow
        'ENTANGLED': '#0373fc',   # Blue
        'RESOLVED': '#27ae60',    # Green
        'DEFERRED': '#95a5a6',    # Gray
        'CANCELLED': '#e74c3c'    # Red
    }
    
    # Extract data for the chart
    labels = list(state_counts.keys())
    sizes = list(state_counts.values())
    colors = [state_colors.get(state, '#7f8c8d') for state in labels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
           startangle=90, shadow=False)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Set title
    ax.set_title('Task State Distribution')
    
    # Convert plot to base64 encoded string
    buffer = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def create_assignee_workload_chart(tasks):
    """Create a bar chart of workload distribution across assignees"""
    # Count tasks and entropy by assignee
    assignee_tasks = {}
    assignee_entropy = {}
    
    for task in tasks:
        assignee = task.assignee or "Unassigned"
        
        if assignee not in assignee_tasks:
            assignee_tasks[assignee] = 0
            assignee_entropy[assignee] = 0
        
        assignee_tasks[assignee] += 1
        assignee_entropy[assignee] += task.entropy
    
    if not assignee_tasks:
        # Return empty chart if no data
        return None
    
    # Sort by task count
    sorted_assignees = sorted(assignee_tasks.keys(), 
                             key=lambda x: assignee_tasks[x], 
                             reverse=True)
    
    # Extract data for the chart
    labels = sorted_assignees
    task_counts = [assignee_tasks[a] for a in sorted_assignees]
    entropy_values = [assignee_entropy[a] for a in sorted_assignees]
    
    # Handle case with too many assignees
    if len(labels) > 10:
        # Show top 9 and group others
        others_tasks = sum(task_counts[9:])
        others_entropy = sum(entropy_values[9:])
        
        labels = labels[:9] + ["Others"]
        task_counts = task_counts[:9] + [others_tasks]
        entropy_values = entropy_values[:9] + [others_entropy]
    
    # Create figure with two Y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot task counts as bars
    x = np.arange(len(labels))
    width = 0.35
    bars = ax1.bar(x, task_counts, width, label='Task Count', color='#3498db')
    ax1.set_ylabel('Number of Tasks')
    ax1.set_title('Workload Distribution by Assignee')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    
    # Create second Y-axis for entropy
    ax2 = ax1.twinx()
    ax2.plot(x, entropy_values, 'ro-', label='Total Entropy')
    ax2.set_ylabel('Total Entropy')
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Convert plot to base64 encoded string
    buffer = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def create_priority_histogram(tasks):
    """Create a histogram of task priorities"""
    if not tasks:
        return None
    
    # Extract priorities
    priorities = [task.priority for task in tasks]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create histogram
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    ax.hist(priorities, bins=bins, edgecolor='black', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Priority')
    ax.set_ylabel('Number of Tasks')
    ax.set_title('Task Priority Distribution')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Convert plot to base64 encoded string
    buffer = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def create_deadline_chart(tasks):
    """Create a chart showing task distribution by deadline"""
    # Filter tasks with deadlines
    tasks_with_deadlines = [task for task in tasks if task.deadline]
    
    if not tasks_with_deadlines:
        return None
    
    # Group tasks by week
    weeks = {}
    today = datetime.now().date()
    
    for task in tasks_with_deadlines:
        deadline = task.deadline.date()
        days_remaining = (deadline - today).days
        
        if days_remaining < 0:
            week_key = "Overdue"
        elif days_remaining < 7:
            week_key = "This Week"
        elif days_remaining < 14:
            week_key = "Next Week"
        elif days_remaining < 30:
            week_key = "This Month"
        else:
            week_key = "Future"
        
        if week_key not in weeks:
            weeks[week_key] = []
        
        weeks[week_key].append(task)
    
    # Order the keys logically
    ordered_keys = ["Overdue", "This Week", "Next Week", "This Month", "Future"]
    # Filter to only include keys that have tasks
    ordered_keys = [key for key in ordered_keys if key in weeks]
    
    # Count tasks in each time period
    counts = [len(weeks[key]) for key in ordered_keys]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create bars with sequential colors
    bars = ax.bar(ordered_keys, counts, color=plt.cm.viridis(np.linspace(0, 0.9, len(ordered_keys))))
    
    # Set labels and title
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Number of Tasks')
    ax.set_title('Task Deadline Distribution')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    # Convert plot to base64 encoded string
    buffer = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def create_entanglement_network_visualization(network_data):
    """Create a network visualization of task entanglements"""
    # This would normally use a network visualization library
    # For the purposes of this prototype, we'll create a simple visualization
    
    nodes = network_data.get("nodes", [])
    edges = network_data.get("edges", [])
    
    if not nodes or not edges:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Compute a simple layout (in a real implementation, use networkx or similar)
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
    
    # Convert plot to base64 encoded string
    buffer = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64
