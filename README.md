# Neuromorphic Quantum-Cognitive Task Management System

A sophisticated task management system inspired by quantum mechanics and cognitive science, featuring neural embeddings, task entanglement, and entropy-based analytics.

## Features

- **Quantum-Inspired Task States:** Tasks exist in superposition (PENDING), entanglement (linked to other tasks), or collapsed states (RESOLVED)
- **Neural Embeddings:** Semantic understanding of task relationships through vector embeddings
- **Task Entanglement:** Discover and manage relationships between related tasks
- **Entropy Analytics:** Track system complexity and identify overloaded work zones
- **Adaptive Similarity Algorithm:** Self-adjusting threshold for task relationship detection
- **Multiverse Paths:** Track multiple potential resolution paths for each task
- **Advanced Visualizations:** Network graphs, entropy trends, and workload distribution

## System Architecture

The system consists of the following components:

1. **Task Multiverse Core:** Manages the quantum task state, relationships, and operations
2. **Embedding Engine:** Provides semantic understanding of task relationships
3. **Persistence Manager:** Handles data persistence and snapshots
4. **Visualization Tools:** Provides visual analytics and insights
5. **FastAPI Backend:** Exposes core functionality via REST API
6. **Streamlit UI:** Interactive web interface for managing tasks and analytics

## Technology Stack

- **Backend:** Python, FastAPI, LangChain integration
- **Embeddings:** SentenceTransformer, FAISS, ChromaDB
- **Frontend:** Streamlit with advanced data visualization
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib for charts and graphs

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (install via pip):
  - streamlit
  - fastapi
  - uvicorn
  - sentence-transformers
  - chromadb
  - faiss-cpu
  - numpy
  - pandas
  - matplotlib
  - requests
  - langchain
  - langchain_openai (optional for LLM integration)

### Installation

1. Clone this repository
2. Install required packages:
```bash
pip install streamlit fastapi uvicorn sentence-transformers chromadb faiss-cpu numpy pandas matplotlib requests
