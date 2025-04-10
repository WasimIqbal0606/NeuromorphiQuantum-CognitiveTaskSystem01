"""
Advanced embedding and similarity engine for Neuromorphic Quantum-Cognitive Task System
"""

import numpy as np
import threading
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    import faiss
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


class EmbeddingEngine:
    """Manages neural embeddings for tasks with adaptive similarity thresholds"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.lock = threading.RLock()
        self.initialized = False
        self.model_name = model_name
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        if not EMBEDDING_AVAILABLE:
            print("⚠️ Embedding libraries not available. Using fallback text matching.")
            return
        
        try:
            # Initialize the embedding model
            self.embedding_model = SentenceTransformer(model_name)
            print(f"✅ Embedding model {model_name} initialized successfully")
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.Client()
            try:
                self.task_collection = self.chroma_client.get_collection("task_metadata")
                print("✅ Using existing Chroma collection")
            except:
                self.task_collection = self.chroma_client.create_collection("task_metadata")
                print("✅ Created new Chroma collection")
            
            # Initialize FAISS
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self.id_to_index = {}  # Map task ID to FAISS index
            self.index_to_id = {}  # Map FAISS index to task ID
            self.next_index = 0  # Next available FAISS index
            print("✅ FAISS index initialized")
            
            # Track similarity statistics for adaptive thresholding
            self.similarity_scores = []
            self.threshold_history = []
            
            self.initialized = True
            
        except Exception as e:
            print(f"⚠️ Error initializing embeddings: {e}")
            print("⚠️ Using fallback text matching without embeddings")
    
    def add_task_embedding(self, task):
        """Add embedding for a task"""
        if not self.initialized:
            return False
        
        with self.lock:
            try:
                # Create embedding for the task
                task.embedding = self.embedding_model.encode(task.description)
                
                # Add to FAISS index
                vector = np.array([task.embedding], dtype=np.float32)
                self.faiss_index.add(vector)
                self.id_to_index[task.id] = self.next_index
                self.index_to_id[self.next_index] = task.id
                self.next_index += 1
                
                # Add to ChromaDB
                self.task_collection.add(
                    documents=[task.description],
                    metadatas=[{
                        "id": task.id,
                        "priority": float(task.priority),
                        "state": task.state,
                        "entropy": float(task.entropy),
                        "assignee": task.assignee or "unassigned",
                        "tags": ",".join(task.tags) if task.tags else ""
                    }],
                    ids=[task.id]
                )
                
                return True
            except Exception as e:
                print(f"⚠️ Error adding task embedding: {e}")
                return False
    
    def update_task_embedding(self, task):
        """Update embedding for a task that has changed"""
        if not self.initialized:
            return False
        
        with self.lock:
            try:
                # Remove old embedding from FAISS
                if task.id in self.id_to_index:
                    old_index = self.id_to_index[task.id]
                    # FAISS doesn't support direct removal, so we'd rebuild the index
                    # This is a simple version - in production, we'd use a more efficient approach
                    # like maintaining a mask of active indices
                    
                # Generate new embedding
                task.embedding = self.embedding_model.encode(task.description)
                
                # Update ChromaDB
                self.task_collection.update(
                    documents=[task.description],
                    metadatas=[{
                        "id": task.id,
                        "priority": float(task.priority),
                        "state": task.state,
                        "entropy": float(task.entropy),
                        "assignee": task.assignee or "unassigned",
                        "tags": ",".join(task.tags) if task.tags else ""
                    }],
                    ids=[task.id]
                )
                
                # For FAISS, we'll just add it again and keep track of the latest index
                vector = np.array([task.embedding], dtype=np.float32)
                self.faiss_index.add(vector)
                
                # Update index mappings
                if task.id in self.id_to_index:
                    old_index = self.id_to_index[task.id]
                    del self.index_to_id[old_index]
                
                self.id_to_index[task.id] = self.next_index
                self.index_to_id[self.next_index] = task.id
                self.next_index += 1
                
                return True
            except Exception as e:
                print(f"⚠️ Error updating task embedding: {e}")
                return False
    
    def update_task_metadata(self, task):
        """Update just the metadata for a task without changing the embedding"""
        if not self.initialized:
            return False
        
        with self.lock:
            try:
                # Update ChromaDB metadata
                self.task_collection.update(
                    metadatas=[{
                        "id": task.id,
                        "priority": float(task.priority),
                        "state": task.state,
                        "entropy": float(task.entropy),
                        "assignee": task.assignee or "unassigned",
                        "tags": ",".join(task.tags) if task.tags else ""
                    }],
                    ids=[task.id]
                )
                
                return True
            except Exception as e:
                print(f"⚠️ Error updating task metadata: {e}")
                return False
    
    def remove_task_embedding(self, task_id):
        """Remove task embedding when a task is deleted"""
        if not self.initialized:
            return False
        
        with self.lock:
            try:
                # Remove from ChromaDB
                self.task_collection.delete(ids=[task_id])
                
                # For FAISS, rebuilding the index would be needed for proper removal
                # Here we just update our mappings and ignore the orphaned vector
                if task_id in self.id_to_index:
                    index = self.id_to_index[task_id]
                    del self.id_to_index[task_id]
                    if index in self.index_to_id:
                        del self.index_to_id[index]
                
                return True
            except Exception as e:
                print(f"⚠️ Error removing task embedding: {e}")
                return False
    
    def find_similar_tasks(self, task, threshold=0.7, max_results=5):
        """Find tasks similar to the given task using vector similarity"""
        if not self.initialized:
            return []
        
        with self.lock:
            try:
                # If task doesn't have an embedding yet, create one
                query_embedding = task.embedding
                if query_embedding is None:
                    query_embedding = self.embedding_model.encode(task.description)
                
                # Prepare query vector for FAISS
                query_vector = np.array([query_embedding], dtype=np.float32)
                
                # Get the total number of vectors in the index
                total_vectors = self.faiss_index.ntotal
                
                # If index is empty, return empty list
                if total_vectors == 0:
                    return []
                
                # Search for similar tasks with FAISS
                k = min(max_results + 1, total_vectors)  # +1 to account for self match
                distances, indices = self.faiss_index.search(query_vector, k)
                
                # Convert distances to similarity scores and map to task IDs
                similar_task_ids = []
                
                for i, idx in enumerate(indices[0]):
                    # Look up the task ID from the index
                    if idx in self.index_to_id:
                        task_id = self.index_to_id[idx]
                        
                        # Skip self
                        if task_id == task.id:
                            continue
                        
                        # Convert L2 distance to similarity score (1 = identical, 0 = completely different)
                        # This is a simple conversion and can be improved
                        similarity = 1.0 - min(1.0, distances[0][i] / (2 * self.embedding_dim))
                        
                        # Track similarity statistics for adaptive thresholding
                        self.similarity_scores.append(similarity)
                        if len(self.similarity_scores) > 100:
                            self.similarity_scores = self.similarity_scores[-100:]
                        
                        if similarity >= threshold:
                            similar_task_ids.append(task_id)
                
                return similar_task_ids
            except Exception as e:
                print(f"⚠️ Error finding similar tasks: {e}")
                return []
    
    def calculate_similarity(self, task1, task2):
        """Calculate similarity between two tasks"""
        if not self.initialized:
            # Fallback to simple text matching
            task1_words = set(task1.description.lower().split())
            task2_words = set(task2.description.lower().split())
            
            if not task1_words or not task2_words:
                return 0.0
            
            return len(task1_words.intersection(task2_words)) / len(task1_words.union(task2_words))
        
        with self.lock:
            try:
                # Make sure both tasks have embeddings
                if task1.embedding is None:
                    task1.embedding = self.embedding_model.encode(task1.description)
                
                if task2.embedding is None:
                    task2.embedding = self.embedding_model.encode(task2.description)
                
                # Calculate cosine similarity
                dot_product = np.dot(task1.embedding, task2.embedding)
                norm1 = np.linalg.norm(task1.embedding)
                norm2 = np.linalg.norm(task2.embedding)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return dot_product / (norm1 * norm2)
            except Exception as e:
                print(f"⚠️ Error calculating similarity: {e}")
                return 0.0
    
    def get_adaptive_threshold(self):
        """Calculate adaptive threshold based on historical similarity scores"""
        if not self.similarity_scores:
            return 0.7  # Default threshold
        
        # Calculate percentile-based threshold
        # Use the 33rd percentile to get a reasonable threshold
        scores = np.array(self.similarity_scores)
        threshold = max(0.65, np.percentile(scores, 33))
        
        # Record threshold history
        self.threshold_history.append({
            "timestamp": datetime.now().isoformat(),
            "threshold": threshold,
            "samples": len(scores)
        })
        
        # Keep only last 20 threshold calculations
        if len(self.threshold_history) > 20:
            self.threshold_history = self.threshold_history[-20:]
        
        return threshold
    
    def contextual_search(self, query_text, filters=None, max_results=10):
        """Search for tasks using natural language query with optional filters"""
        if not self.initialized:
            return []
        
        with self.lock:
            try:
                # Generate embedding for query
                query_embedding = self.embedding_model.encode(query_text)
                
                # Search in ChromaDB, which supports metadata filtering
                where_clause = {}
                if filters:
                    if filters.get('state'):
                        where_clause["state"] = filters['state']
                    if filters.get('assignee'):
                        where_clause["assignee"] = filters['assignee']
                    if filters.get('min_priority') is not None:
                        where_clause["priority"] = {"$gte": filters['min_priority']}
                
                # Prepare query for ChromaDB
                results = self.task_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=max_results,
                    where=where_clause if where_clause else None
                )
                
                # Process results
                if results and results['ids'] and results['distances']:
                    task_results = []
                    
                    for i, task_id in enumerate(results['ids'][0]):
                        distance = results['distances'][0][i]
                        # Convert distance to similarity (assuming normalized L2 distance)
                        similarity = 1.0 - min(1.0, distance / 2.0)
                        
                        task_results.append({
                            'task_id': task_id,
                            'similarity': similarity,
                            'metadata': results['metadatas'][0][i] if 'metadatas' in results else None
                        })
                    
                    return task_results
                
                return []
            except Exception as e:
                print(f"⚠️ Error in contextual search: {e}")
                return []
    
    def get_embedding_statistics(self):
        """Get statistics about the embedding engine"""
        if not self.initialized:
            return {
                "status": "not_initialized",
                "fallback_method": "text_matching",
                "reason": "Required libraries not available"
            }
        
        with self.lock:
            return {
                "status": "initialized",
                "model": self.model_name,
                "embedding_dimension": self.embedding_dim,
                "indexed_tasks": self.faiss_index.ntotal,
                "similarity_threshold": {
                    "current": self.get_adaptive_threshold(),
                    "history": self.threshold_history[-5:] if self.threshold_history else []
                },
                "similarity_distribution": {
                    "mean": np.mean(self.similarity_scores) if self.similarity_scores else None,
                    "median": np.median(self.similarity_scores) if self.similarity_scores else None,
                    "min": min(self.similarity_scores) if self.similarity_scores else None,
                    "max": max(self.similarity_scores) if self.similarity_scores else None,
                    "samples": len(self.similarity_scores)
                }
            }
