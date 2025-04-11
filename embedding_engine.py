"""
Advanced embedding and similarity engine for Neuromorphic Quantum-Cognitive Task System
"""

import numpy as np
import threading
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime

# Wrap imports in try-except to handle unavailable dependencies
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    import faiss
    EMBEDDING_AVAILABLE = True
except ImportError as e:
    EMBEDDING_AVAILABLE = False
    IMPORT_ERROR_MESSAGE = str(e)


class EmbeddingEngine:
    """Manages neural embeddings for tasks with adaptive similarity thresholds"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.lock = threading.RLock()
        self.initialized = False
        self.model_name = model_name
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self.error_reason = None
        self.fallback_method = "text_matching"
        
        if not EMBEDDING_AVAILABLE:
            self.error_reason = f"Required libraries not available: {IMPORT_ERROR_MESSAGE}"
            print(f"⚠️ Embedding libraries not available: {IMPORT_ERROR_MESSAGE}")
            print("⚠️ Using fallback text matching.")
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
            except Exception as ce:
                print(f"Info: {ce}")
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
            self.error_reason = f"Error initializing embedding system: {str(e)}"
            print(f"⚠️ Error initializing embeddings: {e}")
            print("⚠️ Using fallback text matching without embeddings")
    
    def add_task_embedding(self, task):
        """Add embedding for a task"""
        if not self.initialized:
            return False
        
        with self.lock:
            try:
                # Create embedding for the task
                task_embedding = self.embedding_model.encode(task.description)
                task.embedding = task_embedding  # Assign to task object
                
                # Add to FAISS index
                vector = np.array([task_embedding], dtype=np.float32)
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
                # Remove old embedding from ChromaDB if it exists
                try:
                    self.task_collection.delete(ids=[task.id])
                except Exception as e:
                    print(f"Note: Could not delete old task embedding: {e}")
                
                # Generate new embedding
                task_embedding = self.embedding_model.encode(task.description)
                task.embedding = task_embedding  # Update task object
                
                # Update ChromaDB
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
                
                # For FAISS, we'll just add it again and keep track of the latest index
                vector = np.array([task_embedding], dtype=np.float32)
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
                # Check if task exists in collection
                try:
                    results = self.task_collection.get(ids=[task.id])
                    if not results or not results['ids']:
                        print(f"Task {task.id} not found in collection, adding instead")
                        return self.add_task_embedding(task)
                except Exception as e:
                    print(f"Error checking task: {e}")
                    return self.add_task_embedding(task)
                
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
                try:
                    self.task_collection.delete(ids=[task_id])
                except Exception as e:
                    print(f"Note: Could not delete task from ChromaDB: {e}")
                
                # For FAISS, update our mappings and ignore the orphaned vector
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
            # Use fallback text matching
            return self._fallback_find_similar_tasks(task, threshold, max_results)
        
        with self.lock:
            try:
                # If task doesn't have an embedding yet, create one
                query_embedding = getattr(task, 'embedding', None)
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
                            similar_task_ids.append((task_id, similarity))
                
                # Sort by similarity and return just the IDs
                similar_task_ids.sort(key=lambda x: x[1], reverse=True)
                return [task_id for task_id, _ in similar_task_ids[:max_results]]
            except Exception as e:
                print(f"⚠️ Error finding similar tasks: {e}")
                return self._fallback_find_similar_tasks(task, threshold, max_results)
    
    def _fallback_find_similar_tasks(self, task, threshold=0.7, max_results=5):
        """Fallback method to find similar tasks when embeddings are not available"""
        # This would require all tasks to be provided or fetched from a database
        # In a real implementation, you would need to pass the task collection or fetch it
        print("Using fallback text matching for finding similar tasks")
        return []  # In a real implementation, return actual matches
    
    def calculate_similarity(self, task1, task2):
        """Calculate similarity between two tasks"""
        if not self.initialized:
            return self._fallback_calculate_similarity(task1, task2)
        
        with self.lock:
            try:
                # Make sure both tasks have embeddings
                embedding1 = getattr(task1, 'embedding', None)
                if embedding1 is None:
                    embedding1 = self.embedding_model.encode(task1.description)
                    task1.embedding = embedding1
                
                embedding2 = getattr(task2, 'embedding', None)
                if embedding2 is None:
                    embedding2 = self.embedding_model.encode(task2.description)
                    task2.embedding = embedding2
                
                # Calculate cosine similarity
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                similarity = dot_product / (norm1 * norm2)
                
                # Track similarity statistics
                self.similarity_scores.append(similarity)
                if len(self.similarity_scores) > 100:
                    self.similarity_scores = self.similarity_scores[-100:]
                
                return similarity
            except Exception as e:
                print(f"⚠️ Error calculating similarity: {e}")
                return self._fallback_calculate_similarity(task1, task2)
    
    def _fallback_calculate_similarity(self, task1, task2):
        """Enhanced fallback text matching with TF-IDF inspired weighting"""
        def tokenize(text):
            # Remove punctuation and convert to lowercase
            words = ''.join(c.lower() for c in text if c.isalnum() or c.isspace()).split()
            # Remove common words
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are'}
            return [w for w in words if w not in stopwords]
        
        # Get weighted term frequencies
        task1_words = tokenize(task1.description)
        task2_words = tokenize(task2.description)
        
        if not task1_words or not task2_words:
            return 0.0
        
        # Calculate word frequencies
        word_freq1 = {}
        word_freq2 = {}
        for word in task1_words:
            word_freq1[word] = word_freq1.get(word, 0) + 1
        for word in task2_words:
            word_freq2[word] = word_freq2.get(word, 0) + 1
        
        # Calculate similarity with term frequency weighting
        common_words = set(word_freq1.keys()) & set(word_freq2.keys())
        if not common_words:
            return 0.0
        
        similarity = 0
        for word in common_words:
            similarity += min(word_freq1[word], word_freq2[word])
        
        # Normalize by total word count
        max_words = max(len(task1_words), len(task2_words))
        return similarity / max_words
    
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
                "fallback_method": self.fallback_method,
                "reason": self.error_reason or "Unknown reason"
            }
        
        with self.lock:
            stats = {
                "status": "initialized",
                "model": self.model_name,
                "embedding_dimension": self.embedding_dim,
                "indexed_tasks": self.faiss_index.ntotal if hasattr(self, 'faiss_index') else 0,
                "similarity_threshold": {
                    "current": self.get_adaptive_threshold(),
                    "history": self.threshold_history[-5:] if self.threshold_history else []
                },
                "similarity_distribution": {
                    "samples": len(self.similarity_scores)
                }
            }
            
            # Add statistics only if we have similarity scores
            if self.similarity_scores:
                stats["similarity_distribution"].update({
                    "mean": float(np.mean(self.similarity_scores)),
                    "median": float(np.median(self.similarity_scores)),
                    "min": float(min(self.similarity_scores)),
                    "max": float(max(self.similarity_scores))
                })
            
            return stats
