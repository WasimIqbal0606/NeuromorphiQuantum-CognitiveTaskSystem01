"""
Advanced embedding and similarity engine for Neuromorphic Quantum-Cognitive Task System
"""

import numpy as np
import threading
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Optional imports wrapped for fault tolerance
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
        self.embedding_dim = 384
        self.error_reason = None
        self.fallback_method = "text_matching"

        if not EMBEDDING_AVAILABLE:
            self.error_reason = f"Required libraries not available: {IMPORT_ERROR_MESSAGE}"
            print(self.error_reason)
            return

        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.chroma_client = chromadb.Client()

            try:
                self.task_collection = self.chroma_client.get_collection("task_metadata")
            except Exception:
                self.task_collection = self.chroma_client.create_collection("task_metadata")

            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self.id_to_index = {}
            self.index_to_id = {}
            self.next_index = 0

            self.similarity_scores = []
            self.threshold_history = []

            self.initialized = True
        except Exception as e:
            self.error_reason = str(e)
            print(f"Initialization failed: {self.error_reason}")

    def _encode(self, text):
        return self.embedding_model.encode(text) if self.initialized else None

    def _add_faiss(self, task_id, vector):
        self.faiss_index.add(np.array([vector], dtype=np.float32))
        self.id_to_index[task_id] = self.next_index
        self.index_to_id[self.next_index] = task_id
        self.next_index += 1

    def _prepare_metadata(self, task):
        return {
            "id": task.id,
            "priority": float(task.priority),
            "state": task.state,
            "entropy": float(task.entropy),
            "assignee": task.assignee or "unassigned",
            "tags": ",".join(task.tags) if task.tags else ""
        }

    def add_task_embedding(self, task):
        if not self.initialized:
            return False
        with self.lock:
            try:
                task.embedding = self._encode(task.description)
                self._add_faiss(task.id, task.embedding)
                self.task_collection.add(
                    documents=[task.description],
                    metadatas=[self._prepare_metadata(task)],
                    ids=[task.id]
                )
                return True
            except Exception as e:
                print(f"Add failed: {e}")
                return False

    def update_task_embedding(self, task):
        if not self.initialized:
            return False
        with self.lock:
            try:
                self.task_collection.delete(ids=[task.id])
                task.embedding = self._encode(task.description)
                self.task_collection.add(
                    documents=[task.description],
                    metadatas=[self._prepare_metadata(task)],
                    ids=[task.id]
                )
                if task.id in self.id_to_index:
                    old_index = self.id_to_index.pop(task.id)
                    self.index_to_id.pop(old_index, None)
                self._add_faiss(task.id, task.embedding)
                return True
            except Exception as e:
                print(f"Update failed: {e}")
                return False

    def update_task_metadata(self, task):
        if not self.initialized:
            return False
        with self.lock:
            try:
                try:
                    self.task_collection.get(ids=[task.id])
                except:
                    return self.add_task_embedding(task)
                self.task_collection.update(
                    metadatas=[self._prepare_metadata(task)],
                    ids=[task.id]
                )
                return True
            except Exception as e:
                print(f"Metadata update failed: {e}")
                return False

    def remove_task_embedding(self, task_id):
        if not self.initialized:
            return False
        with self.lock:
            try:
                self.task_collection.delete(ids=[task_id])
                if task_id in self.id_to_index:
                    idx = self.id_to_index.pop(task_id)
                    self.index_to_id.pop(idx, None)
                return True
            except Exception as e:
                print(f"Remove failed: {e}")
                return False

    def find_similar_tasks(self, task, threshold=0.7, max_results=5):
        if not self.initialized:
            return []
        with self.lock:
            try:
                query_embedding = getattr(task, 'embedding', None) or self._encode(task.description)
                query_vector = np.array([query_embedding], dtype=np.float32)
                if self.faiss_index.ntotal == 0:
                    return []
                k = min(max_results + 1, self.faiss_index.ntotal)
                distances, indices = self.faiss_index.search(query_vector, k)
                results = []
                for i, idx in enumerate(indices[0]):
                    task_id = self.index_to_id.get(idx)
                    if task_id and task_id != task.id:
                        similarity = 1.0 - min(1.0, distances[0][i] / (2 * self.embedding_dim))
                        self.similarity_scores.append(similarity)
                        if similarity >= threshold:
                            results.append((task_id, similarity))
                results.sort(key=lambda x: x[1], reverse=True)
                return [task_id for task_id, _ in results[:max_results]]
            except Exception as e:
                print(f"Similarity search failed: {e}")
                return []

    def calculate_similarity(self, task1, task2):
        if not self.initialized:
            return 0.0
        with self.lock:
            try:
                emb1 = getattr(task1, 'embedding', None) or self._encode(task1.description)
                emb2 = getattr(task2, 'embedding', None) or self._encode(task2.description)
                dot = np.dot(emb1, emb2)
                norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
                if norm == 0:
                    return 0.0
                similarity = dot / norm
                self.similarity_scores.append(similarity)
                return similarity
            except Exception as e:
                print(f"Similarity calc failed: {e}")
                return 0.0

    def get_adaptive_threshold(self):
        if not self.similarity_scores:
            return 0.7
        threshold = max(0.65, np.percentile(self.similarity_scores, 33))
        self.threshold_history.append({
            "timestamp": datetime.now().isoformat(),
            "threshold": threshold,
            "samples": len(self.similarity_scores)
        })
        if len(self.threshold_history) > 20:
            self.threshold_history = self.threshold_history[-20:]
        return threshold

    def contextual_search(self, query_text, filters=None, max_results=10):
        if not self.initialized:
            return []
        with self.lock:
            try:
                embedding = self._encode(query_text)
                where = {}
                if filters:
                    if filters.get('state'):
                        where['state'] = filters['state']
                    if filters.get('assignee'):
                        where['assignee'] = filters['assignee']
                    if 'min_priority' in filters:
                        where['priority'] = {"$gte": filters['min_priority']}
                results = self.task_collection.query(
                    query_embeddings=[embedding.tolist()],
                    n_results=max_results,
                    where=where if where else None
                )
                task_results = []
                for i, task_id in enumerate(results['ids'][0]):
                    dist = results['distances'][0][i]
                    sim = 1.0 - min(1.0, dist / 2.0)
                    meta = results['metadatas'][0][i] if 'metadatas' in results else None
                    task_results.append({"task_id": task_id, "similarity": sim, "metadata": meta})
                return task_results
            except Exception as e:
                print(f"Contextual search failed: {e}")
                return []

    def get_embedding_statistics(self):
        if not self.initialized:
            return {"status": "not_initialized", "reason": self.error_reason}
        stats = {
            "status": "initialized",
            "model": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "indexed_tasks": self.faiss_index.ntotal,
            "similarity_threshold": {
                "current": self.get_adaptive_threshold(),
                "history": self.threshold_history[-5:]
            },
            "similarity_distribution": {
                "samples": len(self.similarity_scores)
            }
        }
        if self.similarity_scores:
            stats["similarity_distribution"].update({
                "mean": float(np.mean(self.similarity_scores)),
                "median": float(np.median(self.similarity_scores)),
                "min": float(min(self.similarity_scores)),
                "max": float(max(self.similarity_scores))
            })
        return stats
