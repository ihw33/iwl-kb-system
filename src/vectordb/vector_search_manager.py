"""
Unified Vector Search Manager for IWL Knowledge Base
Supports ChromaDB, FAISS, and Annoy backends
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Literal
from datetime import datetime
import os

from .chroma_manager import ChromaManager
from .faiss_manager import FAISSManager
from .annoy_manager import AnnoyManager

class VectorSearchManager:
    """
    Unified interface for vector search operations
    Supports multiple backend implementations
    """
    
    def __init__(
        self,
        backend: Literal["chromadb", "faiss", "annoy"] = "faiss",
        dimension: int = 384,
        persist_directory: str = "./vector_index",
        **kwargs
    ):
        """
        Initialize Vector Search Manager
        
        Args:
            backend: Backend to use ('chromadb', 'faiss', 'annoy')
            dimension: Vector dimension
            persist_directory: Directory to persist index
            **kwargs: Additional backend-specific parameters
        """
        self.backend_type = backend
        self.dimension = dimension
        self.persist_directory = os.path.join(persist_directory, backend)
        
        # Initialize backend
        if backend == "chromadb":
            self.backend = ChromaManager(persist_directory=self.persist_directory)
        elif backend == "faiss":
            index_type = kwargs.get('index_type', 'IVF')
            self.backend = FAISSManager(
                dimension=dimension,
                index_type=index_type,
                persist_directory=self.persist_directory
            )
        elif backend == "annoy":
            metric = kwargs.get('metric', 'angular')
            n_trees = kwargs.get('n_trees', 10)
            self.backend = AnnoyManager(
                dimension=dimension,
                metric=metric,
                n_trees=n_trees,
                persist_directory=self.persist_directory
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Cache for recent searches
        self.search_cache = {}
        self.cache_size = kwargs.get('cache_size', 100)
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        ids: List[str],
        metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add documents with their embeddings
        
        Args:
            documents: List of document texts
            embeddings: Document embeddings
            ids: Document IDs
            metadata: Document metadata
        
        Returns:
            List of document IDs
        """
        if self.backend_type == "chromadb":
            return self.backend.add_documents(
                documents=documents,
                embeddings=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
                ids=ids,
                metadatas=metadata
            )
        elif self.backend_type in ["faiss", "annoy"]:
            indices = self.backend.add_vectors(
                vectors=embeddings,
                ids=ids,
                metadata=metadata
            )
            
            # Build index for Annoy
            if self.backend_type == "annoy":
                self.backend.build_index()
            
            return ids
        
        return []
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        return_docs: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector
            k: Number of results
            filters: Optional metadata filters
            return_docs: Whether to return full documents
        
        Returns:
            List of search results
        """
        # Check cache
        cache_key = self._get_cache_key(query_embedding, k, filters)
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        results = []
        
        if self.backend_type == "chromadb":
            # ChromaDB search
            search_results = self.backend.search(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=filters
            )
            
            if search_results and "documents" in search_results:
                for i in range(len(search_results["documents"][0])):
                    result = {
                        "id": search_results["ids"][0][i] if "ids" in search_results else None,
                        "text": search_results["documents"][0][i],
                        "metadata": search_results["metadatas"][0][i] if "metadatas" in search_results else {},
                        "score": 1.0 - search_results["distances"][0][i] if "distances" in search_results else 0.5
                    }
                    results.append(result)
        
        elif self.backend_type in ["faiss", "annoy"]:
            # FAISS/Annoy search
            if self.backend_type == "faiss":
                indices, distances = self.backend.search(
                    query_vectors=query_embedding.reshape(1, -1),
                    k=k,
                    filters=filters
                )
            else:  # annoy
                indices, distances = self.backend.search(
                    query_vectors=query_embedding.reshape(1, -1),
                    k=k,
                    filters=filters
                )
            
            # Get results for first query
            if indices and len(indices) > 0:
                query_indices = indices[0]
                query_distances = distances[0]
                
                for idx, dist in zip(query_indices, query_distances):
                    if idx >= 0:  # Valid index
                        # Get document info
                        doc_id = self.backend.idx_to_id.get(idx, f"doc_{idx}")
                        
                        # Get metadata
                        meta = {}
                        if hasattr(self.backend, 'metadata') and idx < len(self.backend.metadata):
                            meta = self.backend.metadata[idx]
                        
                        result = {
                            "id": doc_id,
                            "index": idx,
                            "metadata": meta,
                            "score": 1.0 / (1.0 + dist)  # Convert distance to score
                        }
                        results.append(result)
        
        # Update cache
        if len(self.search_cache) >= self.cache_size:
            # Remove oldest entry
            self.search_cache.pop(next(iter(self.search_cache)))
        self.search_cache[cache_key] = results
        
        return results
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch search for multiple queries
        
        Args:
            query_embeddings: Multiple query vectors
            k: Number of results per query
            filters: Optional metadata filters
        
        Returns:
            List of search results for each query
        """
        all_results = []
        
        for query_embedding in query_embeddings:
            results = self.search(query_embedding, k, filters)
            all_results.append(results)
        
        return all_results
    
    def update_document(
        self,
        doc_id: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update document embedding and/or metadata"""
        if self.backend_type == "chromadb":
            update_kwargs = {"ids": [doc_id]}
            if embedding is not None:
                update_kwargs["embeddings"] = [embedding.tolist()]
            if metadata is not None:
                update_kwargs["metadatas"] = [metadata]
            
            self.backend.update_documents(**update_kwargs)
        
        elif self.backend_type == "faiss":
            if embedding is not None:
                self.backend.update_vectors(
                    ids=[doc_id],
                    vectors=embedding.reshape(1, -1),
                    metadata=[metadata] if metadata else None
                )
        
        # Clear cache after update
        self.search_cache.clear()
    
    def delete_documents(self, doc_ids: List[str]):
        """Delete documents by IDs"""
        if self.backend_type == "chromadb":
            self.backend.delete_documents(ids=doc_ids)
        elif self.backend_type == "faiss":
            self.backend.delete_by_ids(doc_ids)
        
        # Clear cache after deletion
        self.search_cache.clear()
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        if self.backend_type == "chromadb":
            result = self.backend.get_documents(ids=[doc_id])
            if result and "documents" in result and result["documents"]:
                return {
                    "id": doc_id,
                    "text": result["documents"][0] if result["documents"] else None,
                    "metadata": result["metadatas"][0] if "metadatas" in result else {}
                }
        elif self.backend_type in ["faiss", "annoy"]:
            results = self.backend.get_by_ids([doc_id])
            if results:
                return results[0]
        
        return None
    
    def _get_cache_key(
        self,
        query_embedding: np.ndarray,
        k: int,
        filters: Optional[Dict]
    ) -> str:
        """Generate cache key for search"""
        # Use first few dimensions of embedding for cache key
        embed_key = "_".join(map(str, query_embedding[:5].round(3)))
        filter_key = str(sorted(filters.items())) if filters else "no_filter"
        return f"{embed_key}_{k}_{filter_key}"
    
    def optimize_index(self):
        """Optimize index for better search performance"""
        if self.backend_type == "annoy":
            # Rebuild with more trees for better accuracy
            print("Rebuilding Annoy index with more trees...")
            self.backend.n_trees = min(self.backend.n_trees * 2, 100)
            self.backend.build_index()
        elif self.backend_type == "faiss":
            print("FAISS index optimization...")
            # Could implement index retraining or parameter tuning
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        stats = {
            "backend": self.backend_type,
            "dimension": self.dimension,
            "cache_size": len(self.search_cache),
            "max_cache_size": self.cache_size
        }
        
        if hasattr(self.backend, 'get_stats'):
            backend_stats = self.backend.get_stats()
            stats.update(backend_stats)
        elif hasattr(self.backend, 'get_collection_stats'):
            backend_stats = self.backend.get_collection_stats()
            stats.update(backend_stats)
        
        return stats
    
    def benchmark_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Dict[str, Any]:
        """Benchmark search performance"""
        import time
        
        start_time = time.time()
        results = self.search(query_embedding, k)
        search_time = time.time() - start_time
        
        return {
            "search_time_ms": search_time * 1000,
            "num_results": len(results),
            "backend": self.backend_type,
            "cached": False
        }
    
    def clear(self):
        """Clear all data"""
        if hasattr(self.backend, 'clear'):
            self.backend.clear()
        elif hasattr(self.backend, 'reset_collection'):
            self.backend.reset_collection()
        
        self.search_cache.clear()