"""
FAISS Manager for IWL Knowledge Base
High-performance vector search optimization
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

class FAISSManager:
    """Manage FAISS operations for optimized vector search"""
    
    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "IVF",
        persist_directory: str = "./faiss_index"
    ):
        """
        Initialize FAISS index
        
        Args:
            dimension: Vector dimension
            index_type: Type of index (Flat, IVF, HNSW)
            persist_directory: Directory to persist index
        """
        self.dimension = dimension
        self.index_type = index_type
        self.persist_directory = persist_directory
        
        # Create directory if not exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize paths
        self.index_path = os.path.join(persist_directory, "faiss.index")
        self.metadata_path = os.path.join(persist_directory, "metadata.pkl")
        self.id_map_path = os.path.join(persist_directory, "id_map.json")
        
        # Initialize index and metadata
        self.index = self._create_index()
        self.metadata = []
        self.id_to_idx = {}  # Map from document ID to FAISS index
        self.idx_to_id = {}  # Map from FAISS index to document ID
        self.next_idx = 0
        
        # Load existing index if available
        self._load_index()
    
    def _create_index(self) -> faiss.Index:
        """Create appropriate FAISS index based on type"""
        if self.index_type == "Flat":
            # Exact search - slower but most accurate
            index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            # Inverted File Index - faster for large datasets
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "HNSW":
            # Hierarchical NSW - very fast approximate search
            index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            # Default to Flat index
            index = faiss.IndexFlatL2(self.dimension)
        
        # Add ID mapping capability
        index = faiss.IndexIDMap(index)
        
        return index
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: List[str],
        metadata: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Add vectors to the index
        
        Args:
            vectors: Numpy array of vectors (n_vectors, dimension)
            ids: List of document IDs
            metadata: List of metadata dictionaries
        
        Returns:
            List of internal indices
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        else:
            vectors = vectors.astype(np.float32)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Train index if needed (for IVF)
        if self.index_type == "IVF" and not self.index.is_trained:
            self.index.train(vectors)
        
        # Generate internal indices
        indices = []
        for i, doc_id in enumerate(ids):
            if doc_id not in self.id_to_idx:
                idx = self.next_idx
                self.id_to_idx[doc_id] = idx
                self.idx_to_id[idx] = doc_id
                self.next_idx += 1
                indices.append(idx)
            else:
                indices.append(self.id_to_idx[doc_id])
        
        # Add vectors with indices
        idx_array = np.array(indices, dtype=np.int64)
        self.index.add_with_ids(vectors, idx_array)
        
        # Store metadata
        for meta in metadata:
            meta['indexed_at'] = datetime.utcnow().isoformat()
            self.metadata.append(meta)
        
        # Persist changes
        self._save_index()
        
        return indices
    
    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Search for similar vectors
        
        Args:
            query_vectors: Query vectors (n_queries, dimension)
            k: Number of results per query
            filters: Optional metadata filters
        
        Returns:
            Tuple of (indices, distances)
        """
        if not isinstance(query_vectors, np.ndarray):
            query_vectors = np.array(query_vectors, dtype=np.float32)
        else:
            query_vectors = query_vectors.astype(np.float32)
        
        # Normalize query vectors
        faiss.normalize_L2(query_vectors)
        
        # Search
        distances, indices = self.index.search(query_vectors, k)
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            filtered_distances = []
            
            for query_idx in range(len(indices)):
                query_results = []
                query_distances = []
                
                for i, idx in enumerate(indices[query_idx]):
                    if idx >= 0 and self._match_filters(int(idx), filters):
                        query_results.append(int(idx))
                        query_distances.append(float(distances[query_idx][i]))
                
                filtered_results.append(query_results)
                filtered_distances.append(query_distances)
            
            return filtered_results, filtered_distances
        
        # Convert to lists
        indices_list = [[int(idx) if idx >= 0 else -1 for idx in row] for row in indices]
        distances_list = [[float(d) for d in row] for row in distances]
        
        return indices_list, distances_list
    
    def _match_filters(self, idx: int, filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters"""
        if idx >= len(self.metadata):
            return False
        
        meta = self.metadata[idx]
        for key, value in filters.items():
            if key not in meta or meta[key] != value:
                return False
        
        return True
    
    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Get vectors and metadata by document IDs"""
        results = []
        
        for doc_id in ids:
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                if idx < len(self.metadata):
                    result = {
                        'id': doc_id,
                        'index': idx,
                        'metadata': self.metadata[idx]
                    }
                    results.append(result)
        
        return results
    
    def update_vectors(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Update existing vectors"""
        # Remove old vectors
        self.delete_by_ids(ids)
        
        # Add new vectors
        if metadata is None:
            metadata = [{} for _ in ids]
        
        self.add_vectors(vectors, ids, metadata)
    
    def delete_by_ids(self, ids: List[str]):
        """Delete vectors by document IDs"""
        indices_to_remove = []
        
        for doc_id in ids:
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                indices_to_remove.append(idx)
                
                # Clean up mappings
                del self.id_to_idx[doc_id]
                del self.idx_to_id[idx]
        
        if indices_to_remove:
            # FAISS doesn't support deletion, so we need to rebuild
            self._rebuild_index_excluding(indices_to_remove)
    
    def _rebuild_index_excluding(self, exclude_indices: List[int]):
        """Rebuild index excluding certain indices"""
        # Get all vectors except excluded ones
        all_vectors = []
        all_ids = []
        all_metadata = []
        
        for idx, doc_id in self.idx_to_id.items():
            if idx not in exclude_indices:
                # Note: We'd need to store vectors to fully rebuild
                # For now, we'll mark as deleted in metadata
                if idx < len(self.metadata):
                    self.metadata[idx]['deleted'] = True
        
        self._save_index()
    
    def _save_index(self):
        """Save index and metadata to disk"""
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save ID mappings
        with open(self.id_map_path, 'w') as f:
            json.dump({
                'id_to_idx': self.id_to_idx,
                'idx_to_id': {str(k): v for k, v in self.idx_to_id.items()},
                'next_idx': self.next_idx
            }, f)
    
    def _load_index(self):
        """Load index and metadata from disk"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        
        if os.path.exists(self.id_map_path):
            with open(self.id_map_path, 'r') as f:
                data = json.load(f)
                self.id_to_idx = data.get('id_to_idx', {})
                self.idx_to_id = {int(k): v for k, v in data.get('idx_to_id', {}).items()}
                self.next_idx = data.get('next_idx', 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'total_vectors': self.index.ntotal,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            'total_documents': len(self.id_to_idx),
            'persist_directory': self.persist_directory
        }
    
    def clear(self):
        """Clear all data"""
        self.index = self._create_index()
        self.metadata = []
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.next_idx = 0
        
        # Remove persistent files
        for path in [self.index_path, self.metadata_path, self.id_map_path]:
            if os.path.exists(path):
                os.remove(path)