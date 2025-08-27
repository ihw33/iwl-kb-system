"""
Annoy Manager for IWL Knowledge Base
Approximate Nearest Neighbors for fast similarity search
"""

from annoy import AnnoyIndex
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pickle

class AnnoyManager:
    """Manage Annoy operations for fast approximate nearest neighbor search"""
    
    def __init__(
        self,
        dimension: int = 384,
        metric: str = "angular",
        n_trees: int = 10,
        persist_directory: str = "./annoy_index"
    ):
        """
        Initialize Annoy index
        
        Args:
            dimension: Vector dimension
            metric: Distance metric ('angular', 'euclidean', 'manhattan', 'hamming', 'dot')
            n_trees: Number of trees for index (more trees = better accuracy, slower build)
            persist_directory: Directory to persist index
        """
        self.dimension = dimension
        self.metric = metric
        self.n_trees = n_trees
        self.persist_directory = persist_directory
        
        # Create directory if not exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize paths
        self.index_path = os.path.join(persist_directory, "annoy.ann")
        self.metadata_path = os.path.join(persist_directory, "metadata.pkl")
        self.id_map_path = os.path.join(persist_directory, "id_map.json")
        
        # Initialize index
        self.index = AnnoyIndex(dimension, metric)
        self.metadata = []
        self.id_to_idx = {}  # Map from document ID to Annoy index
        self.idx_to_id = {}  # Map from Annoy index to document ID
        self.next_idx = 0
        self.is_built = False
        
        # Load existing index if available
        self._load_index()
    
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
        if self.is_built:
            print("Warning: Index is already built. Rebuilding with new vectors...")
            self._rebuild_with_new_vectors(vectors, ids, metadata)
            return list(range(len(ids)))
        
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        
        indices = []
        for i, (vec, doc_id) in enumerate(zip(vectors, ids)):
            if doc_id not in self.id_to_idx:
                idx = self.next_idx
                self.index.add_item(idx, vec.tolist())
                
                # Update mappings
                self.id_to_idx[doc_id] = idx
                self.idx_to_id[idx] = doc_id
                self.next_idx += 1
                
                # Store metadata
                meta = metadata[i].copy()
                meta['indexed_at'] = datetime.utcnow().isoformat()
                meta['doc_id'] = doc_id
                self.metadata.append(meta)
                
                indices.append(idx)
            else:
                # Update existing vector
                idx = self.id_to_idx[doc_id]
                indices.append(idx)
        
        return indices
    
    def build_index(self):
        """Build the index after adding all vectors"""
        if not self.is_built:
            self.index.build(self.n_trees)
            self.is_built = True
            self._save_index()
    
    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 5,
        search_k: int = -1,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Search for similar vectors
        
        Args:
            query_vectors: Query vectors (n_queries, dimension)
            k: Number of results per query
            search_k: Number of nodes to inspect (default: n_trees * k)
            filters: Optional metadata filters
        
        Returns:
            Tuple of (indices, distances)
        """
        if not self.is_built:
            self.build_index()
        
        if not isinstance(query_vectors, np.ndarray):
            query_vectors = np.array(query_vectors, dtype=np.float32)
        
        if search_k == -1:
            search_k = self.n_trees * k * 10
        
        all_indices = []
        all_distances = []
        
        for query_vec in query_vectors:
            # Search in Annoy
            indices, distances = self.index.get_nns_by_vector(
                query_vec.tolist(),
                k,
                search_k=search_k,
                include_distances=True
            )
            
            # Apply filters if provided
            if filters:
                filtered_indices = []
                filtered_distances = []
                
                for idx, dist in zip(indices, distances):
                    if self._match_filters(idx, filters):
                        filtered_indices.append(idx)
                        filtered_distances.append(dist)
                
                all_indices.append(filtered_indices)
                all_distances.append(filtered_distances)
            else:
                all_indices.append(indices)
                all_distances.append(distances)
        
        return all_indices, all_distances
    
    def _match_filters(self, idx: int, filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters"""
        # Find metadata for this index
        for meta in self.metadata:
            if meta.get('doc_id') == self.idx_to_id.get(idx):
                for key, value in filters.items():
                    if key not in meta or meta[key] != value:
                        return False
                return True
        return False
    
    def get_vector_by_id(self, doc_id: str) -> Optional[np.ndarray]:
        """Get vector by document ID"""
        if doc_id in self.id_to_idx:
            idx = self.id_to_idx[doc_id]
            return np.array(self.index.get_item_vector(idx))
        return None
    
    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Get vectors and metadata by document IDs"""
        results = []
        
        for doc_id in ids:
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                vector = self.index.get_item_vector(idx)
                
                # Find metadata
                meta = None
                for m in self.metadata:
                    if m.get('doc_id') == doc_id:
                        meta = m
                        break
                
                result = {
                    'id': doc_id,
                    'index': idx,
                    'vector': vector,
                    'metadata': meta
                }
                results.append(result)
        
        return results
    
    def find_similar(
        self,
        doc_id: str,
        k: int = 5,
        include_self: bool = False
    ) -> Tuple[List[str], List[float]]:
        """Find similar documents to a given document"""
        if doc_id not in self.id_to_idx:
            return [], []
        
        if not self.is_built:
            self.build_index()
        
        idx = self.id_to_idx[doc_id]
        n = k + 1 if not include_self else k
        
        indices, distances = self.index.get_nns_by_item(
            idx, n, include_distances=True
        )
        
        # Convert indices to document IDs
        doc_ids = []
        final_distances = []
        
        for i, dist in zip(indices, distances):
            if not include_self and i == idx:
                continue
            if i in self.idx_to_id:
                doc_ids.append(self.idx_to_id[i])
                final_distances.append(dist)
        
        return doc_ids, final_distances
    
    def _rebuild_with_new_vectors(
        self,
        new_vectors: np.ndarray,
        new_ids: List[str],
        new_metadata: List[Dict[str, Any]]
    ):
        """Rebuild index with new vectors"""
        # Create new index
        new_index = AnnoyIndex(self.dimension, self.metric)
        
        # Add existing vectors
        for idx in range(self.next_idx):
            if idx in self.idx_to_id:
                vec = self.index.get_item_vector(idx)
                new_index.add_item(idx, vec)
        
        # Add new vectors
        for vec, doc_id, meta in zip(new_vectors, new_ids, new_metadata):
            if doc_id not in self.id_to_idx:
                idx = self.next_idx
                new_index.add_item(idx, vec.tolist())
                
                self.id_to_idx[doc_id] = idx
                self.idx_to_id[idx] = doc_id
                self.next_idx += 1
                
                meta['indexed_at'] = datetime.utcnow().isoformat()
                meta['doc_id'] = doc_id
                self.metadata.append(meta)
        
        # Build new index
        new_index.build(self.n_trees)
        self.index = new_index
        self.is_built = True
        
        self._save_index()
    
    def _save_index(self):
        """Save index and metadata to disk"""
        if self.is_built:
            # Save Annoy index
            self.index.save(self.index_path)
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save ID mappings
        with open(self.id_map_path, 'w') as f:
            json.dump({
                'id_to_idx': self.id_to_idx,
                'idx_to_id': {str(k): v for k, v in self.idx_to_id.items()},
                'next_idx': self.next_idx,
                'is_built': self.is_built
            }, f)
    
    def _load_index(self):
        """Load index and metadata from disk"""
        if os.path.exists(self.index_path):
            self.index.load(self.index_path)
            self.is_built = True
            
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        
        if os.path.exists(self.id_map_path):
            with open(self.id_map_path, 'r') as f:
                data = json.load(f)
                self.id_to_idx = data.get('id_to_idx', {})
                self.idx_to_id = {int(k): v for k, v in data.get('idx_to_id', {}).items()}
                self.next_idx = data.get('next_idx', 0)
                self.is_built = data.get('is_built', False)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'metric': self.metric,
            'dimension': self.dimension,
            'n_trees': self.n_trees,
            'total_vectors': self.next_idx,
            'is_built': self.is_built,
            'total_documents': len(self.id_to_idx),
            'persist_directory': self.persist_directory
        }
    
    def clear(self):
        """Clear all data"""
        self.index = AnnoyIndex(self.dimension, self.metric)
        self.metadata = []
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.next_idx = 0
        self.is_built = False
        
        # Remove persistent files
        for path in [self.index_path, self.metadata_path, self.id_map_path]:
            if os.path.exists(path):
                os.remove(path)