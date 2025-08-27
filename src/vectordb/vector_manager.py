"""
Enhanced Vector Manager with ChromaDB + FAISS/Annoy optimization
Provides hybrid search capabilities for better performance
"""

import numpy as np
import faiss
from annoy import AnnoyIndex
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Literal
import uuid
from datetime import datetime
import pickle
import os

class HybridVectorManager:
    """Hybrid vector database manager combining ChromaDB with FAISS/Annoy"""
    
    def __init__(
        self, 
        persist_directory: str = "./vector_db",
        index_type: Literal["faiss", "annoy", "chromadb"] = "faiss",
        embedding_dim: int = 384
    ):
        """Initialize hybrid vector manager"""
        self.persist_directory = persist_directory
        self.index_type = index_type
        self.embedding_dim = embedding_dim
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB for metadata storage
        self.chroma_client = chromadb.PersistentClient(
            path=os.path.join(persist_directory, "chromadb"),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self._init_chroma_collection()
        
        # Initialize vector index based on type
        if index_type == "faiss":
            self._init_faiss_index()
        elif index_type == "annoy":
            self._init_annoy_index()
        
        # Document mapping
        self.doc_id_mapping = {}
        self.reverse_mapping = {}
        self._load_mappings()
    
    def _init_chroma_collection(self):
        """Initialize ChromaDB collection"""
        try:
            collection = self.chroma_client.get_collection(name="iwl_hybrid")
        except:
            collection = self.chroma_client.create_collection(
                name="iwl_hybrid",
                metadata={"description": "IWL hybrid vector storage"}
            )
        return collection
    
    def _init_faiss_index(self):
        """Initialize FAISS index for fast similarity search"""
        self.faiss_index_path = os.path.join(self.persist_directory, "faiss.index")
        
        if os.path.exists(self.faiss_index_path):
            # Load existing index
            self.faiss_index = faiss.read_index(self.faiss_index_path)
        else:
            # Create new index with L2 distance
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            # For large-scale, use IVF index
            # nlist = 100
            # self.faiss_index = faiss.IndexIVFFlat(
            #     faiss.IndexFlatL2(self.embedding_dim), 
            #     self.embedding_dim, 
            #     nlist
            # )
    
    def _init_annoy_index(self):
        """Initialize Annoy index for approximate nearest neighbors"""
        self.annoy_index_path = os.path.join(self.persist_directory, "annoy.ann")
        self.annoy_index = AnnoyIndex(self.embedding_dim, 'angular')
        
        if os.path.exists(self.annoy_index_path):
            self.annoy_index.load(self.annoy_index_path)
    
    def _save_mappings(self):
        """Save document ID mappings"""
        mapping_path = os.path.join(self.persist_directory, "mappings.pkl")
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'doc_id_mapping': self.doc_id_mapping,
                'reverse_mapping': self.reverse_mapping
            }, f)
    
    def _load_mappings(self):
        """Load document ID mappings"""
        mapping_path = os.path.join(self.persist_directory, "mappings.pkl")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                data = pickle.load(f)
                self.doc_id_mapping = data['doc_id_mapping']
                self.reverse_mapping = data['reverse_mapping']
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents with optimized indexing"""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # Add to ChromaDB for metadata
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        # Add to FAISS/Annoy for fast search
        embeddings_array = np.array(embeddings).astype('float32')
        
        if self.index_type == "faiss":
            # Add to FAISS index
            start_idx = self.faiss_index.ntotal
            self.faiss_index.add(embeddings_array)
            
            # Update mappings
            for i, doc_id in enumerate(ids):
                faiss_idx = start_idx + i
                self.doc_id_mapping[faiss_idx] = doc_id
                self.reverse_mapping[doc_id] = faiss_idx
            
            # Save index
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            
        elif self.index_type == "annoy":
            # Add to Annoy index
            for i, (embedding, doc_id) in enumerate(zip(embeddings, ids)):
                idx = len(self.doc_id_mapping)
                self.annoy_index.add_item(idx, embedding)
                self.doc_id_mapping[idx] = doc_id
                self.reverse_mapping[doc_id] = idx
            
            # Build and save index
            self.annoy_index.build(10)  # 10 trees
            self.annoy_index.save(self.annoy_index_path)
        
        self._save_mappings()
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Optimized hybrid search"""
        query_array = np.array([query_embedding]).astype('float32')
        
        if self.index_type == "faiss":
            # FAISS search
            distances, indices = self.faiss_index.search(query_array, k * 2)  # Get more for filtering
            
            # Map indices to document IDs
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx in self.doc_id_mapping:
                    doc_id = self.doc_id_mapping[idx]
                    
                    # Get metadata from ChromaDB
                    doc_data = self.collection.get(ids=[doc_id])
                    
                    if doc_data and doc_data['documents']:
                        metadata = doc_data['metadatas'][0] if doc_data['metadatas'] else {}
                        
                        # Apply filters if provided
                        if filters:
                            if not all(metadata.get(k) == v for k, v in filters.items()):
                                continue
                        
                        results.append({
                            'id': doc_id,
                            'content': doc_data['documents'][0],
                            'metadata': metadata,
                            'distance': float(dist),
                            'score': 1.0 / (1.0 + float(dist))  # Convert distance to score
                        })
                        
                        if len(results) >= k:
                            break
            
            return results
            
        elif self.index_type == "annoy":
            # Annoy search
            indices, distances = self.annoy_index.get_nns_by_vector(
                query_embedding, 
                k * 2, 
                include_distances=True
            )
            
            results = []
            for idx, dist in zip(indices, distances):
                if idx in self.doc_id_mapping:
                    doc_id = self.doc_id_mapping[idx]
                    
                    # Get metadata from ChromaDB
                    doc_data = self.collection.get(ids=[doc_id])
                    
                    if doc_data and doc_data['documents']:
                        metadata = doc_data['metadatas'][0] if doc_data['metadatas'] else {}
                        
                        # Apply filters
                        if filters:
                            if not all(metadata.get(k) == v for k, v in filters.items()):
                                continue
                        
                        results.append({
                            'id': doc_id,
                            'content': doc_data['documents'][0],
                            'metadata': metadata,
                            'distance': dist,
                            'score': 1.0 - dist  # Angular distance to score
                        })
                        
                        if len(results) >= k:
                            break
            
            return results
            
        else:
            # Fallback to ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filters
            )
            
            formatted_results = []
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0,
                        'score': 1.0 - results['distances'][0][i] if results['distances'] else 1.0
                    })
            
            return formatted_results
    
    def delete_documents(self, ids: List[str]):
        """Delete documents from all indices"""
        # Delete from ChromaDB
        self.collection.delete(ids=ids)
        
        # Update FAISS/Annoy indices
        # Note: FAISS and Annoy don't support direct deletion,
        # so we need to rebuild the index
        if self.index_type in ["faiss", "annoy"]:
            # Remove from mappings
            for doc_id in ids:
                if doc_id in self.reverse_mapping:
                    idx = self.reverse_mapping[doc_id]
                    del self.doc_id_mapping[idx]
                    del self.reverse_mapping[doc_id]
            
            self._save_mappings()
            
            # For production, implement periodic index rebuilding
            # self._rebuild_index()
    
    def update_document(
        self,
        doc_id: str,
        document: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None
    ):
        """Update a document"""
        # Update in ChromaDB
        update_data = {}
        if document:
            update_data['documents'] = [document]
        if embedding:
            update_data['embeddings'] = [embedding]
        if metadata:
            update_data['metadatas'] = [metadata]
        
        if update_data:
            self.collection.update(ids=[doc_id], **update_data)
        
        # Update FAISS/Annoy if embedding changed
        if embedding and self.index_type in ["faiss", "annoy"]:
            if doc_id in self.reverse_mapping:
                # For simplicity, delete and re-add
                # In production, implement more efficient update
                pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = {
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'persist_directory': self.persist_directory
        }
        
        # ChromaDB stats
        collection_count = self.collection.count()
        stats['chromadb_documents'] = collection_count
        
        # FAISS stats
        if self.index_type == "faiss":
            stats['faiss_vectors'] = self.faiss_index.ntotal
            stats['faiss_index_trained'] = self.faiss_index.is_trained
        
        # Annoy stats
        elif self.index_type == "annoy":
            stats['annoy_items'] = len(self.doc_id_mapping)
            stats['annoy_trees'] = 10  # Default value we use
        
        stats['total_mappings'] = len(self.doc_id_mapping)
        
        return stats
    
    def optimize_index(self):
        """Optimize the index for better performance"""
        if self.index_type == "faiss":
            # Add index optimization like clustering for IVF
            if hasattr(self.faiss_index, 'train'):
                # Train the index if it's an IVF index
                pass
        elif self.index_type == "annoy":
            # Rebuild with more trees for better accuracy
            self.annoy_index.build(50)
            self.annoy_index.save(self.annoy_index_path)