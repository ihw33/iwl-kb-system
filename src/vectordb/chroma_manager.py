"""
ChromaDB Manager for IWL Knowledge Base
Handles vector database operations
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import os

class ChromaManager:
    """Manage ChromaDB operations for the knowledge base"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client"""
        self.persist_directory = persist_directory
        
        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize or get collection
        self.collection_name = "iwl_knowledge"
        self.collection = self._get_or_create_collection()
        
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            print(f"Using existing collection: {self.collection_name}")
        except:
            # Create new collection if doesn't exist
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "IWL educational content",
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            print(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the collection"""
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Ensure metadatas exist
        if metadatas is None:
            metadatas = [{"created_at": datetime.utcnow().isoformat()} for _ in documents]
        
        # Add documents to collection
        if embeddings:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        else:
            # Let ChromaDB generate embeddings using default embedding function
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        return ids
    
    def search(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Search for similar documents"""
        if query_texts:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
        elif query_embeddings:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
        else:
            raise ValueError("Either query_texts or query_embeddings must be provided")
        
        return results
    
    def get_documents(self, ids: List[str]) -> Dict[str, Any]:
        """Get specific documents by IDs"""
        return self.collection.get(ids=ids)
    
    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict]] = None
    ):
        """Update existing documents"""
        update_data = {}
        if documents:
            update_data["documents"] = documents
        if embeddings:
            update_data["embeddings"] = embeddings
        if metadatas:
            update_data["metadatas"] = metadatas
        
        if update_data:
            self.collection.update(ids=ids, **update_data)
    
    def delete_documents(self, ids: List[str]):
        """Delete documents by IDs"""
        self.collection.delete(ids=ids)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        count = self.collection.count()
        
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "persist_directory": self.persist_directory,
            "metadata": self.collection.metadata
        }
    
    def reset_collection(self):
        """Reset the entire collection (use with caution)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except:
            pass
        
        self.collection = self._get_or_create_collection()
        print("Collection reset complete")
    
    def list_all_documents(self, limit: int = 100) -> Dict[str, Any]:
        """List all documents in the collection"""
        return self.collection.get(limit=limit)