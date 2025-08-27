"""
Embedding Pipeline for IWL Knowledge Base
Handles text embedding generation
"""

import os
from typing import List, Optional, Dict, Any
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

class EmbeddingPipeline:
    """Manage embedding generation for documents"""
    
    def __init__(
        self,
        model_type: str = "sentence-transformer",  # "openai" or "sentence-transformer"
        model_name: Optional[str] = None
    ):
        """Initialize embedding model"""
        self.model_type = model_type
        
        if model_type == "openai":
            # Use OpenAI embeddings
            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key:
                openai.api_key = self.api_key
                self.model_name = model_name or "text-embedding-ada-002"
                self.embedding_function = self._embed_openai
            else:
                print("Warning: OpenAI API key not found, falling back to sentence-transformer")
                self.model_type = "sentence-transformer"
        
        if self.model_type == "sentence-transformer":
            # Use local sentence transformer
            self.model_name = model_name or "all-MiniLM-L6-v2"
            try:
                self.model = SentenceTransformer(self.model_name)
                self.embedding_function = self._embed_sentence_transformer
            except Exception as e:
                print(f"Warning: Failed to load {self.model_name}: {e}")
                print("Using simple embedding fallback")
                self.model = None
                self.embedding_function = self._embed_simple
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            print(f"OpenAI embedding error: {e}")
            # Fallback to sentence transformer
            return self._embed_sentence_transformer(texts)
    
    def _embed_sentence_transformer(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence transformer"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def _embed_simple(self, texts: List[str]) -> List[List[float]]:
        """Simple embedding fallback using character frequency"""
        import hashlib
        embeddings = []
        for text in texts:
            # Create a simple embedding based on character frequency
            # This is just for demo purposes
            text_hash = hashlib.md5(text.encode()).hexdigest()
            embedding = [float(int(c, 16)) / 15.0 for c in text_hash[:384]]  # 384-dimensional embedding
            embeddings.append(embedding)
        return embeddings
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            return []
        
        # Use configured embedding function
        return self.embedding_function(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        chunks = self.text_splitter.split_text(text)
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "chunk_index": i,
                "chunk_total": len(chunks),
                "chunk_size": len(chunk)
            }
            
            # Add provided metadata
            if metadata:
                chunk_metadata.update(metadata)
            
            chunk_data.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        return chunk_data
    
    def process_document(
        self,
        document: str,
        metadata: Optional[Dict] = None,
        generate_embeddings: bool = True
    ) -> List[Dict[str, Any]]:
        """Process a document: chunk and optionally embed"""
        # Chunk the document
        chunks = self.chunk_text(document, metadata)
        
        # Generate embeddings if requested
        if generate_embeddings:
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embed_texts(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
        
        return chunks
    
    def batch_process_documents(
        self,
        documents: List[Dict[str, Any]],
        generate_embeddings: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple documents in batch"""
        all_chunks = []
        
        for doc in documents:
            doc_text = doc.get("content", "")
            doc_metadata = doc.get("metadata", {})
            
            # Add document ID to metadata if available
            if "id" in doc:
                doc_metadata["document_id"] = doc["id"]
            if "title" in doc:
                doc_metadata["document_title"] = doc["title"]
            
            chunks = self.process_document(
                doc_text,
                doc_metadata,
                generate_embeddings
            )
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        info = {
            "model_type": self.model_type,
            "model_name": self.model_name
        }
        
        if self.model_type == "sentence-transformer":
            info["embedding_dimension"] = self.model.get_sentence_embedding_dimension()
            info["max_seq_length"] = self.model.max_seq_length
        elif self.model_type == "openai":
            info["embedding_dimension"] = 1536  # Ada-002 dimension
            info["max_tokens"] = 8191  # Ada-002 max tokens
        
        return info