"""
IWL Knowledge Base API v2
Enhanced with multiple vector search backends
"""

from fastapi import FastAPI, HTTPException, status, Query, Path, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import os
import uuid
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import components
import sys
from pathlib import Path as FilePath
sys.path.append(str(FilePath(__file__).parent.parent))

from vectordb.vector_search_manager import VectorSearchManager
from embeddings.embedding_pipeline import EmbeddingPipeline
from rag.rag_pipeline import RAGPipeline

# Configuration
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")  # chromadb, faiss, annoy
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# Initialize components
vector_manager = VectorSearchManager(
    backend=VECTOR_BACKEND,
    dimension=EMBEDDING_DIM,
    persist_directory="./vector_index",
    index_type="HNSW" if VECTOR_BACKEND == "faiss" else None,
    n_trees=50 if VECTOR_BACKEND == "annoy" else None
)

embedding_pipeline = EmbeddingPipeline(model_type="sentence-transformer")

# For RAG, we need to adapt the vector manager
class VectorManagerAdapter:
    """Adapter to make VectorSearchManager compatible with RAGPipeline"""
    
    def __init__(self, vector_manager):
        self.vector_manager = vector_manager
    
    def search(self, query_embeddings, n_results, where=None):
        results = self.vector_manager.search(
            query_embedding=query_embeddings[0],
            k=n_results,
            filters=where
        )
        
        # Format for RAG pipeline
        return {
            "documents": [[r.get("text", "") for r in results]],
            "metadatas": [[r.get("metadata", {}) for r in results]],
            "distances": [[1.0 - r.get("score", 0.5) for r in results]],
            "ids": [[r.get("id", "") for r in results]]
        }
    
    def add_documents(self, documents, embeddings, metadatas, ids):
        return self.vector_manager.add_documents(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadata=metadatas
        )
    
    def get_collection_stats(self):
        return self.vector_manager.get_stats()

vector_adapter = VectorManagerAdapter(vector_manager)
rag_pipeline = RAGPipeline(vector_adapter, embedding_pipeline)

# Initialize FastAPI app
app = FastAPI(
    title="IWL Knowledge Base API v2",
    description="Enhanced RAG-based educational content management with optimized vector search",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
documents_db = {}
documents_content = {}  # Store actual content

# Models
class DocumentBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1, max_length=50)
    level: str = Field(default="intermediate", pattern="^(beginner|intermediate|advanced)$")
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentCreate(DocumentBase):
    pass

class DocumentUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    category: Optional[str] = Field(None, min_length=1, max_length=50)
    level: Optional[str] = Field(None, pattern="^(beginner|intermediate|advanced)$")
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class Document(DocumentBase):
    id: str
    created_at: datetime
    updated_at: datetime
    embedding_status: str = "pending"

class SearchRequest(BaseModel):
    query: str
    backend: Optional[Literal["chromadb", "faiss", "annoy"]] = None
    top_k: int = Field(default=5, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None
    include_content: bool = True

class BatchSearchRequest(BaseModel):
    queries: List[str]
    backend: Optional[Literal["chromadb", "faiss", "annoy"]] = None
    top_k: int = Field(default=5, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None

class IndexRequest(BaseModel):
    content: str
    metadata: Dict[str, Any]
    title: Optional[str] = None
    auto_chunk: bool = True
    chunk_size: int = 500
    chunk_overlap: int = 100

class RAGRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    include_sources: bool = True

class BenchmarkRequest(BaseModel):
    test_queries: List[str]
    backends: List[Literal["chromadb", "faiss", "annoy"]] = ["faiss"]
    top_k: int = 5

# Health Check
@app.get("/health")
async def health_check():
    """Check service health"""
    return {
        "status": "healthy",
        "service": "iwl-knowledge-base-v2",
        "version": "2.0.0",
        "vector_backend": VECTOR_BACKEND,
        "documents_count": len(documents_db),
        "vector_stats": vector_manager.get_stats()
    }

# CRUD Operations
@app.post("/api/v2/documents", response_model=Document)
async def create_document(document: DocumentCreate):
    """Create a new document with automatic embedding"""
    doc_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    # Create document
    new_doc = Document(
        id=doc_id,
        title=document.title,
        content=document.content,
        category=document.category,
        level=document.level,
        tags=document.tags,
        metadata=document.metadata,
        created_at=now,
        updated_at=now,
        embedding_status="processing"
    )
    
    # Store document
    documents_db[doc_id] = new_doc
    documents_content[doc_id] = document.content
    
    # Generate embedding and index
    try:
        embedding = embedding_pipeline.embed_query(document.content)
        
        # Prepare metadata
        meta = {
            "title": document.title,
            "category": document.category,
            "level": document.level,
            "tags": document.tags,
            **document.metadata
        }
        
        # Add to vector store
        vector_manager.add_documents(
            documents=[document.content],
            embeddings=np.array([embedding]),
            ids=[doc_id],
            metadata=[meta]
        )
        
        new_doc.embedding_status = "indexed"
    except Exception as e:
        new_doc.embedding_status = f"error: {str(e)}"
    
    return new_doc

@app.get("/api/v2/documents", response_model=List[Document])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    category: Optional[str] = None,
    level: Optional[str] = None,
    tag: Optional[str] = None
):
    """List documents with filtering"""
    docs = list(documents_db.values())
    
    # Apply filters
    if category:
        docs = [d for d in docs if d.category == category]
    if level:
        docs = [d for d in docs if d.level == level]
    if tag:
        docs = [d for d in docs if tag in d.tags]
    
    # Sort by created_at
    docs.sort(key=lambda x: x.created_at, reverse=True)
    
    return docs[skip:skip+limit]

@app.get("/api/v2/documents/{document_id}", response_model=Document)
async def get_document(document_id: str):
    """Get document by ID"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    return documents_db[document_id]

@app.put("/api/v2/documents/{document_id}", response_model=Document)
async def update_document(document_id: str, update: DocumentUpdate):
    """Update document and re-index if content changed"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents_db[document_id]
    update_data = update.dict(exclude_unset=True)
    
    # Check if content is being updated
    content_updated = "content" in update_data
    
    # Update fields
    for field, value in update_data.items():
        setattr(doc, field, value)
    
    doc.updated_at = datetime.utcnow()
    
    # Re-index if content changed
    if content_updated:
        documents_content[document_id] = doc.content
        
        try:
            embedding = embedding_pipeline.embed_query(doc.content)
            
            meta = {
                "title": doc.title,
                "category": doc.category,
                "level": doc.level,
                "tags": doc.tags,
                **doc.metadata
            }
            
            vector_manager.update_document(
                doc_id=document_id,
                embedding=np.array(embedding),
                metadata=meta
            )
            
            doc.embedding_status = "indexed"
        except Exception as e:
            doc.embedding_status = f"error: {str(e)}"
    
    return doc

@app.delete("/api/v2/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete document and its embeddings"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete from vector store
    vector_manager.delete_documents([document_id])
    
    # Delete from storage
    del documents_db[document_id]
    if document_id in documents_content:
        del documents_content[document_id]
    
    return {"status": "deleted", "id": document_id}

# Search Operations
@app.post("/api/v2/search")
async def semantic_search(request: SearchRequest):
    """Perform semantic search with specified backend"""
    # Generate query embedding
    query_embedding = embedding_pipeline.embed_query(request.query)
    
    # Search
    results = vector_manager.search(
        query_embedding=np.array(query_embedding),
        k=request.top_k,
        filters=request.filters,
        return_docs=request.include_content
    )
    
    # Enhance results with content if requested
    if request.include_content:
        for result in results:
            if result.get("id") in documents_content:
                result["content"] = documents_content[result["id"]]
    
    return {
        "query": request.query,
        "backend": VECTOR_BACKEND,
        "results": results,
        "total": len(results)
    }

@app.post("/api/v2/batch_search")
async def batch_semantic_search(request: BatchSearchRequest):
    """Batch semantic search for multiple queries"""
    # Generate embeddings for all queries
    embeddings = embedding_pipeline.embed_texts(request.queries)
    
    # Batch search
    all_results = vector_manager.batch_search(
        query_embeddings=np.array(embeddings),
        k=request.top_k,
        filters=request.filters
    )
    
    return {
        "queries": request.queries,
        "backend": VECTOR_BACKEND,
        "results": all_results,
        "total_queries": len(request.queries)
    }

# RAG Operations
@app.post("/api/v2/rag/query")
async def rag_query(request: RAGRequest):
    """Process RAG query with optimized retrieval"""
    start_time = time.time()
    
    # Execute RAG query
    result = rag_pipeline.query(
        question=request.question,
        n_results=request.top_k,
        temperature=request.temperature,
        include_sources=request.include_sources
    )
    
    # Add timing information
    result["response_time_ms"] = (time.time() - start_time) * 1000
    result["backend"] = VECTOR_BACKEND
    
    return result

# Indexing Operations
@app.post("/api/v2/index")
async def index_content(request: IndexRequest):
    """Index new content with automatic chunking"""
    doc_id = str(uuid.uuid4())
    
    # Process with embedding pipeline
    chunks = embedding_pipeline.process_document(
        document=request.content,
        metadata=request.metadata,
        generate_embeddings=True
    )
    
    # Add to vector store
    chunk_ids = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        
        vector_manager.add_documents(
            documents=[chunk["text"]],
            embeddings=np.array([chunk["embedding"]]),
            ids=[chunk_id],
            metadata=[chunk["metadata"]]
        )
        
        chunk_ids.append(chunk_id)
    
    return {
        "status": "indexed",
        "document_id": doc_id,
        "chunks_created": len(chunks),
        "chunk_ids": chunk_ids,
        "title": request.title or request.metadata.get("title", "Untitled")
    }

# Benchmark Operations
@app.post("/api/v2/benchmark")
async def benchmark_search(request: BenchmarkRequest):
    """Benchmark different vector search backends"""
    results = {}
    
    for backend_name in request.backends:
        # Create temporary backend
        temp_manager = VectorSearchManager(
            backend=backend_name,
            dimension=EMBEDDING_DIM,
            persist_directory=f"./temp_bench_{backend_name}"
        )
        
        # Add sample data
        sample_texts = [
            "Python programming basics",
            "Machine learning fundamentals",
            "Web development with FastAPI",
            "Data structures and algorithms",
            "Cloud computing essentials"
        ]
        
        sample_embeddings = embedding_pipeline.embed_texts(sample_texts)
        sample_ids = [f"bench_{i}" for i in range(len(sample_texts))]
        sample_metadata = [{"title": text} for text in sample_texts]
        
        temp_manager.add_documents(
            documents=sample_texts,
            embeddings=np.array(sample_embeddings),
            ids=sample_ids,
            metadata=sample_metadata
        )
        
        # Benchmark queries
        backend_results = []
        for query in request.test_queries:
            query_embedding = embedding_pipeline.embed_query(query)
            
            bench_result = temp_manager.benchmark_search(
                query_embedding=np.array(query_embedding),
                k=request.top_k
            )
            bench_result["query"] = query
            backend_results.append(bench_result)
        
        results[backend_name] = {
            "queries": backend_results,
            "avg_time_ms": sum(r["search_time_ms"] for r in backend_results) / len(backend_results),
            "total_documents": len(sample_texts)
        }
        
        # Cleanup
        temp_manager.clear()
    
    return {
        "benchmark_results": results,
        "test_queries": request.test_queries,
        "top_k": request.top_k
    }

# Statistics
@app.get("/api/v2/stats")
async def get_statistics():
    """Get comprehensive system statistics"""
    vector_stats = vector_manager.get_stats()
    
    return {
        "documents": {
            "total": len(documents_db),
            "by_category": {},
            "by_level": {}
        },
        "vector_search": vector_stats,
        "embedding_model": embedding_pipeline.get_model_info(),
        "backend": VECTOR_BACKEND,
        "dimension": EMBEDDING_DIM
    }

# Admin Operations
@app.post("/api/v2/admin/optimize")
async def optimize_index():
    """Optimize vector search index"""
    vector_manager.optimize_index()
    return {"status": "optimized", "backend": VECTOR_BACKEND}

@app.post("/api/v2/admin/clear_cache")
async def clear_cache():
    """Clear search cache"""
    vector_manager.search_cache.clear()
    return {"status": "cache_cleared"}

@app.delete("/api/v2/admin/reset")
async def reset_system():
    """Reset entire system (use with caution!)"""
    vector_manager.clear()
    documents_db.clear()
    documents_content.clear()
    return {"status": "system_reset"}

# Import numpy
import numpy as np

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        reload=True
    )