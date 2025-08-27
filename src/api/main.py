"""
IWL Knowledge Base API
Main FastAPI application
"""

from fastapi import FastAPI, HTTPException, status, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="IWL Knowledge Base API",
    description="RAG-based educational content management and search",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo (replace with actual database)
documents_db = {}
chunks_db = {}

# Request/Response Models
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
    chunk_ids: List[str] = Field(default_factory=list)

class ChunkBase(BaseModel):
    content: str
    document_id: str
    position: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Chunk(ChunkBase):
    id: str
    embedding: Optional[List[float]] = None
    created_at: datetime

class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 5
    include_metadata: bool = True

class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]

class RAGRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None
    model: str = "gpt-4"
    temperature: float = 0.7

class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class IndexRequest(BaseModel):
    content: str
    metadata: Dict[str, Any]
    chunk_size: int = 1000
    chunk_overlap: int = 200

# Health Check
@app.get("/health")
async def health_check():
    """Check service health"""
    return {
        "status": "healthy",
        "service": "iwl-knowledge-base",
        "version": "0.1.0",
        "documents_count": len(documents_db),
        "chunks_count": len(chunks_db)
    }

# ==================== CRUD ENDPOINTS ====================

# CREATE - Document
@app.post("/api/documents", response_model=Document, status_code=status.HTTP_201_CREATED)
async def create_document(document: DocumentCreate):
    """Create a new document"""
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
        chunk_ids=[]
    )
    
    # Simple chunking (every 500 characters)
    chunk_size = 500
    chunks = []
    for i in range(0, len(document.content), chunk_size):
        chunk_id = str(uuid.uuid4())
        chunk = Chunk(
            id=chunk_id,
            content=document.content[i:i+chunk_size],
            document_id=doc_id,
            position=i // chunk_size,
            metadata={"doc_title": document.title},
            created_at=now
        )
        chunks.append(chunk)
        chunks_db[chunk_id] = chunk
        new_doc.chunk_ids.append(chunk_id)
    
    documents_db[doc_id] = new_doc
    return new_doc

# READ - List all documents
@app.get("/api/documents", response_model=List[Document])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    category: Optional[str] = None,
    level: Optional[str] = None,
    tag: Optional[str] = None
):
    """List all documents with optional filtering"""
    docs = list(documents_db.values())
    
    # Apply filters
    if category:
        docs = [d for d in docs if d.category == category]
    if level:
        docs = [d for d in docs if d.level == level]
    if tag:
        docs = [d for d in docs if tag in d.tags]
    
    # Sort by created_at (newest first)
    docs.sort(key=lambda x: x.created_at, reverse=True)
    
    # Apply pagination
    return docs[skip:skip+limit]

# READ - Get single document
@app.get("/api/documents/{document_id}", response_model=Document)
async def get_document(document_id: str = Path(..., description="The document ID")):
    """Get a specific document by ID"""
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    return documents_db[document_id]

# UPDATE - Update document
@app.put("/api/documents/{document_id}", response_model=Document)
async def update_document(
    document_id: str,
    document_update: DocumentUpdate
):
    """Update a document"""
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    doc = documents_db[document_id]
    update_data = document_update.dict(exclude_unset=True)
    
    # Update fields
    for field, value in update_data.items():
        setattr(doc, field, value)
    
    doc.updated_at = datetime.utcnow()
    
    # If content changed, update chunks
    if "content" in update_data:
        # Delete old chunks
        for chunk_id in doc.chunk_ids:
            if chunk_id in chunks_db:
                del chunks_db[chunk_id]
        
        # Create new chunks
        doc.chunk_ids = []
        chunk_size = 500
        for i in range(0, len(doc.content), chunk_size):
            chunk_id = str(uuid.uuid4())
            chunk = Chunk(
                id=chunk_id,
                content=doc.content[i:i+chunk_size],
                document_id=document_id,
                position=i // chunk_size,
                metadata={"doc_title": doc.title},
                created_at=datetime.utcnow()
            )
            chunks_db[chunk_id] = chunk
            doc.chunk_ids.append(chunk_id)
    
    return doc

# DELETE - Delete document
@app.delete("/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str):
    """Delete a document and its chunks"""
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    # Delete chunks
    doc = documents_db[document_id]
    for chunk_id in doc.chunk_ids:
        if chunk_id in chunks_db:
            del chunks_db[chunk_id]
    
    # Delete document
    del documents_db[document_id]
    return None

# ==================== CHUNK ENDPOINTS ====================

# GET - List chunks for a document
@app.get("/api/documents/{document_id}/chunks", response_model=List[Chunk])
async def get_document_chunks(document_id: str):
    """Get all chunks for a specific document"""
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    doc = documents_db[document_id]
    chunks = [chunks_db[chunk_id] for chunk_id in doc.chunk_ids if chunk_id in chunks_db]
    chunks.sort(key=lambda x: x.position)
    return chunks

# GET - Single chunk
@app.get("/api/chunks/{chunk_id}", response_model=Chunk)
async def get_chunk(chunk_id: str):
    """Get a specific chunk by ID"""
    if chunk_id not in chunks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk with ID {chunk_id} not found"
        )
    return chunks_db[chunk_id]

# Search Endpoints
@app.post("/api/search", response_model=List[SearchResult])
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search on knowledge base
    """
    # TODO: Implement actual search logic
    # This is a placeholder response
    return [
        SearchResult(
            id="doc_001",
            content="Sample content matching your query",
            score=0.95,
            metadata={"title": "Sample Document", "level": "beginner"}
        )
    ]

# RAG Endpoints
@app.post("/api/rag/query", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    Process a question using RAG (Retrieval-Augmented Generation)
    """
    # TODO: Implement actual RAG logic
    # This is a placeholder response
    return RAGResponse(
        answer="This is a sample answer to your question based on the knowledge base.",
        sources=["doc_001", "doc_002"],
        confidence=0.85
    )

# Content Management
@app.post("/api/index")
async def index_content(request: IndexRequest):
    """
    Index new content into the knowledge base
    """
    # TODO: Implement actual indexing logic
    return {
        "status": "success",
        "message": "Content indexed successfully",
        "document_id": "doc_new_001",
        "chunks_created": 5
    }

@app.get("/api/content/{content_id}")
async def get_content(content_id: str):
    """
    Retrieve specific content by ID
    """
    # TODO: Implement actual content retrieval
    return {
        "id": content_id,
        "title": "Sample Content",
        "content": "This is the content you requested",
        "metadata": {
            "created_at": "2025-08-27",
            "level": "intermediate",
            "tags": ["python", "programming"]
        }
    }

# ==================== BATCH OPERATIONS ====================

@app.post("/api/documents/batch", response_model=List[Document])
async def create_documents_batch(documents: List[DocumentCreate]):
    """Create multiple documents in a single request"""
    created_docs = []
    for doc_data in documents:
        doc_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        new_doc = Document(
            id=doc_id,
            title=doc_data.title,
            content=doc_data.content,
            category=doc_data.category,
            level=doc_data.level,
            tags=doc_data.tags,
            metadata=doc_data.metadata,
            created_at=now,
            updated_at=now,
            chunk_ids=[]
        )
        
        # Create chunks
        chunk_size = 500
        for i in range(0, len(doc_data.content), chunk_size):
            chunk_id = str(uuid.uuid4())
            chunk = Chunk(
                id=chunk_id,
                content=doc_data.content[i:i+chunk_size],
                document_id=doc_id,
                position=i // chunk_size,
                metadata={"doc_title": doc_data.title},
                created_at=now
            )
            chunks_db[chunk_id] = chunk
            new_doc.chunk_ids.append(chunk_id)
        
        documents_db[doc_id] = new_doc
        created_docs.append(new_doc)
    
    return created_docs

@app.delete("/api/documents/batch")
async def delete_documents_batch(document_ids: List[str]):
    """Delete multiple documents in a single request"""
    deleted_count = 0
    not_found_ids = []
    
    for doc_id in document_ids:
        if doc_id in documents_db:
            # Delete chunks
            doc = documents_db[doc_id]
            for chunk_id in doc.chunk_ids:
                if chunk_id in chunks_db:
                    del chunks_db[chunk_id]
            # Delete document
            del documents_db[doc_id]
            deleted_count += 1
        else:
            not_found_ids.append(doc_id)
    
    return {
        "deleted": deleted_count,
        "not_found": not_found_ids,
        "total_requested": len(document_ids)
    }

# Statistics & Monitoring
@app.get("/api/stats")
async def get_statistics():
    """
    Get knowledge base statistics
    """
    # Calculate statistics
    categories = {}
    levels = {}
    all_tags = set()
    
    for doc in documents_db.values():
        categories[doc.category] = categories.get(doc.category, 0) + 1
        levels[doc.level] = levels.get(doc.level, 0) + 1
        all_tags.update(doc.tags)
    
    latest_doc = max(documents_db.values(), key=lambda x: x.created_at) if documents_db else None
    
    return {
        "total_documents": len(documents_db),
        "total_chunks": len(chunks_db),
        "categories": categories,
        "levels": levels,
        "unique_tags": len(all_tags),
        "tags": list(all_tags),
        "last_indexed": latest_doc.created_at.isoformat() if latest_doc else None,
        "avg_chunks_per_doc": len(chunks_db) / len(documents_db) if documents_db else 0
    }

# Embedding Endpoints
@app.post("/api/embed")
async def generate_embedding(text: str):
    """
    Generate embedding for given text
    """
    # TODO: Implement actual embedding generation
    return {
        "text": text,
        "embedding": [0.1] * 1536,  # Placeholder embedding
        "model": "text-embedding-ada-002",
        "dimension": 1536
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )