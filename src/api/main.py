"""
IWL Knowledge Base API - Simplified CRUD
FastAPI application with basic CRUD endpoints
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

# Initialize FastAPI app
app = FastAPI(
    title="IWL Knowledge Base API",
    description="Simplified CRUD API for educational content management",
    version="1.0.0"
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

# Pydantic Models
class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1, max_length=50)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    category: Optional[str] = Field(None, min_length=1, max_length=50)
    metadata: Optional[Dict[str, Any]] = None

class Document(BaseModel):
    id: str
    title: str
    content: str
    category: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

# ==================== CRUD ENDPOINTS ====================

# GET - Read single document
@app.get("/api/documents/{document_id}", response_model=Document)
async def get_document(document_id: str):
    """Get a specific document by ID"""
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    return documents_db[document_id]

# POST - Create document
@app.post("/api/documents", response_model=Document, status_code=status.HTTP_201_CREATED)
async def create_document(document: DocumentCreate):
    """Create a new document"""
    doc_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    new_doc = Document(
        id=doc_id,
        title=document.title,
        content=document.content,
        category=document.category,
        metadata=document.metadata,
        created_at=now,
        updated_at=now
    )
    
    documents_db[doc_id] = new_doc
    return new_doc

# PUT - Update document
@app.put("/api/documents/{document_id}", response_model=Document)
async def update_document(document_id: str, document_update: DocumentUpdate):
    """Update an existing document"""
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
    return doc

# DELETE - Delete document
@app.delete("/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str):
    """Delete a document"""
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    del documents_db[document_id]
    return None

# Health Check
@app.get("/health")
async def health_check():
    """Check service health"""
    return {
        "status": "healthy",
        "service": "iwl-knowledge-base",
        "version": "1.0.0",
        "documents_count": len(documents_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002,
        reload=True
    )