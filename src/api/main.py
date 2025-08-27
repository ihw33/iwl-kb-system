"""
IWL Knowledge Base API - Database-backed CRUD
FastAPI application with Redis/PostgreSQL storage
"""

from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import database
from database import get_db

# Initialize FastAPI app
app = FastAPI(
    title="IWL Knowledge Base API",
    description="Database-backed CRUD API for educational content management",
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

# Get database instance
db = get_db()

# Pydantic Models
class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1, max_length=50)
    level: str = Field(default="intermediate", pattern="^(beginner|intermediate|advanced)$")
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    category: Optional[str] = Field(None, min_length=1, max_length=50)
    level: Optional[str] = Field(None, pattern="^(beginner|intermediate|advanced)$")
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class Document(BaseModel):
    id: str
    title: str
    content: str
    category: str
    level: str = "intermediate"
    tags: List[str] = []
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

# ==================== CRUD ENDPOINTS ====================

# GET - List all documents
@app.get("/api/documents", response_model=List[Document])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    category: Optional[str] = None
):
    """List all documents with optional filtering"""
    docs = db.list_documents(skip=skip, limit=limit, category=category)
    
    # Convert to Document models
    documents = []
    for doc in docs:
        # Ensure all required fields exist
        if "level" not in doc:
            doc["level"] = "intermediate"
        if "tags" not in doc:
            doc["tags"] = []
        documents.append(Document(**doc))
    
    return documents

# GET - Read single document
@app.get("/api/documents/{document_id}", response_model=Document)
async def get_document(document_id: str):
    """Get a specific document by ID"""
    doc = db.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    # Ensure all required fields exist
    if "level" not in doc:
        doc["level"] = "intermediate"
    if "tags" not in doc:
        doc["tags"] = []
    
    return Document(**doc)

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
        level=document.level,
        tags=document.tags,
        metadata=document.metadata,
        created_at=now,
        updated_at=now
    )
    
    # Store in database
    doc_dict = new_doc.dict()
    success = db.create_document(doc_id, doc_dict)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create document"
        )
    
    return new_doc

# PUT - Update document
@app.put("/api/documents/{document_id}", response_model=Document)
async def update_document(document_id: str, document_update: DocumentUpdate):
    """Update an existing document"""
    doc = db.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    update_data = document_update.dict(exclude_unset=True)
    update_data["updated_at"] = datetime.utcnow()
    
    # Update in database
    success = db.update_document(document_id, update_data)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document"
        )
    
    # Return updated document
    updated_doc = db.get_document(document_id)
    
    # Ensure all required fields exist
    if "level" not in updated_doc:
        updated_doc["level"] = "intermediate"
    if "tags" not in updated_doc:
        updated_doc["tags"] = []
    
    return Document(**updated_doc)

# DELETE - Delete document
@app.delete("/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str):
    """Delete a document"""
    success = db.delete_document(document_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    return None

# ==================== ADDITIONAL ENDPOINTS ====================

# Health Check
@app.get("/health")
async def health_check():
    """Check service health"""
    return {
        "status": "healthy",
        "service": "iwl-knowledge-base",
        "version": "2.0.0",
        "documents_count": db.count_documents(),
        "database": db.__class__.__name__
    }

# Search documents
@app.get("/api/search", response_model=List[Document])
async def search_documents(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100)
):
    """Search documents by text query"""
    docs = db.search_documents(query=q, limit=limit)
    
    documents = []
    for doc in docs:
        # Ensure all required fields exist
        if "level" not in doc:
            doc["level"] = "intermediate"
        if "tags" not in doc:
            doc["tags"] = []
        documents.append(Document(**doc))
    
    return documents

# Count documents
@app.get("/api/stats")
async def get_statistics():
    """Get database statistics"""
    return {
        "total_documents": db.count_documents(),
        "database_backend": db.__class__.__name__,
        "categories": {
            "programming": db.count_documents("programming"),
            "ml": db.count_documents("ml"),
            "web": db.count_documents("web"),
            "other": db.count_documents("other")
        }
    }

# Batch create documents
@app.post("/api/documents/batch", response_model=List[Document])
async def create_documents_batch(documents: List[DocumentCreate]):
    """Create multiple documents in batch"""
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
            updated_at=now
        )
        
        # Store in database
        doc_dict = new_doc.dict()
        success = db.create_document(doc_id, doc_dict)
        
        if success:
            created_docs.append(new_doc)
    
    return created_docs

# Clear all documents (admin only)
@app.delete("/api/admin/clear", include_in_schema=False)
async def clear_all_documents(admin_key: str = Query(...)):
    """Clear all documents - requires admin key"""
    if admin_key != "admin123":  # In production, use proper authentication
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin key"
        )
    
    success = db.clear_all()
    if success:
        return {"message": "All documents cleared"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear documents"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002,
        reload=True
    )