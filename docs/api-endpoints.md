# IWL Knowledge Base API Endpoints

## Base URL
- Development: `http://127.0.0.1:8002`
- Production: TBD

## Interactive Documentation
- Swagger UI: http://127.0.0.1:8002/docs
- ReDoc: http://127.0.0.1:8002/redoc

## Endpoints

### Health Check
```http
GET /health
```
Response:
```json
{
  "status": "healthy",
  "service": "iwl-knowledge-base",
  "version": "0.1.0",
  "documents_count": 0,
  "chunks_count": 0
}
```

### Document CRUD Operations

#### Create Document
```http
POST /api/documents
Content-Type: application/json

{
  "title": "Document Title",
  "content": "Document content...",
  "category": "programming",
  "level": "intermediate",
  "tags": ["tag1", "tag2"],
  "metadata": {
    "author": "Author Name"
  }
}
```

#### List Documents
```http
GET /api/documents?skip=0&limit=10&category=programming&level=intermediate&tag=python
```

#### Get Single Document
```http
GET /api/documents/{document_id}
```

#### Update Document
```http
PUT /api/documents/{document_id}
Content-Type: application/json

{
  "title": "Updated Title",
  "tags": ["new", "tags"]
}
```

#### Delete Document
```http
DELETE /api/documents/{document_id}
```

### Batch Operations

#### Create Multiple Documents
```http
POST /api/documents/batch
Content-Type: application/json

[
  {
    "title": "Document 1",
    "content": "Content 1",
    "category": "web",
    "level": "beginner",
    "tags": ["tag1"]
  },
  {
    "title": "Document 2",
    "content": "Content 2",
    "category": "devops",
    "level": "advanced",
    "tags": ["tag2"]
  }
]
```

#### Delete Multiple Documents
```http
DELETE /api/documents/batch
Content-Type: application/json

["document_id_1", "document_id_2"]
```

### Chunk Operations

#### Get Document Chunks
```http
GET /api/documents/{document_id}/chunks
```

#### Get Single Chunk
```http
GET /api/chunks/{chunk_id}
```

### Search Operations

#### Semantic Search
```http
POST /api/search
Content-Type: application/json

{
  "query": "비동기 프로그래밍",
  "filters": {
    "category": "programming"
  },
  "top_k": 5,
  "include_metadata": true
}
```

### RAG Operations

#### RAG Query
```http
POST /api/rag/query
Content-Type: application/json

{
  "question": "FastAPI에서 dependency injection은 어떻게 동작하나요?",
  "context": {
    "user_level": "advanced"
  },
  "model": "gpt-4",
  "temperature": 0.7
}
```

### Statistics

#### Get Statistics
```http
GET /api/stats
```
Response:
```json
{
  "total_documents": 10,
  "total_chunks": 45,
  "categories": {
    "programming": 5,
    "web": 3,
    "devops": 2
  },
  "levels": {
    "beginner": 3,
    "intermediate": 4,
    "advanced": 3
  },
  "unique_tags": 15,
  "tags": ["python", "fastapi", "docker", ...],
  "last_indexed": "2025-08-27T02:00:00",
  "avg_chunks_per_doc": 4.5
}
```

### Other Operations

#### Index Content
```http
POST /api/index
Content-Type: application/json

{
  "content": "Content to index",
  "metadata": {
    "source": "manual"
  },
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

#### Generate Embedding
```http
POST /api/embed
Content-Type: application/json

{
  "text": "Text to embed"
}
```

## Error Responses

### 404 Not Found
```json
{
  "detail": "Document with ID {id} not found"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "field"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

## Testing

### Using curl
```bash
# Health check
curl http://127.0.0.1:8002/health

# Create document
curl -X POST http://127.0.0.1:8002/api/documents \
  -H "Content-Type: application/json" \
  -d '{"title":"Test","content":"Content","category":"test","level":"beginner","tags":["test"]}'

# List documents
curl http://127.0.0.1:8002/api/documents

# Get statistics
curl http://127.0.0.1:8002/api/stats
```

### Using Python test script
```bash
python test_crud.py
```

## Notes

- All document content is automatically chunked (500 characters per chunk)
- Chunks are updated automatically when document content changes
- Currently using in-memory storage (will be replaced with persistent database)
- Embeddings and RAG features are placeholders (to be implemented)