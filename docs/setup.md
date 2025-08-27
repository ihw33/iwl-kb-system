# IWL Knowledge Base Setup Guide

## Prerequisites

### System Requirements
- Python 3.11+
- Docker & Docker Compose
- 8GB+ RAM
- 20GB+ Storage

### API Keys Required
```bash
# Create .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
PINECONE_API_KEY=...  # If using Pinecone
```

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/ihw33/iwl-kb-system.git
cd iwl-kb-system
```

### 2. Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Vector Database Setup

#### Option A: ChromaDB (Local)
```bash
# No additional setup required
# ChromaDB will create local storage automatically
```

#### Option B: Pinecone (Cloud)
```bash
# Sign up at https://www.pinecone.io/
# Create index with dimension 1536 (for Ada-002)
# Add API key to .env file
```

### 4. Initialize Database
```bash
# Run initialization script
python scripts/init_db.py

# Load sample content
python scripts/load_samples.py
```

### 5. Start Services
```bash
# Development mode
uvicorn src.api.main:app --reload --port 8000

# Production mode
docker-compose up -d
```

## Detailed Configuration

### Vector Database Configuration

#### ChromaDB Settings
```python
# config/chroma.yaml
chroma:
  persist_directory: "./chroma_db"
  collection_name: "iwl_knowledge"
  embedding_function: "openai"
  distance_metric: "cosine"
```

#### Pinecone Settings
```python
# config/pinecone.yaml
pinecone:
  environment: "us-east-1"
  index_name: "iwl-knowledge"
  dimension: 1536
  metric: "cosine"
  replicas: 1
```

### Embedding Configuration
```python
# config/embeddings.yaml
embeddings:
  model: "text-embedding-ada-002"  # OpenAI
  chunk_size: 1000
  chunk_overlap: 200
  batch_size: 100
```

### RAG Configuration
```python
# config/rag.yaml
rag:
  llm_model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000
  top_k_retrieval: 5
  context_window: 4000
```

## Content Management

### Adding Content
```bash
# Single document
python scripts/add_content.py --file path/to/document.md

# Batch import
python scripts/batch_import.py --dir content/courses/

# From URL
python scripts/import_url.py --url https://example.com/content
```

### Content Structure
```markdown
# content/courses/python/basics.md
---
title: Python Basics
level: beginner
language: ko
tags: [python, programming, basics]
version: 1.0
---

## Course Content
...
```

## Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Load Testing
```bash
locust -f tests/load/locustfile.py --host http://localhost:8000
```

## Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Metrics
```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Custom dashboard
open http://localhost:3000/dashboard
```

## Troubleshooting

### Common Issues

#### Issue: Embedding API Rate Limit
```bash
# Solution: Implement rate limiting
export EMBEDDING_RATE_LIMIT=10  # requests per second
```

#### Issue: Vector DB Connection Failed
```bash
# Check connection
python scripts/check_db.py

# Reset database
python scripts/reset_db.py --confirm
```

#### Issue: Memory Issues
```bash
# Increase Docker memory
docker-compose down
# Edit docker-compose.yml - add memory limits
docker-compose up -d
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python src/api/main.py
```

## Docker Deployment

### Build Image
```bash
docker build -t iwl-kb-system:latest .
```

### Run Container
```bash
docker run -d \
  --name iwl-kb \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  iwl-kb-system:latest
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://...
    volumes:
      - ./data:/app/data
    depends_on:
      - postgres
      - redis
  
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: iwl_kb
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

## Production Checklist

### Security
- [ ] API keys in secrets manager
- [ ] HTTPS enabled
- [ ] Rate limiting configured
- [ ] Input validation active
- [ ] Audit logging enabled

### Performance
- [ ] Caching configured
- [ ] Database indexed
- [ ] CDN setup (if needed)
- [ ] Auto-scaling configured

### Monitoring
- [ ] Health checks active
- [ ] Alerts configured
- [ ] Metrics dashboard
- [ ] Log aggregation

### Backup
- [ ] Database backup scheduled
- [ ] Vector DB snapshots
- [ ] Configuration backed up
- [ ] Disaster recovery tested

## Support

### Documentation
- [Architecture Guide](./architecture.md)
- [API Documentation](./api-spec.md)
- [Contributing Guide](../CONTRIBUTING.md)

### Issues
- GitHub Issues: https://github.com/ihw33/iwl-kb-system/issues
- Discord: [Join our server](#)

---

*Setup Guide Version: 1.0*
*Last Updated: 2025-08-27*