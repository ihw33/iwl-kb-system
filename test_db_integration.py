#!/usr/bin/env python
"""
Test Database Integration
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8002"

def test_database_integration():
    """Test complete database integration flow"""
    
    print("=" * 50)
    print("Database Integration Test")
    print("=" * 50)
    
    # 1. Check health and database backend
    print("\n1. Checking health and database...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Database backend: {data['database']}")
    print(f"✅ Document count: {data['documents_count']}")
    
    # 2. Create a document
    print("\n2. Creating document...")
    doc = {
        "title": "Test Document with DB",
        "content": "This document is stored in the database backend.",
        "category": "test",
        "level": "intermediate",
        "tags": ["database", "test"],
        "metadata": {"db_test": True}
    }
    
    response = requests.post(f"{BASE_URL}/api/documents", json=doc)
    assert response.status_code == 201
    doc_id = response.json()["id"]
    print(f"✅ Created document: {doc_id}")
    
    # 3. Get the document
    print("\n3. Retrieving document...")
    response = requests.get(f"{BASE_URL}/api/documents/{doc_id}")
    assert response.status_code == 200
    retrieved_doc = response.json()
    assert retrieved_doc["title"] == doc["title"]
    print(f"✅ Retrieved: {retrieved_doc['title']}")
    
    # 4. Update the document
    print("\n4. Updating document...")
    update = {"level": "advanced", "tags": ["database", "test", "updated"]}
    response = requests.put(f"{BASE_URL}/api/documents/{doc_id}", json=update)
    assert response.status_code == 200
    updated_doc = response.json()
    assert updated_doc["level"] == "advanced"
    assert "updated" in updated_doc["tags"]
    print(f"✅ Updated level: {updated_doc['level']}")
    print(f"✅ Updated tags: {updated_doc['tags']}")
    
    # 5. List documents
    print("\n5. Listing documents...")
    response = requests.get(f"{BASE_URL}/api/documents")
    assert response.status_code == 200
    docs_list = response.json()
    assert len(docs_list) > 0
    print(f"✅ Found {len(docs_list)} documents")
    
    # 6. Search documents
    print("\n6. Searching documents...")
    response = requests.get(f"{BASE_URL}/api/search?q=database")
    assert response.status_code == 200
    search_results = response.json()
    assert len(search_results) > 0
    print(f"✅ Search found {len(search_results)} results")
    
    # 7. Get statistics
    print("\n7. Getting statistics...")
    response = requests.get(f"{BASE_URL}/api/stats")
    assert response.status_code == 200
    stats = response.json()
    print(f"✅ Total documents: {stats['total_documents']}")
    print(f"✅ Backend: {stats['database_backend']}")
    
    # 8. Batch create
    print("\n8. Batch creating documents...")
    batch_docs = [
        {
            "title": f"Batch Doc {i}",
            "content": f"Batch content {i}",
            "category": "batch",
            "level": "beginner",
            "tags": [f"batch{i}"],
            "metadata": {"batch": i}
        } for i in range(3)
    ]
    
    response = requests.post(f"{BASE_URL}/api/documents/batch", json=batch_docs)
    assert response.status_code == 200
    created_batch = response.json()
    assert len(created_batch) == 3
    print(f"✅ Created {len(created_batch)} documents in batch")
    
    # 9. List to verify all documents
    print("\n9. Verifying all documents...")
    response = requests.get(f"{BASE_URL}/api/documents?limit=100")
    assert response.status_code == 200
    all_docs = response.json()
    print(f"✅ Total documents in database: {len(all_docs)}")
    
    # 10. Delete the test document
    print("\n10. Deleting test document...")
    response = requests.delete(f"{BASE_URL}/api/documents/{doc_id}")
    assert response.status_code == 204
    
    # Verify deletion
    response = requests.get(f"{BASE_URL}/api/documents/{doc_id}")
    assert response.status_code == 404
    print(f"✅ Document {doc_id} deleted successfully")
    
    # 11. Final health check
    print("\n11. Final health check...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Final document count: {data['documents_count']}")
    
    print("\n" + "=" * 50)
    print("✅ All database integration tests passed!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    test_database_integration()