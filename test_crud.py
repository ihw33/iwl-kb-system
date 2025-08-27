#!/usr/bin/env python
"""
Test script for CRUD operations
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8002"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_create_document():
    """Test document creation"""
    print("Testing document creation...")
    
    document = {
        "title": "Python 비동기 프로그래밍",
        "content": "Python의 async/await를 사용한 비동기 프로그래밍은 I/O 바운드 작업에서 효율적입니다. " * 50,
        "category": "programming",
        "level": "advanced",
        "tags": ["python", "async", "programming"],
        "metadata": {
            "author": "IWL Team",
            "version": "1.0"
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/documents", json=document)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 201:
        doc = response.json()
        print(f"Created document ID: {doc['id']}")
        print(f"Number of chunks: {len(doc['chunk_ids'])}")
        return doc['id']
    else:
        print(f"Error: {response.text}")
        return None

def test_list_documents():
    """Test listing documents"""
    print("Testing document listing...")
    
    response = requests.get(f"{BASE_URL}/api/documents")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        docs = response.json()
        print(f"Total documents: {len(docs)}")
        for doc in docs:
            print(f"  - {doc['title']} (ID: {doc['id']})")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_get_document(doc_id):
    """Test getting a specific document"""
    print(f"Testing get document {doc_id}...")
    
    response = requests.get(f"{BASE_URL}/api/documents/{doc_id}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        doc = response.json()
        print(f"Title: {doc['title']}")
        print(f"Category: {doc['category']}")
        print(f"Level: {doc['level']}")
        print(f"Tags: {doc['tags']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_update_document(doc_id):
    """Test updating a document"""
    print(f"Testing update document {doc_id}...")
    
    update_data = {
        "title": "Python 비동기 프로그래밍 (수정됨)",
        "tags": ["python", "async", "programming", "updated"]
    }
    
    response = requests.put(f"{BASE_URL}/api/documents/{doc_id}", json=update_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        doc = response.json()
        print(f"Updated title: {doc['title']}")
        print(f"Updated tags: {doc['tags']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_get_chunks(doc_id):
    """Test getting document chunks"""
    print(f"Testing get chunks for document {doc_id}...")
    
    response = requests.get(f"{BASE_URL}/api/documents/{doc_id}/chunks")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        chunks = response.json()
        print(f"Total chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"  Chunk {i}: {chunk['content'][:50]}...")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_batch_create():
    """Test batch document creation"""
    print("Testing batch document creation...")
    
    documents = [
        {
            "title": "FastAPI 기초",
            "content": "FastAPI는 현대적인 웹 API 프레임워크입니다.",
            "category": "web",
            "level": "beginner",
            "tags": ["fastapi", "web", "api"]
        },
        {
            "title": "Docker 컨테이너",
            "content": "Docker를 사용한 컨테이너화 전략과 베스트 프랙티스.",
            "category": "devops",
            "level": "intermediate",
            "tags": ["docker", "containers", "devops"]
        }
    ]
    
    response = requests.post(f"{BASE_URL}/api/documents/batch", json=documents)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        docs = response.json()
        print(f"Created {len(docs)} documents")
        return [doc['id'] for doc in docs]
    else:
        print(f"Error: {response.text}")
        return []
    print("-" * 50)

def test_statistics():
    """Test statistics endpoint"""
    print("Testing statistics...")
    
    response = requests.get(f"{BASE_URL}/api/stats")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        stats = response.json()
        print(json.dumps(stats, indent=2))
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_delete_document(doc_id):
    """Test deleting a document"""
    print(f"Testing delete document {doc_id}...")
    
    response = requests.delete(f"{BASE_URL}/api/documents/{doc_id}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 204:
        print("Document deleted successfully")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def main():
    print("=" * 50)
    print("IWL Knowledge Base CRUD Tests")
    print("=" * 50)
    print()
    
    # Test health
    test_health()
    
    # Test single document CRUD
    doc_id = test_create_document()
    if doc_id:
        test_get_document(doc_id)
        test_update_document(doc_id)
        test_get_chunks(doc_id)
    
    # Test batch operations
    batch_ids = test_batch_create()
    
    # List all documents
    test_list_documents()
    
    # Get statistics
    test_statistics()
    
    # Cleanup - delete test documents
    if doc_id:
        test_delete_document(doc_id)
    
    if batch_ids:
        print(f"Cleaning up {len(batch_ids)} batch documents...")
        response = requests.delete(f"{BASE_URL}/api/documents/batch", json=batch_ids)
        print(f"Cleanup status: {response.status_code}")
    
    print("\n" + "=" * 50)
    print("Tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()