#!/usr/bin/env python
"""
Simple test script for KB System v2
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8003"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Health: {data['status']}, Backend: {data['vector_backend']}")
    return True

def test_document_crud():
    """Test document CRUD operations"""
    print("\nTesting document CRUD...")
    
    # Create
    doc = {
        "title": "Test Document",
        "content": "This is a test document about Python programming.",
        "category": "programming",
        "level": "beginner",
        "tags": ["python", "test"]
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/documents", json=doc)
    assert response.status_code == 200
    doc_id = response.json()["id"]
    print(f"✅ Created document: {doc_id}")
    
    # Read
    response = requests.get(f"{BASE_URL}/api/v2/documents/{doc_id}")
    assert response.status_code == 200
    print(f"✅ Retrieved document: {response.json()['title']}")
    
    # Update
    update = {"level": "intermediate"}
    response = requests.put(f"{BASE_URL}/api/v2/documents/{doc_id}", json=update)
    assert response.status_code == 200
    print(f"✅ Updated document level to: {response.json()['level']}")
    
    # Delete
    response = requests.delete(f"{BASE_URL}/api/v2/documents/{doc_id}")
    assert response.status_code == 200
    print(f"✅ Deleted document: {doc_id}")
    
    return True

def test_search():
    """Test search functionality"""
    print("\nTesting search...")
    
    # Create test documents
    documents = [
        {
            "title": "Python Basics",
            "content": "Python is a high-level programming language known for its simplicity.",
            "category": "programming",
            "level": "beginner",
            "tags": ["python"]
        },
        {
            "title": "Machine Learning",
            "content": "Machine learning enables computers to learn from data without explicit programming.",
            "category": "ai",
            "level": "intermediate",
            "tags": ["ml", "ai"]
        },
        {
            "title": "Web Development",
            "content": "FastAPI is a modern web framework for building APIs with Python.",
            "category": "web",
            "level": "intermediate",
            "tags": ["web", "fastapi"]
        }
    ]
    
    doc_ids = []
    for doc in documents:
        response = requests.post(f"{BASE_URL}/api/v2/documents", json=doc)
        if response.status_code == 200:
            doc_ids.append(response.json()["id"])
    
    print(f"✅ Created {len(doc_ids)} test documents")
    
    # Wait for indexing
    time.sleep(1)
    
    # Search
    search_request = {
        "query": "Python programming",
        "top_k": 3
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/search", json=search_request)
    assert response.status_code == 200
    results = response.json()["results"]
    print(f"✅ Search returned {len(results)} results")
    
    # Cleanup
    for doc_id in doc_ids:
        requests.delete(f"{BASE_URL}/api/v2/documents/{doc_id}")
    
    return True

def test_indexing():
    """Test content indexing"""
    print("\nTesting content indexing...")
    
    index_request = {
        "content": "This is a test document for indexing. It contains information about various topics.",
        "metadata": {"category": "test"},
        "title": "Test Index"
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/index", json=index_request)
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Indexed document: {data['document_id']}, Chunks: {data['chunks_created']}")
    
    return True

def test_rag():
    """Test RAG functionality"""
    print("\nTesting RAG...")
    
    # Index some content first
    contents = [
        {
            "content": "Python is a versatile programming language used for web development, data science, and automation.",
            "metadata": {"title": "Python Overview"}
        },
        {
            "content": "FastAPI is built on top of Starlette and Pydantic, offering high performance and automatic documentation.",
            "metadata": {"title": "FastAPI Details"}
        }
    ]
    
    for content in contents:
        requests.post(f"{BASE_URL}/api/v2/index", json=content)
    
    time.sleep(1)
    
    # RAG query
    rag_request = {
        "question": "What is Python used for?",
        "top_k": 3,
        "temperature": 0.5
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/rag/query", json=rag_request)
    assert response.status_code == 200
    data = response.json()
    print(f"✅ RAG answer received, confidence: {data.get('confidence', 0):.2f}")
    
    return True

def test_benchmark():
    """Test benchmark functionality"""
    print("\nTesting benchmark...")
    
    benchmark_request = {
        "test_queries": ["Python", "Machine Learning", "Web Development"],
        "backends": ["faiss"],
        "top_k": 3
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/benchmark", json=benchmark_request)
    assert response.status_code == 200
    data = response.json()
    
    if "faiss" in data["benchmark_results"]:
        avg_time = data["benchmark_results"]["faiss"]["avg_time_ms"]
        print(f"✅ Benchmark complete - FAISS avg time: {avg_time:.2f}ms")
    
    return True

def test_stats():
    """Test statistics endpoint"""
    print("\nTesting statistics...")
    
    response = requests.get(f"{BASE_URL}/api/v2/stats")
    assert response.status_code == 200
    data = response.json()
    
    print(f"✅ Stats retrieved:")
    print(f"   - Backend: {data['backend']}")
    print(f"   - Dimension: {data['dimension']}")
    print(f"   - Documents: {data['documents']['total']}")
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("IWL Knowledge Base v2 Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Document CRUD", test_document_crud),
        ("Search", test_search),
        ("Indexing", test_indexing),
        ("RAG", test_rag),
        ("Benchmark", test_benchmark),
        ("Statistics", test_stats)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)