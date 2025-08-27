#!/usr/bin/env python
"""
Test script for Vector DB and RAG operations
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8002"

def test_vectordb_stats():
    """Test Vector DB statistics"""
    print("Testing Vector DB stats...")
    response = requests.get(f"{BASE_URL}/api/vectordb/stats")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        stats = response.json()
        print(json.dumps(stats, indent=2))
    print("-" * 50)

def test_index_content():
    """Test content indexing into Vector DB"""
    print("Testing content indexing...")
    
    documents = [
        {
            "content": """Python은 배우기 쉽고 강력한 프로그래밍 언어입니다. 
            Python의 주요 특징으로는 간결한 문법, 동적 타이핑, 풍부한 표준 라이브러리가 있습니다.
            데이터 과학, 웹 개발, 인공지능 등 다양한 분야에서 널리 사용됩니다.""",
            "metadata": {
                "title": "Python 프로그래밍 소개",
                "category": "programming",
                "level": "beginner",
                "tags": ["python", "programming", "basics"]
            }
        },
        {
            "content": """FastAPI는 현대적이고 빠른 웹 API 프레임워크입니다.
            Python 3.6+ 기반으로 작성되었으며, 타입 힌트를 활용하여 자동 문서 생성이 가능합니다.
            비동기 처리를 기본으로 지원하며, 높은 성능을 제공합니다.""",
            "metadata": {
                "title": "FastAPI 프레임워크",
                "category": "web",
                "level": "intermediate",
                "tags": ["fastapi", "web", "api", "python"]
            }
        },
        {
            "content": """Vector Database는 벡터 임베딩을 저장하고 검색하는 데이터베이스입니다.
            ChromaDB, Pinecone, Weaviate 등이 대표적인 벡터 데이터베이스입니다.
            의미 기반 검색과 RAG 시스템 구축에 필수적인 컴포넌트입니다.""",
            "metadata": {
                "title": "Vector Database 개념",
                "category": "infrastructure",
                "level": "advanced",
                "tags": ["vector-db", "chromadb", "embeddings"]
            }
        }
    ]
    
    indexed_ids = []
    for doc in documents:
        response = requests.post(f"{BASE_URL}/api/index", json=doc)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Indexed: {result['message']}")
            print(f"  Document ID: {result['document_id']}")
            print(f"  Chunks created: {result['chunks_created']}")
            indexed_ids.append(result['document_id'])
        else:
            print(f"  Error: {response.text}")
        print()
    
    print(f"Total documents indexed: {len(indexed_ids)}")
    print("-" * 50)
    return indexed_ids

def test_semantic_search():
    """Test semantic search"""
    print("Testing semantic search...")
    
    queries = [
        "Python 프로그래밍의 특징은?",
        "FastAPI 비동기 처리",
        "벡터 데이터베이스 종류",
        "웹 API 개발"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        request_data = {
            "query": query,
            "top_k": 3,
            "include_metadata": True
        }
        
        response = requests.post(f"{BASE_URL}/api/search", json=request_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f}")
                print(f"     Content: {result['content'][:100]}...")
                if result.get('metadata'):
                    print(f"     Metadata: {result['metadata']}")
        else:
            print(f"Error: {response.text}")
    
    print("-" * 50)

def test_rag_query():
    """Test RAG queries"""
    print("Testing RAG queries...")
    
    questions = [
        {
            "question": "Python의 주요 특징을 설명해주세요",
            "temperature": 0.5
        },
        {
            "question": "FastAPI와 다른 웹 프레임워크의 차이점은?",
            "temperature": 0.7
        },
        {
            "question": "Vector Database를 사용하는 이유는 무엇인가요?",
            "temperature": 0.6
        }
    ]
    
    for q_data in questions:
        print(f"\nQuestion: {q_data['question']}")
        
        response = requests.post(f"{BASE_URL}/api/rag/query", json=q_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Sources: {result['sources']}")
        else:
            print(f"Error: {response.text}")
    
    print("-" * 50)

def test_vectordb_operations():
    """Test various Vector DB operations"""
    print("Testing Vector DB operations...")
    
    # 1. Get initial stats
    response = requests.get(f"{BASE_URL}/api/vectordb/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"Initial documents: {stats['total_documents']}")
    
    # 2. Index a test document
    test_doc = {
        "content": "테스트 문서입니다. Vector DB 테스트를 위한 내용입니다.",
        "metadata": {"title": "Test Document", "test": True}
    }
    response = requests.post(f"{BASE_URL}/api/index", json=test_doc)
    
    # 3. Get updated stats
    response = requests.get(f"{BASE_URL}/api/vectordb/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"After indexing: {stats['total_documents']}")
        print(f"Embedding model: {stats.get('embedding_model', {})}")
    
    print("-" * 50)

def main():
    print("=" * 50)
    print("IWL Knowledge Base Vector DB & RAG Tests")
    print("=" * 50)
    print()
    
    # Test Vector DB stats
    test_vectordb_stats()
    
    # Index sample documents
    print("\n📚 Indexing sample documents...")
    indexed_ids = test_index_content()
    
    # Wait for indexing to complete
    print("Waiting for indexing to complete...")
    time.sleep(2)
    
    # Test semantic search
    print("\n🔍 Testing semantic search...")
    test_semantic_search()
    
    # Test RAG queries
    print("\n🤖 Testing RAG queries...")
    test_rag_query()
    
    # Test Vector DB operations
    print("\n🗄️ Testing Vector DB operations...")
    test_vectordb_operations()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()