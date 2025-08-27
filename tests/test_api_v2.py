"""
API v2 Tests
Tests for the enhanced Knowledge Base API with multiple vector backends
"""

import pytest
import requests
import json
import time
from typing import Dict, List, Any

BASE_URL = "http://127.0.0.1:8003"


class TestHealthCheck:
    """Test health check endpoints"""
    
    def test_health_endpoint(self):
        """Test /health endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "vector_backend" in data
        assert "vector_stats" in data


class TestDocumentCRUD:
    """Test document CRUD operations"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_document = {
            "title": "Python Programming Guide",
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "category": "programming",
            "level": "beginner",
            "tags": ["python", "programming", "tutorial"],
            "metadata": {"author": "Test Author", "version": "1.0"}
        }
    
    def test_create_document(self):
        """Test document creation"""
        response = requests.post(
            f"{BASE_URL}/api/v2/documents",
            json=self.test_document
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert data["title"] == self.test_document["title"]
        assert data["embedding_status"] in ["indexed", "processing"]
        
        return data["id"]
    
    def test_list_documents(self):
        """Test listing documents"""
        # Create a few documents
        for i in range(3):
            doc = self.test_document.copy()
            doc["title"] = f"Document {i}"
            requests.post(f"{BASE_URL}/api/v2/documents", json=doc)
        
        # List documents
        response = requests.get(f"{BASE_URL}/api/v2/documents")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_get_document(self):
        """Test getting single document"""
        # Create document
        create_response = requests.post(
            f"{BASE_URL}/api/v2/documents",
            json=self.test_document
        )
        doc_id = create_response.json()["id"]
        
        # Get document
        response = requests.get(f"{BASE_URL}/api/v2/documents/{doc_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == doc_id
        assert data["title"] == self.test_document["title"]
    
    def test_update_document(self):
        """Test updating document"""
        # Create document
        create_response = requests.post(
            f"{BASE_URL}/api/v2/documents",
            json=self.test_document
        )
        doc_id = create_response.json()["id"]
        
        # Update document
        update_data = {
            "title": "Updated Python Guide",
            "level": "intermediate"
        }
        
        response = requests.put(
            f"{BASE_URL}/api/v2/documents/{doc_id}",
            json=update_data
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["title"] == "Updated Python Guide"
        assert data["level"] == "intermediate"
    
    def test_delete_document(self):
        """Test deleting document"""
        # Create document
        create_response = requests.post(
            f"{BASE_URL}/api/v2/documents",
            json=self.test_document
        )
        doc_id = create_response.json()["id"]
        
        # Delete document
        response = requests.delete(f"{BASE_URL}/api/v2/documents/{doc_id}")
        assert response.status_code == 200
        
        # Verify deletion
        get_response = requests.get(f"{BASE_URL}/api/v2/documents/{doc_id}")
        assert get_response.status_code == 404


class TestSearchOperations:
    """Test search functionality"""
    
    def setup_method(self):
        """Setup test documents"""
        self.documents = [
            {
                "title": "Python Basics",
                "content": "Python is a versatile programming language used for web development, data science, and automation.",
                "category": "programming",
                "level": "beginner",
                "tags": ["python", "basics"]
            },
            {
                "title": "Machine Learning Introduction",
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "category": "ml",
                "level": "intermediate",
                "tags": ["ml", "ai"]
            },
            {
                "title": "Web Development with FastAPI",
                "content": "FastAPI is a modern web framework for building APIs with Python based on standard Python type hints.",
                "category": "web",
                "level": "intermediate",
                "tags": ["fastapi", "web", "python"]
            }
        ]
        
        # Create documents
        self.doc_ids = []
        for doc in self.documents:
            response = requests.post(f"{BASE_URL}/api/v2/documents", json=doc)
            if response.status_code == 200:
                self.doc_ids.append(response.json()["id"])
    
    def test_semantic_search(self):
        """Test semantic search"""
        search_request = {
            "query": "How to build web applications with Python?",
            "top_k": 3,
            "include_content": True
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v2/search",
            json=search_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert len(data["results"]) <= 3
        assert data["query"] == search_request["query"]
    
    def test_batch_search(self):
        """Test batch semantic search"""
        batch_request = {
            "queries": [
                "Python programming",
                "Machine learning basics",
                "Web API development"
            ],
            "top_k": 2
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v2/batch_search",
            json=batch_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 3
        assert all(len(r) <= 2 for r in data["results"])
    
    def test_filtered_search(self):
        """Test search with filters"""
        search_request = {
            "query": "programming concepts",
            "top_k": 5,
            "filters": {"category": "programming"}
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v2/search",
            json=search_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        # Results should be filtered by category


class TestRAGOperations:
    """Test RAG functionality"""
    
    def setup_method(self):
        """Setup test documents for RAG"""
        documents = [
            {
                "content": """Python is an interpreted, high-level programming language. 
                It emphasizes code readability and uses significant indentation. 
                Python supports multiple programming paradigms including procedural, 
                object-oriented, and functional programming.""",
                "metadata": {
                    "title": "Python Overview",
                    "category": "programming"
                }
            },
            {
                "content": """FastAPI is a modern, fast web framework for building APIs. 
                It's based on standard Python type hints and offers automatic API documentation. 
                FastAPI is built on top of Starlette and Pydantic.""",
                "metadata": {
                    "title": "FastAPI Framework",
                    "category": "web"
                }
            }
        ]
        
        # Index documents
        for doc in documents:
            requests.post(f"{BASE_URL}/api/v2/index", json=doc)
    
    def test_rag_query(self):
        """Test RAG query processing"""
        rag_request = {
            "question": "What is Python and what are its main features?",
            "top_k": 3,
            "temperature": 0.5,
            "include_sources": True
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v2/rag/query",
            json=rag_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "response_time_ms" in data
        assert data["response_time_ms"] > 0
    
    def test_rag_with_context(self):
        """Test RAG with additional context"""
        rag_request = {
            "question": "Compare Python and FastAPI",
            "context": {"focus": "web development"},
            "top_k": 5,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v2/rag/query",
            json=rag_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data


class TestIndexingOperations:
    """Test content indexing"""
    
    def test_index_content(self):
        """Test content indexing with auto-chunking"""
        index_request = {
            "content": """This is a long document about artificial intelligence.
            AI has revolutionized many industries including healthcare, finance, and transportation.
            Machine learning algorithms can now recognize patterns in vast amounts of data.
            Deep learning has enabled breakthroughs in computer vision and natural language processing.
            The future of AI looks promising with continued advances in technology.""",
            "metadata": {
                "category": "ai",
                "importance": "high"
            },
            "title": "AI Revolution",
            "auto_chunk": True,
            "chunk_size": 100,
            "chunk_overlap": 20
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v2/index",
            json=index_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "indexed"
        assert "document_id" in data
        assert "chunks_created" in data
        assert data["chunks_created"] > 0
        assert len(data["chunk_ids"]) == data["chunks_created"]


class TestBenchmarkOperations:
    """Test benchmarking functionality"""
    
    def test_benchmark_backends(self):
        """Test backend benchmarking"""
        benchmark_request = {
            "test_queries": [
                "Python programming",
                "Machine learning",
                "Web development"
            ],
            "backends": ["faiss"],  # Test with available backend
            "top_k": 3
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v2/benchmark",
            json=benchmark_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "benchmark_results" in data
        assert "faiss" in data["benchmark_results"]
        
        faiss_results = data["benchmark_results"]["faiss"]
        assert "avg_time_ms" in faiss_results
        assert faiss_results["avg_time_ms"] > 0
        assert len(faiss_results["queries"]) == 3


class TestAdminOperations:
    """Test admin operations"""
    
    def test_optimize_index(self):
        """Test index optimization"""
        response = requests.post(f"{BASE_URL}/api/v2/admin/optimize")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "optimized"
    
    def test_clear_cache(self):
        """Test cache clearing"""
        response = requests.post(f"{BASE_URL}/api/v2/admin/clear_cache")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "cache_cleared"
    
    def test_get_stats(self):
        """Test statistics endpoint"""
        response = requests.get(f"{BASE_URL}/api/v2/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "documents" in data
        assert "vector_search" in data
        assert "embedding_model" in data
        assert "backend" in data


class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_complete_workflow(self):
        """Test complete document lifecycle"""
        # 1. Create document
        document = {
            "title": "Complete Guide to Python",
            "content": "Python is a powerful programming language with many applications.",
            "category": "programming",
            "level": "beginner",
            "tags": ["python", "guide"]
        }
        
        create_response = requests.post(
            f"{BASE_URL}/api/v2/documents",
            json=document
        )
        assert create_response.status_code == 200
        doc_id = create_response.json()["id"]
        
        # 2. Search for document
        search_request = {
            "query": "Python programming guide",
            "top_k": 5
        }
        
        search_response = requests.post(
            f"{BASE_URL}/api/v2/search",
            json=search_request
        )
        assert search_response.status_code == 200
        
        # 3. RAG query
        rag_request = {
            "question": "What is Python used for?",
            "top_k": 3
        }
        
        rag_response = requests.post(
            f"{BASE_URL}/api/v2/rag/query",
            json=rag_request
        )
        assert rag_response.status_code == 200
        
        # 4. Update document
        update_data = {"level": "intermediate"}
        update_response = requests.put(
            f"{BASE_URL}/api/v2/documents/{doc_id}",
            json=update_data
        )
        assert update_response.status_code == 200
        
        # 5. Delete document
        delete_response = requests.delete(
            f"{BASE_URL}/api/v2/documents/{doc_id}"
        )
        assert delete_response.status_code == 200
    
    def test_performance_under_load(self):
        """Test system performance with multiple documents"""
        # Create multiple documents
        n_documents = 50
        doc_ids = []
        
        print(f"\nCreating {n_documents} documents...")
        for i in range(n_documents):
            doc = {
                "title": f"Document {i}",
                "content": f"This is the content for document {i}. It contains various information about topic {i % 10}.",
                "category": f"category_{i % 5}",
                "level": ["beginner", "intermediate", "advanced"][i % 3],
                "tags": [f"tag_{i % 10}", f"tag_{i % 7}"]
            }
            
            response = requests.post(f"{BASE_URL}/api/v2/documents", json=doc)
            if response.status_code == 200:
                doc_ids.append(response.json()["id"])
        
        print(f"Created {len(doc_ids)} documents")
        
        # Perform searches
        n_searches = 10
        search_times = []
        
        print(f"Performing {n_searches} searches...")
        for i in range(n_searches):
            start_time = time.time()
            
            search_request = {
                "query": f"information about topic {i}",
                "top_k": 10
            }
            
            response = requests.post(
                f"{BASE_URL}/api/v2/search",
                json=search_request
            )
            
            search_time = time.time() - start_time
            search_times.append(search_time)
            
            assert response.status_code == 200
        
        avg_search_time = sum(search_times) / len(search_times)
        print(f"Average search time: {avg_search_time * 1000:.2f}ms")
        
        # Performance assertions
        assert avg_search_time < 1.0  # Should complete within 1 second
        
        # Cleanup
        print("Cleaning up...")
        for doc_id in doc_ids:
            requests.delete(f"{BASE_URL}/api/v2/documents/{doc_id}")


def run_api_tests():
    """Run all API tests"""
    print("Starting API v2 Tests...")
    print("=" * 50)
    
    # Make sure server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("❌ Server not responding properly")
            return
    except requests.exceptions.RequestException:
        print("❌ Server not running. Please start with:")
        print("   python src/api/v2_main.py")
        return
    
    print("✅ Server is running")
    
    # Run test classes
    test_classes = [
        TestHealthCheck,
        TestDocumentCRUD,
        TestSearchOperations,
        TestRAGOperations,
        TestIndexingOperations,
        TestBenchmarkOperations,
        TestAdminOperations,
        TestEndToEnd
    ]
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        instance = test_class()
        
        # Run all test methods
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    # Setup if exists
                    if hasattr(instance, "setup_method"):
                        instance.setup_method()
                    
                    # Run test
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✅ {method_name}")
                    
                except AssertionError as e:
                    print(f"  ❌ {method_name}: {e}")
                except Exception as e:
                    print(f"  ❌ {method_name}: Unexpected error: {e}")
    
    print("\n" + "=" * 50)
    print("API v2 Tests Completed!")


if __name__ == "__main__":
    run_api_tests()