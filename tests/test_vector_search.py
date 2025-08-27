#!/usr/bin/env python
"""
Test suite for Vector Search functionality
Tests FAISS, Annoy, and ChromaDB integrations
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectordb.vector_manager import HybridVectorManager
from src.embeddings.embedding_pipeline import EmbeddingPipeline

class TestVectorSearch:
    """Test class for vector search operations"""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.test_docs = [
            {
                "id": "doc1",
                "content": "Python is a high-level programming language.",
                "metadata": {"category": "programming", "level": "beginner"}
            },
            {
                "id": "doc2", 
                "content": "Machine learning is a subset of artificial intelligence.",
                "metadata": {"category": "ai", "level": "intermediate"}
            },
            {
                "id": "doc3",
                "content": "FastAPI is a modern web framework for building APIs.",
                "metadata": {"category": "web", "level": "intermediate"}
            }
        ]
        
        cls.embedding_pipeline = EmbeddingPipeline(model_type="sentence-transformer")
    
    def test_faiss_vector_manager(self):
        """Test FAISS-based vector manager"""
        manager = HybridVectorManager(
            persist_directory="./test_vector_db/faiss",
            index_type="faiss",
            embedding_dim=384
        )
        
        # Generate embeddings
        contents = [doc["content"] for doc in self.test_docs]
        embeddings = self.embedding_pipeline.embed_texts(contents)
        
        # Add documents
        ids = manager.add_documents(
            documents=contents,
            embeddings=embeddings,
            metadatas=[doc["metadata"] for doc in self.test_docs],
            ids=[doc["id"] for doc in self.test_docs]
        )
        
        assert len(ids) == 3
        
        # Test search
        query = "programming language"
        query_embedding = self.embedding_pipeline.embed_query(query)
        results = manager.search(query_embedding, k=2)
        
        assert len(results) <= 2
        assert results[0]["id"] in ["doc1", "doc3"]
        assert "score" in results[0]
        
        # Test statistics
        stats = manager.get_statistics()
        assert stats["faiss_vectors"] == 3
        assert stats["index_type"] == "faiss"
        
        print("✅ FAISS vector manager test passed")
    
    def test_annoy_vector_manager(self):
        """Test Annoy-based vector manager"""
        manager = HybridVectorManager(
            persist_directory="./test_vector_db/annoy",
            index_type="annoy",
            embedding_dim=384
        )
        
        # Generate embeddings
        contents = [doc["content"] for doc in self.test_docs]
        embeddings = self.embedding_pipeline.embed_texts(contents)
        
        # Add documents
        ids = manager.add_documents(
            documents=contents,
            embeddings=embeddings,
            metadatas=[doc["metadata"] for doc in self.test_docs],
            ids=[doc["id"] for doc in self.test_docs]
        )
        
        assert len(ids) == 3
        
        # Test search
        query = "artificial intelligence"
        query_embedding = self.embedding_pipeline.embed_query(query)
        results = manager.search(query_embedding, k=2)
        
        assert len(results) <= 2
        assert "score" in results[0]
        
        # Test statistics
        stats = manager.get_statistics()
        assert stats["annoy_items"] == 3
        assert stats["index_type"] == "annoy"
        
        print("✅ Annoy vector manager test passed")
    
    def test_chromadb_fallback(self):
        """Test ChromaDB fallback mode"""
        manager = HybridVectorManager(
            persist_directory="./test_vector_db/chromadb",
            index_type="chromadb",
            embedding_dim=384
        )
        
        # Generate embeddings
        contents = [doc["content"] for doc in self.test_docs]
        embeddings = self.embedding_pipeline.embed_texts(contents)
        
        # Add documents
        ids = manager.add_documents(
            documents=contents,
            embeddings=embeddings,
            metadatas=[doc["metadata"] for doc in self.test_docs],
            ids=[doc["id"] for doc in self.test_docs]
        )
        
        assert len(ids) == 3
        
        # Test search
        query = "web framework"
        query_embedding = self.embedding_pipeline.embed_query(query)
        results = manager.search(query_embedding, k=2)
        
        assert len(results) <= 2
        assert "score" in results[0]
        
        print("✅ ChromaDB fallback test passed")
    
    def test_filtered_search(self):
        """Test search with metadata filters"""
        manager = HybridVectorManager(
            persist_directory="./test_vector_db/filtered",
            index_type="faiss",
            embedding_dim=384
        )
        
        # Generate embeddings
        contents = [doc["content"] for doc in self.test_docs]
        embeddings = self.embedding_pipeline.embed_texts(contents)
        
        # Add documents
        manager.add_documents(
            documents=contents,
            embeddings=embeddings,
            metadatas=[doc["metadata"] for doc in self.test_docs],
            ids=[doc["id"] for doc in self.test_docs]
        )
        
        # Search with filter
        query = "programming"
        query_embedding = self.embedding_pipeline.embed_query(query)
        results = manager.search(
            query_embedding, 
            k=3,
            filters={"level": "intermediate"}
        )
        
        # Should only return intermediate level documents
        for result in results:
            assert result["metadata"]["level"] == "intermediate"
        
        print("✅ Filtered search test passed")
    
    def test_performance_comparison(self):
        """Compare performance of different index types"""
        import time
        
        # Generate test data
        n_docs = 100
        test_contents = [f"Document {i}: " + " ".join(np.random.choice(
            ["python", "programming", "machine", "learning", "data", "science", 
             "api", "web", "framework", "database"], 10))
            for i in range(n_docs)]
        test_embeddings = [np.random.randn(384).tolist() for _ in range(n_docs)]
        
        results = {}
        
        for index_type in ["faiss", "annoy", "chromadb"]:
            manager = HybridVectorManager(
                persist_directory=f"./test_vector_db/perf_{index_type}",
                index_type=index_type,
                embedding_dim=384
            )
            
            # Measure indexing time
            start_time = time.time()
            manager.add_documents(
                documents=test_contents,
                embeddings=test_embeddings
            )
            index_time = time.time() - start_time
            
            # Measure search time
            query_embedding = np.random.randn(384).tolist()
            start_time = time.time()
            for _ in range(10):
                manager.search(query_embedding, k=5)
            search_time = (time.time() - start_time) / 10
            
            results[index_type] = {
                "index_time": index_time,
                "search_time": search_time
            }
            
            print(f"  {index_type.upper()}:")
            print(f"    Index time: {index_time:.3f}s")
            print(f"    Avg search time: {search_time*1000:.2f}ms")
        
        print("✅ Performance comparison completed")
        return results

def run_all_tests():
    """Run all vector search tests"""
    print("=" * 50)
    print("Vector Search Test Suite")
    print("=" * 50)
    
    test_suite = TestVectorSearch()
    test_suite.setup_class()
    
    try:
        print("\n1. Testing FAISS Vector Manager...")
        test_suite.test_faiss_vector_manager()
        
        print("\n2. Testing Annoy Vector Manager...")
        test_suite.test_annoy_vector_manager()
        
        print("\n3. Testing ChromaDB Fallback...")
        test_suite.test_chromadb_fallback()
        
        print("\n4. Testing Filtered Search...")
        test_suite.test_filtered_search()
        
        print("\n5. Performance Comparison...")
        test_suite.test_performance_comparison()
        
        print("\n" + "=" * 50)
        print("✅ All vector search tests passed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()