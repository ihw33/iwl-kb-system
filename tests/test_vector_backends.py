"""
Comprehensive tests for vector search backends
Tests ChromaDB, FAISS, and Annoy implementations
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vectordb.faiss_manager import FAISSManager
from vectordb.annoy_manager import AnnoyManager
from vectordb.chroma_manager import ChromaManager
from vectordb.vector_search_manager import VectorSearchManager


class TestFAISSManager:
    """Test FAISS backend functionality"""
    
    @pytest.fixture
    def faiss_manager(self):
        """Create FAISS manager with temporary directory"""
        temp_dir = tempfile.mkdtemp()
        manager = FAISSManager(
            dimension=128,
            index_type="Flat",
            persist_directory=temp_dir
        )
        yield manager
        shutil.rmtree(temp_dir)
    
    def test_add_vectors(self, faiss_manager):
        """Test adding vectors to FAISS index"""
        # Create test data
        vectors = np.random.randn(5, 128).astype(np.float32)
        ids = [f"doc_{i}" for i in range(5)]
        metadata = [{"title": f"Document {i}"} for i in range(5)]
        
        # Add vectors
        indices = faiss_manager.add_vectors(vectors, ids, metadata)
        
        assert len(indices) == 5
        assert faiss_manager.index.ntotal == 5
    
    def test_search(self, faiss_manager):
        """Test vector search in FAISS"""
        # Add test vectors
        vectors = np.random.randn(10, 128).astype(np.float32)
        ids = [f"doc_{i}" for i in range(10)]
        metadata = [{"category": "test" if i < 5 else "other"} for i in range(10)]
        
        faiss_manager.add_vectors(vectors, ids, metadata)
        
        # Search
        query = np.random.randn(1, 128).astype(np.float32)
        indices, distances = faiss_manager.search(query, k=3)
        
        assert len(indices[0]) == 3
        assert len(distances[0]) == 3
    
    def test_search_with_filters(self, faiss_manager):
        """Test filtered search"""
        # Add test vectors
        vectors = np.random.randn(10, 128).astype(np.float32)
        ids = [f"doc_{i}" for i in range(10)]
        metadata = [{"category": "A" if i < 5 else "B"} for i in range(10)]
        
        faiss_manager.add_vectors(vectors, ids, metadata)
        
        # Search with filter
        query = np.random.randn(1, 128).astype(np.float32)
        indices, distances = faiss_manager.search(
            query, k=5, filters={"category": "A"}
        )
        
        # Should only return category A documents
        assert all(idx < 5 for idx in indices[0] if idx >= 0)
    
    def test_persistence(self, faiss_manager):
        """Test index persistence"""
        # Add vectors
        vectors = np.random.randn(5, 128).astype(np.float32)
        ids = [f"doc_{i}" for i in range(5)]
        metadata = [{"title": f"Document {i}"} for i in range(5)]
        
        faiss_manager.add_vectors(vectors, ids, metadata)
        
        # Save
        faiss_manager._save_index()
        
        # Create new manager with same directory
        new_manager = FAISSManager(
            dimension=128,
            persist_directory=faiss_manager.persist_directory
        )
        
        assert new_manager.index.ntotal == 5
        assert "doc_0" in new_manager.id_to_idx


class TestAnnoyManager:
    """Test Annoy backend functionality"""
    
    @pytest.fixture
    def annoy_manager(self):
        """Create Annoy manager with temporary directory"""
        temp_dir = tempfile.mkdtemp()
        manager = AnnoyManager(
            dimension=128,
            metric="angular",
            n_trees=10,
            persist_directory=temp_dir
        )
        yield manager
        shutil.rmtree(temp_dir)
    
    def test_add_and_build(self, annoy_manager):
        """Test adding vectors and building index"""
        # Create test data
        vectors = np.random.randn(5, 128).astype(np.float32)
        ids = [f"doc_{i}" for i in range(5)]
        metadata = [{"title": f"Document {i}"} for i in range(5)]
        
        # Add vectors
        indices = annoy_manager.add_vectors(vectors, ids, metadata)
        assert len(indices) == 5
        
        # Build index
        annoy_manager.build_index()
        assert annoy_manager.is_built
    
    def test_search(self, annoy_manager):
        """Test vector search in Annoy"""
        # Add test vectors
        vectors = np.random.randn(10, 128).astype(np.float32)
        ids = [f"doc_{i}" for i in range(10)]
        metadata = [{"category": "test"} for i in range(10)]
        
        annoy_manager.add_vectors(vectors, ids, metadata)
        annoy_manager.build_index()
        
        # Search
        query = np.random.randn(1, 128).astype(np.float32)
        indices, distances = annoy_manager.search(query, k=3)
        
        assert len(indices[0]) == 3
        assert len(distances[0]) == 3
    
    def test_find_similar(self, annoy_manager):
        """Test finding similar documents"""
        # Add test vectors
        vectors = np.eye(10, 128).astype(np.float32)  # Orthogonal vectors
        ids = [f"doc_{i}" for i in range(10)]
        metadata = [{"title": f"Document {i}"} for i in range(10)]
        
        annoy_manager.add_vectors(vectors, ids, metadata)
        annoy_manager.build_index()
        
        # Find similar to first document
        similar_ids, distances = annoy_manager.find_similar("doc_0", k=3)
        
        assert len(similar_ids) <= 3
        assert "doc_0" not in similar_ids  # Should exclude self by default
    
    def test_persistence(self, annoy_manager):
        """Test index persistence"""
        # Add vectors and build
        vectors = np.random.randn(5, 128).astype(np.float32)
        ids = [f"doc_{i}" for i in range(5)]
        metadata = [{"title": f"Document {i}"} for i in range(5)]
        
        annoy_manager.add_vectors(vectors, ids, metadata)
        annoy_manager.build_index()
        
        # Create new manager with same directory
        new_manager = AnnoyManager(
            dimension=128,
            persist_directory=annoy_manager.persist_directory
        )
        
        assert new_manager.is_built
        assert "doc_0" in new_manager.id_to_idx


class TestVectorSearchManager:
    """Test unified vector search manager"""
    
    @pytest.fixture(params=["faiss", "annoy"])
    def vector_manager(self, request):
        """Create vector manager with different backends"""
        temp_dir = tempfile.mkdtemp()
        manager = VectorSearchManager(
            backend=request.param,
            dimension=128,
            persist_directory=temp_dir
        )
        yield manager
        shutil.rmtree(temp_dir)
    
    def test_add_documents(self, vector_manager):
        """Test adding documents"""
        # Create test data
        documents = ["Document 1", "Document 2", "Document 3"]
        embeddings = np.random.randn(3, 128)
        ids = ["doc1", "doc2", "doc3"]
        metadata = [{"title": f"Doc {i}"} for i in range(3)]
        
        # Add documents
        result_ids = vector_manager.add_documents(
            documents, embeddings, ids, metadata
        )
        
        assert result_ids == ids
    
    def test_search(self, vector_manager):
        """Test document search"""
        # Add documents
        documents = [f"Document {i}" for i in range(10)]
        embeddings = np.random.randn(10, 128)
        ids = [f"doc{i}" for i in range(10)]
        metadata = [{"index": i} for i in range(10)]
        
        vector_manager.add_documents(documents, embeddings, ids, metadata)
        
        # Search
        query_embedding = np.random.randn(128)
        results = vector_manager.search(query_embedding, k=5)
        
        assert len(results) <= 5
        assert all("id" in r for r in results)
        assert all("score" in r for r in results)
    
    def test_batch_search(self, vector_manager):
        """Test batch search"""
        # Add documents
        documents = [f"Document {i}" for i in range(10)]
        embeddings = np.random.randn(10, 128)
        ids = [f"doc{i}" for i in range(10)]
        metadata = [{"index": i} for i in range(10)]
        
        vector_manager.add_documents(documents, embeddings, ids, metadata)
        
        # Batch search
        query_embeddings = np.random.randn(3, 128)
        results = vector_manager.batch_search(query_embeddings, k=3)
        
        assert len(results) == 3
        assert all(len(r) <= 3 for r in results)
    
    def test_update_document(self, vector_manager):
        """Test document update"""
        # Add initial document
        documents = ["Initial content"]
        embeddings = np.random.randn(1, 128)
        ids = ["doc1"]
        metadata = [{"version": 1}]
        
        vector_manager.add_documents(documents, embeddings, ids, metadata)
        
        # Update document
        new_embedding = np.random.randn(128)
        new_metadata = {"version": 2}
        
        vector_manager.update_document(
            "doc1", embedding=new_embedding, metadata=new_metadata
        )
        
        # Verify update
        doc = vector_manager.get_document("doc1")
        if doc:
            assert doc.get("metadata", {}).get("version") == 2
    
    def test_delete_documents(self, vector_manager):
        """Test document deletion"""
        # Add documents
        documents = ["Doc1", "Doc2", "Doc3"]
        embeddings = np.random.randn(3, 128)
        ids = ["doc1", "doc2", "doc3"]
        metadata = [{"index": i} for i in range(3)]
        
        vector_manager.add_documents(documents, embeddings, ids, metadata)
        
        # Delete one document
        vector_manager.delete_documents(["doc2"])
        
        # Verify deletion
        doc = vector_manager.get_document("doc2")
        assert doc is None or "deleted" in doc.get("metadata", {})
    
    def test_cache_functionality(self, vector_manager):
        """Test search caching"""
        # Add documents
        documents = [f"Document {i}" for i in range(5)]
        embeddings = np.random.randn(5, 128)
        ids = [f"doc{i}" for i in range(5)]
        metadata = [{"index": i} for i in range(5)]
        
        vector_manager.add_documents(documents, embeddings, ids, metadata)
        
        # First search (not cached)
        query = np.random.randn(128)
        start = time.time()
        results1 = vector_manager.search(query, k=3)
        time1 = time.time() - start
        
        # Second search (should be cached)
        start = time.time()
        results2 = vector_manager.search(query, k=3)
        time2 = time.time() - start
        
        # Cache should make second search faster
        assert results1 == results2
        # Note: Cache lookup should be faster, but timing can be unreliable in tests
    
    def test_benchmark(self, vector_manager):
        """Test benchmark functionality"""
        # Add documents
        documents = [f"Document {i}" for i in range(10)]
        embeddings = np.random.randn(10, 128)
        ids = [f"doc{i}" for i in range(10)]
        metadata = [{"index": i} for i in range(10)]
        
        vector_manager.add_documents(documents, embeddings, ids, metadata)
        
        # Run benchmark
        query = np.random.randn(128)
        bench_result = vector_manager.benchmark_search(query, k=5)
        
        assert "search_time_ms" in bench_result
        assert "num_results" in bench_result
        assert "backend" in bench_result
        assert bench_result["search_time_ms"] >= 0
    
    def test_stats(self, vector_manager):
        """Test statistics retrieval"""
        # Add some documents
        documents = ["Doc1", "Doc2"]
        embeddings = np.random.randn(2, 128)
        ids = ["doc1", "doc2"]
        metadata = [{"index": i} for i in range(2)]
        
        vector_manager.add_documents(documents, embeddings, ids, metadata)
        
        # Get stats
        stats = vector_manager.get_stats()
        
        assert "backend" in stats
        assert "dimension" in stats
        assert stats["dimension"] == 128


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_faiss_vs_annoy_accuracy(self):
        """Compare search accuracy between FAISS and Annoy"""
        dimension = 128
        n_docs = 100
        n_queries = 10
        k = 5
        
        # Generate test data
        np.random.seed(42)
        documents = [f"Document {i}" for i in range(n_docs)]
        embeddings = np.random.randn(n_docs, dimension)
        ids = [f"doc{i}" for i in range(n_docs)]
        metadata = [{"index": i} for i in range(n_docs)]
        
        # Create both managers
        temp_dir_faiss = tempfile.mkdtemp()
        temp_dir_annoy = tempfile.mkdtemp()
        
        faiss_manager = VectorSearchManager(
            backend="faiss",
            dimension=dimension,
            persist_directory=temp_dir_faiss
        )
        
        annoy_manager = VectorSearchManager(
            backend="annoy",
            dimension=dimension,
            persist_directory=temp_dir_annoy
        )
        
        # Add same documents to both
        faiss_manager.add_documents(documents, embeddings, ids, metadata)
        annoy_manager.add_documents(documents, embeddings, ids, metadata)
        
        # Compare search results
        query_embeddings = np.random.randn(n_queries, dimension)
        
        for query in query_embeddings:
            faiss_results = faiss_manager.search(query, k=k)
            annoy_results = annoy_manager.search(query, k=k)
            
            # Check that both return results
            assert len(faiss_results) > 0
            assert len(annoy_results) > 0
            
            # Check overlap (Annoy is approximate, so may not be identical)
            faiss_ids = {r["id"] for r in faiss_results[:3]}
            annoy_ids = {r["id"] for r in annoy_results[:3]}
            overlap = len(faiss_ids & annoy_ids)
            
            # Should have some overlap in top results
            assert overlap > 0
        
        # Cleanup
        shutil.rmtree(temp_dir_faiss)
        shutil.rmtree(temp_dir_annoy)
    
    def test_performance_comparison(self):
        """Compare performance between backends"""
        dimension = 128
        n_docs = 1000
        n_queries = 10
        k = 10
        
        # Generate test data
        documents = [f"Document {i}" for i in range(n_docs)]
        embeddings = np.random.randn(n_docs, dimension)
        ids = [f"doc{i}" for i in range(n_docs)]
        metadata = [{"index": i} for i in range(n_docs)]
        
        results = {}
        
        for backend in ["faiss", "annoy"]:
            temp_dir = tempfile.mkdtemp()
            manager = VectorSearchManager(
                backend=backend,
                dimension=dimension,
                persist_directory=temp_dir
            )
            
            # Measure indexing time
            start = time.time()
            manager.add_documents(documents, embeddings, ids, metadata)
            index_time = time.time() - start
            
            # Measure search time
            query_embeddings = np.random.randn(n_queries, dimension)
            start = time.time()
            for query in query_embeddings:
                manager.search(query, k=k)
            search_time = time.time() - start
            
            results[backend] = {
                "index_time": index_time,
                "search_time": search_time,
                "avg_search_time": search_time / n_queries
            }
            
            shutil.rmtree(temp_dir)
        
        # Print results for debugging
        print("\nPerformance Comparison:")
        for backend, metrics in results.items():
            print(f"{backend}:")
            print(f"  Index time: {metrics['index_time']:.3f}s")
            print(f"  Total search time: {metrics['search_time']:.3f}s")
            print(f"  Avg search time: {metrics['avg_search_time']*1000:.3f}ms")
        
        # Both should complete reasonably fast
        assert all(m["index_time"] < 10.0 for m in results.values())
        assert all(m["avg_search_time"] < 1.0 for m in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])