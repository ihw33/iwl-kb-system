#!/usr/bin/env python
"""
Test suite for CRUD API endpoints
Tests all basic CRUD operations
"""

import pytest
import requests
import json
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:8002"

class TestCRUDAPI:
    """Test class for CRUD operations"""
    
    @classmethod
    def setup_class(cls):
        """Setup test data"""
        cls.test_document = {
            "title": "Python Programming Guide",
            "content": "Python is a versatile programming language used for web development, data science, and automation.",
            "category": "programming",
            "metadata": {
                "level": "beginner",
                "tags": ["python", "programming", "tutorial"]
            }
        }
        cls.created_doc_id = None
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "documents_count" in data
    
    def test_create_document(self):
        """Test POST - Create document"""
        response = requests.post(
            f"{BASE_URL}/api/documents",
            json=self.test_document
        )
        assert response.status_code == 201
        
        data = response.json()
        assert "id" in data
        assert data["title"] == self.test_document["title"]
        assert data["content"] == self.test_document["content"]
        assert data["category"] == self.test_document["category"]
        assert "created_at" in data
        assert "updated_at" in data
        
        # Store ID for other tests
        TestCRUDAPI.created_doc_id = data["id"]
        return data["id"]
    
    def test_get_document(self):
        """Test GET - Read single document"""
        # First create a document
        doc_id = self.test_create_document()
        
        # Now get it
        response = requests.get(f"{BASE_URL}/api/documents/{doc_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == doc_id
        assert data["title"] == self.test_document["title"]
    
    def test_get_nonexistent_document(self):
        """Test GET with non-existent ID"""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = requests.get(f"{BASE_URL}/api/documents/{fake_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_update_document(self):
        """Test PUT - Update document"""
        # Create a document first
        doc_id = self.test_create_document()
        
        # Update it
        update_data = {
            "title": "Updated Python Guide",
            "content": "Updated content about Python programming"
        }
        
        response = requests.put(
            f"{BASE_URL}/api/documents/{doc_id}",
            json=update_data
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["title"] == update_data["title"]
        assert data["content"] == update_data["content"]
        assert data["category"] == self.test_document["category"]  # Unchanged
        assert data["updated_at"] > data["created_at"]
    
    def test_partial_update(self):
        """Test PUT with partial update"""
        doc_id = self.test_create_document()
        
        # Update only title
        update_data = {"title": "Partially Updated Title"}
        
        response = requests.put(
            f"{BASE_URL}/api/documents/{doc_id}",
            json=update_data
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["title"] == update_data["title"]
        assert data["content"] == self.test_document["content"]  # Unchanged
    
    def test_delete_document(self):
        """Test DELETE - Delete document"""
        # Create a document
        doc_id = self.test_create_document()
        
        # Delete it
        response = requests.delete(f"{BASE_URL}/api/documents/{doc_id}")
        assert response.status_code == 204
        
        # Verify it's deleted
        response = requests.get(f"{BASE_URL}/api/documents/{doc_id}")
        assert response.status_code == 404
    
    def test_delete_nonexistent_document(self):
        """Test DELETE with non-existent ID"""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = requests.delete(f"{BASE_URL}/api/documents/{fake_id}")
        assert response.status_code == 404

def test_crud_workflow():
    """Test complete CRUD workflow"""
    # 1. Create
    create_response = requests.post(
        f"{BASE_URL}/api/documents",
        json={
            "title": "Complete Workflow Test",
            "content": "Testing the complete CRUD workflow",
            "category": "test",
            "metadata": {"test": True}
        }
    )
    assert create_response.status_code == 201
    doc_id = create_response.json()["id"]
    
    # 2. Read
    read_response = requests.get(f"{BASE_URL}/api/documents/{doc_id}")
    assert read_response.status_code == 200
    assert read_response.json()["title"] == "Complete Workflow Test"
    
    # 3. Update
    update_response = requests.put(
        f"{BASE_URL}/api/documents/{doc_id}",
        json={"title": "Updated Workflow Test"}
    )
    assert update_response.status_code == 200
    assert update_response.json()["title"] == "Updated Workflow Test"
    
    # 4. Delete
    delete_response = requests.delete(f"{BASE_URL}/api/documents/{doc_id}")
    assert delete_response.status_code == 204
    
    # 5. Verify deletion
    verify_response = requests.get(f"{BASE_URL}/api/documents/{doc_id}")
    assert verify_response.status_code == 404

def test_validation():
    """Test input validation"""
    # Test empty title
    response = requests.post(
        f"{BASE_URL}/api/documents",
        json={
            "title": "",  # Empty title
            "content": "Some content",
            "category": "test"
        }
    )
    assert response.status_code == 422  # Validation error
    
    # Test missing required field
    response = requests.post(
        f"{BASE_URL}/api/documents",
        json={
            "title": "Test",
            # Missing content
            "category": "test"
        }
    )
    assert response.status_code == 422

if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("IWL Knowledge Base CRUD API Test Suite")
    print("=" * 50)
    
    # Run basic tests
    test_suite = TestCRUDAPI()
    test_suite.setup_class()
    
    try:
        print("\n1. Testing Health Check...")
        test_suite.test_health_check()
        print("   ✅ Health check passed")
        
        print("\n2. Testing Create Document...")
        doc_id = test_suite.test_create_document()
        print(f"   ✅ Document created: {doc_id}")
        
        print("\n3. Testing Get Document...")
        test_suite.test_get_document()
        print("   ✅ Document retrieved")
        
        print("\n4. Testing Update Document...")
        test_suite.test_update_document()
        print("   ✅ Document updated")
        
        print("\n5. Testing Delete Document...")
        test_suite.test_delete_document()
        print("   ✅ Document deleted")
        
        print("\n6. Testing Complete Workflow...")
        test_crud_workflow()
        print("   ✅ Complete workflow passed")
        
        print("\n7. Testing Input Validation...")
        test_validation()
        print("   ✅ Validation tests passed")
        
        print("\n" + "=" * 50)
        print("✅ All CRUD API tests passed successfully!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)