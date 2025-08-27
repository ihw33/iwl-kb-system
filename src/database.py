"""
Database connection and management
Supports Redis and PostgreSQL backends
"""

import os
import json
import redis
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
from dotenv import load_dotenv

load_dotenv()

# Database backend selection
DB_BACKEND = os.getenv("DB_BACKEND", "redis")  # redis or postgresql


class RedisDatabase:
    """Redis database manager for documents"""
    
    def __init__(self):
        """Initialize Redis connection"""
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_db = int(os.getenv("REDIS_DB", 0))
        self.redis_password = os.getenv("REDIS_PASSWORD")
        
        # Connect to Redis
        self.client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            decode_responses=True
        )
        
        # Test connection
        try:
            self.client.ping()
            print(f"✅ Connected to Redis at {self.redis_host}:{self.redis_port}")
        except redis.ConnectionError:
            print(f"⚠️ Redis connection failed. Using in-memory fallback.")
            self._use_memory_fallback = True
            self._memory_store = {}
        else:
            self._use_memory_fallback = False
    
    def _get_key(self, doc_id: str) -> str:
        """Get Redis key for document"""
        return f"doc:{doc_id}"
    
    def create_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Create a new document"""
        if self._use_memory_fallback:
            self._memory_store[doc_id] = document
            return True
        
        key = self._get_key(doc_id)
        # Store as JSON
        doc_json = json.dumps(document, default=str)
        result = self.client.set(key, doc_json)
        
        # Add to index
        self.client.sadd("doc:index", doc_id)
        
        # Update category index
        if "category" in document:
            self.client.sadd(f"category:{document['category']}", doc_id)
        
        return result
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        if self._use_memory_fallback:
            return self._memory_store.get(doc_id)
        
        key = self._get_key(doc_id)
        doc_json = self.client.get(key)
        
        if doc_json:
            return json.loads(doc_json)
        return None
    
    def update_document(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing document"""
        doc = self.get_document(doc_id)
        if not doc:
            return False
        
        # Update document
        doc.update(updates)
        doc["updated_at"] = datetime.utcnow().isoformat()
        
        if self._use_memory_fallback:
            self._memory_store[doc_id] = doc
            return True
        
        # Save back to Redis
        key = self._get_key(doc_id)
        doc_json = json.dumps(doc, default=str)
        return self.client.set(key, doc_json)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        if self._use_memory_fallback:
            if doc_id in self._memory_store:
                del self._memory_store[doc_id]
                return True
            return False
        
        # Get document for cleanup
        doc = self.get_document(doc_id)
        
        # Delete from Redis
        key = self._get_key(doc_id)
        result = self.client.delete(key)
        
        # Remove from indices
        self.client.srem("doc:index", doc_id)
        
        if doc and "category" in doc:
            self.client.srem(f"category:{doc['category']}", doc_id)
        
        return result > 0
    
    def list_documents(
        self,
        skip: int = 0,
        limit: int = 10,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List documents with pagination"""
        if self._use_memory_fallback:
            docs = list(self._memory_store.values())
            if category:
                docs = [d for d in docs if d.get("category") == category]
            return docs[skip:skip+limit]
        
        # Get document IDs
        if category:
            doc_ids = self.client.smembers(f"category:{category}")
        else:
            doc_ids = self.client.smembers("doc:index")
        
        # Convert to list and paginate
        doc_ids = list(doc_ids)[skip:skip+limit]
        
        # Fetch documents
        documents = []
        for doc_id in doc_ids:
            doc = self.get_document(doc_id)
            if doc:
                documents.append(doc)
        
        # Sort by created_at (newest first)
        documents.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return documents
    
    def count_documents(self, category: Optional[str] = None) -> int:
        """Count total documents"""
        if self._use_memory_fallback:
            if category:
                return len([d for d in self._memory_store.values() 
                          if d.get("category") == category])
            return len(self._memory_store)
        
        if category:
            return self.client.scard(f"category:{category}")
        return self.client.scard("doc:index")
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Simple text search in documents"""
        documents = []
        
        if self._use_memory_fallback:
            for doc in self._memory_store.values():
                if query.lower() in str(doc).lower():
                    documents.append(doc)
                    if len(documents) >= limit:
                        break
        else:
            # Get all document IDs
            doc_ids = self.client.smembers("doc:index")
            
            for doc_id in doc_ids:
                doc = self.get_document(doc_id)
                if doc and query.lower() in str(doc).lower():
                    documents.append(doc)
                    if len(documents) >= limit:
                        break
        
        return documents
    
    def clear_all(self) -> bool:
        """Clear all documents (use with caution!)"""
        if self._use_memory_fallback:
            self._memory_store.clear()
            return True
        
        # Get all document IDs
        doc_ids = self.client.smembers("doc:index")
        
        # Delete all documents
        for doc_id in doc_ids:
            self.delete_document(doc_id)
        
        # Clear indices
        self.client.delete("doc:index")
        
        # Clear category indices
        for key in self.client.keys("category:*"):
            self.client.delete(key)
        
        return True


class PostgreSQLDatabase:
    """PostgreSQL database manager using SQLAlchemy"""
    
    def __init__(self):
        """Initialize PostgreSQL connection"""
        from sqlalchemy import create_engine, Column, String, DateTime, JSON, text
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.orm import sessionmaker
        
        # Database configuration
        db_host = os.getenv("POSTGRES_HOST", "localhost")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        db_name = os.getenv("POSTGRES_DB", "iwl_kb")
        db_user = os.getenv("POSTGRES_USER", "postgres")
        db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        
        # Create connection string
        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        # Create engine
        try:
            self.engine = create_engine(db_url, echo=False)
            self.Session = sessionmaker(bind=self.engine)
            self.Base = declarative_base()
            
            # Define Document model
            class Document(self.Base):
                __tablename__ = 'documents'
                
                id = Column(String, primary_key=True)
                title = Column(String)
                content = Column(String)
                category = Column(String, index=True)
                level = Column(String)
                metadata = Column(JSON)
                created_at = Column(DateTime)
                updated_at = Column(DateTime)
            
            self.Document = Document
            
            # Create tables
            self.Base.metadata.create_all(self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print(f"✅ Connected to PostgreSQL at {db_host}:{db_port}/{db_name}")
            self._connected = True
            
        except Exception as e:
            print(f"⚠️ PostgreSQL connection failed: {e}")
            print("Using in-memory fallback")
            self._connected = False
            self._memory_store = {}
    
    def create_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Create a new document"""
        if not self._connected:
            self._memory_store[doc_id] = document
            return True
        
        session = self.Session()
        try:
            db_doc = self.Document(
                id=doc_id,
                title=document.get("title"),
                content=document.get("content"),
                category=document.get("category"),
                level=document.get("level"),
                metadata=document.get("metadata", {}),
                created_at=document.get("created_at"),
                updated_at=document.get("updated_at")
            )
            session.add(db_doc)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error creating document: {e}")
            return False
        finally:
            session.close()
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        if not self._connected:
            return self._memory_store.get(doc_id)
        
        session = self.Session()
        try:
            doc = session.query(self.Document).filter_by(id=doc_id).first()
            if doc:
                return {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "category": doc.category,
                    "level": doc.level,
                    "metadata": doc.metadata or {},
                    "created_at": doc.created_at,
                    "updated_at": doc.updated_at
                }
            return None
        finally:
            session.close()
    
    def update_document(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing document"""
        if not self._connected:
            if doc_id in self._memory_store:
                self._memory_store[doc_id].update(updates)
                self._memory_store[doc_id]["updated_at"] = datetime.utcnow()
                return True
            return False
        
        session = self.Session()
        try:
            doc = session.query(self.Document).filter_by(id=doc_id).first()
            if doc:
                for key, value in updates.items():
                    if hasattr(doc, key):
                        setattr(doc, key, value)
                doc.updated_at = datetime.utcnow()
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error updating document: {e}")
            return False
        finally:
            session.close()
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        if not self._connected:
            if doc_id in self._memory_store:
                del self._memory_store[doc_id]
                return True
            return False
        
        session = self.Session()
        try:
            doc = session.query(self.Document).filter_by(id=doc_id).first()
            if doc:
                session.delete(doc)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error deleting document: {e}")
            return False
        finally:
            session.close()
    
    def list_documents(
        self,
        skip: int = 0,
        limit: int = 10,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List documents with pagination"""
        if not self._connected:
            docs = list(self._memory_store.values())
            if category:
                docs = [d for d in docs if d.get("category") == category]
            return docs[skip:skip+limit]
        
        session = self.Session()
        try:
            query = session.query(self.Document)
            
            if category:
                query = query.filter_by(category=category)
            
            query = query.order_by(self.Document.created_at.desc())
            query = query.offset(skip).limit(limit)
            
            documents = []
            for doc in query:
                documents.append({
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "category": doc.category,
                    "level": doc.level,
                    "metadata": doc.metadata or {},
                    "created_at": doc.created_at,
                    "updated_at": doc.updated_at
                })
            
            return documents
        finally:
            session.close()
    
    def count_documents(self, category: Optional[str] = None) -> int:
        """Count total documents"""
        if not self._connected:
            if category:
                return len([d for d in self._memory_store.values() 
                          if d.get("category") == category])
            return len(self._memory_store)
        
        session = self.Session()
        try:
            query = session.query(self.Document)
            if category:
                query = query.filter_by(category=category)
            return query.count()
        finally:
            session.close()
    
    def clear_all(self) -> bool:
        """Clear all documents (use with caution!)"""
        if not self._connected:
            self._memory_store.clear()
            return True
        
        session = self.Session()
        try:
            session.query(self.Document).delete()
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error clearing documents: {e}")
            return False
        finally:
            session.close()


# Factory function to get database instance
def get_database():
    """Get database instance based on configuration"""
    if DB_BACKEND == "postgresql":
        return PostgreSQLDatabase()
    else:
        # Default to Redis
        return RedisDatabase()


# Singleton instance
_db_instance = None


def get_db():
    """Get singleton database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = get_database()
    return _db_instance