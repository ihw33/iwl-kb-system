#!/usr/bin/env python
"""
Initialize the vector database for IWL Knowledge Base
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

def init_chromadb():
    """Initialize ChromaDB"""
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Create persistent client
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create collection for IWL knowledge
        collection_name = "iwl_knowledge"
        
        # Delete if exists (for fresh start)
        try:
            client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except:
            pass
        
        # Create new collection
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "IWL educational content"}
        )
        
        print(f"✅ ChromaDB initialized successfully")
        print(f"   Collection: {collection_name}")
        print(f"   Path: ./chroma_db")
        
        return client, collection
        
    except Exception as e:
        print(f"❌ Failed to initialize ChromaDB: {e}")
        return None, None

def init_pinecone():
    """Initialize Pinecone (if API key is available)"""
    api_key = os.getenv("PINECONE_API_KEY")
    
    if not api_key:
        print("⚠️ Pinecone API key not found, skipping Pinecone initialization")
        return None
    
    try:
        from pinecone import Pinecone, ServerlessSpec
        
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        index_name = "iwl-knowledge"
        
        # Check if index exists
        if index_name not in pc.list_indexes().names():
            # Create index
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI Ada-002 dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print(f"✅ Pinecone index created: {index_name}")
        else:
            print(f"✅ Pinecone index already exists: {index_name}")
        
        return pc.Index(index_name)
        
    except Exception as e:
        print(f"⚠️ Pinecone initialization failed: {e}")
        return None

def create_sample_data():
    """Create sample data structure"""
    sample_content = [
        {
            "id": "sample_001",
            "title": "Python 기초",
            "content": "Python은 배우기 쉽고 강력한 프로그래밍 언어입니다.",
            "metadata": {
                "level": "beginner",
                "language": "ko",
                "category": "programming"
            }
        },
        {
            "id": "sample_002",
            "title": "FastAPI 소개",
            "content": "FastAPI는 현대적이고 빠른 웹 API 프레임워크입니다.",
            "metadata": {
                "level": "intermediate",
                "language": "ko",
                "category": "web"
            }
        }
    ]
    
    return sample_content

def main():
    """Main initialization function"""
    print("=" * 50)
    print("IWL Knowledge Base Initialization")
    print("=" * 50)
    
    # Check environment
    print("\n📋 Checking environment...")
    print(f"   Python version: {sys.version.split()[0]}")
    print(f"   Working directory: {os.getcwd()}")
    
    # Initialize vector databases
    print("\n🗄️ Initializing Vector Databases...")
    
    # ChromaDB (local)
    chroma_client, chroma_collection = init_chromadb()
    
    # Pinecone (cloud)
    pinecone_index = init_pinecone()
    
    # Create sample data
    print("\n📝 Creating sample data...")
    samples = create_sample_data()
    print(f"   Created {len(samples)} sample documents")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ Initialization Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run: python scripts/load_samples.py")
    print("2. Start API: uvicorn src.api.main:app --reload")
    print("3. Test: curl http://localhost:8000/health")

if __name__ == "__main__":
    main()