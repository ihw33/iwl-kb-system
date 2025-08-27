"""
RAG (Retrieval-Augmented Generation) Pipeline for IWL Knowledge Base
Combines retrieval with LLM generation
"""

import os
from typing import List, Dict, Any, Optional
import openai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()

class RAGPipeline:
    """Manage RAG operations for the knowledge base"""
    
    def __init__(
        self,
        chroma_manager,
        embedding_pipeline,
        llm_provider: str = "openai",  # "openai", "anthropic", "gemini"
        model_name: Optional[str] = None
    ):
        """Initialize RAG pipeline"""
        self.chroma_manager = chroma_manager
        self.embedding_pipeline = embedding_pipeline
        self.llm_provider = llm_provider
        
        # Configure LLM
        if llm_provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key:
                openai.api_key = self.api_key
                self.model_name = model_name or "gpt-4"
            else:
                print("Warning: OpenAI API key not found")
                self.model_name = None
        else:
            # Add support for other providers later
            self.model_name = None
        
        # Initialize prompt templates
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompt templates"""
        self.qa_prompt = PromptTemplate(
            template="""You are an AI assistant for the IWL (IdeaWorkLab) educational platform.
Use the following context to answer the user's question accurately and helpfully.

Context:
{context}

Question: {question}

Instructions:
1. Answer based on the provided context
2. If the context doesn't contain enough information, say so
3. Be concise but comprehensive
4. Use Korean if the question is in Korean

Answer:""",
            input_variables=["context", "question"]
        )
        
        self.summarize_prompt = PromptTemplate(
            template="""Summarize the following educational content concisely:

Content:
{content}

Summary:""",
            input_variables=["content"]
        )
    
    def retrieve_context(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context from the vector database"""
        # Generate query embedding
        query_embedding = self.embedding_pipeline.embed_query(query)
        
        # Search in ChromaDB
        results = self.chroma_manager.search(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters
        )
        
        # Format results
        formatted_results = []
        if results and "documents" in results:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i] if "ids" in results else None,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if "metadatas" in results else {},
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
        
        return formatted_results
    
    def generate_answer(
        self,
        question: str,
        context: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Generate answer using LLM"""
        if not self.model_name:
            return "LLM not configured. Please set up API keys."
        
        # Format prompt
        prompt = self.qa_prompt.format(
            context=context,
            question=question
        )
        
        try:
            if self.llm_provider == "openai":
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant for IWL educational platform."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            else:
                return "LLM provider not supported yet."
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(
        self,
        question: str,
        n_results: int = 5,
        filters: Optional[Dict] = None,
        temperature: float = 0.7,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Execute full RAG query"""
        # Retrieve relevant context
        retrieved_docs = self.retrieve_context(question, n_results, filters)
        
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "No relevant information found in the knowledge base.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Combine retrieved texts as context
        context = "\n\n".join([doc["text"] for doc in retrieved_docs])
        
        # Generate answer
        answer = self.generate_answer(question, context, temperature)
        
        # Calculate confidence based on retrieval scores
        avg_distance = sum(doc.get("distance", 1.0) for doc in retrieved_docs) / len(retrieved_docs)
        confidence = max(0.0, min(1.0, 1.0 - avg_distance))
        
        # Prepare response
        response = {
            "question": question,
            "answer": answer,
            "confidence": confidence
        }
        
        if include_sources:
            response["sources"] = [
                {
                    "id": doc.get("id"),
                    "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "relevance_score": 1.0 - doc.get("distance", 1.0)
                }
                for doc in retrieved_docs
            ]
        
        return response
    
    def index_document(
        self,
        document: Dict[str, Any],
        generate_summary: bool = False
    ) -> Dict[str, Any]:
        """Index a new document into the knowledge base"""
        # Process document into chunks with embeddings
        chunks = self.embedding_pipeline.process_document(
            document["content"],
            document.get("metadata", {}),
            generate_embeddings=True
        )
        
        # Prepare data for ChromaDB
        texts = []
        embeddings = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document.get('id', 'doc')}_{i}"
            texts.append(chunk["text"])
            embeddings.append(chunk["embedding"])
            
            # Combine document and chunk metadata
            metadata = chunk["metadata"].copy()
            metadata["document_id"] = document.get("id")
            metadata["document_title"] = document.get("title")
            metadatas.append(metadata)
            ids.append(chunk_id)
        
        # Add to ChromaDB
        added_ids = self.chroma_manager.add_documents(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        result = {
            "document_id": document.get("id"),
            "chunks_created": len(added_ids),
            "chunk_ids": added_ids
        }
        
        # Generate summary if requested
        if generate_summary and self.model_name:
            summary = self._generate_summary(document["content"])
            result["summary"] = summary
        
        return result
    
    def _generate_summary(self, content: str) -> str:
        """Generate a summary of the content"""
        if not self.model_name:
            return "Summary generation not available"
        
        prompt = self.summarize_prompt.format(content=content[:2000])
        
        try:
            if self.llm_provider == "openai":
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful summarizer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=200
                )
                return response.choices[0].message.content
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
        
        return "Summary not available"
    
    def update_document(
        self,
        document_id: str,
        new_content: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Update an existing document"""
        # Delete old chunks
        # Note: In production, you'd want to query for existing chunk IDs
        # For now, we'll assume chunk IDs follow the pattern
        
        # Re-index the document
        document = {
            "id": document_id,
            "content": new_content,
            "metadata": metadata or {}
        }
        
        return self.index_document(document)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        stats = self.chroma_manager.get_collection_stats()
        stats["embedding_model"] = self.embedding_pipeline.get_model_info()
        stats["llm_provider"] = self.llm_provider
        stats["llm_model"] = self.model_name
        
        return stats