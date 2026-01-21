import asyncio
from typing import Dict, Any, List
from app.services.pipelines.traditional_rag import traditional_rag
from app.services.vector_stores.vector_store_factory import VectorStoreFactory
from app.services.preprocessors.unified_processor import UnifiedDocumentProcessor
from app.services.retrievers.retrieval_service import RetrievalService
from app.providers.factory import LLMProviderFactory
from app.config.settings import settings
import time
import hashlib


class RAGQueryHelper:
    """Helper class for consistent RAG queries across agents."""
    
    def __init__(self):
        self.vector_store = VectorStoreFactory.create_vector_store(settings)
        self.llm_provider = LLMProviderFactory.create_provider(
            settings.DEFAULT_LLM_PROVIDER,
            settings
        )
        self.document_processor = UnifiedDocumentProcessor(
            vector_store=self.vector_store,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.retrieval_service = RetrievalService(
            vector_store=self.vector_store,
            llm_provider=self.llm_provider
        )
    
    async def query_document(
        self,
        query: str,
        document_url: str,
        k: int = 5,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a RAG query against a document.
        
        Args:
            query: The question or query to ask
            document_url: URL or path to the document
            k: Number of relevant chunks to retrieve
            use_cache: Whether to use cached results
            
        Returns:
            Dict containing success status, answer, metadata, and raw response
        """
        try:
            # Generate unique document ID for caching
            document_id = hashlib.sha256(document_url.encode()).hexdigest()[:16]
            
            answers, metadata, raw_response = await traditional_rag(
                document_id=document_id,
                document_url=document_url,
                questions=[query],
                k=k,
                vector_store=self.vector_store,
                document_processor=self.document_processor,
                retrieval_service=self.retrieval_service,
                settings=settings
            )
            
            return {
                "success": True,
                "answer": answers[0] if answers else "",
                "metadata": metadata,
                "raw_response": raw_response,
                "query": query,
                "document_url": document_url,
                "chunks_retrieved": k
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": "",
                "metadata": {},
                "raw_response": None,
                "query": query,
                "document_url": document_url
            }
    
    async def batch_query_document(
        self,
        queries: List[str],
        document_url: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple RAG queries against a document efficiently.
        
        Args:
            queries: List of questions or queries
            document_url: URL or path to the document
            k: Number of relevant chunks to retrieve per query
            
        Returns:
            List of query results
        """
        try:
            document_id = hashlib.sha256(document_url.encode()).hexdigest()[:16]
            
            answers, metadata, raw_response = await traditional_rag(
                document_id=document_id,
                document_url=document_url,
                questions=queries,
                k=k,
                vector_store=self.vector_store,
                document_processor=self.document_processor,
                retrieval_service=self.retrieval_service,
                settings=settings
            )
            
            results = []
            for i, query in enumerate(queries):
                result = {
                    "success": True,
                    "answer": answers[i] if i < len(answers) else "",
                    "metadata": metadata,
                    "raw_response": raw_response,
                    "query": query,
                    "document_url": document_url,
                    "chunks_retrieved": k
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            # Return error for all queries
            error_result = {
                "success": False,
                "error": str(e),
                "answer": "",
                "metadata": {},
                "raw_response": None,
                "document_url": document_url
            }
            
            return [
                {**error_result, "query": query}
                for query in queries
            ]
    
