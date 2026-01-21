"""
Multi-key reranker services for the Insurance RAG API

This module implements multi-key Pinecone reranker management and Cohere reranker
as a fallback service, with unified reranker interface for primary/fallback switching.
"""

import asyncio
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

import cohere
from pinecone import Pinecone

from app.config import Config

logger = logging.getLogger(__name__)


@dataclass
class RerankerResult:
    """Result from reranking operation"""
    index: int
    score: float
    document: str
    service_used: str  # "pinecone" or "cohere"


class PineconeRerankerManager:
    """
    Manages multiple Pinecone reranker API keys with async key cycling and fallback logic.
    Cycles through 10 dedicated reranker keys separate from the main vector DB key.
    """
    
    def __init__(self, reranker_api_keys: List[str]):
        """
        Initialize with list of Pinecone reranker API keys.
        
        Args:
            reranker_api_keys: List of Pinecone reranker API keys
        """
        self.reranker_keys = [key for key in reranker_api_keys if key and key.strip()]
        self.current_key_index = 0
        
        if not self.reranker_keys:
            logger.warning("No Pinecone reranker API keys provided")
        else:
            logger.info(f"PineconeRerankerManager initialized with {len(self.reranker_keys)} keys")
    
    async def get_next_reranker_key(self) -> str:
        """
        Get the next reranker API key in rotation.
        
        Returns:
            Next reranker API key
            
        Raises:
            Exception: If no reranker keys are available
        """
        if not self.reranker_keys:
            raise Exception("No Pinecone reranker API keys configured")
        
        # Get next key in rotation
        key = self.reranker_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.reranker_keys)
        
        return key
    
    async def rerank_with_fallback(self, query: str, documents: List[str], top_n: int) -> List[RerankerResult]:
        """
        Rerank documents using Pinecone reranker with key cycling and fallback logic.
        Tries each key until success or all keys fail.
        
        Args:
            query: Query text for reranking
            documents: List of document texts to rerank
            top_n: Number of top results to return
            
        Returns:
            List of RerankerResult objects
            
        Raises:
            Exception: If all reranker keys fail
        """
        if not self.reranker_keys:
            raise Exception("No Pinecone reranker API keys available")
        
        if not documents:
            logger.warning("No documents provided for reranking")
            return []
        
        # Try each key until success
        last_error = None
        keys_tried = 0
        
        for _ in range(len(self.reranker_keys)):
            try:
                # Get next key
                api_key = await self.get_next_reranker_key()
                keys_tried += 1
                
                logger.debug(f"Trying Pinecone reranker with key {keys_tried}/{len(self.reranker_keys)}")
                
                # Create Pinecone client with this reranker key
                pc = Pinecone(api_key=api_key)
                
                # Prepare documents for reranking
                rerank_documents = []
                for i, doc_text in enumerate(documents):
                    rerank_documents.append({
                        "id": str(i),
                        "text": doc_text
                    })
                
                # Perform reranking
                ranked_results = await asyncio.to_thread(
                    pc.inference.rerank,
                    model=Config.PINECONE_RERANK_MODEL,
                    query=query,
                    documents=rerank_documents,
                    top_n=min(top_n, len(documents)),
                    rank_fields=["text"],
                    return_documents=True
                )
                
                # Convert results to RerankerResult objects
                results = []
                for result in ranked_results.data:
                    results.append(RerankerResult(
                        index=int(result.document.id),
                        score=result.score,
                        document=result.document.text,
                        service_used="pinecone"
                    ))
                
                logger.info(f"Pinecone reranking successful with key {keys_tried}, returned {len(results)} results")
                return results
                
            except Exception as e:
                last_error = e
                logger.warning(f"Pinecone reranker key {keys_tried} failed: {e}")
                continue
        
        # All keys failed
        error_msg = f"All {len(self.reranker_keys)} Pinecone reranker keys failed. Last error: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)


class CohereRerankerService:
    """
    Cohere reranker service using cohere.AsyncClientV2.
    Manages multiple Cohere API keys with cycling and fallback logic.
    """
    
    def __init__(self, api_keys: List[str]):
        """
        Initialize Cohere reranker service.
        
        Args:
            api_keys: List of Cohere API keys
        """
        self.api_keys = [key for key in api_keys if key and key.strip()]
        self.current_key_index = 0
        
        if not self.api_keys:
            logger.warning("No Cohere API keys provided")
        else:
            logger.info(f"CohereRerankerService initialized with {len(self.api_keys)} keys")
    
    async def get_next_key(self) -> str:
        """
        Get the next Cohere API key in rotation.
        
        Returns:
            Next Cohere API key
            
        Raises:
            Exception: If no API keys are available
        """
        if not self.api_keys:
            raise Exception("No Cohere API keys available")
        
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key
    
    async def rerank(self, query: str, documents: List[str], top_n: int) -> List[RerankerResult]:
        """
        Rerank documents using Cohere reranker with key cycling.
        
        Args:
            query: Query text for reranking
            documents: List of document texts to rerank
            top_n: Number of top results to return
            
        Returns:
            List of RerankerResult objects
            
        Raises:
            Exception: If all Cohere API keys fail
        """
        if not self.api_keys:
            raise Exception("No Cohere API keys available")
        
        if not documents:
            logger.warning("No documents provided for Cohere reranking")
            return []
        
        # Try each API key until one works
        for attempt in range(len(self.api_keys)):
            try:
                api_key = await self.get_next_key()
                client = cohere.AsyncClientV2(api_key=api_key)
                
                logger.debug(f"Reranking {len(documents)} documents with Cohere (attempt {attempt + 1})")
                
                # Perform reranking using Cohere
                response = await client.rerank(
                    model=Config.COHERE_RERANK_MODEL,
                    query=query,
                    documents=documents,
                    top_n=min(top_n, len(documents))
                )
                
                # Convert results to RerankerResult objects
                results = []
                for result in response.results:
                    results.append(RerankerResult(
                        index=result.index,
                        score=result.relevance_score,
                        document=result.document.text if hasattr(result.document, 'text') else documents[result.index],
                        service_used="cohere"
                    ))
                
                logger.info(f"Cohere reranking successful, returned {len(results)} results")
                return results
                
            except Exception as e:
                logger.warning(f"Cohere reranking failed with key {attempt + 1}: {e}")
                if attempt == len(self.api_keys) - 1:
                    logger.error(f"All Cohere API keys failed")
                    raise Exception(f"All Cohere API keys failed: {e}")
                continue


class RerankerService:
    """
    Unified reranker service that handles primary/fallback switching between
    Pinecone and Cohere reranker services.
    """
    
    def __init__(self):
        """Initialize unified reranker service with primary and fallback services."""
        # Get reranker keys
        pinecone_reranker_keys = Config.get_pinecone_reranker_keys()
        cohere_api_keys = Config.get_cohere_api_keys()
        
        # Initialize services
        self.pinecone_manager = PineconeRerankerManager(pinecone_reranker_keys)
        self.cohere_service = CohereRerankerService(cohere_api_keys)
        
        # Determine primary and secondary services
        self.primary_service = Config.PRIMARY_RERANKER_SERVICE.lower()
        self.secondary_service = Config.SECONDARY_RERANKER_SERVICE.lower()
        
        logger.info(f"RerankerService initialized - Primary: {self.primary_service}, Secondary: {self.secondary_service}")
    
    async def rerank(self, query: str, documents: List[str], top_n: int) -> List[RerankerResult]:
        """
        Rerank documents using primary service with fallback to secondary service.
        
        Args:
            query: Query text for reranking
            documents: List of document texts to rerank
            top_n: Number of top results to return
            
        Returns:
            List of RerankerResult objects
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []
        
        # Try primary service first
        try:
            logger.debug(f"Attempting reranking with primary service: {self.primary_service}")
            
            if self.primary_service == "pinecone":
                return await self.pinecone_manager.rerank_with_fallback(query, documents, top_n)
            elif self.primary_service == "cohere":
                return await self.cohere_service.rerank(query, documents, top_n)
            else:
                logger.warning(f"Unknown primary reranker service: {self.primary_service}")
                raise Exception(f"Unknown primary reranker service: {self.primary_service}")
                
        except Exception as primary_error:
            logger.warning(f"Primary reranker service ({self.primary_service}) failed: {primary_error}")
            
            # Try secondary service
            try:
                logger.info(f"Falling back to secondary service: {self.secondary_service}")
                
                if self.secondary_service == "pinecone":
                    return await self.pinecone_manager.rerank_with_fallback(query, documents, top_n)
                elif self.secondary_service == "cohere":
                    return await self.cohere_service.rerank(query, documents, top_n)
                else:
                    logger.warning(f"Unknown secondary reranker service: {self.secondary_service}")
                    raise Exception(f"Unknown secondary reranker service: {self.secondary_service}")
                    
            except Exception as secondary_error:
                logger.error(f"Secondary reranker service ({self.secondary_service}) also failed: {secondary_error}")
                
                # Both services failed - return empty results
                logger.error("Both primary and secondary reranker services failed")
                raise Exception(f"All reranker services failed. Primary ({self.primary_service}): {primary_error}. Secondary ({self.secondary_service}): {secondary_error}")
    
    def is_available(self) -> bool:
        """
        Check if at least one reranker service is available.
        
        Returns:
            True if at least one service is available, False otherwise
        """
        pinecone_available = len(self.pinecone_manager.reranker_keys) > 0
        cohere_available = len(self.cohere_service.api_keys) > 0
        
        return pinecone_available or cohere_available
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of all reranker services.
        
        Returns:
            Dictionary with service status information
        """
        return {
            "primary_service": self.primary_service,
            "secondary_service": self.secondary_service,
            "pinecone_keys_count": len(self.pinecone_manager.reranker_keys),
            "cohere_keys_count": len(self.cohere_service.api_keys),
            "overall_available": self.is_available()
        }