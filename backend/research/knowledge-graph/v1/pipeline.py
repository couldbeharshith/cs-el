"""
Main Pipeline Module
Orchestrates document processing, chunking, and knowledge graph building
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import PolicyConfig
from document_processor import AzureDocumentProcessor
from chunking import PolicyChunker, PolicyEntityExtractor
from graph_builder import PolicyGraphBuilder
from query_processor import PolicyQueryProcessor

logger = logging.getLogger(__name__)


class PolicyKnowledgeGraphPipeline:
    """Main pipeline for processing policy documents"""
    
    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
        self.config.validate()
        
        # Initialize components
        self.doc_processor = AzureDocumentProcessor(
            endpoint=self.config.azure_endpoint,
            key=self.config.azure_key,
            model="prebuilt-layout"
        )
        
        self.chunker = PolicyChunker(self.config)
        self.entity_extractor = PolicyEntityExtractor()
        
        self.graph_builder = PolicyGraphBuilder(
            neo4j_uri=self.config.neo4j_uri,
            neo4j_user=self.config.neo4j_user,
            neo4j_password=self.config.neo4j_password,
            openai_api_key=self.config.openai_api_key,
            openai_model=self.config.openai_model,
            embedding_model=self.config.openai_embedding_model
        )
        
        self.query_processor = PolicyQueryProcessor(self.config, self.graph_builder)
    
    async def process_policy_document(
        self,
        file_path: str,
        policy_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a policy document and add to knowledge graph"""
        logger.info(f"Processing policy document: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Policy document not found: {file_path}")
        
        try:
            # Step 1: Extract text using Azure OCR (markdown mode)
            logger.info("Step 1: Extracting text with Azure OCR...")
            content = self.doc_processor.process_file(file_path)
            logger.info(f"Extracted {len(content)} characters")
            
            # Step 2: Chunk the document
            logger.info("Step 2: Chunking document...")
            chunks = self.chunker.chunk_document(content, policy_info)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Extract entities from chunks
            logger.info("Step 3: Extracting entities...")
            for i, chunk in enumerate(chunks):
                chunks[i] = self.entity_extractor.extract_entities(chunk)
            
            total_entities = sum(len(chunk.entities) for chunk in chunks)
            logger.info(f"Extracted {total_entities} entity types across all chunks")
            
            # Step 4: Add to knowledge graph
            logger.info("Step 4: Adding to knowledge graph...")
            if not policy_info:
                policy_info = {
                    "title": Path(file_path).stem,
                    "uin": "UNKNOWN",
                    "filename": Path(file_path).name
                }
            
            result = await self.graph_builder.add_policy_to_graph(chunks, policy_info)
            
            # Add summary information
            result.update({
                "file_processed": file_path,
                "chunks_created": len(chunks),
                "total_entities": total_entities,
                "content_length": len(content)
            })
            
            logger.info(f"Policy processing complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
    
    async def process_multiple_documents(
        self,
        file_paths: List[str],
        policy_infos: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple policy documents"""
        results = []
        
        for i, file_path in enumerate(file_paths):
            policy_info = policy_infos[i] if policy_infos and i < len(policy_infos) else None
            try:
                result = await self.process_policy_document(file_path, policy_info)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({
                    "file_processed": file_path,
                    "error": str(e),
                    "episodes_added": 0,
                    "entities_added": 0
                })
        
        return results
    
    async def answer_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Answer multiple queries about policies"""
        results = []
        
        for query in queries:
            logger.info(f"Processing query: {query[:100]}...")
            try:
                answer = await self.query_processor.answer_query(query)
                results.append(answer)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "answer": "Error processing query"
                })
        
        return results
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        await self.graph_builder.initialize()
        return self.graph_builder.get_graph_stats()
    
    async def search_policies(self, search_query: str) -> List[Dict[str, Any]]:
        """Search across all policies in the knowledge graph"""
        return await self.graph_builder.search(search_query)
    
    async def close(self):
        """Clean up resources"""
        await self.graph_builder.close()


# Convenience functions
async def process_single_document(file_path: str, policy_info: Optional[Dict] = None) -> Dict[str, Any]:
    """Convenience function to process a single document"""
    pipeline = PolicyKnowledgeGraphPipeline()
    try:
        result = await pipeline.process_policy_document(file_path, policy_info)
        return result
    finally:
        await pipeline.close()


async def quick_search(query: str) -> List[Dict[str, Any]]:
    """Convenience function for quick searches"""
    pipeline = PolicyKnowledgeGraphPipeline()
    try:
        return await pipeline.search_policies(query)
    finally:
        await pipeline.close()