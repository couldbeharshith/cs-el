from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.preprocessors.azure import AzureDocumentProcessor
from app.services.preprocessors.Markdown_chunker import InsuranceDocumentChunker
from app.services.vector_stores.base_vector_store import BaseVectorStore
from app.config.settings import settings


class DocumentEmbedder:
    def __init__(
        self,
        vector_store: BaseVectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 100,
        azure_processor: Optional[AzureDocumentProcessor] = None,
        use_advanced_chunking: bool = True
    ):
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.use_advanced_chunking = use_advanced_chunking
        
        self.azure_processor = azure_processor or AzureDocumentProcessor()
        
        # Initialize advanced chunker (primary)
        if self.use_advanced_chunking:
            try:
                self.advanced_chunker = InsuranceDocumentChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    min_chunk_size=300,
                    use_semantic_chunker=True
                )
                print("Advanced Markdown chunker initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize advanced chunker: {e}. Using fallback.")
                self.advanced_chunker = None
                self.use_advanced_chunking = False
        else:
            self.advanced_chunker = None
        
        # Initialize fallback text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_and_embed_file(self, file_path: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        if not self.azure_processor.is_supported_format(file_path):
            raise ValueError(f"Unsupported file format: {Path(file_path).suffix}")
        
        # Check if we already have this file cached in the vector store
        if self.vector_store.has_cache(file_path):
            print(f"Cache found for file: {file_path}")
            if self.vector_store.load_from_cache(file_path):
                # Return cache hit response - no new processing needed
                return {
                    "success": True,
                    "chunk_count": 0,  # No new chunks processed
                    "chunks_processed": 0,
                    "document_ids": [],  # No new document IDs created
                    "vector_ids": [],
                    "vector_store": self.vector_store.store_type,
                    "document_id": (metadata or {}).get("document_id"),
                    "chunking_method": "cache_hit",
                    "cache_used": True
                }
        
        # No cache found, proceed with Azure processing
        documents = self.azure_processor.process_file(file_path)
        extracted_data = self.azure_processor.extract_text_and_metadata(documents)
        
        # Only include JSON-serializable metadata, exclude any LangChain objects
        safe_extracted_metadata = {}
        for key, value in extracted_data["metadata"].items():
            try:
                import json
                json.dumps(value)  # Test if value is JSON serializable
                safe_extracted_metadata[key] = value
            except (TypeError, ValueError):
                continue  # Skip non-serializable values
        
        return self._embed_content(
            content=extracted_data["text"],
            base_metadata={
                "source": file_path,
                "page_count": extracted_data["page_count"],
                **(metadata or {}),
                **safe_extracted_metadata
            }
        )
    
    def process_and_embed_url(self, url: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        # Check if we already have this URL cached in the vector store
        if self.vector_store.has_cache(url):
            print(f"Cache found for URL: {url}")
            if self.vector_store.load_from_cache(url):
                # Return cache hit response - no new processing needed
                return {
                    "success": True,
                    "chunk_count": 0,  # No new chunks processed
                    "chunks_processed": 0,
                    "document_ids": [],  # No new document IDs created
                    "vector_ids": [],
                    "vector_store": self.vector_store.store_type,
                    "document_id": (metadata or {}).get("document_id"),
                    "chunking_method": "cache_hit",
                    "cache_used": True
                }
        
        # No cache found, proceed with Azure processing
        documents = self.azure_processor.process_url(url)
        extracted_data = self.azure_processor.extract_text_and_metadata(documents)
        
        # Only include JSON-serializable metadata, exclude any LangChain objects
        safe_extracted_metadata = {}
        for key, value in extracted_data["metadata"].items():
            try:
                import json
                json.dumps(value)  # Test if value is JSON serializable
                safe_extracted_metadata[key] = value
            except (TypeError, ValueError):
                continue  # Skip non-serializable values
        
        return self._embed_content(
            content=extracted_data["text"],
            base_metadata={
                "source": url,
                "page_count": extracted_data["page_count"],
                **(metadata or {}),
                **safe_extracted_metadata
            }
        )
    
    def process_and_embed_bytes(self, file_content: bytes, filename: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        documents = self.azure_processor.process_bytes(file_content, filename)
        extracted_data = self.azure_processor.extract_text_and_metadata(documents)
        
        return self._embed_content(
            content=extracted_data["text"],
            base_metadata={
                "source": filename,
                "page_count": extracted_data["page_count"],
                **(metadata or {}),
                **extracted_data["metadata"]
            }
        )
    
    def _embed_content(self, content: str, base_metadata: Dict) -> Dict[str, Any]:
        if not content.strip():
            return {
                "success": False,
                "error": "No content to process",
                "chunk_count": 0,
                "chunks_processed": 0,
                "document_ids": [],
                "vector_store": self.vector_store.store_type,
                "document_id": base_metadata.get("document_id"),
                "cache_used": False
            }
        
        # Try advanced chunking first
        if self.use_advanced_chunking and self.advanced_chunker:
            try:
                # Get chunks with rich metadata from advanced chunker
                chunks_with_metadata = self.advanced_chunker.chunk_with_metadata(content)
                
                if chunks_with_metadata:
                    document_ids, cache_used = self._embed_advanced_chunks_in_batches(chunks_with_metadata, base_metadata)
                    
                    return {
                        "success": True,
                        "chunk_count": len(chunks_with_metadata),
                        "chunks_processed": len(chunks_with_metadata),
                        "document_ids": document_ids,
                        "vector_ids": document_ids,
                        "vector_store": self.vector_store.store_type,
                        "document_id": base_metadata.get("document_id"),
                        "chunking_method": "advanced_markdown_chunker",
                        "cache_used": cache_used
                    }
                
            except Exception as e:
                print(f"Advanced chunking failed: {e}. Falling back to basic chunking.")
                # Fall through to fallback chunking
        
        # Fallback to basic chunking
        chunks = self.text_splitter.split_text(content)
        
        if not chunks:
            return {
                "success": False,
                "error": "No chunks generated",
                "chunk_count": 0,
                "chunks_processed": 0,
                "document_ids": [],
                "vector_store": self.vector_store.store_type,
                "document_id": base_metadata.get("document_id"),
                "cache_used": False
            }
        
        document_ids, cache_used = self._embed_chunks_in_batches(chunks, base_metadata)
        
        return {
            "success": True,
            "chunk_count": len(chunks),
            "chunks_processed": len(chunks),
            "document_ids": document_ids,
            "vector_ids": document_ids,
            "vector_store": self.vector_store.store_type,
            "document_id": base_metadata.get("document_id"),
            "chunking_method": "fallback_recursive_splitter",
            "cache_used": cache_used
        }
    
    def _embed_chunks_in_batches(self, chunks: List[str], base_metadata: Dict) -> tuple[List[str], bool]:
        all_ids = []
        total_chunks = len(chunks)
        cache_used_overall = False
        
        for i in range(0, total_chunks, self.batch_size):
            batch_end = min(i + self.batch_size, total_chunks)
            batch_chunks = chunks[i:batch_end]
            
            # Simple metadata like the old document_processor
            batch_metadatas = []
            for j, chunk in enumerate(batch_chunks):
                chunk_metadata = {
                    "document_id": base_metadata.get("document_id"),
                    "source": base_metadata.get("source"),
                    "chunk_index": i + j,
                    "total_chunks": total_chunks,
                    "page": base_metadata.get("page_count", 0)
                }
                batch_metadatas.append(chunk_metadata)
            
            batch_ids, batch_cache_used = self.vector_store.add_documents(
                texts=batch_chunks,
                metadatas=batch_metadatas
            )
            
            all_ids.extend(batch_ids)
            if batch_cache_used:
                cache_used_overall = True
        
        return all_ids, cache_used_overall
    
    def _embed_advanced_chunks_in_batches(self, chunks_with_metadata: List[Tuple], base_metadata: Dict) -> tuple[List[str], bool]:
        """Embed chunks from advanced chunker with rich metadata"""
        all_ids = []
        total_chunks = len(chunks_with_metadata)
        cache_used_overall = False
        
        for i in range(0, total_chunks, self.batch_size):
            batch_end = min(i + self.batch_size, total_chunks)
            batch_chunks = chunks_with_metadata[i:batch_end]
            
            batch_texts = []
            batch_metadatas = []
            
            for j, (chunk_text, chunk_metadata) in enumerate(batch_chunks):
                # Combine base metadata with advanced chunker metadata
                enhanced_metadata = {
                    # Base metadata
                    "document_id": base_metadata.get("document_id"),
                    "source": base_metadata.get("source"),
                    "page_count": base_metadata.get("page_count", 0),
                    
                    # Basic chunk info
                    "chunk_index": i + j,
                    "total_chunks": total_chunks,
                    
                    # Advanced chunker metadata (convert dataclass to dict if needed)
                    "chunk_id": chunk_metadata.chunk_id,
                    "chunk_type": chunk_metadata.chunk_type,
                    "importance_score": chunk_metadata.importance_score,
                    "section_hierarchy": chunk_metadata.section_hierarchy,
                    "has_tables": chunk_metadata.has_tables,
                    "table_count": chunk_metadata.table_count,
                    "has_monetary_values": chunk_metadata.has_monetary_values,
                    "monetary_amounts": chunk_metadata.monetary_amounts,
                    "has_exclusions": chunk_metadata.has_exclusions,
                    "exclusion_phrases": chunk_metadata.exclusion_phrases,
                    "policy_terms": chunk_metadata.policy_terms,
                    "cross_references": chunk_metadata.cross_references,
                    "original_headers": chunk_metadata.original_headers,
                    "chunk_position": chunk_metadata.chunk_position,
                    "has_important_notes": getattr(chunk_metadata, 'has_important_notes', False),
                    "important_notes": getattr(chunk_metadata, 'important_notes', []),
                    "has_structured_lists": getattr(chunk_metadata, 'has_structured_lists', False),
                    "list_types": getattr(chunk_metadata, 'list_types', []),
                    # Structure-aware fields propagated for retrieval
                    "element_type": getattr(chunk_metadata, 'element_type', 'text'),
                    "table_id": getattr(chunk_metadata, 'table_id', None),
                    "table_headers": getattr(chunk_metadata, 'table_headers', None),
                    "row_index": getattr(chunk_metadata, 'row_index', None),
                    "column_data": getattr(chunk_metadata, 'column_data', None),
                    "is_complete_table": getattr(chunk_metadata, 'is_complete_table', False),
                    "footnote_ref": getattr(chunk_metadata, 'footnote_ref', None),
                    "cell_reference": getattr(chunk_metadata, 'cell_reference', None),
                }
                
                batch_texts.append(chunk_text)
                batch_metadatas.append(enhanced_metadata)
            
            # Embed batch with enhanced metadata
            batch_ids, batch_cache_used = self.vector_store.add_documents(
                texts=batch_texts,
                metadatas=batch_metadatas
            )
            
            all_ids.extend(batch_ids)
            if batch_cache_used:
                cache_used_overall = True
        
        return all_ids, cache_used_overall
    
    async def _embed_chunks_in_batches_async(self, chunks: List[str], base_metadata: Dict) -> tuple[List[str], bool]:
        all_ids = []
        total_chunks = len(chunks)
        cache_used_overall = False
        
        for i in range(0, total_chunks, self.batch_size):
            batch_end = min(i + self.batch_size, total_chunks)
            batch_chunks = chunks[i:batch_end]
            
            # Simple metadata like the old document_processor
            batch_metadatas = []
            for j, chunk in enumerate(batch_chunks):
                chunk_metadata = {
                    "document_id": base_metadata.get("document_id"),
                    "source": base_metadata.get("source"),
                    "chunk_index": i + j,
                    "total_chunks": total_chunks,
                    "page": base_metadata.get("page_count", 0)
                }
                batch_metadatas.append(chunk_metadata)
            
            if hasattr(self.vector_store, 'aadd_documents'):
                batch_ids, batch_cache_used = await self.vector_store.aadd_documents(
                    texts=batch_chunks,
                    metadatas=batch_metadatas
                )
            else:
                batch_ids, batch_cache_used = self.vector_store.add_documents(
                    texts=batch_chunks,
                    metadatas=batch_metadatas
                )
            
            all_ids.extend(batch_ids)
            if batch_cache_used:
                cache_used_overall = True
        
        return all_ids, cache_used_overall
    
    async def _embed_advanced_chunks_in_batches_async(self, chunks_with_metadata: List[Tuple], base_metadata: Dict) -> tuple[List[str], bool]:
        """Embed chunks from advanced chunker with rich metadata (async)"""
        all_ids = []
        total_chunks = len(chunks_with_metadata)
        cache_used_overall = False
        
        for i in range(0, total_chunks, self.batch_size):
            batch_end = min(i + self.batch_size, total_chunks)
            batch_chunks = chunks_with_metadata[i:batch_end]
            
            batch_texts = []
            batch_metadatas = []
            
            for j, (chunk_text, chunk_metadata) in enumerate(batch_chunks):
                # Combine base metadata with advanced chunker metadata
                enhanced_metadata = {
                    # Base metadata
                    "document_id": base_metadata.get("document_id"),
                    "source": base_metadata.get("source"),
                    "page_count": base_metadata.get("page_count", 0),
                    
                    # Basic chunk info
                    "chunk_index": i + j,
                    "total_chunks": total_chunks,
                    
                    # Advanced chunker metadata (convert dataclass to dict if needed)
                    "chunk_id": chunk_metadata.chunk_id,
                    "chunk_type": chunk_metadata.chunk_type,
                    "importance_score": chunk_metadata.importance_score,
                    "section_hierarchy": chunk_metadata.section_hierarchy,
                    "has_tables": chunk_metadata.has_tables,
                    "table_count": chunk_metadata.table_count,
                    "has_monetary_values": chunk_metadata.has_monetary_values,
                    "monetary_amounts": chunk_metadata.monetary_amounts,
                    "has_exclusions": chunk_metadata.has_exclusions,
                    "exclusion_phrases": chunk_metadata.exclusion_phrases,
                    "policy_terms": chunk_metadata.policy_terms,
                    "cross_references": chunk_metadata.cross_references,
                    "original_headers": chunk_metadata.original_headers,
                    "chunk_position": chunk_metadata.chunk_position,
                    "has_important_notes": getattr(chunk_metadata, 'has_important_notes', False),
                    "important_notes": getattr(chunk_metadata, 'important_notes', []),
                    "has_structured_lists": getattr(chunk_metadata, 'has_structured_lists', False),
                    "list_types": getattr(chunk_metadata, 'list_types', []),
                    # Structure-aware fields propagated for retrieval
                    "element_type": getattr(chunk_metadata, 'element_type', 'text'),
                    "table_id": getattr(chunk_metadata, 'table_id', None),
                    "table_headers": getattr(chunk_metadata, 'table_headers', None),
                    "row_index": getattr(chunk_metadata, 'row_index', None),
                    "column_data": getattr(chunk_metadata, 'column_data', None),
                    "is_complete_table": getattr(chunk_metadata, 'is_complete_table', False),
                    "footnote_ref": getattr(chunk_metadata, 'footnote_ref', None),
                    "cell_reference": getattr(chunk_metadata, 'cell_reference', None),
                }
                
                batch_texts.append(chunk_text)
                batch_metadatas.append(enhanced_metadata)
            
            # Embed batch with enhanced metadata
            if hasattr(self.vector_store, 'aadd_documents'):
                batch_ids, batch_cache_used = await self.vector_store.aadd_documents(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
            else:
                batch_ids, batch_cache_used = self.vector_store.add_documents(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
            
            all_ids.extend(batch_ids)
            if batch_cache_used:
                cache_used_overall = True
        
        return all_ids, cache_used_overall
    
    async def process_and_embed_file_async(self, file_path: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        if not self.azure_processor.is_supported_format(file_path):
            raise ValueError(f"Unsupported file format: {Path(file_path).suffix}")
        
        # Check if we already have this file cached in the vector store
        if self.vector_store.has_cache(file_path):
            print(f"Cache found for file: {file_path}")
            if self.vector_store.load_from_cache(file_path):
                # Return cache hit response - no new processing needed
                return {
                    "success": True,
                    "chunk_count": 0,  # No new chunks processed
                    "chunks_processed": 0,
                    "document_ids": [],  # No new document IDs created
                    "vector_ids": [],
                    "vector_store": self.vector_store.store_type,
                    "document_id": (metadata or {}).get("document_id"),
                    "chunking_method": "cache_hit",
                    "cache_used": True
                }
        
        # No cache found, proceed with Azure processing
        documents = self.azure_processor.process_file(file_path)
        extracted_data = self.azure_processor.extract_text_and_metadata(documents)
        
        safe_extracted_metadata = {}
        for key, value in extracted_data["metadata"].items():
            try:
                import json
                json.dumps(value) 
                safe_extracted_metadata[key] = value
            except (TypeError, ValueError):
                continue  
        
        return await self._embed_content_async(
            content=extracted_data["text"],
            base_metadata={
                "source": file_path,
                "page_count": extracted_data["page_count"],
                **(metadata or {}),
                **safe_extracted_metadata
            }
        )
    
    async def process_and_embed_url_async(self, url: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        # Check if we already have this URL cached in the vector store
        if self.vector_store.has_cache(url):
            print(f"Cache found for URL: {url}")
            if self.vector_store.load_from_cache(url):
                # Return cache hit response - no new processing needed
                return {
                    "success": True,
                    "chunk_count": 0,  # No new chunks processed
                    "chunks_processed": 0,
                    "document_ids": [],  # No new document IDs created
                    "vector_ids": [],
                    "vector_store": self.vector_store.store_type,
                    "document_id": (metadata or {}).get("document_id"),
                    "chunking_method": "cache_hit",
                    "cache_used": True
                }
        
        # No cache found, proceed with Azure processing
        documents = self.azure_processor.process_url(url)
        extracted_data = self.azure_processor.extract_text_and_metadata(documents)
        
        safe_extracted_metadata = {}
        for key, value in extracted_data["metadata"].items():
            try:
                import json
                json.dumps(value)  
                safe_extracted_metadata[key] = value
            except (TypeError, ValueError):
                continue  
        
        return await self._embed_content_async(
            content=extracted_data["text"],
            base_metadata={
                "source": url,
                "page_count": extracted_data["page_count"],
                **(metadata or {}),
                **safe_extracted_metadata
            }
        )
    
    async def _embed_content_async(self, content: str, base_metadata: Dict) -> Dict[str, Any]:
        if not content.strip():
            return {
                "success": False,
                "error": "No content to process",
                "chunk_count": 0,
                "chunks_processed": 0,
                "document_ids": [],
                "vector_store": self.vector_store.store_type,
                "document_id": base_metadata.get("document_id"),
                "cache_used": False
            }
        
        # Try advanced chunking first
        if self.use_advanced_chunking and self.advanced_chunker:
            try:
                # Get chunks with rich metadata from advanced chunker
                chunks_with_metadata = self.advanced_chunker.chunk_with_metadata(content)
                
                if chunks_with_metadata:
                    document_ids, cache_used = await self._embed_advanced_chunks_in_batches_async(chunks_with_metadata, base_metadata)
                    
                    return {
                        "success": True,
                        "chunk_count": len(chunks_with_metadata),
                        "chunks_processed": len(chunks_with_metadata),
                        "document_ids": document_ids,
                        "vector_ids": document_ids,
                        "vector_store": self.vector_store.store_type,
                        "document_id": base_metadata.get("document_id"),
                        "chunking_method": "advanced_markdown_chunker",
                        "cache_used": cache_used
                    }
                
            except Exception as e:
                print(f"Advanced chunking failed: {e}. Falling back to basic chunking.")
                # Fall through to fallback chunking
        
        # Fallback to basic chunking
        chunks = self.text_splitter.split_text(content)
        
        if not chunks:
            return {
                "success": False,
                "error": "No chunks generated",
                "chunk_count": 0,
                "chunks_processed": 0,
                "document_ids": [],
                "vector_store": self.vector_store.store_type,
                "document_id": base_metadata.get("document_id"),
                "cache_used": False
            }
        
        document_ids, cache_used = await self._embed_chunks_in_batches_async(chunks, base_metadata)
        
        return {
            "success": True,
            "chunk_count": len(chunks),
            "chunks_processed": len(chunks),
            "document_ids": document_ids,
            "vector_ids": document_ids,
            "vector_store": self.vector_store.store_type,
            "document_id": base_metadata.get("document_id"),
            "chunking_method": "fallback_recursive_splitter",
            "cache_used": cache_used
        }
    
    def get_document_count(self) -> int:
        return self.vector_store.get_document_count()
    
    def clear_all_documents(self) -> bool:
        return self.vector_store.delete_all_documents()
    
    def get_supported_formats(self) -> List[str]:
        return self.azure_processor.get_supported_formats()
