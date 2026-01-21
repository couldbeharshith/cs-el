from typing import List, Dict, Optional, Any, Union
from app.services.preprocessors.azure import AzureDocumentProcessor
from app.services.vector_stores.base_vector_store import BaseVectorStore
from app.services.preprocessors.document_embedder import DocumentEmbedder
from app.config.settings import settings


class UnifiedDocumentProcessor:
    def __init__(
        self,
        vector_store: BaseVectorStore,
        chunk_size: int = None,
        chunk_overlap: int = None,
        batch_size: int = 100,
        azure_processor: Optional[AzureDocumentProcessor] = None,
        use_advanced_chunking: bool = True
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.use_advanced_chunking = use_advanced_chunking
        
        self.embedder = DocumentEmbedder(
            vector_store=vector_store,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            batch_size=batch_size,
            azure_processor=azure_processor,
            use_advanced_chunking=self.use_advanced_chunking
        )
    
    def process_document_from_path(self, file_path: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        return self.embedder.process_and_embed_file(file_path, metadata)
    
    def process_document_from_url(self, url: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        return self.embedder.process_and_embed_url(url, metadata)
    
    def process_document_from_bytes(self, file_content: bytes, filename: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        return self.embedder.process_and_embed_bytes(file_content, filename, metadata)
    
    async def process_document_from_path_async(self, file_path: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        return await self.embedder.process_and_embed_file_async(file_path, metadata)
    
    async def process_document_from_url_async(self, url: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        return await self.embedder.process_and_embed_url_async(url, metadata)
    
    def get_document_count(self) -> int:
        return self.embedder.get_document_count()
    
    def clear_all_documents(self) -> bool:
        return self.embedder.clear_all_documents()
    
    def get_supported_formats(self) -> List[str]:
        return self.embedder.get_supported_formats()
    
    def get_chunking_info(self) -> Dict[str, Any]:
        """Get information about the current chunking configuration"""
        return {
            "use_advanced_chunking": self.use_advanced_chunking,
            "advanced_chunker_available": self.embedder.advanced_chunker is not None,
            "fallback_chunker_available": self.embedder.text_splitter is not None,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunking_method": "advanced_markdown" if (self.use_advanced_chunking and self.embedder.advanced_chunker) else "fallback_recursive"
        }
