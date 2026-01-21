import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_core.documents import Document
from app.config.settings import settings


class AzureDocumentProcessor:
    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_model: str = "prebuilt-layout",
        mode: str = "markdown",
        analysis_features: Optional[List[str]] = None
    ):
        self.api_endpoint = api_endpoint or settings.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
        self.api_key = api_key or settings.AZURE_DOCUMENT_INTELLIGENCE_KEY
        self.api_model = api_model
        self.mode = mode
        self.analysis_features = analysis_features or []
        
        if not self.api_endpoint or not self.api_key:
            raise ValueError("Azure Document Intelligence endpoint and key are required")
    
    def process_file(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        loader = AzureAIDocumentIntelligenceLoader(
            api_endpoint=self.api_endpoint,
            api_key=self.api_key,
            file_path=file_path,
            api_model=self.api_model,
            mode=self.mode,
            analysis_features=self.analysis_features
        )
        
        return loader.load()
    
    def process_url(self, url: str) -> List[Document]:
        loader = AzureAIDocumentIntelligenceLoader(
            api_endpoint=self.api_endpoint,
            api_key=self.api_key,
            url_path=url,
            api_model=self.api_model,
            mode=self.mode,
            analysis_features=self.analysis_features
        )
        
        return loader.load()
    
    def process_bytes(self, file_content: bytes, filename: str = "document") -> List[Document]:
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            return self.process_file(temp_file_path)
        finally:
            os.unlink(temp_file_path)
    
    def extract_text_and_metadata(self, documents: List[Document]) -> Dict[str, Any]:
        if not documents:
            return {"text": "", "metadata": {}, "page_count": 0}
        
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        combined_metadata = {}
        
        for doc in documents:
            serializable_metadata = {}
            for key, value in doc.metadata.items():
                try:
                    import json
                    json.dumps(value) 
                    serializable_metadata[key] = value
                except (TypeError, ValueError):
                    continue
            combined_metadata.update(serializable_metadata)
        
        return {
            "text": combined_text,
            "metadata": combined_metadata,
            "page_count": len(documents) if self.mode == "page" else 1
        }
    
    def get_supported_formats(self) -> List[str]:
        return [
            "pdf", "jpeg", "jpg", "png", "bmp", "tiff", "heif",
            "docx", "xlsx", "pptx", "html"
        ]
    
    def is_supported_format(self, file_path: str) -> bool:
        extension = Path(file_path).suffix.lower().lstrip('.')
        return extension in self.get_supported_formats()
