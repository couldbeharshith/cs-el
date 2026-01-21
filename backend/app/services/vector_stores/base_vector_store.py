from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from langchain_core.documents import Document

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(
        self, 
        texts: List[str], 
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ) -> Tuple[List[str], bool]:
        pass
    
    @abstractmethod
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        pass
    
    @abstractmethod
    def as_retriever(self, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def delete_documents(
        self, 
        ids: List[str]
    ) -> bool:
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        pass
    
    @abstractmethod
    def delete_all_documents(self) -> bool:
        pass
    
    def supports_caching(self) -> bool:
        return False
    
    def load_from_cache(self, document_url: str) -> bool:
        return False
    
    def save_to_cache(self, document_url: str) -> bool:
        return False
    
    def has_cache(self, document_url: str) -> bool:
        return False
    
    def clear_cache(self, document_url: Optional[str] = None) -> bool:
        return False
