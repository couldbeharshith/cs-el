from langchain_core.vectorstores import InMemoryVectorStore
from app.services.vector_stores.base_vector_store import BaseVectorStore
from app.services.vector_stores.vector_store_cache import VectorStoreCache
from app.services.embedders.embedding_factory import get_embedding_model
from app.services.embedders.langchain_wrapper import LangChainEmbeddingWrapper
from typing import List, Dict, Optional, Any
from langchain_core.documents import Document
from app.config.settings import settings
from pathlib import Path
import uuid

class InMemoryVectorStoreService(BaseVectorStore):
    def __init__(
        self, 
        embedding_model: str = "text-embedding-3-small"
    ):
        self.embedding_model = embedding_model
        
        embedder = get_embedding_model(embedding_model)
        self.embeddings = LangChainEmbeddingWrapper(embedder)
        
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
        
        self.store_type = "inmemory"
        
        self.cache_manager = VectorStoreCache()
        
        print("Initialized InMemory vector store with caching support")
    
    def add_documents(
        self, 
        texts: List[str], 
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ) -> tuple[List[str], bool]:
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Extract source URL for potential caching after successful addition
        source_url = None
        if metadatas and len(metadatas) > 0:
            source_url = metadatas[0].get("source")
        
        print(f"Adding {len(texts)} documents to InMemory vector store...")
        
        try:
            documents = [
                Document(id=doc_id, page_content=text, metadata=metadata)
                for doc_id, text, metadata in zip(ids, texts, metadatas)
            ]
            
            added_ids = self.vector_store.add_documents(documents)
            
            # Save to cache if we have a source URL and successfully added documents
            if source_url and added_ids:
                try:
                    self.save_to_cache(source_url)
                    print(f"Cached vector store for future use: {source_url[:50]}...")
                except Exception as cache_error:
                    print(f"Warning: Failed to cache vector store: {cache_error}")
                    # Don't fail the whole operation if caching fails
            
            print(f"Successfully added {len(added_ids)} documents to InMemory vector store")
            return added_ids, False  # cache_used is always False since caching is handled in DocumentEmbedder
            
        except Exception as e:
            print(f"Error adding documents to InMemory vector store: {e}")
            raise
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query, 
                k=k
            )
            return results
            
        except Exception as e:
            print(f"Error during similarity search with score: {e}")
            return []
    
    def delete_documents(
        self, 
        ids: List[str]
    ) -> bool:
        try:
            self.vector_store.delete(ids=ids)
            print(f"Deleted {len(ids)} documents from InMemory vector store")
            return True
            
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    def get_document_count(self) -> int:
        try:
            count = len(self.vector_store.store)
            return count
            
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    def delete_all_documents(self) -> bool:
        try:
            all_ids = list(self.vector_store.store.keys())
            if all_ids:
                self.vector_store.delete(ids=all_ids)
            
            print(f"Deleted all documents from InMemory vector store")
            return True
            
        except Exception as e:
            print(f"Error deleting all documents: {e}")
            return False
    
    async def aadd_documents(
        self, 
        texts: List[str], 
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ) -> tuple[List[str], bool]:
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Extract source URL for potential caching after successful addition
        source_url = None
        if metadatas and len(metadatas) > 0:
            source_url = metadatas[0].get("source")
        
        print(f"Adding {len(texts)} documents to InMemory vector store (async)...")
        
        try:
            documents = [
                Document(id=doc_id, page_content=text, metadata=metadata)
                for doc_id, text, metadata in zip(ids, texts, metadatas)
            ]
            
            added_ids = await self.vector_store.aadd_documents(documents)
            
            # Save to cache if we have a source URL and successfully added documents
            if source_url and added_ids:
                try:
                    self.save_to_cache(source_url)
                    print(f"Cached vector store for future use: {source_url[:50]}...")
                except Exception as cache_error:
                    print(f"Warning: Failed to cache vector store: {cache_error}")
                    # Don't fail the whole operation if caching fails
            
            print(f"Successfully added {len(added_ids)} documents to InMemory vector store (async)")
            return added_ids, False  # cache_used is always False since caching is handled in DocumentEmbedder
            
        except Exception as e:
            print(f"Error adding documents to InMemory vector store: {e}")
            raise
    
    
    async def asimilarity_search(
        self, 
        query: str, 
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        try:
            results = await self.vector_store.asimilarity_search(
                query=query, 
                k=k
            )
            return results
            
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
    
    async def asimilarity_search_with_score(
        self, 
        query: str, 
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        try:
            results = await self.vector_store.asimilarity_search_with_score(
                query=query, 
                k=k
            )
            return results
            
        except Exception as e:
            print(f"Error during similarity search with score: {e}")
            return []
    
    async def adelete_documents(
        self, 
        ids: List[str]
    ) -> bool:
        try:
            await self.vector_store.adelete(ids=ids)
            print(f"Deleted {len(ids)} documents from InMemory vector store (async)")
            return True
            
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    async def aget_document_count(self) -> int:
        try:
            count = len(self.vector_store.store)
            return count
            
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    async def adelete_all_documents(self) -> bool:
        try:
            all_ids = list(self.vector_store.store.keys())
            if all_ids:
                await self.vector_store.adelete(ids=all_ids)
            
            print(f"Deleted all documents from InMemory vector store (async)")
            return True
            
        except Exception as e:
            print(f"Error deleting all documents: {e}")
            return False
    
    def as_retriever(self, **kwargs) -> Any:
        return self.vector_store.as_retriever(**kwargs)
    
    def dump_to_file(self, file_path: str) -> bool:
        try:
            self.vector_store.dump(file_path)
            print(f"Dumped vector store to: {file_path}")
            return True
        except Exception as e:
            print(f"Error dumping vector store: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: str, embedding_model: str = "text-embedding-3-small") -> 'InMemoryVectorStoreService':
        try:
            embedder = get_embedding_model(embedding_model)
            embeddings = LangChainEmbeddingWrapper(embedder)
            
            vector_store = InMemoryVectorStore.load(file_path, embedding=embeddings)
            
            service = cls.__new__(cls)
            service.embedding_model = embedding_model
            service.embeddings = embeddings
            service.vector_store = vector_store
            service.store_type = "inmemory"
            
            print(f"Loaded vector store from: {file_path}")
            return service
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise
    
    def get_temp_dump_path(self) -> str:
        cache_dir = Path("vector_store_cache")
        cache_dir.mkdir(exist_ok=True)
        temp_file = f"temp_vector_store_{uuid.uuid4().hex[:8]}.vs"
        return str(cache_dir / temp_file)
    
    def supports_caching(self) -> bool:
        return True
    
    def load_from_cache(self, document_url: str) -> bool:
        try:
            cached_path = self.cache_manager.get_cache_path(document_url)
            if cached_path:
                print(f"Loading cached vector store for: {document_url[:50]}...")
                
                self.vector_store = InMemoryVectorStore.load(cached_path, embedding=self.embeddings)
                
                print("Successfully loaded cached vector store")
                return True
            return False
        except Exception as e:
            print(f"Failed to load cached vector store: {e}")
            return False
    
    def save_to_cache(self, document_url: str) -> bool:
        try:
            temp_path = self.get_temp_dump_path()
            if self.dump_to_file(temp_path):
                success = self.cache_manager.cache_vector_store(document_url, temp_path)
                if success:
                    print("Successfully cached vector store for future use")
                return success
            return False
        except Exception as e:
            print(f"Failed to cache vector store: {e}")
            return False
    
    def has_cache(self, document_url: str) -> bool:
        return self.cache_manager.has_cached_store(document_url)
    
