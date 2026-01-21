from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client, Client
from app.services.vector_stores.base_vector_store import BaseVectorStore
from app.services.embedders.embedding_factory import get_embedding_model
from app.services.embedders.langchain_wrapper import LangChainEmbeddingWrapper
from typing import List, Dict, Optional, Any
from langchain_core.documents import Document
from app.config.settings import settings
import uuid

class SupabaseVectorStoreService(BaseVectorStore):
    def __init__(
        self, 
        supabase_url: str,
        supabase_key: str,
        embedding_model: str = "text-embedding-3-small",
        table_name: str = "documents",
        query_name: str = "match_documents"
    ):
    
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.embedding_model = embedding_model or getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small")
        self.table_name = table_name
        self.query_name = query_name
        
        self.supabase_client: Client = create_client(supabase_url, supabase_key)
        embedder = get_embedding_model(self.embedding_model)
        self.embeddings = LangChainEmbeddingWrapper(embedder)
        
        self.vector_store = SupabaseVectorStore(
            client=self.supabase_client,
            embedding=self.embeddings,
            table_name=table_name,
            query_name=query_name,
            chunk_size=settings.CHUNK_SIZE,
        )
        
        self.store_type = "supabase"
        
        self._verify_database_setup()
    
    def _verify_database_setup(self):
        try:
            result = self.supabase_client.table(self.table_name).select("id").limit(1).execute()
            print(f"Supabase table '{self.table_name}' is accessible")
        except Exception as e:
            print(f"Warning: Could not verify Supabase table '{self.table_name}': {e}")
            print("Make sure you have created the documents table with the pgvector extension")
    
    def add_documents(
        self, 
        texts: List[str], 
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ) -> tuple[List[str], bool]:
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        print(f"Adding {len(texts)} documents to Supabase...")
        
        try:
            documents = [
                Document(page_content=text, metadata=metadata)
                for text, metadata in zip(texts, metadatas)
            ]
            
            added_ids = self.vector_store.add_documents(documents, ids=ids)
            
            print(f"Successfully added {len(added_ids)} documents to Supabase")
            return added_ids, False  # cache_used is always False since caching is handled in DocumentEmbedder
            
        except Exception as e:
            print(f"Error adding documents to Supabase: {e}")
            raise e
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        search_kwargs = {"k": k}
        
        if filter:
            search_kwargs["filter"] = filter
        
        return self.vector_store.similarity_search_with_relevance_scores(query, **search_kwargs)
    
    async def asimilarity_search_with_score(
        self, 
        query: str, 
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        try:
            # Since Supabase doesn't support async operations, 
            # we just call the sync version
            results = self.similarity_search_with_score(
                query=query, 
                k=k,
                filter=filter
            )
            return results
            
        except Exception as e:
            print(f"Error during similarity search with score (async): {e}")
            return []
    
    def as_retriever(self, **kwargs) -> Any:
        return self.vector_store.as_retriever(**kwargs)
    
    def delete_documents(
        self, 
        ids: List[str]
    ) -> bool:
        try:
            result = self.supabase_client.table(self.table_name).delete().in_("id", ids).execute()
            
            if result.data:
                print(f"Deleted {len(result.data)} documents from Supabase")
                return True
            else:
                print("No documents were deleted (they may not exist)")
                return False
                
        except Exception as e:
            print(f"Error deleting documents from Supabase: {e}")
            return False
    
    def get_document_count(self) -> int:
        try:
            query = self.supabase_client.table(self.table_name).select("id", count="exact")
            
            result = query.execute()
            return result.count if result.count is not None else 0
            
        except Exception as e:
            print(f"Error getting document count from Supabase: {e}")
            return 0
    
    def delete_all_documents(self) -> bool:
        try:
            print(f"Deleting all documents from Supabase table: {self.table_name}")
            
            result = self.supabase_client.table(self.table_name).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            deleted_count = len(result.data) if result.data else 0
            print(f"Deleted all {deleted_count} documents from Supabase")
            
            return True
            
        except Exception as e:
            print(f"Error deleting all documents from Supabase: {e}")
            return False
    
    
    def supports_caching(self) -> bool:
        """Supabase supports caching by checking if documents exist in the database"""
        return True
    
    def has_cache(self, document_url: str) -> bool:
        """Check if documents for the given URL (via document_id hash) exist in Supabase"""
        try:
            # Generate document_id from URL hash (same logic as in rag.py)
            import hashlib
            document_id = hashlib.sha256(document_url.encode()).hexdigest()[:16]
            
            # Query for documents with this document_id
            result = self.supabase_client.table(self.table_name).select("id").eq("metadata->>document_id", document_id).limit(1).execute()
            
            has_cache = len(result.data) > 0
            if has_cache:
                print(f"Found cached documents for URL: {document_url[:50]}...")
            else:
                print(f"No cached documents found for URL: {document_url[:50]}...")
            
            return has_cache
            
        except Exception as e:
            print(f"Error checking cache for URL: {e}")
            return False
    
    def load_from_cache(self, document_url: str) -> bool:
        """For Supabase, loading from cache means the documents are already in the database"""
        try:
            has_cache = self.has_cache(document_url)
            if has_cache:
                print("Documents are already available in Supabase (cached)")
            return has_cache
            
        except Exception as e:
            print(f"Error loading from cache: {e}")
            return False
    
    
    
