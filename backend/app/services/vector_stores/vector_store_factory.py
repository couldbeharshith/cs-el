from app.services.vector_stores.supabase_vector_store import SupabaseVectorStoreService
from app.services.vector_stores.inmemory_vector_store import InMemoryVectorStoreService
from app.services.vector_stores.base_vector_store import BaseVectorStore
from app.config.settings import Settings

class VectorStoreFactory:
    
    _instances = {}
    
    @staticmethod
    def create_vector_store(settings: Settings) -> BaseVectorStore:
        vector_store_type = settings.DEFAULT_VECTOR_STORE.lower()

        # Build a cache key that changes when table/query/model changes to avoid stale instances
        if vector_store_type == "supabase":
            cache_key = f"{vector_store_type}:{settings.SUPABASE_TABLE_NAME}:{settings.SUPABASE_QUERY_NAME}:{settings.EMBEDDING_MODEL}"
        elif vector_store_type == "inmemory":
            cache_key = f"{vector_store_type}:{settings.EMBEDDING_MODEL}"
        else:
            cache_key = vector_store_type
        
        if cache_key in VectorStoreFactory._instances:
            return VectorStoreFactory._instances[cache_key]
        
        if vector_store_type == "supabase":
            instance = VectorStoreFactory._create_supabase_store(settings)
        elif vector_store_type == "inmemory":
            instance = VectorStoreFactory._create_inmemory_store(settings)
        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}. Supported types: 'supabase', 'inmemory'")
        
        VectorStoreFactory._instances[cache_key] = instance
        return instance
    
    @staticmethod
    def _create_supabase_store(settings: Settings) -> SupabaseVectorStoreService:
        if not settings.SUPABASE_URL:
            raise ValueError("SUPABASE_URL is required for Supabase vector store")
        
        supabase_key = settings.SUPABASE_SERVICE_KEY or settings.SUPABASE_ANON_KEY
        if not supabase_key:
            raise ValueError("Either SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY is required for Supabase vector store")
        
        return SupabaseVectorStoreService(
            supabase_url=settings.SUPABASE_URL,
            supabase_key=supabase_key,
            embedding_model=settings.EMBEDDING_MODEL,
            table_name=settings.SUPABASE_TABLE_NAME,
            query_name=settings.SUPABASE_QUERY_NAME
        )
    
    @staticmethod
    def _create_inmemory_store(settings: Settings) -> InMemoryVectorStoreService:
        return InMemoryVectorStoreService(
            embedding_model=settings.EMBEDDING_MODEL
        )
