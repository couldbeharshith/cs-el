from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv
import os
load_dotenv("../../.env")

class Settings(BaseSettings):
    # API Configuration 
    PROJECT_NAME: str = "Intelligent RAG API"
    
    # Environment Configuration
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")  # production, development
    
    # Vector Store Configuration 
    DEFAULT_VECTOR_STORE: str = os.getenv("DEFAULT_VECTOR_STORE", "supabase")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # LLM Providers
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    
    # Gemini Configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    
    # Groq Configuration
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # LM Studio Configuration
    LMSTUDIO_API_KEY: str = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
    LMSTUDIO_MODEL: str = os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-4b")
    LMSTUDIO_BASE_URL: str = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    
    # RAG Server Configuration
    RAG_API_KEY: Optional[str] = os.getenv("RAG_API_KEY")

    # Azure Document Intelligence Configuration 
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: Optional[str] = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    AZURE_DOCUMENT_INTELLIGENCE_KEY: Optional[str] = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    # Supabase Configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY")
    SUPABASE_TABLE_NAME: str = os.getenv("SUPABASE_TABLE_NAME", "documents")
    SUPABASE_QUERY_NAME: str = os.getenv("SUPABASE_QUERY_NAME", "match_documents")
    ENABLE_REQUEST_LOGGING: bool = os.getenv("ENABLE_REQUEST_LOGGING", "true").lower() == "true"

    # Processing Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Caching Configuration
    ENABLE_CACHING: bool = True  # Enable/disable caching
    CACHE_MIN_CHUNKS: int = 0  # Only cache docs with >0 chunks


    class Config:
        env_file = ".env"

settings = Settings()
