"""
Centralized Configuration Management for Insurance RAG API

This module consolidates all configuration settings, environment variables,
and prompt management for the simplified API architecture.
"""

import os
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env.rag (in the rag_server directory)
env_path = Path(__file__).parent.parent / '.env.rag'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


class Config:
    """Centralized configuration management class"""
    
    # CORE API CONFIGURATION
    API_KEY: str = os.getenv("API_KEY", "")
    APP_NAME: str = os.getenv("APP_NAME", "Insurance RAG API")
    APP_VERSION: str = os.getenv("APP_VERSION", "4.0.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "7860"))
    
    # LLM CONFIGURATION
    
    # Multi-key configuration
    GEMINI_API_KEY_1: str = os.getenv("GEMINI_API_KEY_1", os.getenv("GEMINI_API_KEY", "NO_API_KEY"))
    GEMINI_API_KEY_2: str = os.getenv("GEMINI_API_KEY_2", "")
    GEMINI_API_KEY_3: str = os.getenv("GEMINI_API_KEY_3", "")
    GEMINI_API_KEY_4: str = os.getenv("GEMINI_API_KEY_4", "")
    GEMINI_API_KEY_5: str = os.getenv("GEMINI_API_KEY_5", "")
    GEMINI_API_KEY_6: str = os.getenv("GEMINI_API_KEY_6", "")
    GEMINI_API_KEY_7: str = os.getenv("GEMINI_API_KEY_7", "")
    GEMINI_API_KEY_8: str = os.getenv("GEMINI_API_KEY_8", "")
    GEMINI_API_KEY_9: str = os.getenv("GEMINI_API_KEY_9", "")
    GEMINI_API_KEY_10: str = os.getenv("GEMINI_API_KEY_10", "")
    GEMINI_API_KEY_11: str = os.getenv("GEMINI_API_KEY_11", "")
    GEMINI_API_KEY_12: str = os.getenv("GEMINI_API_KEY_12", "")
    GEMINI_API_KEY_13: str = os.getenv("GEMINI_API_KEY_13", "")
    GEMINI_API_KEY_14: str = os.getenv("GEMINI_API_KEY_14", "")
    GEMINI_API_KEY_15: str = os.getenv("GEMINI_API_KEY_15", "")
    GEMINI_API_KEY_16: str = os.getenv("GEMINI_API_KEY_16", "")
    GEMINI_API_KEY_17: str = os.getenv("GEMINI_API_KEY_17", "")
    GEMINI_API_KEY_18: str = os.getenv("GEMINI_API_KEY_18", "")
    GEMINI_API_KEY_19: str = os.getenv("GEMINI_API_KEY_19", "")
    GEMINI_API_KEY_20: str = os.getenv("GEMINI_API_KEY_20", "")
    GEMINI_API_KEY_21: str = os.getenv("GEMINI_API_KEY_21", "")
    GEMINI_API_KEY_22: str = os.getenv("GEMINI_API_KEY_22", "")
    GEMINI_API_KEY_23: str = os.getenv("GEMINI_API_KEY_23", "")
    GEMINI_API_KEY_24: str = os.getenv("GEMINI_API_KEY_24", "")
    GEMINI_API_KEY_25: str = os.getenv("GEMINI_API_KEY_25", "")
    
    GROQ_API_KEY_1: str = os.getenv("GROQ_API_KEY_1", "")
    GROQ_API_KEY_2: str = os.getenv("GROQ_API_KEY_2", "")
    GROQ_API_KEY_3: str = os.getenv("GROQ_API_KEY_3", "")
    GROQ_API_KEY_4: str = os.getenv("GROQ_API_KEY_4", "")
    GROQ_API_KEY_5: str = os.getenv("GROQ_API_KEY_5", "")
    GROQ_API_KEY_6: str = os.getenv("GROQ_API_KEY_6", "")
    GROQ_API_KEY_7: str = os.getenv("GROQ_API_KEY_7", "")
    GROQ_API_KEY_8: str = os.getenv("GROQ_API_KEY_8", "")
    GROQ_API_KEY_9: str = os.getenv("GROQ_API_KEY_9", "")
    GROQ_API_KEY_10: str = os.getenv("GROQ_API_KEY_10", "")
    GROQ_API_KEY_11: str = os.getenv("GROQ_API_KEY_11", "")
    GROQ_API_KEY_12: str = os.getenv("GROQ_API_KEY_12", "")
    GROQ_API_KEY_13: str = os.getenv("GROQ_API_KEY_13", "")
    GROQ_API_KEY_14: str = os.getenv("GROQ_API_KEY_14", "")
    GROQ_API_KEY_15: str = os.getenv("GROQ_API_KEY_15", "")
    
    PRIMARY_LLM_SERVICE: str = os.getenv("PRIMARY_LLM_SERVICE", "gemini").lower()
    
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # LLM parameters
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))
    GEMINI_TOP_P: float = float(os.getenv("GEMINI_TOP_P", "0.8"))
    
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "2048"))
    GROQ_TOP_P: float = float(os.getenv("GROQ_TOP_P", "0.8"))
    
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "insurance-rag")
    PINECONE_HOST_URL: str = os.getenv("PINECONE_HOST_URL", "")
    PINECONE_MAIN_NAMESPACE: str = os.getenv("PINECONE_MAIN_NAMESPACE", "documents")
    
    # Pinecone reranker API keys (dedicated for reranking operations)
    PINECONE_RERANKER_API_KEY_1: str = os.getenv("PINECONE_RERANKER_API_KEY_1", "")
    PINECONE_RERANKER_API_KEY_2: str = os.getenv("PINECONE_RERANKER_API_KEY_2", "")
    PINECONE_RERANKER_API_KEY_3: str = os.getenv("PINECONE_RERANKER_API_KEY_3", "")
    PINECONE_RERANKER_API_KEY_4: str = os.getenv("PINECONE_RERANKER_API_KEY_4", "")
    PINECONE_RERANKER_API_KEY_5: str = os.getenv("PINECONE_RERANKER_API_KEY_5", "")
    PINECONE_RERANKER_API_KEY_6: str = os.getenv("PINECONE_RERANKER_API_KEY_6", "")
    PINECONE_RERANKER_API_KEY_7: str = os.getenv("PINECONE_RERANKER_API_KEY_7", "")
    PINECONE_RERANKER_API_KEY_8: str = os.getenv("PINECONE_RERANKER_API_KEY_8", "")
    PINECONE_RERANKER_API_KEY_9: str = os.getenv("PINECONE_RERANKER_API_KEY_9", "")
    PINECONE_RERANKER_API_KEY_10: str = os.getenv("PINECONE_RERANKER_API_KEY_10", "")
    PINECONE_RERANKER_API_KEY_11: str = os.getenv("PINECONE_RERANKER_API_KEY_11", "")
    
    # Pinecone search and reranking configuration
    PINECONE_RERANKING_ENABLED: bool = os.getenv("PINECONE_RERANKING_ENABLED", "false").lower() == "true"
    PINECONE_TOP_K: int = int(os.getenv("PINECONE_TOP_K", "15"))
    PINECONE_RERANK_TOP_N: int = int(os.getenv("PINECONE_RERANK_TOP_N", "4"))
    PINECONE_RERANK_MODEL: str = os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3")
    
    # COHERE CONFIGURATION
    COHERE_API_KEY_1: str = os.getenv("COHERE_API_KEY_1", "")
    COHERE_API_KEY_2: str = os.getenv("COHERE_API_KEY_2", "")
    COHERE_API_KEY_3: str = os.getenv("COHERE_API_KEY_3", "")
    COHERE_API_KEY_4: str = os.getenv("COHERE_API_KEY_4", "")
    COHERE_API_KEY_5: str = os.getenv("COHERE_API_KEY_5", "")
    COHERE_API_KEY_6: str = os.getenv("COHERE_API_KEY_6", "")
    COHERE_API_KEY_7: str = os.getenv("COHERE_API_KEY_7", "")
    COHERE_API_KEY_8: str = os.getenv("COHERE_API_KEY_8", "")
    COHERE_API_KEY_9: str = os.getenv("COHERE_API_KEY_9", "")
    COHERE_API_KEY_10: str = os.getenv("COHERE_API_KEY_10", "")
    COHERE_API_KEY_11: str = os.getenv("COHERE_API_KEY_11", "")
    COHERE_API_KEY_12: str = os.getenv("COHERE_API_KEY_12", "")
    COHERE_API_KEY_13: str = os.getenv("COHERE_API_KEY_13", "")
    COHERE_RERANK_MODEL: str = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    
    # RERANKER SERVICE CONFIGURATION
    PRIMARY_RERANKER_SERVICE: str = os.getenv("PRIMARY_RERANKER_SERVICE", "pinecone")
    SECONDARY_RERANKER_SERVICE: str = os.getenv("SECONDARY_RERANKER_SERVICE", "cohere")
    
    # RELEVANCE CHECK CONFIGURATION
    RELEVANCE_CHECK_ENABLED: bool = os.getenv("RELEVANCE_CHECK_ENABLED", "true").lower() == "true"
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.255")) 
    
    # PLATFORM CONFIGURATION
    PLATFORM: str = os.getenv("PLATFORM", "local")  # vercel, railway, huggingface, or local
    
    # PERFORMANCE CONFIGURATION
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "25"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "200"))  # Increased to 200 seconds
    MAX_DOCUMENT_SIZE_MB: int = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "1024"))  # Increased to 1GB
    MAX_QUESTIONS_PER_REQUEST: int = int(os.getenv("MAX_QUESTIONS_PER_REQUEST", "50"))
    MAX_QUESTION_LENGTH: int = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))
    MAX_URL_LENGTH: int = int(os.getenv("MAX_URL_LENGTH", "2048"))
    
    # Direct Gemini upload threshold (files smaller than this will be uploaded directly to Gemini)
    DIRECT_GEMINI_UPLOAD_THRESHOLD_MB: float = float(os.getenv("DIRECT_GEMINI_UPLOAD_THRESHOLD_MB", "0.5"))
    
    # TIKTOKEN CONFIGURATION
    TIKTOKEN_CHUNK_SIZE: int = int(os.getenv("TIKTOKEN_CHUNK_SIZE", "400"))
    TIKTOKEN_CHUNK_OVERLAP: int = int(os.getenv("TIKTOKEN_CHUNK_OVERLAP", "50"))
    TIKTOKEN_ENCODING: str = os.getenv("TIKTOKEN_ENCODING", "cl100k_base")
    
    # PYMUPDF CONFIGURATION
    USE_MARKDOWN_EXTRACTION: bool = os.getenv("USE_MARKDOWN_EXTRACTION", "true").lower() == "true"
    
    # LOGGING CONFIGURATION
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # RETRY CONFIGURATION
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    
    # Security headers
    SECURITY_HEADERS: dict = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY", 
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    }
    
    # PROMPT MANAGEMENT
    _prompts: Optional[Dict[str, Any]] = None
    _prompts_file_path: str = "app/prompts.json"
    
    @classmethod
    def load_prompts(cls) -> Dict[str, Any]:
        """Load prompts from JSON file with caching."""
        if cls._prompts is None:
            try:
                prompts_path = Path(cls._prompts_file_path)
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    cls._prompts = json.load(f)
                logger.info(f"Loaded prompts from {cls._prompts_file_path}")
            except Exception as e:
                logger.error(f"Failed to load prompts from {cls._prompts_file_path}: {e}")
                raise RuntimeError(f"Critical error: Cannot load prompts from {cls._prompts_file_path}: {e}")
        
        return cls._prompts
    
    @classmethod
    def get_prompt(cls, prompt_name: str, **kwargs) -> str:
        """
        Get a prompt template and format it with provided variables.
        
        Args:
            prompt_name: Name of the prompt template
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted prompt string
        """
        prompts = cls.load_prompts()
        
        if prompt_name not in prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found in prompts.json")
        
        prompt_config = prompts[prompt_name]
        template = prompt_config["template"]
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Missing variable {e} for prompt '{prompt_name}'")
        except Exception as e:
            raise RuntimeError(f"Error formatting prompt '{prompt_name}': {e}")
    
    @classmethod
    def get_prompt_config(cls, prompt_name: str) -> Dict[str, Any]:
        """
        Get the full configuration for a prompt.
        
        Args:
            prompt_name: Name of the prompt
            
        Returns:
            Prompt configuration dictionary
        """
        prompts = cls.load_prompts()
        if prompt_name not in prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found in prompts.json")
        return prompts[prompt_name]
    
    @classmethod
    def reload_prompts(cls) -> None:
        """Force reload prompts from file."""
        cls._prompts = None
        cls.load_prompts()
        logger.info("Prompts reloaded from file")
    
    @classmethod
    def load_api_keys(cls) -> tuple[List[str], List[str]]:
        """
        Load Gemini and Groq API keys from environment variables.
        
        Returns:
            Tuple of (gemini_keys, groq_keys) lists
        """
        gemini_keys = []
        groq_keys = []
        
        # Load Gemini keys
        for i in range(1, 26):  # GEMINI_API_KEY_1 through GEMINI_API_KEY_25
            key = getattr(cls, f"GEMINI_API_KEY_{i}")
            if key and key != "NO_API_KEY":
                gemini_keys.append(key)
        
        # Load Groq keys
        for i in range(1, 16):  # GROQ_API_KEY_1 through GROQ_API_KEY_15
            key = getattr(cls, f"GROQ_API_KEY_{i}")
            if key:
                groq_keys.append(key)
        
        return gemini_keys, groq_keys
    
    @classmethod
    def get_gemini_api_keys(cls) -> List[str]:
        """
        Get list of Gemini API keys.
        
        Returns:
            List of Gemini API keys
        """
        gemini_keys, _ = cls.load_api_keys()
        return gemini_keys
    
    @classmethod
    def get_groq_api_keys(cls) -> List[str]:
        """
        Get list of Groq API keys.
        
        Returns:
            List of Groq API keys
        """
        _, groq_keys = cls.load_api_keys()
        return groq_keys
    
    @classmethod
    def get_pinecone_reranker_keys(cls) -> List[str]:
        """
        Get list of Pinecone reranker API keys.
        
        Returns:
            List of Pinecone reranker API keys
        """
        reranker_keys = []
        
        # Load Pinecone reranker keys
        for i in range(1, 12):  # PINECONE_RERANKER_API_KEY_1 through PINECONE_RERANKER_API_KEY_11
            key = getattr(cls, f"PINECONE_RERANKER_API_KEY_{i}")
            if key:
                reranker_keys.append(key)
        
        return reranker_keys
    
    @classmethod
    def get_cohere_api_keys(cls) -> List[str]:
        """
        Get list of Cohere API keys.
        
        Returns:
            List of Cohere API keys
        """
        cohere_keys = []
        
        # Load Cohere API keys
        for i in range(1, 14):  # COHERE_API_KEY_1 through COHERE_API_KEY_13
            key = getattr(cls, f"COHERE_API_KEY_{i}")
            if key:
                cohere_keys.append(key)
        
        return cohere_keys
    
    @classmethod
    def get_reranker_service_priority(cls) -> tuple[str, str]:
        """
        Get primary and secondary reranker service based on configuration.
        
        Returns:
            Tuple of (primary_service, secondary_service)
        """
        primary = cls.PRIMARY_RERANKER_SERVICE.lower()
        secondary = cls.SECONDARY_RERANKER_SERVICE.lower()
        
        # Validate services
        valid_services = ["pinecone", "cohere"]
        if primary not in valid_services:
            logger.warning(f"Invalid PRIMARY_RERANKER_SERVICE '{primary}', defaulting to pinecone")
            primary = "pinecone"
        if secondary not in valid_services:
            logger.warning(f"Invalid SECONDARY_RERANKER_SERVICE '{secondary}', defaulting to cohere")
            secondary = "cohere"
        
        return primary, secondary
    
    @classmethod
    def get_reranker_keys_with_fallback(cls) -> tuple[List[str], List[str], str, str]:
        """
        Get reranker API keys with fallback logic.
        Returns primary service keys first, then secondary service keys.
        
        Returns:
            Tuple of (primary_keys, secondary_keys, primary_service, secondary_service)
        """
        primary_service, secondary_service = cls.get_reranker_service_priority()
        
        # Get keys for each service
        if primary_service == "pinecone":
            primary_keys = cls.get_pinecone_reranker_keys()
        else:  # cohere
            primary_keys = cls.get_cohere_api_keys()
        
        if secondary_service == "pinecone":
            secondary_keys = cls.get_pinecone_reranker_keys()
        else:  # cohere
            secondary_keys = cls.get_cohere_api_keys()
        
        return primary_keys, secondary_keys, primary_service, secondary_service
    
    @classmethod
    def get_llm_service_priority(cls) -> tuple[str, str]:
        """
        Get primary and secondary LLM service based on configuration.
        
        Returns:
            Tuple of (primary_service, secondary_service)
        """
        primary = cls.PRIMARY_LLM_SERVICE
        if primary == "gemini":
            return "gemini", "groq"
        elif primary == "groq":
            return "groq", "gemini"
        else:
            logger.warning(f"Invalid PRIMARY_LLM_SERVICE '{primary}', defaulting to gemini")
            return "gemini", "groq"
    
    @classmethod
    def validate_required_config(cls) -> List[str]:
        """
        Validate that all required configuration values are set.
        
        Returns:
            List of missing configuration keys
        """
        required_configs = [
            ("API_KEY", cls.API_KEY),
            ("PINECONE_API_KEY", cls.PINECONE_API_KEY),
            ("PINECONE_HOST_URL", cls.PINECONE_HOST_URL),
        ]
        
        missing_configs = []
        for config_name, config_value in required_configs:
            if not config_value:
                missing_configs.append(config_name)
        
        # Check if we have at least one API key for each service
        gemini_keys, groq_keys = cls.load_api_keys()
        if not gemini_keys:
            missing_configs.append("GEMINI_API_KEYS (at least one required)")
        if not groq_keys:
            missing_configs.append("GROQ_API_KEYS (at least one required)")
        
        return missing_configs
    
    @classmethod
    def validate_tiktoken_config(cls) -> List[str]:
        """
        Validate TikToken and PyMuPDF configuration parameters.
        
        Returns:
            List of validation errors
        """
        validation_errors = []
        
        # Validate TIKTOKEN_CHUNK_SIZE
        if cls.TIKTOKEN_CHUNK_SIZE <= 0:
            validation_errors.append("TIKTOKEN_CHUNK_SIZE must be a positive integer")
        
        # Validate TIKTOKEN_CHUNK_OVERLAP
        if cls.TIKTOKEN_CHUNK_OVERLAP < 0:
            validation_errors.append("TIKTOKEN_CHUNK_OVERLAP must be a non-negative integer")
        
        if cls.TIKTOKEN_CHUNK_OVERLAP >= cls.TIKTOKEN_CHUNK_SIZE:
            validation_errors.append("TIKTOKEN_CHUNK_OVERLAP must be less than TIKTOKEN_CHUNK_SIZE")
        
        # Validate TIKTOKEN_ENCODING
        valid_encodings = ["cl100k_base", "p50k_base", "r50k_base", "p50k_edit"]
        if cls.TIKTOKEN_ENCODING not in valid_encodings:
            validation_errors.append(f"TIKTOKEN_ENCODING must be one of: {', '.join(valid_encodings)}")
        

        
        return validation_errors
    
    @classmethod
    def get_config_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of current configuration (excluding sensitive data).
        
        Returns:
            Configuration summary dictionary
        """
        gemini_keys, groq_keys = cls.load_api_keys()
        primary_keys, secondary_keys = cls.get_pinecone_reranker_keys(), cls.get_cohere_api_keys()
        primary_service, secondary_service = cls.get_llm_service_priority()
        
        return {
            "app_name": cls.APP_NAME,
            "app_version": cls.APP_VERSION,
            "platform": cls.PLATFORM,
            "primary_llm_service": primary_service,
            "secondary_llm_service": secondary_service,
            "gemini_model": cls.GEMINI_MODEL,
            "groq_model": cls.GROQ_MODEL,
            "gemini_keys_count": len(gemini_keys),
            "groq_keys_count": len(groq_keys),
            "pinecone_index": cls.PINECONE_INDEX_NAME,
            "max_concurrent_requests": cls.MAX_CONCURRENT_REQUESTS,
            "max_document_size_mb": cls.MAX_DOCUMENT_SIZE_MB,
            "log_level": cls.LOG_LEVEL,
            "debug": cls.DEBUG,
            "tiktoken_chunk_size": cls.TIKTOKEN_CHUNK_SIZE,
            "tiktoken_chunk_overlap": cls.TIKTOKEN_CHUNK_OVERLAP,
            "tiktoken_encoding": cls.TIKTOKEN_ENCODING,
            "use_markdown_extraction": cls.USE_MARKDOWN_EXTRACTION,
            "relevance_check_enabled": cls.RELEVANCE_CHECK_ENABLED,
            "relevance_threshold": cls.RELEVANCE_THRESHOLD,
            "pinecone_reranking_enabled": cls.PINECONE_RERANKING_ENABLED,
            "pinecone_top_k": cls.PINECONE_TOP_K,
            "pinecone_rerank_top_n": cls.PINECONE_RERANK_TOP_N,
            "pinecone_rerank_model": cls.PINECONE_RERANK_MODEL,
            "pinecone_rerank_keys_count" : len(primary_keys),
            "cohere_keys_count": len(secondary_keys),
            "cohere_reranker": cls.COHERE_RERANK_MODEL,
            "direct_gemini_upload_threshold_mb": cls.DIRECT_GEMINI_UPLOAD_THRESHOLD_MB
        }
    
    @classmethod
    def get_document_hash(cls, content: bytes) -> str:
        """
        Generate a fast hash of document content for namespace generation.
        
        Args:
            content: Document content as bytes
            
        Returns:
            First 16 characters of SHA-256 hash
        """
        if not content:
            return "empty_doc"
        
        # Use SHA-256 for consistent hashing
        hash_obj = hashlib.sha256(content)
        full_hash = hash_obj.hexdigest()
        
        # Return first 16 characters for namespace
        return full_hash[:16]
    
    @classmethod
    def get_document_namespace(cls, content: bytes) -> str:
        """
        Generate a Pinecone namespace for document content.
        
        Args:
            content: Document content as bytes
            
        Returns:
            Namespace string in format 'doc_{hash}'
        """
        doc_hash = cls.get_document_hash(content)
        return f"doc_{doc_hash}"
    
    @classmethod
    def is_serverless(cls) -> bool:
        """Check if running in serverless environment."""
        return cls.PLATFORM.lower() in ["vercel", "huggingface"]


# Create global config instance for easy access
config = Config()