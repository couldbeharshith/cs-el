from typing import Optional, Dict, Any
from .base_embedder import BaseEmbedder
from .openai_embedder import OpenAIEmbedder


def get_embedding_model(
    name: str, 
    api_key: Optional[str] = None,
    **kwargs
) -> BaseEmbedder:
    name = name.lower().strip()
    
    if name in ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]:
        return OpenAIEmbedder(model=name, api_key=api_key)
    
    else:
        supported_models = [
            "text-embedding-3-small",
            "text-embedding-3-large", 
            "text-embedding-ada-002",
            "bge-m3"
        ]
        raise ValueError(
            f"Unsupported embedding model: {name}. "
            f"Supported models: {', '.join(supported_models)}"
        )
