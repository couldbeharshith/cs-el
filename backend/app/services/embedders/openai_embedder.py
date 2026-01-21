from typing import List, Union
from langchain_openai import OpenAIEmbeddings
from .base_embedder import BaseEmbedder
import os
from app.config.settings import settings


class OpenAIEmbedder(BaseEmbedder):
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self.model = model
        self.api_key = api_key or settings.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=self.api_key
        )
        
        self._dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 1536,
        }
    
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.embeddings.embed_documents(texts)
        return embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimensions.get(self.model, 1536)
    
    @property
    def model_name(self) -> str:
        return self.model
