from typing import List
from langchain_core.embeddings import Embeddings
from .base_embedder import BaseEmbedder


class LangChainEmbeddingWrapper(Embeddings):
    
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed_single(text)
