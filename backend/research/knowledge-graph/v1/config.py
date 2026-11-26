"""
Configuration classes for the Insurance Policy Knowledge Graph Builder
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class PolicyConfig:
    """Configuration for policy processing"""
    # Azure OCR settings
    azure_endpoint: str = field(default_factory=lambda: os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"))
    azure_key: str = field(default_factory=lambda: os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"))
    azure_model: str = "prebuilt-layout"
    azure_mode: str = "markdown"
    
    # OpenAI settings
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4"))
    openai_embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"))
    
    # Neo4j settings
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD"))
    
    # Processing settings
    chunk_size: int = 2000  # Smaller chunks for better granularity
    chunk_overlap: int = 200
    max_tokens_per_episode: int = 6000  # Graphiti token limit
    
    def validate(self):
        """Validate required configuration"""
        if not self.azure_endpoint or not self.azure_key:
            raise ValueError("Azure Document Intelligence credentials required")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required")
        if not self.neo4j_password:
            raise ValueError("Neo4j password required")