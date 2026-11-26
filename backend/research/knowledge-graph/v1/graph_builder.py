"""
Knowledge Graph Builder Module
Uses Graphiti with HTTP fallback for Neo4j Aura
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json
import requests
import base64

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

from chunking import PolicyChunk

logger = logging.getLogger(__name__)


class HttpNeo4jClient:
    """HTTP client for Neo4j Aura when Bolt protocol fails"""
    
    def __init__(self, uri: str, username: str, password: str):
        print(f"DEBUG: Original URI: {uri}")  # Debug line
        # Extract HTTP endpoint from Neo4j URI
        if uri.startswith('neo4j+s://'):
            host = uri.replace('neo4j+s://', '')  # Remove prefix
            # For Neo4j Aura, try the correct HTTP API endpoint
            self.http_url = f"https://{host}/db/data/transaction/commit"
            # Alternative endpoint to try
            self.alt_http_url = f"https://{host}/db/neo4j/tx/commit"
        elif uri.startswith('neo4j://'):
            host = uri.replace('neo4j://', '')   # Remove prefix
            self.http_url = f"https://{host}/db/data/transaction/commit"
            self.alt_http_url = f"https://{host}/db/neo4j/tx/commit"
        elif uri.startswith('bolt+s://'):
            host = uri.replace('bolt+s://', '')  # Remove prefix
            self.http_url = f"https://{host}/db/data/transaction/commit"
            self.alt_http_url = f"https://{host}/db/neo4j/tx/commit"
        elif uri.startswith('bolt://'):
            host = uri.replace('bolt://', '')   # Remove prefix
            # For local Neo4j, assume HTTP is available
            self.http_url = f"http://{host}/db/neo4j/tx/commit"
            self.alt_http_url = None
        else:
            self.http_url = f"{uri}/db/data/transaction/commit"
            self.alt_http_url = None
        print(f"DEBUG: HTTP URL: {self.http_url}")  # Debug line
        
        # Create authorization header
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {encoded_credentials}"
        }
    
    def run_query(self, query: str, parameters: Dict = None) -> Dict:
        """Execute a Cypher query via HTTP"""
        payload = {
            "statements": [
                {
                    "statement": query,
                    "parameters": parameters or {}
                }
            ]
        }
        
        try:
            response = requests.post(
                self.http_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # If primary URL fails and we have an alternative, try it
            if hasattr(self, 'alt_http_url') and self.alt_http_url:
                try:
                    print(f"Primary URL failed, trying alternative: {self.alt_http_url}")
                    response = requests.post(
                        self.alt_http_url,
                        headers=self.headers,
                        json=payload
                    )
                    response.raise_for_status()
                    return response.json()
                except Exception as alt_e:
                    logger.error(f"Both HTTP endpoints failed. Primary: {e}, Alternative: {alt_e}")
                    raise e
            else:
                logger.error(f"HTTP query failed: {e}")
                raise


class PolicyGraphBuilder:
    """Build knowledge graph from policy documents using Graphiti with HTTP fallback"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 openai_api_key: str, openai_model: str = "gpt-4",
                 embedding_model: str = "text-embedding-ada-002"):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.embedding_model = embedding_model
        
        # HTTP client for direct Neo4j access
        self.http_client = HttpNeo4jClient(neo4j_uri, neo4j_user, neo4j_password)
        
        # Try to initialize Graphiti (may fail with Aura)
        self.graphiti = None
        self._graphiti_available = False
        self._initialized = False
    
    async def initialize(self):
        """Initialize graph database and Graphiti if possible"""
        if self._initialized:
            return
        
        try:
            # Try to initialize Graphiti
            llm_config = LLMConfig(
                api_key=self.openai_api_key,
                model=self.openai_model
            )
            
            embedder_config = OpenAIEmbedderConfig(
                api_key=self.openai_api_key,
                model=self.embedding_model
            )
            
            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=OpenAIClient(llm_config),
                embedder=OpenAIEmbedder(embedder_config)
            )
            
            await self.graphiti.build_indices_and_constraints()
            self._graphiti_available = True
            logger.info("Graphiti initialized successfully")
        except Exception as e:
            logger.warning(f"Graphiti initialization failed: {e}")
            logger.info("Falling back to HTTP API for Neo4j operations")
            self._graphiti_available = False
        
        # Initialize basic graph structure via HTTP
        self._create_basic_schema()
        self._initialized = True
    
    def _create_basic_schema(self):
        """Create basic schema via HTTP"""
        try:
            # Create indices
            schema_queries = [
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX episode_index IF NOT EXISTS FOR (ep:Episode) ON (ep.index)",
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT episode_id IF NOT EXISTS FOR (ep:Episode) REQUIRE ep.id IS UNIQUE"
            ]
            
            for query in schema_queries:
                self.http_client.run_query(query)
            
            logger.info("Basic schema created via HTTP")
        except Exception as e:
            logger.warning(f"Schema creation failed: {e}")
    
    async def add_policy_to_graph(self, chunks: List[PolicyChunk], policy_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add policy chunks to knowledge graph"""
        await self.initialize()
        
        results = {
            "episodes_added": 0,
            "entities_added": 0,
            "errors": []
        }
        
        if self._graphiti_available:
            # Use Graphiti for full processing
            results = await self._add_via_graphiti(chunks, policy_info)
        else:
            # Use HTTP API with simulated entity extraction
            results = self._add_via_http(chunks, policy_info)
        
        return results
    
    async def _add_via_graphiti(self, chunks: List[PolicyChunk], policy_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add chunks using Graphiti"""
        results = {"episodes_added": 0, "errors": []}
        
        for chunk in chunks:
            try:
                episode_content = self._prepare_episode_content(chunk, policy_info)
                episode_id = f"{policy_info.get('uin', 'policy')}_{chunk.index}_{datetime.now().timestamp()}"
                
                await self.graphiti.add_episode(
                    name=episode_id,
                    episode_body=episode_content,
                    source=EpisodeType.text,
                    source_description=f"{policy_info.get('title', 'Policy')} - Section {chunk.index}",
                    reference_time=datetime.now(timezone.utc)
                )
                
                results["episodes_added"] += 1
                logger.info(f"Added episode {episode_id} via Graphiti")
                
            except Exception as e:
                error_msg = f"Failed to add chunk {chunk.index}: {str(e)}"
                results["errors"].append(error_msg)
        
        return results
    
    def _add_via_http(self, chunks: List[PolicyChunk], policy_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add chunks using HTTP API"""
        results = {"episodes_added": 0, "entities_added": 0, "errors": []}
        
        for chunk in chunks:
            try:
                # Create Episode node
                episode_query = """
                CREATE (ep:Episode {
                    id: $episode_id,
                    index: $index,
                    content: $content,
                    section_title: $section_title,
                    policy_title: $policy_title,
                    policy_uin: $policy_uin,
                    created_at: datetime()
                })
                RETURN ep
                """
                
                episode_id = f"{policy_info.get('uin', 'policy')}_{chunk.index}"
                self.http_client.run_query(episode_query, {
                    "episode_id": episode_id,
                    "index": chunk.index,
                    "content": chunk.content[:1000],  # Truncate for storage
                    "section_title": chunk.section_title,
                    "policy_title": policy_info.get('title', 'Unknown'),
                    "policy_uin": policy_info.get('uin', 'Unknown')
                })
                
                # Create Entity nodes and relationships
                for entity_type, entities in chunk.entities.items():
                    for entity_name in entities:
                        try:
                            entity_query = """
                            MERGE (ent:Entity {name: $name, type: $type})
                            WITH ent
                            MATCH (ep:Episode {id: $episode_id})
                            MERGE (ep)-[:MENTIONS]->(ent)
                            RETURN ent, ep
                            """
                            
                            self.http_client.run_query(entity_query, {
                                "name": entity_name,
                                "type": entity_type,
                                "episode_id": episode_id
                            })
                            
                            results["entities_added"] += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to add entity {entity_name}: {e}")
                
                results["episodes_added"] += 1
                logger.info(f"Added episode {episode_id} via HTTP")
                
            except Exception as e:
                error_msg = f"Failed to add chunk {chunk.index}: {str(e)}"
                results["errors"].append(error_msg)
        
        return results
    
    def _prepare_episode_content(self, chunk: PolicyChunk, policy_info: Dict) -> str:
        """Prepare episode content with metadata"""
        metadata = {
            "policy": policy_info.get("title", "Unknown"),
            "uin": policy_info.get("uin", "Unknown"),
            "section": chunk.section_title,
            "entities": chunk.entities
        }
        
        return f"[METADATA: {json.dumps(metadata)}]\n\n{chunk.content}"
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Search the knowledge graph"""
        await self.initialize()
        
        if self._graphiti_available:
            try:
                results = await self.graphiti.search(query)
                return [
                    {
                        "fact": r.fact,
                        "uuid": str(r.uuid),
                        "valid_at": getattr(r, 'valid_at', None),
                        "invalid_at": getattr(r, 'invalid_at', None)
                    }
                    for r in results
                ]
            except Exception as e:
                logger.warning(f"Graphiti search failed: {e}")
        
        # Fallback to HTTP search
        return self._search_via_http(query)
    
    def _search_via_http(self, query: str) -> List[Dict[str, Any]]:
        """Search via HTTP API"""
        search_query = """
        MATCH (ep:Episode)-[:MENTIONS]->(ent:Entity)
        WHERE toLower(ep.content) CONTAINS toLower($query)
           OR toLower(ent.name) CONTAINS toLower($query)
           OR toLower(ep.section_title) CONTAINS toLower($query)
        RETURN ep.content as content, ep.section_title as section,
               collect(ent.name) as entities
        LIMIT 10
        """
        
        try:
            result = self.http_client.run_query(search_query, {"query": query})
            
            facts = []
            if result.get("results") and result["results"][0].get("data"):
                for record in result["results"][0]["data"]:
                    row = record["row"]
                    facts.append({
                        "fact": f"Section: {row[1]}\nContent: {row[0][:500]}...\nEntities: {', '.join(row[2])}",
                        "uuid": f"search_{len(facts)}",
                        "valid_at": None,
                        "invalid_at": None
                    })
            
            return facts
        except Exception as e:
            logger.error(f"HTTP search failed: {e}")
            return []
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics via HTTP"""
        try:
            stats_query = """
            MATCH (n)
            UNWIND labels(n) as label
            RETURN label, count(n) as count
            ORDER BY count DESC
            """
            
            result = self.http_client.run_query(stats_query)
            stats = {}
            
            if result.get("results") and result["results"][0].get("data"):
                for record in result["results"][0]["data"]:
                    row = record["row"]
                    stats[row[0]] = row[1]
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    async def close(self):
        """Close graph connection"""
        if self._graphiti_available and self.graphiti:
            try:
                await self.graphiti.close()
            except Exception as e:
                logger.warning(f"Error closing Graphiti: {e}")
        self._initialized = False