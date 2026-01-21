import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv

from config import PolicyConfig
from chunking import PolicyChunker, PolicyEntityExtractor
from query_api_client import AuraQueryAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ingest")


def read_extracted_markdown(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def create_schema(client: AuraQueryAPI):
    schema_queries = [
        "CREATE INDEX episode_id IF NOT EXISTS FOR (e:Episode) ON (e.id)",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (ent:Entity) ON (ent.name)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (ent:Entity) ON (ent.type)"
    ]
    for q in schema_queries:
        client.execute(q)


def clear_previous_policy(client: AuraQueryAPI):
    # Only clear Episodes and Entities created by this ingestion (identified by source='extracted_markdown')
    client.execute(
        """
        MATCH (e:Episode {source: $src}) DETACH DELETE e
        """,
        {"src": "extracted_markdown"},
    )
    client.execute(
        """
        MATCH (ent:Entity {source: $src}) DETACH DELETE ent
        """,
        {"src": "extracted_markdown"},
    )


def ingest_chunks(client: AuraQueryAPI, content: str, policy_info: Dict[str, Any]) -> Dict[str, int]:
    cfg = PolicyConfig()
    chunker = PolicyChunker(cfg)
    extractor = PolicyEntityExtractor()

    chunks = chunker.chunk_document(content, policy_info)

    episodes = 0
    edges = 0
    entities_cache = set()  # (name,type)

    for chunk in chunks:
        chunk = extractor.extract_entities(chunk)
        episode_id = f"{policy_info.get('uin','UNKNOWN')}_{chunk.index}"

        # Create or reuse Episode (idempotent)
        r1 = client.execute(
            """
            MERGE (ep:Episode {id: $id})
            ON CREATE SET ep.index = $idx,
                          ep.section_title = $title,
                          ep.content = $content,
                          ep.policy_title = $policy_title,
                          ep.policy_uin = $policy_uin,
                          ep.source = 'extracted_markdown',
                          ep.created_at = datetime()
            ON MATCH SET  ep.section_title = coalesce($title, ep.section_title),
                          ep.content = coalesce($content, ep.content)
            """,
            {
                "id": episode_id,
                "idx": chunk.index,
                "title": chunk.section_title or f"Section {chunk.index}",
                "content": (chunk.content or "")[:1200],
                "policy_title": policy_info.get("title", "Unknown"),
                "policy_uin": policy_info.get("uin", "Unknown"),
            },
        )
        if r1.success:
            episodes += 1

        # Entities + relationships
        for etype, names in chunk.entities.items():
            for name in names:
                key = (name, etype)
                if key not in entities_cache:
                    client.execute(
                        """
                        MERGE (ent:Entity {name: $name, type: $type})
                        ON CREATE SET ent.source = 'extracted_markdown', ent.created_at = datetime()
                        """,
                        {"name": name, "type": etype}
                    )
                    entities_cache.add(key)
                # relation
                rrel = client.execute(
                    """
                    MATCH (ep:Episode {id: $id})
                    MATCH (ent:Entity {name: $name, type: $type})
                    MERGE (ep)-[:MENTIONS]->(ent)
                    """,
                    {"id": episode_id, "name": name, "type": etype}
                )
                if rrel.success:
                    edges += 1

    return {"episodes": episodes, "relationships": edges, "unique_entities": len(entities_cache)}


def get_counts(client: AuraQueryAPI) -> Dict[str, int]:
    nodes = client.execute("MATCH (n) RETURN count(n) as c")
    rels = client.execute("MATCH ()-[r]->() RETURN count(r) as c")
    n = nodes.values[0][0] if nodes.success and nodes.values else 0
    r = rels.values[0][0] if rels.success and rels.values else 0
    return {"nodes": n, "relationships": r}


def main():
    load_dotenv()
    cfg = PolicyConfig()

    client = AuraQueryAPI(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password)
    print(f"Using Aura Query API endpoint: {client.url}")
    assert client.ping(), "Failed to connect to Aura Query API. Check credentials or IP allowlist."

    # Create schema if needed
    create_schema(client)

    # Read markdown
    content = read_extracted_markdown("extracted_policy.md")

    # Optionally clear previous ingestion from markdown
    if os.getenv("CLEAR_BEFORE_INGEST", "false").lower() in {"1", "true", "yes"}:
        print("CLEAR_BEFORE_INGEST is true — removing previous 'extracted_markdown' nodes...")
        clear_previous_policy(client)
    else:
        print("Skipping clear step (set CLEAR_BEFORE_INGEST=true to enable). Using MERGE for idempotency.")

    # Policy metadata (adjustable)
    policy_info = {
        "title": "OICL Happy Family Floater Policy 2021",
        "uin": "OICHLIP22010V042223",
    }

    summary = ingest_chunks(client, content, policy_info)
    print(f"Ingestion summary: {summary}")

    counts = get_counts(client)
    print(f"Current DB counts: Nodes ({counts['nodes']}), Relationships ({counts['relationships']})")
    exp_nodes = os.getenv("EXPECTED_NODES")
    exp_rels = os.getenv("EXPECTED_RELATIONSHIPS")
    if exp_nodes and exp_rels:
        try:
            en = int(exp_nodes)
            er = int(exp_rels)
            status = "MATCH" if (counts["nodes"] == en and counts["relationships"] == er) else "DIFFERS"
            print(f"Expected totals: Nodes ({en}), Relationships ({er}) -> {status}")
        except ValueError:
            pass

    # Persist a minimal report
    with open("INGEST_REPORT.md", "w", encoding="utf-8") as f:
        f.write(f"""
# Ingestion Report

- Source file: extracted_policy.md
- Policy: {policy_info['title']} ({policy_info['uin']})
- Episodes created this run: {summary['episodes']}
- Unique entities created this run: {summary['unique_entities']}
- Relationships created this run: {summary['relationships']}
- Database totals now: Nodes ({counts['nodes']}), Relationships ({counts['relationships']})
- Timestamp: {datetime.utcnow().isoformat()}Z

## Environment

- CLEAR_BEFORE_INGEST: {os.getenv('CLEAR_BEFORE_INGEST', 'false')}
- EXPECTED_NODES: {os.getenv('EXPECTED_NODES', '')}
- EXPECTED_RELATIONSHIPS: {os.getenv('EXPECTED_RELATIONSHIPS', '')}

## Quick checks

- Episodes
```
MATCH (e:Episode) RETURN count(e) as episodes
```
- Entities
```
MATCH (ent:Entity) RETURN count(ent) as entities
```
- Sample
```
MATCH (e:Episode)-[:MENTIONS]->(ent:Entity)
RETURN e.section_title as section, ent.type as type, ent.name as name
LIMIT 10
```
""")

    print("Report written to INGEST_REPORT.md")


if __name__ == "__main__":
    main()
