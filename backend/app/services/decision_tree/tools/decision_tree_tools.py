import json
import os
import hashlib
from typing import Dict, Any
from agno.tools import tool


def create_query_tool(rag_helper):
    @tool(description="Query the insurance document using RAG to get relevant information")
    async def query_document(
        query: str,
        document_url: str,
        k: int = 8
    ) -> Dict[str, Any]:
        result = await rag_helper.query_document(
            query=query,
            document_url=document_url,
            k=k
        )
        
        return {
            "success": result.get("success", False),
            "content": result.get("answer", ""),
            "query": query,
            "document_url": document_url,
            "chunks_retrieved": k
        }
    return query_document


def create_save_tool(trees_folder, document_url, timestamp):
    @tool(description="Save decision tree JSON to the trees folder")
    async def save_decision_tree(
        topic: str,
        tree_json: str
    ) -> Dict[str, Any]:
        if not topic or not tree_json:
            return {"success": False, "error": "Missing topic or tree_json"}
        
        try:
            tree_data = json.loads(tree_json)
            
            # Generate hash of document URL
            url_hash = hashlib.sha256(document_url.encode()).hexdigest()[:16]
            
            # Create path: trees_folder/url_hash/timestamp/topic/
            base_path = os.path.join(trees_folder, url_hash, timestamp)
            topic_folder = os.path.join(base_path, topic)
            os.makedirs(topic_folder, exist_ok=True)
            
            file_path = os.path.join(topic_folder, f"{topic}.json")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(tree_data, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "file_path": file_path,
                "topic": topic,
                "nodes_count": len(tree_data.get("nodes", []))
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON: {str(e)}",
                "topic": topic
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "topic": topic
            }
    return save_decision_tree