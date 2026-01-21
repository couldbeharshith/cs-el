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
        raw_response = result.get("raw_response", {})
        debug_info = raw_response.get("debug_info", [])
        context_documents = debug_info[0].get("context_documents", []) if debug_info else []
        # print(context_documents)

        response = {
            "success": result.get("success", False),
            "content": result.get("answer", ""),
            "context_documents": context_documents,
            "query": query,
            "document_url": document_url,
            "chunks_retrieved": k
        }
        print(f"Query Tool Result: {response['success']}, Retrieved {len(context_documents)} context documents.")
        # print(result)
        return response
    return query_document
