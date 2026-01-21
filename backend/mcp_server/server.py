from typing import Union, List, Dict, Any, Optional
from fastmcp import FastMCP
from mcp_server.config.mcp_settings import MCP_SERVER_PORT
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("rag-server")

# RAG Server Configuration
RAG_SERVER_URL = "http://localhost:7860/api/v1/hackrx/run"
RAG_SERVER_TIMEOUT = 300  # 5 minutes timeout
RAG_API_KEY = os.getenv("RAG_API_KEY")


@mcp.tool(description="Process a document from URL and answer questions using the Insurance RAG API. Supports PDF, DOCX, XLSX, PPTX, TXT, images, audio, and video files.")
async def process_document_rag(
    document_url: str,
    questions: Union[str, List[str]],
):
    """
    Process a document using the Insurance RAG API.
    
    Args:
        document_url: URL to the document (https/http, 10-2048 chars)
        questions: Single question or list of questions (1-50 items, each 3-1000 chars)
    
    Returns:
        Dictionary with answers list or error information
    """
    if isinstance(questions, str):
        questions = [questions]
    
    if not RAG_API_KEY:
        return {
            "success": False,
            "error": "RAG_API_KEY not configured in environment"
        }
    
    try:
        # Prepare request
        rag_request = {
            "documents": document_url,
            "questions": questions
        }
        
        # Call RAG server
        async with httpx.AsyncClient(timeout=RAG_SERVER_TIMEOUT) as client:
            response = await client.post(
                RAG_SERVER_URL,
                json=rag_request,
                headers={
                    "Authorization": f"Bearer {RAG_API_KEY}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"RAG server returned status {response.status_code}",
                    "details": response.text
                }
            
            result = response.json()
            return {
                "success": True,
                "answers": result.get("answers", []),
                "document_url": document_url
            }
    
    except httpx.TimeoutException:
        return {
            "success": False,
            "error": "RAG server request timed out (5 minute limit)"
        }
    except httpx.RequestError as e:
        return {
            "success": False,
            "error": f"Failed to connect to RAG server: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

def run_server(port: int = None):
    mcp.run(transport="streamable-http", port=port or MCP_SERVER_PORT)