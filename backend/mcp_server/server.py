from fastmcp import FastMCP

from integrations.insurance_rag import InsuranceRagError, answer_questions
from mcp_server.config.mcp_settings import MCP_SERVER_PORT

mcp = FastMCP("rag-server")


@mcp.tool(description="Process a document from URL and answer questions using the Insurance RAG API. Supports PDF, DOCX, XLSX, PPTX, TXT, images, audio, and video files.")
async def process_document_rag(document_url: str, questions: str | list[str]) -> dict:
    try:
        result = await answer_questions(
            document_url, [questions] if isinstance(questions, str) else questions
        )
    except InsuranceRagError as exc:
        return {"success": False, "error": str(exc)}

    return {"success": True, "answers": result["answers"], "document_url": document_url}


def run_server(port: int | None = None):
    mcp.run(transport="streamable-http", port=port or MCP_SERVER_PORT)
