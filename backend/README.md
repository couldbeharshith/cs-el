# Backend API

A FastAPI-based backend for the financial agent application.

## Installation

```bash
# Install uv if not already installed
pip install uv

# Create a virtual environment with Python 3.12
uv venv --python=3.12

# Install dependencies
uv pip install -r requirements.txt
```

## Running

```bash
# Start the FastAPI server
uv run main.py
```

The API will be available at `http://localhost:8000`

## Running the MCP server (needed for the frontend chat ui)

This project ships an MCP server for tool-style integrations. It's separate from the FastAPI app and not started by `uv run main.py`.

```bash
# From backend/
python run_mcp.py
```

By default it listens on port 8001 (see `mcp_server/config/mcp_settings.py`).

## API endpoints

- GET `/` — Basic info
- GET `/health` — Quick health check
- POST `/rag/run` — RAG endpoint (currently stubbed for external API integration)

### POST /rag/run

This endpoint is currently stubbed and ready for your external RAG API integration.

Request body:

```json
{
	"documents": "https://.../documents/document.pdf",
	"questions": [
		"What is the main topic?",
		"What are the key points?"
	],
	"k": 8,
	"processing_mode": "traditional"
}
```

Response (development mode):

```json
{
	"success": true,
	"answers": ["...", "..."],
	"processing_time": 0.01,
	"document_metadata": {"status": "RAG API not configured"},
	"raw_response": {"message": "This endpoint is ready for your RAG API integration"}
}
```

## Integrating Your RAG API

To integrate your own RAG API:

1. Open `backend/app/api/v1/endpoints/rag.py`
2. Replace the placeholder implementation in the `run_rag` function with your API call
3. Update the MCP server tool in `backend/mcp_server/server.py` if needed

Example integration:
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "YOUR_RAG_API_URL",
        json={
            "documents": request.documents,
            "questions": request.questions,
            "k": request.k,
            "processing_mode": request.processing_mode
        }
    )
    return response.json()
```
