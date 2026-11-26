# Python services

Use Python 3.12 and give the main backend, insurance worker, and screener their own virtual environments. Run each command from its service directory.

| Service | Working directory | Entry point | Default port |
| --- | --- | --- | --- |
| RAG gateway | `backend/` | `python main.py` | 8000 |
| Document MCP | `backend/` | `python run_mcp.py` | 8001 |
| Computer MCP | `backend/` | `python run_computer_mcp.py` | 8002 |
| Insurance worker | `backend/rag_server/` | `python app.py` | 7860 |
| Screener API | `backend/screener_server/` | `python server.py` | 8080 |

## Gateway and document MCP

From `backend/`:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-gateway.txt
cp .env.example .env
python main.py
```

Start the MCP service in another terminal with the same environment. Set `RAG_API_KEY` to the worker’s `API_KEY`. Set `RAG_SERVER_URL` to override the default worker endpoint and `MCP_SERVER_PORT` to override the MCP port. The gateway exposes `GET /`, `GET /health`, `GET /rag/health`, and `POST /rag/run`. Run the gateway and MCP fixtures with `python -m unittest discover -s tests`.

Use the full `requirements.txt` for the computer MCP and offline preprocessing and retrieval scripts. The computer MCP’s Notepad and desktop workflows require Windows. See the [Windows launch scripts](START_HERE.md).

## Insurance worker and screener

In `rag_server/`, install its `requirements.txt` and copy `.env.template` to `.env.rag`. Supply a worker token, a Pinecone API key and index host, and the Gemini and Groq credentials described in the [worker README](rag_server/README.md).

In `screener_server/`, install its `requirements.txt`, copy `.env.example` to `.env.screener`, and supply Screener.in credentials. Install Chrome for browser-backed endpoints. Set `GOOGLE_API_KEY` for `/query_agent`. The lookup and direct chart endpoints use the Screener.in HTTP API.
