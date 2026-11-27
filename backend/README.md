# Python services

Install `uv`, copy [the shared template](../.env.example) to `.env` at the repository root, and fill in the credentials for the services you use. From the repository root:

```sh
uv run --script backend/serve.py
```

The launcher installs each service's Python 3.12 dependencies into `backend/.venv-services/` on the first run. It starts the stack in one terminal, waits for each endpoint to respond, and stops the services it started when you press Ctrl+C or when a service exits. All local listeners bind to `127.0.0.1`.

| Local endpoint | Default port | Port setting |
| --- | --- | --- |
| RAG gateway | 8000 | `BACKEND_PORT` |
| Document MCP at `/mcp` | 8001 | `MCP_SERVER_PORT` |
| Computer MCP at `/mcp` (Windows) | 8002 | `COMPUTER_MCP_PORT` |
| Insurance worker | 7860 | `RAG_WORKER_PORT` |
| Screener API | 8080 | `SCREENER_PORT` |

The frontend's existing defaults work with these ports. If you change a frontend-facing port, update `RAG_MCP_URL`, `COMPUTER_MCP_URL`, or `SCREENER_API_URL` in the root `.env` as applicable. Set `RAG_SERVER_URL` to use a remote insurance worker. Otherwise the gateway and document MCP use the worker started by the launcher.

- Use `--without-worker` with a remote insurance worker, or when working on features that do not use insurance RAG.
- Use `--without-computer` to skip Windows desktop tools.
- Use `--check` to start the stack, verify readiness, and shut it down. Readiness probes check the local endpoints without making application requests to external providers.
- Use `--setup-only` to prepare dependencies in advance.

The PowerShell and Batch start scripts also run the same launcher. If a port is occupied, stop the older service in its terminal or change the corresponding port setting.

## Standalone services

For debugging or deployment, use Python 3.12 with the requirements for the service you are running. Run the entry point from its service directory.

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
python main.py
```

Start the MCP service in another terminal with the same environment. Set `RAG_API_KEY` to the bearer token shared with the insurance worker. Set `RAG_SERVER_URL` to override the default worker endpoint and `MCP_SERVER_PORT` to override the MCP port. The gateway exposes `GET /`, `GET /health`, `GET /rag/health`, and `POST /rag/run`. Run the gateway and MCP fixtures with `python -m unittest discover -s tests`.

Use `requirements-computer.txt` for the standalone computer MCP on Windows. Use the full `requirements.txt` for offline preprocessing and retrieval scripts.

## Insurance worker and screener

In `rag_server/`, install its `requirements.txt`. In the root `.env`, supply the shared `RAG_API_KEY`, a Pinecone API key and index host, and the Gemini and Groq credentials described in the [worker README](rag_server/README.md).

In `screener_server/`, install its `requirements.txt` and supply `SCREENER_EMAIL` and `SCREENER_PASSWORD` in the root `.env`. Install Chrome for browser-backed endpoints. Set `GOOGLE_API_KEY` for `/query_agent`. The lookup and direct chart endpoints use the Screener.in HTTP API.
