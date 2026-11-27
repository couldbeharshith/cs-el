# Working in Sampada AI

Preserve the existing chat workflows, API and MCP contracts, dashboard routes, and stored conversation compatibility. Make small, reviewable changes and verify them through the normal entry points.

## Repository map

- `frontend/app` contains the Next.js routes. Chat presentation, state, prompts, providers, and tools live under `frontend/features/chat`. Keep provider credentials, database administration, and tool execution on the server.
- `frontend/components` contains shared UI, navigation, and dashboard components. Reuse accessible primitives and the theme tokens in `app/globals.css`.
- `frontend/lib/supabase` contains connection code, queries, and the application schema.
- `backend/app` is the main FastAPI gateway and contains retrieval, vector-store, and decision-tree code.
- `backend/integrations` contains shared clients for upstream services.
- `backend/mcp_server` contains separate RAG and Windows computer MCP entry points.
- `backend/serve.py` starts the local backend and manages its Python environments. The insurance worker and screener still run from their own directories so standalone imports and deployment entry points resolve correctly.
- `backend/research` contains offline knowledge-graph notebooks and ingestion prototypes.
- `backend/scripts` contains offline tooling. Keep scripts and research code usable independently of the web request path.

## Checks

Run `npm run lint`, `npm run typecheck`, and `npm run build` from `frontend/`. Exercise changed flows in a browser at desktop and mobile widths, including keyboard navigation, loading, and failure states.

Use Python 3.12 for backend work. Run backend checks from the owning service directory. `backend/test_api.py` and `backend/screener_server/test_api.py` are live smoke scripts. Run them against the intended environment with explicit integration credentials. Use the browser fixtures from [frontend/README.md](frontend/README.md#browser-tests) and the gateway suite from [backend/README.md](backend/README.md#gateway-and-document-mcp) for repeatable tests.

Keep local secrets in the ignored repository-root `.env`. The default application identity is shared in demo mode. Use isolated projects and synthetic data for development and browser tests. Keep the Windows computer MCP on a trusted local machine.
