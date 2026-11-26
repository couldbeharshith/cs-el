# Working in FinSync AI

Preserve the existing chat workflows, API and MCP contracts, dashboard routes, and stored conversation compatibility. Make small, reviewable changes and verify them through the normal entry points.

## Repository map

- `frontend/app` contains the Next.js routes. Chat-specific presentation and hooks live under `app/chat`. Keep provider credentials, database administration, and tool execution on the server.
- `frontend/components` contains shared UI, navigation, and dashboard components. Reuse accessible primitives and the theme tokens in `app/globals.css`.
- `frontend/lib/supabase` contains connection code, queries, and the application schema.
- `backend/app` is the main FastAPI gateway and contains retrieval, vector-store, decision-tree, and knowledge-graph code.
- `backend/mcp_server` contains separate RAG and Windows computer MCP entry points.
- `backend/rag_server` and `backend/screener_server` have separate Python environments and settings. Start each from its own directory so its imports and configuration resolve correctly.
- `backend/scripts` contains offline tooling. Keep scripts and research code usable independently of the web request path.

## Checks

Run `npm run lint`, `npx tsc --noEmit`, and `npm run build` from `frontend/`. Exercise changed flows in a browser at desktop and mobile widths, including keyboard navigation, loading, and failure states.

Use Python 3.12 for backend work. Run backend checks from the owning service directory. `backend/test_api.py` and `backend/screener_server/test_api.py` are live smoke scripts. Run them against the intended environment with explicit integration credentials. Prefer local fixtures for repeatable tests.

Keep secrets in service-local environment files. The default application identity is shared in demo mode. Use isolated projects and synthetic data for development and browser tests. Keep the Windows computer MCP on a trusted local machine.
