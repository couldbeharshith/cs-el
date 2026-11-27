# Sampada AI

Sampada AI is a financial research workspace for analyzing markets, screening companies, and asking questions about insurance documents. Its chat streams model responses, displays tool results and charts, and uses Supabase for conversation history. The dashboard contains a sample Indian-market snapshot.

## Repository layout

| Directory | Responsibility |
| --- | --- |
| [frontend](frontend) | Next.js routes, chat feature, shared UI, browser tests, and Supabase schema |
| [backend/app](backend/app) | FastAPI gateway and reusable document retrieval and decision-tree services |
| [backend/integrations](backend/integrations) | Shared gateway and MCP clients for upstream services |
| [backend/mcp_server](backend/mcp_server) | Document RAG and Windows computer tools over MCP |
| [backend/rag_server](backend/rag_server) | Insurance document processing, vector search, reranking, and answer generation |
| [backend/screener_server](backend/screener_server) | Screener.in integration and company data API |
| [backend/scripts](backend/scripts) | Offline decision-tree and diagram utilities |
| [backend/research](backend/research) | Knowledge-graph notebooks and policy ingestion prototypes |

## Run the web app

Use Node.js 22 or later. From the repository root, copy the [environment template](.env.example) to `.env` with `cp .env.example .env` or, in PowerShell, `Copy-Item .env.example .env`. Set `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, and `OPENAI_API_KEY` in that file.

For a fresh Supabase project, run the [conversation schema](frontend/lib/supabase/migrations/schema.sql) in the SQL editor. Review it against your existing schema before applying it to a populated project. The web app uses a shared demo identity, so run it in a trusted environment with synthetic data. The optional database-query tool requires the server-only `SUPABASE_ADMIN` credential and an existing `execute_sql` RPC.

```sh
cd frontend
npm ci
npm run dev
```

Open [localhost:3000](http://localhost:3000). The home route opens chat. Use `/dashboard` for the market snapshot and `/about` for product information. Restart the web app after editing `.env` during development.

All services in this checkout read the root `.env`, and process environment variables take priority over entries with the same name. If you're upgrading, move values from the old frontend and backend environment files into the root file, reconcile duplicate keys, and archive the old files securely. Use `RAG_API_KEY` for the local insurance worker's bearer token and rename its `HOST` and `PORT` entries to `RAG_WORKER_HOST` and `RAG_WORKER_PORT`.

## Run backend integrations

Use Python 3.12 and a separate virtual environment for each requirements file. See [backend setup](backend/README.md) for per-service launch instructions and [Windows scripts](backend/START_HERE.md) for starting all services together. See [insurance RAG setup](backend/rag_server/README.md) and [RAG API examples](backend/rag_server/usage.md) for document types and credentials.

The Next.js chat API calls model providers directly. It discovers document and computer tools via MCP and calls the screener service for company analysis. The main FastAPI gateway and document MCP both forward document requests to the insurance RAG service. Keep computer automation on a trusted Windows host.

## Verify changes

```sh
cd frontend
npm run lint
npm run typecheck
npm run build
```

Run the [browser suite](frontend/README.md#browser-tests) for desktop and mobile coverage and the [gateway tests](backend/README.md#gateway-and-document-mcp) for API and MCP checks. See [AGENTS.md](AGENTS.md) for development guidance.
