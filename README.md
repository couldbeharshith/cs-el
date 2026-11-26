# FinSync AI

FinSync AI is a financial research workspace for analyzing markets, screening companies, and asking questions about insurance documents. Its chat streams model responses, displays tool results and charts, and uses Supabase for conversation history. The dashboard contains a sample Indian-market snapshot.

## Repository layout

| Directory | Responsibility |
| --- | --- |
| [frontend](frontend) | Next.js web application, streaming chat API, voice transcription, and Supabase access |
| [backend/app](backend/app) | FastAPI gateway and reusable document retrieval and decision-tree services |
| [backend/mcp_server](backend/mcp_server) | Document RAG and Windows computer tools over MCP |
| [backend/rag_server](backend/rag_server) | Insurance document processing, vector search, reranking, and answer generation |
| [backend/screener_server](backend/screener_server) | Screener.in integration and company data API |
| [backend/scripts](backend/scripts) | Offline decision-tree and diagram utilities |

## Run the web app

Use Node.js 22 or later. From the repository root:

```sh
cd frontend
npm ci
cp env.example .env.local
npm run dev
```

Open [localhost:3000](http://localhost:3000). The home route opens chat. Use `/dashboard` for the market snapshot and `/about` for product information.

Configure a Supabase project for conversation storage. Set `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` in `frontend/.env.local`. The existing web app uses a shared demo identity, so run it in a trusted environment with synthetic data. Set `OPENAI_API_KEY` for the default chat profiles and voice transcription. Set only the provider and tool credentials relevant to your setup. Leave admin credentials server-side.

## Run backend integrations

Use Python 3.12 and a separate virtual environment for each requirements file. See [backend setup](backend/START_HERE.md) for per-service launch instructions and Windows scripts. See [insurance RAG setup](backend/rag_server/README.md) and [RAG API examples](backend/rag_server/usage.md) for document types and credentials.

The Next.js chat API calls model providers directly. It discovers document and computer tools via MCP and calls the screener service for company analysis. The main FastAPI gateway and document MCP both forward document requests to the insurance RAG service. Keep computer automation on a trusted Windows host.

## Verify changes

```sh
cd frontend
npm run lint
npx tsc --noEmit
npm run build
```

The main API exposes `/health` and `/docs`. The RAG worker exposes its own `/health`. Live smoke scripts in `backend/` and `backend/screener_server/` require their target services and credentials. See [AGENTS.md](AGENTS.md) for development guidance.
