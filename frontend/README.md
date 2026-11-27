# Web application

Use Node.js 22 or newer. See the [repository README](../README.md) for Supabase setup and the optional backend services.

Create and fill in the repository-root `.env` from [the shared template](../.env.example), then run these commands from `frontend/`.

```sh
npm ci
npm run dev
```

The `app/` directory owns routes and request handlers. The chat workspace is in `features/chat/`, shared UI is in `components/`, and database access is in `lib/supabase/`.

Chat defaults to OpenAI. Set the API key for each provider you use in the repository-root `.env`.

| Chat option | Model | Environment variable |
| --- | --- | --- |
| OpenAI | GPT-6 Luna | `OPENAI_API_KEY` |
| Gemini | Gemini 3.8 Flash | `GOOGLE_GENERATIVE_AI_API_KEY` |
| Groq | GPT-OSS 120B | `GROQ_API_KEY` |

## Browser tests

The suite runs Chromium in desktop and mobile configurations. It uses a real Postgres database and the checked-in schema through PostgREST, plus local HTTP fixtures for the three chat providers and transcription. The suite creates disposable conversations.

```sh
npx playwright install chromium
docker compose -f tests/e2e/compose.yml up -d --wait
npm run test:e2e
docker compose -f tests/e2e/compose.yml down
```

To test a production build, build with `NEXT_PUBLIC_SUPABASE_URL=http://127.0.0.1:4316` and `NEXT_PUBLIC_SUPABASE_ANON_KEY=local-test`, then run `E2E_PRODUCTION=1 npm run test:e2e`. Playwright writes screenshots and failure traces to `test-results/`.
