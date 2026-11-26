import { defineConfig, devices } from "@playwright/test";

const baseURL = "http://127.0.0.1:4315";
const servicesURL = "http://127.0.0.1:4316";
const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;

export default defineConfig({
  testDir: "./tests/e2e",
  workers: 1,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    launchOptions: {
      executablePath,
      args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
    },
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: [
    { command: "node tests/e2e/services.mjs", url: `${servicesURL}/health`, timeout: 30_000 },
    {
      command: process.env.E2E_PRODUCTION === "1" ? "npm run start -- --hostname 127.0.0.1 --port 4315" : "npm run dev -- --hostname 127.0.0.1 --port 4315",
      url: `${baseURL}/about`,
      timeout: 90_000,
      env: {
        NEXT_PUBLIC_SUPABASE_URL: servicesURL,
        NEXT_PUBLIC_SUPABASE_ANON_KEY: "local-test",
        OPENAI_API_KEY: "local-test",
        SCREENER_API_URL: servicesURL,
        OPENAI_BASE_URL: `${servicesURL}/v1`,
        RAG_MCP_URL: "",
        COMPUTER_MCP_URL: "",
      },
    },
  ],
});
