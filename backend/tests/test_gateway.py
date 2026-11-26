import os
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import httpx
from fastmcp import Client

from app.main import app
from mcp_server.server import mcp

REQUEST = {
    "documents": "https://example.test/policy.pdf",
    "questions": ["What is covered?"],
}
ANSWER = "The fixture policy covers fire damage."


class GatewayTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        )
        self.addAsyncCleanup(self.client.aclose)

    async def test_health_and_missing_credentials(self):
        response = await self.client.get("/health")
        self.assertEqual(response.json(), {"status": "healthy"})
        with patch.dict(os.environ, {"RAG_API_KEY": ""}):
            response = await self.client.post("/rag/run", json=REQUEST)
        self.assertEqual(response.status_code, 503)

    def mock_worker(self, handler):
        self.enterContext(patch.dict(os.environ, {
            "RAG_API_KEY": "fixture-token",
            "RAG_SERVER_URL": "http://worker/api/v1/hackrx/run",
        }))
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.enterContext(patch("integrations.insurance_rag.httpx.AsyncClient", return_value=upstream))

    async def test_gateway_forwards_request_and_returns_answer(self):
        def worker(request):
            self.assertEqual(request.headers["Authorization"], "Bearer fixture-token")
            self.assertEqual(request.url.path, "/api/v1/hackrx/run")
            return httpx.Response(200, json={"answers": [ANSWER]})

        self.mock_worker(worker)
        response = await self.client.post("/rag/run", json=REQUEST)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answers"], [ANSWER])
        self.assertIn("processing_time", response.json())

    async def test_gateway_returns_upstream_status_without_raw_error(self):
        self.mock_worker(lambda _: httpx.Response(429, text="provider response"))
        response = await self.client.post("/rag/run", json=REQUEST)
        self.assertEqual(response.status_code, 429)
        self.assertNotIn("provider response", response.text)

    async def test_gateway_handles_invalid_worker_response(self):
        self.mock_worker(lambda _: httpx.Response(200, json={"answers": []}))
        response = await self.client.post("/rag/run", json=REQUEST)
        self.assertEqual(response.status_code, 502)

    async def test_mcp_discovers_and_calls_document_tool(self):
        self.mock_worker(lambda _: httpx.Response(200, json={"answers": [ANSWER]}))
        async with Client(mcp) as client:
            tools = await client.list_tools()
            self.assertIn("process_document_rag", [tool.name for tool in tools])
            result = await client.call_tool(
                "process_document_rag",
                {"document_url": REQUEST["documents"], "questions": REQUEST["questions"][0]},
            )
        self.assertFalse(result.is_error)
        self.assertIn(ANSWER, result.content[0].text)
