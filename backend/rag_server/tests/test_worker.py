from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi.testclient import TestClient

from app.config import Config
from app.main import app
from app.gemini import generate_text
from app.services import DocumentService

BODY = {
    "documents": "https://example.test/policy.txt",
    "questions": ["What is covered?", "What is excluded?"],
}


class DownloadTests(IsolatedAsyncioTestCase):
    async def test_returns_content_and_rejects_oversized_chunked_download(self):
        async def small(_request):
            return web.Response(text="A small policy")

        async def large(request):
            response = web.StreamResponse()
            await response.prepare(request)
            await response.write(b"a" * 600_000)
            await response.write(b"b" * 600_000)
            return response

        source = web.Application()
        source.router.add_get("/small.txt", small)
        source.router.add_get("/large.txt", large)
        downloader = DocumentService.__new__(DocumentService)
        with patch.object(Config, "MAX_DOCUMENT_SIZE_MB", 1):
            async with TestServer(source) as server:
                content, content_type = await downloader.fetch_document(str(server.make_url("/small.txt")))
                oversized = await downloader.fetch_document(str(server.make_url("/large.txt")))

        self.assertEqual(content, b"A small policy")
        self.assertTrue(content_type.startswith("text/plain"))
        self.assertEqual(oversized, (b"__FILE_TOO_LARGE__", "application/oversized"))


class WorkerTests(TestCase):
    def test_lifespan_and_authenticated_document_request(self):
        fixture = b"Fire damage is covered. Flood damage is excluded."
        answers = ["Fire damage is covered.", "Flood damage is excluded."]
        index = Mock()
        vector_service = SimpleNamespace(index=index)
        fetch = AsyncMock(return_value=(fixture, "text/plain"))
        model = AsyncMock(side_effect=answers)

        with (
            patch.object(Config, "API_KEY", "fixture-token"),
            patch.object(Config, "validate_required_config", return_value=[]),
            patch("app.main.VectorService", return_value=vector_service),
            patch("app.services.TikTokenChunker.create_from_config", return_value=None),
            patch("app.services.DocumentService.fetch_document", fetch),
            patch("app.services.LLMService.generate_direct_gemini_answer", model),
            TestClient(app) as client,
        ):
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.post("/api/v1/hackrx/run", json=BODY).status_code, 401)
            fetch.assert_not_awaited()
            response = client.post(
                "/api/v1/hackrx/run",
                json=BODY,
                headers={"Authorization": "Bearer fixture-token"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["answers"], answers)

        index.close.assert_called_once()
        fetch.assert_awaited_once_with(BODY["documents"])
        self.assertEqual(model.await_count, 2)


class GeminiTests(IsolatedAsyncioTestCase):
    async def test_text_requests_use_their_own_credentials_and_settings(self):
        generation = AsyncMock(return_value=SimpleNamespace(text="Test answer"))
        client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generation)))
        with patch("app.gemini.create_client", return_value=client) as create:
            answer = await generate_text("fixture-key", "Test prompt", safety_threshold="BLOCK_MEDIUM_AND_ABOVE")
        create.assert_called_once_with("fixture-key")
        self.assertEqual(answer, "Test answer")
        self.assertEqual(generation.call_args.kwargs["contents"], "Test prompt")
        self.assertEqual(len(generation.call_args.kwargs["config"].safety_settings), 4)
