import os

import httpx

from env_config import load_root_env


load_root_env()
DEFAULT_URL = "http://localhost:7860/api/v1/hackrx/run"


class InsuranceRagError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


async def answer_questions(document_url: str, questions: list[str]) -> dict:
    api_key = os.getenv("RAG_API_KEY")
    if not api_key:
        raise InsuranceRagError(503, "Insurance RAG credentials are missing")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=5)) as client:
            response = await client.post(
                os.getenv("RAG_SERVER_URL") or DEFAULT_URL,
                json={"documents": document_url, "questions": questions},
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.TimeoutException as exc:
        raise InsuranceRagError(504, "Insurance RAG request timed out") from exc
    except httpx.RequestError as exc:
        raise InsuranceRagError(503, "Insurance RAG is unreachable") from exc

    if not response.is_success:
        status = response.status_code
        raise InsuranceRagError(status if 400 <= status < 600 else 502,
                                f"Insurance RAG returned HTTP {status}")

    try:
        result = response.json()
        answers = result["answers"]
        if not isinstance(answers, list) or len(answers) != len(questions):
            raise ValueError("Unexpected answer count")
        if not all(isinstance(answer, str) for answer in answers):
            raise ValueError("Unexpected answer type")
    except (ValueError, KeyError, TypeError) as exc:
        raise InsuranceRagError(502, "Insurance RAG returned an invalid response") from exc

    return result
