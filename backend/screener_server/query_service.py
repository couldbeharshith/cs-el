from google import genai
from google.genai.types import HttpOptions, HttpRetryOptions
from schemas import Screener_Query, keywords

class ScreenerQueryService:
    def __init__(self, api_key: str):
        self.client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(api_version="v1beta", timeout=12_000, retry_options=HttpRetryOptions(attempts=1)),
        )
        self.system_prompt = f"""Translate the request into one Screener.in query.
Use the exact available field names and preserve the requested filters and thresholds.
Combine conditions with AND or OR where appropriate.

Available fields and operators:
{keywords}
"""

    def generate_screener_query(self, user_query: str) -> Screener_Query:
        response = self.client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=user_query,
            config={
                "system_instruction": self.system_prompt,
                "thinking_config": {"thinking_level": "minimal"},
                "response_mime_type": "application/json",
                "response_json_schema": Screener_Query.model_json_schema(),
            },
        )
        return Screener_Query.model_validate_json(response.text)
