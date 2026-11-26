from google import genai
from google.genai import types

from app.config import Config

_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)


def create_client(api_key: str):
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))


async def generate_text(api_key: str, prompt: str, *, safety_threshold: str) -> str:
    client = create_client(api_key)
    response = await client.aio.models.generate_content(
        model=Config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=Config.GEMINI_TEMPERATURE,
            top_p=Config.GEMINI_TOP_P,
            max_output_tokens=Config.GEMINI_MAX_TOKENS,
            safety_settings=[
                types.SafetySetting(category=category, threshold=safety_threshold)
                for category in _SAFETY_CATEGORIES
            ],
        ),
    )
    if not response or not response.text:
        raise ValueError("Gemini returned no text")
    return response.text.strip()
