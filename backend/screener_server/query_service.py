from google import genai
from credentials import google_api
from schemas import Screener_Query
from scraper import login, run_custom_query

class ScreenerQueryService:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.prompt = "You are an expert in extracting financial data from screener.in. Given a user query, you need to provide the relevant screener.in query"

    def generate_screener_query(self, user_query: str) -> Screener_Query:
        response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=user_query,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": Screener_Query.model_json_schema(),
            },
        )
        result = Screener_Query.model_validate_json(response.text)
        return result

    def process_query(self, user_query: str) -> str:
        screener_query_obj = self.generate_screener_query(user_query)
        print("Generated Screener Query:", screener_query_obj.screener_query)
        output = run_custom_query(login(), screener_query_obj.screener_query)
        return output , screener_query_obj
