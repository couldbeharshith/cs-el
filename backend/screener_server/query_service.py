from google import genai
from credentials import google_api
from schemas import Screener_Query, keywords
from scraper import login, run_custom_query

class ScreenerQueryService:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.system_prompt = f"""You are an expert financial analyst specializing in screener.in query generation.

Your task is to convert natural language queries into precise screener.in queries that identify fundamentally strong companies.

⚠️ CRITICAL: You MUST ONLY use the exact parameter names listed below. Do NOT use any other parameter names or variations.

AVAILABLE PARAMETERS (USE THESE EXACT NAMES ONLY):
{keywords}

QUERY GENERATION RULES:
1. **STRICT PARAMETER USAGE**: Only use parameters from the list above - use the EXACT spelling and capitalization
2. **Always filter for quality**: Include filters that screen out weak/small companies
3. **Use 5-10+ parameters**: Combine multiple metrics for comprehensive screening
4. **Logical operators**: Use AND, OR, >, <, =, >=, <= appropriately (as shown in the Operations section above)
5. **Reasonable thresholds**: Set realistic values based on industry standards
6. **Sector-specific logic**: Adjust parameters based on the sector (e.g., banks need different metrics than tech companies)

EXAMPLES OF VALID QUERIES (using only parameters from the list):
- "Market Capitalization > 10000 AND Return on equity > 15 AND Debt to equity < 1 AND Sales growth 5Years > 10"
- "Return on capital employed > 20 AND Price to Earning < 25 AND Profit growth 3Years > 15 AND Current ratio > 1.5"
- "Sales > 1000 AND OPM > 15 AND Debt to equity < 0.5 AND Promoter holding > 50 AND Return on equity > 18"

SECTOR-SPECIFIC CONSIDERATIONS (using only available parameters):
- **Banks/NBFCs**: Focus on Return on assets, Return on equity, Debt to equity, Net profit
- **Manufacturing**: Focus on OPM, Inventory turnover ratio, Working capital, Sales growth
- **IT/Services**: Focus on OPM, Return on equity, Sales growth, Profit growth
- **Pharma**: Focus on OPM, Sales growth, Profit growth, Exports percentage

⚠️ REMINDER: Double-check that every parameter in your query exists in the AVAILABLE PARAMETERS list above.

Given the user's query, generate a screener.in query that will identify the most relevant fundamentally strong companies."""

    def generate_screener_query(self, user_query: str) -> Screener_Query:
        # Combine system prompt with user query
        full_prompt = f"{self.system_prompt}\n\nUSER QUERY: {user_query}\n\nGenerate the screener.in query using ONLY the parameters from the available list:"
        
        response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=full_prompt,
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
