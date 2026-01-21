import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env.screener file
env_path = Path(__file__).parent / '.env.screener'
load_dotenv(dotenv_path=env_path)

email = os.getenv("SCREENER_EMAIL", "")
password = os.getenv("SCREENER_PASSWORD", "")
google_api = os.getenv("GOOGLE_API_KEY", "")
