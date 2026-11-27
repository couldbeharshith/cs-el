import os
from pathlib import Path

from dotenv import load_dotenv


backend_dir = Path(__file__).resolve().parent.parent
if backend_dir.name == "backend":
    load_dotenv(backend_dir.parent / ".env")

email = os.getenv("SCREENER_EMAIL", "")
password = os.getenv("SCREENER_PASSWORD", "")
google_api = os.getenv("GOOGLE_API_KEY", "")
