"""
Hugging Face Spaces entry point for Insurance RAG API
This file is required by Hugging Face Spaces to run the application
"""

import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import and run the FastAPI app
from app.main import app

if __name__ == "__main__":
    import uvicorn
    from app.config import Config

    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
 
