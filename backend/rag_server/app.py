"""
Hugging Face Spaces entry point for Insurance RAG API
This file is required by Hugging Face Spaces to run the application
"""

import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Set platform environment variable for Hugging Face Spaces
os.environ['PLATFORM'] = 'huggingface'

# Import and run the FastAPI app
from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
 
