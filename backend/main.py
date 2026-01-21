import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run( host="0.0.0.0", port=8000, reload=True, app="app.main:app", log_level="info")
