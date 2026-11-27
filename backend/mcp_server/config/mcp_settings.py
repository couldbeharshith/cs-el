import os

from env_config import load_root_env


load_root_env()
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8001"))
