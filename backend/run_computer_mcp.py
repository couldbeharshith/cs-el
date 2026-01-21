import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from mcp_server.computer_server import run_server


def main():
    try:
        print("=" * 50)
        print("Computer/Notepad MCP Server")
        print("=" * 50)
        run_server()
        return 0
    except KeyboardInterrupt:
        print("\nServer stopped by user", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

