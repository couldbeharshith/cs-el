# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["python-dotenv==1.1.1"]
# ///

"""Start and supervise the local backend from one terminal."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

from env_config import load_root_env
from service_runtime import Service, check_ports, run_stack


BACKEND = Path(__file__).resolve().parent
WINDOWS = sys.platform == "win32"


def port(key: str, default: int) -> int:
    try:
        value = int(os.environ.get(key) or default)
        if 1 <= value <= 65535:
            return value
    except ValueError:
        pass
    raise ValueError(f"{key} must be a port from 1 to 65535")


def services_for(args: argparse.Namespace) -> list[Service]:
    services = [
        Service(
            "Gateway", "gateway", BACKEND, "app.main:app",
            port("BACKEND_PORT", 8000), "/health",
        ),
        Service(
            "Document MCP", "gateway", BACKEND, "mcp_server.server:create_http_app",
            port("MCP_SERVER_PORT", 8001), "/mcp", factory=True,
        ),
        Service(
            "Screener", "screener", BACKEND / "screener_server", "server:app",
            port("SCREENER_PORT", 8080), "/",
        ),
    ]
    if not args.without_worker:
        services.insert(0, Service(
            "Insurance worker", "insurance", BACKEND / "rag_server", "app.main:app",
            port("RAG_WORKER_PORT", 7860), "/health",
        ))
    if WINDOWS and not args.without_computer:
        services.append(Service(
            "Computer MCP", "gateway", BACKEND, "mcp_server.computer_server:create_http_app",
            port("COMPUTER_MCP_PORT", 8002), "/mcp", factory=True,
        ))
    return services


def prepare_runtimes(services: list[Service], with_computer: bool) -> dict[str, Path]:
    requirements = {
        "gateway": BACKEND / ("requirements-computer.txt" if with_computer else "requirements-gateway.txt"),
        "insurance": BACKEND / "rag_server" / "requirements.txt",
        "screener": BACKEND / "screener_server" / "requirements.txt",
    }
    interpreters = {}
    for runtime in dict.fromkeys(service.runtime for service in services):
        environment = BACKEND / ".venv-services" / runtime
        python = environment / ("Scripts/python.exe" if WINDOWS else "bin/python")
        print(f"Preparing {runtime} dependencies", flush=True)
        if not python.is_file():
            subprocess.run(["uv", "venv", "--python", "3.12", str(environment)], check=True)
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "-r", str(requirements[runtime])],
            check=True,
        )
        interpreters[runtime] = python
    return interpreters


def stop_on_term(_signal, _frame):
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-worker", action="store_true", help="skip the local insurance worker")
    parser.add_argument("--without-computer", action="store_true", help="skip the Windows computer MCP")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--setup-only", action="store_true", help="prepare dependencies and exit")
    mode.add_argument("--check", action="store_true", help="start, verify readiness, and shut down")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop_on_term)

    load_root_env()
    try:
        services = services_for(args)
        if not args.setup_only:
            check_ports(services)
        interpreters = prepare_runtimes(services, WINDOWS and not args.without_computer)
        if not args.setup_only:
            env = os.environ.copy()
            worker = next((service for service in services if service.runtime == "insurance"), None)
            if worker and not env.get("RAG_SERVER_URL"):
                env["RAG_SERVER_URL"] = f"{worker.base_url}/api/v1/hackrx/run"
            run_stack(services, interpreters, env, check_only=args.check)
    except KeyboardInterrupt:
        print("\nBackend stopped.", flush=True)
    except subprocess.CalledProcessError as exc:
        print(f"Backend dependency setup failed (exit {exc.returncode}).", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Backend failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
