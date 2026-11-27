"""The process lifecycle shared by the local backend services."""

import asyncio
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from threading import Event, Thread
from time import monotonic, sleep


HOST = "127.0.0.1"
WINDOWS = sys.platform == "win32"


@dataclass(frozen=True)
class Service:
    name: str
    runtime: str
    directory: Path
    app: str
    port: int
    health_path: str
    factory: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{HOST}:{self.port}"


def check_ports(services: list[Service]) -> None:
    ports: dict[int, str] = {}
    for service in services:
        if service.port in ports:
            raise RuntimeError(
                f"{service.name} and {ports[service.port]} both use port {service.port}."
            )
        ports[service.port] = service.name
        with socket.socket() as probe:
            if WINDOWS:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((HOST, service.port))
            except OSError as exc:
                raise RuntimeError(f"{service.name} cannot bind {service.base_url}: {exc}") from exc


def is_ready(service: Service) -> bool:
    connection = HTTPConnection(HOST, service.port, timeout=0.5)
    try:
        connection.request("GET", service.health_path, headers={"Accept": "application/json"})
        with connection.getresponse() as response:
            # A GET without an MCP session or SSE Accept header is rejected by
            # the active MCP transport without allocating a session.
            return response.status in (400, 406) if service.factory else response.status == 200
    except (OSError, HTTPException):
        return False
    finally:
        connection.close()


def run_stack(
    services: list[Service], interpreters: dict[str, Path], env: dict[str, str], *, check_only: bool
) -> None:
    processes: dict[Service, subprocess.Popen[bytes]] = {}
    try:
        for service in services:
            print(f"Starting {service.name} at {service.base_url}", flush=True)
            processes[service] = subprocess.Popen(
                [
                    str(interpreters[service.runtime]), "-u", str(Path(__file__).resolve()),
                    service.app, str(service.port), str(int(service.factory)),
                ],
                cwd=service.directory,
                env=env,
                stdin=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0,
                start_new_session=not WINDOWS,
            )
        pending = list(services)
        deadline = monotonic() + 90
        while True:
            for service, process in processes.items():
                if (exit_code := process.poll()) is not None:
                    raise RuntimeError(f"{service.name} stopped unexpectedly (exit {exit_code}).")
            if pending:
                pending = [service for service in pending if not is_ready(service)]
                if not pending:
                    print("Backend ready.", flush=True)
                    if check_only:
                        return
                    print("Press Ctrl+C to stop all services.", flush=True)
                elif monotonic() >= deadline:
                    names = ", ".join(service.name for service in pending)
                    raise RuntimeError(f"Startup timed out waiting for {names}.")
            sleep(0.2)
    finally:
        stop_stack(list(processes.values()))


def stop_stack(processes: list[subprocess.Popen[bytes]]) -> None:
    # Closing the pipe requests graceful shutdown even when the launcher runs
    # without a console. It also works when the parent's pipe closes on exit.
    for process in processes:
        process.stdin.close()
    deadline = monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - monotonic()))
        except subprocess.TimeoutExpired:
            if WINDOWS:
                subprocess.run(
                    ["taskkill", "/pid", str(process.pid), "/t", "/f"],
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if process.poll() is None:
                    process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()


def run_service(app: str, port: str, factory: str) -> int:
    # The worker and gateway both have an "app" package. Resolve each one from
    # its own service directory, as their standalone commands do.
    sys.path[0] = os.getcwd()
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(
        app, host=HOST, port=int(port), factory=factory == "1",
        access_log=False, timeout_graceful_shutdown=5,
    ))
    stopping = Event()

    def watch_parent() -> None:
        os.read(sys.stdin.fileno(), 1)
        stopping.set()

    async def stop_after_startup() -> None:
        # Setting should_exit during Uvicorn startup can bypass ASGI shutdown.
        while not (stopping.is_set() and server.started):
            await asyncio.sleep(0.1)
        server.should_exit = True

    async def serve() -> None:
        monitor = asyncio.create_task(stop_after_startup())
        try:
            await server.serve()
        finally:
            monitor.cancel()

    Thread(target=watch_parent, daemon=True).start()
    asyncio.run(serve())
    return 0 if server.started or stopping.is_set() else 1


if __name__ == "__main__":
    sys.exit(run_service(*sys.argv[1:]))
