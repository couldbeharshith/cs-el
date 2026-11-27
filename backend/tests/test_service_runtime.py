from contextlib import ExitStack
from pathlib import Path
import os
import socket
import sys
from unittest import TestCase

from service_runtime import HOST, Service, check_ports, run_stack


BACKEND = Path(__file__).resolve().parents[1]


def free_ports() -> tuple[int, int]:
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(2)]
        for listener in sockets:
            listener.bind((HOST, 0))
        return tuple(listener.getsockname()[1] for listener in sockets)


class ServiceRuntimeTests(TestCase):
    def assert_stopped(self, *ports: int) -> None:
        for port in ports:
            with socket.socket() as connection:
                connection.settimeout(1)
                self.assertNotEqual(connection.connect_ex((HOST, port)), 0)

    def test_checks_gateway_and_document_mcp_then_releases_ports(self):
        gateway_port, mcp_port = free_ports()
        services = [
            Service("Gateway", "gateway", BACKEND, "app.main:app", gateway_port, "/health"),
            Service(
                "Document MCP", "gateway", BACKEND, "mcp_server.server:create_http_app",
                mcp_port, "/mcp", factory=True,
            ),
        ]
        run_stack(services, {"gateway": Path(sys.executable)}, dict(os.environ), check_only=True)
        self.assert_stopped(gateway_port, mcp_port)

    def test_failed_service_stops_the_rest_of_the_stack(self):
        gateway_port, failing_port = free_ports()
        services = [
            Service("Gateway", "gateway", BACKEND, "app.main:app", gateway_port, "/health"),
            Service("Broken service", "gateway", BACKEND, "missing_module:app", failing_port, "/health"),
        ]
        with self.assertRaisesRegex(RuntimeError, "Broken service stopped unexpectedly"):
            run_stack(services, {"gateway": Path(sys.executable)}, dict(os.environ), check_only=True)
        self.assert_stopped(gateway_port, failing_port)

    def test_occupied_port_is_reported_without_stopping_its_listener(self):
        with socket.socket() as listener:
            listener.bind((HOST, 0))
            listener.listen()
            port = listener.getsockname()[1]
            service = Service("Gateway", "gateway", BACKEND, "app.main:app", port, "/health")
            with self.assertRaisesRegex(RuntimeError, "Gateway cannot bind"):
                check_ports([service])
            with socket.create_connection((HOST, port), timeout=1):
                pass
