import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.stream_ingest import PersistentLocalAPI


class KeepAliveHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    connections = set()
    requests = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        type(self).connections.add(id(self.connection))
        type(self).requests += 1
        data = json.dumps({"healthy": True, "committed_events": 0}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_persistent_client_reuses_one_tcp_socket():
    KeepAliveHandler.connections = set()
    KeepAliveHandler.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), KeepAliveHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = PersistentLocalAPI(
        f"http://127.0.0.1:{server.server_port}", attempts=2
    )
    try:
        for _ in range(20):
            assert client.request("/health")["healthy"] is True
        assert KeepAliveHandler.requests == 20
        assert len(KeepAliveHandler.connections) == 1
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_persistent_client_rejects_nonlocal_url():
    with pytest.raises(ValueError):
        PersistentLocalAPI("http://example.com:5050")
