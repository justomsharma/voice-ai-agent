"""http_post against a real (local, plain-HTTP) server: timings and network errors."""

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from stt import SttError, http_post


class Echo(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        reply = f"got {len(body)} bytes, auth={self.headers['Authorization']}".encode()
        self.send_response(201)
        self.send_header("X-Request-Id", "req_9")
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):  # keep test output clean
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def test_post_round_trip_with_timings(server):
    r = http_post(f"{server}/v1/x?a=1", {"Authorization": "Bearer t"}, b"x" * 1000, 5)
    assert r.status == 201 and r.body == b"got 1000 bytes, auth=Bearer t"
    assert r.headers["x-request-id"] == "req_9"  # names lower-cased
    t = r.timings
    assert min(t.connect_s, t.upload_s, t.wait_s, t.download_s) >= 0
    assert t.total_s == pytest.approx(t.connect_s + t.upload_s + t.wait_s + t.download_s)


def test_refused_connection_is_a_retryable_network_error():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here now
    with pytest.raises(SttError) as e:
        # Windows retries a refused SYN for ~2 s before giving up, so allow more.
        http_post(f"http://127.0.0.1:{port}/", {}, b"", 10)
    assert e.value.kind == "network" and e.value.retryable


def test_silent_server_times_out():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()  # accepts the TCP connection but never answers
    try:
        with pytest.raises(SttError) as e:
            http_post(f"http://127.0.0.1:{s.getsockname()[1]}/", {}, b"hi", 0.3)
        assert e.value.kind == "timeout" and e.value.retryable
    finally:
        s.close()
