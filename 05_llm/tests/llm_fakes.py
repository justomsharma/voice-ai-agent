"""Shared fakes: stand-ins for the network so tests need no key and no internet."""

import io
import json
import time

from llm import HttpStream


def sse(*events: dict) -> bytes:
    """Encode events the way the server sends them: event + data + blank line."""
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


class CutAfter(io.BytesIO):
    """A body that drops the connection after `n` lines."""

    def __init__(self, data: bytes, n: int, error: Exception | None = None) -> None:
        super().__init__(data)
        self.n = n
        self.error = error or ConnectionResetError("connection reset by peer")

    def readline(self, *args) -> bytes:
        if self.n == 0:
            raise self.error
        self.n -= 1
        return super().readline(*args)


def response(status: int = 200, body: dict | bytes | io.BytesIO = b"{}",
             headers: dict | None = None) -> HttpStream:
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    if isinstance(body, bytes):
        body = io.BytesIO(body)
    return HttpStream(status, headers or {}, body, time.perf_counter(), connect_s=0.05, wait_s=0.2)


class FakeOpen:
    """Records each request and replies from a script (one reply per call)."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, dict, bytes, float]] = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, body, timeout))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def body(self, i: int = -1) -> dict:
        return json.loads(self.calls[i][2])
