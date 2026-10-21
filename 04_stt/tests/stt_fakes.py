"""Shared fakes: stand-ins for the network so tests need no key and no internet."""

import json

import numpy as np

from audio_format import AudioFormat
from audio_prep import AudioContract, prepare
from stt import HttpResponse, Timings


def tone(seconds: float, rate: int = 16000, amplitude: int = 8000, freq: float = 300) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    return np.round(amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16).reshape(-1, 1)


def silence(seconds: float, rate: int = 16000) -> np.ndarray:
    return np.zeros((round(seconds * rate), 1), np.int16)


def prepared(seconds: float = 1.0):
    return prepare(AudioFormat(16000, 1), tone(seconds), AudioContract(16000))


class FakePost:
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


def reply(status: int = 200, body: dict | str | bytes = b"{}", headers: dict | None = None) -> HttpResponse:
    if isinstance(body, dict):
        body = json.dumps(body)
    if isinstance(body, str):
        body = body.encode()
    return HttpResponse(status, headers or {}, body, Timings(0.05, 0.01, 0.4, 0.001))
