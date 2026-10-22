"""The LLM interface every provider plugs into, plus what they all share:
API keys, one timed HTTPS request that can be read as it arrives (SSE),
error kinds and retries.

    chat.py / stream.py / llm_eval.py
            │  instructions + messages
            ▼
    client.complete(...)  or  client.stream(..., on_text)  ->  Reply
            (openai_llm / fake)

Provider-specific code (URL, JSON shapes, event names) lives in ONE file
per provider. Everything else only talks to this interface, so comparing
providers is a flag, not a rewrite.

Two ways to get an answer
-------------------------
    complete()  one request, one JSON body at the end. You see nothing
                until the model has written the LAST word.
    stream()    the same request with "stream": true. The server keeps the
                connection open and sends small events as the model writes
                (Server-Sent Events). `on_text` is called for every piece.

Why http.client and not the SDK?
--------------------------------
So each wait can be timed on its own, exactly like Stage 04:

    connect      DNS + TCP + TLS handshake
    wait         request sent -> response headers (the model has started)
    first text   request start -> first visible word   <- what a caller feels
    total        request start -> last byte
"""

import http.client
import json
import os
import socket
import ssl
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.parse import urlsplit

from config import Settings

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
MAX_RETRY_WAIT_S = 10.0
ROLES = ("user", "assistant")


class LlmError(Exception):
    """One error type with a `kind`, so callers can react without parsing text.

    kind: auth | rate_limit | bad_request | server | timeout | network |
          bad_response | stream_cut | failed
    retryable:    worth trying again (a busy server), or not (a wrong key)
    partial_text: text that already arrived before things went wrong. It is
                  NOT an answer: never save it to history as if it were.
    """

    def __init__(self, kind: str, message: str, retryable: bool = False,
                 status: int | None = None, retry_after: float | None = None,
                 partial_text: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after
        self.partial_text = partial_text

    def __str__(self) -> str:
        code = f" HTTP {self.status}" if self.status else ""
        return f"[{self.kind}{code}] {super().__str__()}"


@dataclass(frozen=True)
class Message:
    role: str   # "user" or "assistant". The instructions travel separately.
    text: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, not {self.role!r}")


@dataclass(frozen=True)
class Usage:
    """Token counts as the API reports them. None = the API didn't say."""
    input_tokens: int | None = None      # instructions + history + new message
    output_tokens: int | None = None     # includes reasoning tokens
    reasoning_tokens: int | None = None  # hidden "thinking", billed as output
    cached_tokens: int | None = None     # input the provider had cached (cheaper, faster)
    total_tokens: int | None = None


@dataclass(frozen=True)
class Timings:
    connect_s: float = 0.0              # setup + DNS + TCP + TLS
    wait_s: float = 0.0                 # request sent -> response headers
    first_text_s: float | None = None   # request start -> first visible text
    total_s: float = 0.0                # request start -> last byte


@dataclass
class Reply:
    text: str
    provider: str
    model: str
    status: str = "completed"           # "completed" or "incomplete"
    reason: str | None = None           # why incomplete: "max_output_tokens", ...
    timings: Timings = field(default_factory=Timings)
    usage: Usage = field(default_factory=Usage)
    attempts: int = 1
    request_id: str | None = None       # quote this when asking the provider for help
    response_id: str | None = None

    @property
    def complete(self) -> bool:
        """Only a completed reply is an answer. An incomplete one was cut off."""
        return self.status == "completed"


OnText = Callable[[str], None]


class LlmClient(Protocol):
    """All a provider must offer."""
    name: str
    settings: Settings

    def complete(self, instructions: str, messages: list[Message]) -> Reply: ...

    def stream(self, instructions: str, messages: list[Message], on_text: OnText) -> Reply: ...

    def describe_request(self, instructions: str, messages: list[Message], stream: bool) -> str:
        """What would be sent (no key, no network): for --dry-run."""
        ...


# ---------------------------------------------------------------- keys

def load_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines, same rules as 04_stt/stt.py (copied so this
    stage runs without the audio code). Comments, blanks, quotes; nothing else."""
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if value[:1] in ("'", '"'):
            value = value[1:].split(value[0], 1)[0]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key] = value
    return values


def api_key(var: str, env: dict | None = None, env_file: Path | None = None) -> str:
    """Read a key from the environment, else from .env. Never printed or logged."""
    env = os.environ if env is None else env
    # ENV_FILE is looked up now, not at import, so tests can point it elsewhere.
    value = (env.get(var) or load_env_file(env_file or ENV_FILE).get(var) or "").strip()
    if not value:
        raise LlmError("auth", f"{var} is not set. Copy .env.example to .env and paste "
                               "your key there (or set the environment variable).")
    if not all(33 <= ord(c) <= 126 for c in value):
        raise LlmError("auth", f"{var} contains spaces, line breaks or non-ASCII characters; "
                               "re-paste the key on one line")
    return value


# ---------------------------------------------------------------- HTTP

class HttpStream:
    """An open response: status and headers have arrived, the body may still
    be coming. Read it all at once (`read`) or line by line (`lines`)."""

    def __init__(self, status: int, headers: dict[str, str], body: BinaryIO, t0: float,
                 connect_s: float = 0.0, wait_s: float = 0.0,
                 close: Callable[[], None] | None = None) -> None:
        self.status = status
        self.headers = headers          # lower-case names
        self.t0 = t0                    # perf_counter() when the request started
        self.connect_s = connect_s
        self.wait_s = wait_s
        self._body = body
        self._close = close

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def read(self) -> bytes:
        with _read_errors():
            return self._body.read()

    def lines(self) -> Iterator[bytes]:
        while True:
            with _read_errors():
                line = self._body.readline()
            if not line:
                return
            yield line

    def close(self) -> None:
        if self._close:
            self._close()


@contextmanager
def _read_errors() -> Iterator[None]:
    """Turn socket trouble while reading the body into an LlmError."""
    try:
        yield
    except (TimeoutError, socket.timeout) as e:
        raise LlmError("timeout", "the server went quiet mid-response", retryable=True) from e
    except (OSError, http.client.HTTPException) as e:
        # Wi-Fi dropped, server closed the socket, chunked body cut short ...
        raise LlmError("network", f"{type(e).__name__}: {e}", retryable=True) from e


Open = Callable[[str, dict[str, str], bytes, float], HttpStream]
_tls: ssl.SSLContext | None = None


def tls_context() -> ssl.SSLContext:
    """Built once and reused: loading the system's trusted certificates takes
    20-100 ms on Windows, and paying that on every turn is pure waste."""
    global _tls
    if _tls is None:
        _tls = ssl.create_default_context()
    return _tls


def http_open(url: str, headers: dict[str, str], body: bytes, timeout: float) -> HttpStream:
    """Send one HTTPS POST and return as soon as the response headers arrive.

    `timeout` limits each socket wait (connect, first byte, every read after
    that), not the whole request: a server trickling one byte every 29 s would
    never trip a 30 s timeout.
    """
    t0 = time.perf_counter()
    u = urlsplit(url)
    path = u.path + (f"?{u.query}" if u.query else "")
    if u.scheme == "http":   # only for tests against a local server
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    else:
        conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=timeout,
                                           context=tls_context())
    try:
        conn.connect()
        t1 = time.perf_counter()
        conn.putrequest("POST", path)
        for name, value in {**headers, "Content-Length": str(len(body))}.items():
            conn.putheader(name, value)
        conn.endheaders()
        conn.send(body)
        t2 = time.perf_counter()
        resp = conn.getresponse()            # blocks until the response headers arrive
        t3 = time.perf_counter()
    except (TimeoutError, socket.timeout) as e:
        conn.close()
        raise LlmError("timeout", f"no answer within {timeout:g} s", retryable=True) from e
    except ValueError:
        conn.close()
        # http.client's "Invalid header value b'Bearer sk-...'" would print
        # the key, so don't chain or repeat its message.
        raise LlmError("bad_request", "invalid request header (check the API key for "
                                      "stray characters)") from None
    except (OSError, http.client.HTTPException) as e:
        conn.close()
        raise LlmError("network", f"{type(e).__name__}: {e}", retryable=True) from e
    return HttpStream(resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp, t0,
                      connect_s=t1 - t0, wait_s=t3 - t2, close=conn.close)


def error_from_status(status: int, headers: dict[str, str], body: bytes) -> LlmError:
    """An HTTP error status -> LlmError with the API's own message."""
    message = body[:300].decode("utf-8", "replace")
    code = ""
    try:
        err = json.loads(body)
        if isinstance(err.get("error"), dict):            # OpenAI style
            message = err["error"].get("message") or message
            code = err["error"].get("code") or err["error"].get("type") or ""
    except (ValueError, AttributeError):
        pass
    retry_after = None
    if "retry-after" in headers:
        try:
            retry_after = float(headers["retry-after"])
        except ValueError:
            pass
    if status in (401, 403):
        return LlmError("auth", f"key rejected: {message}", status=status)
    if status == 429:
        # Out of credit also comes back as 429, and retrying can't fix that.
        quota = "quota" in str(code) or "quota" in message.lower()
        return LlmError("rate_limit", message, retryable=not quota, status=status,
                        retry_after=retry_after)
    if status == 408:
        return LlmError("timeout", message, retryable=True, status=status)
    if status >= 500:
        return LlmError("server", message, retryable=True, status=status, retry_after=retry_after)
    return LlmError("bad_request", message, status=status)  # 400 / 404 model ...: our fault


def parse_json(data: bytes | str, what: str = "response") -> dict:
    try:
        obj = json.loads(data)
    except ValueError as e:
        raise LlmError("bad_response", f"{what} is not JSON: {data[:100]!r}") from e
    if not isinstance(obj, dict):
        raise LlmError("bad_response", f"{what}: expected a JSON object, got {type(obj).__name__}")
    return obj


# ---------------------------------------------------------------- SSE

def iter_sse(lines: Iterable[bytes]) -> Iterator[tuple[str, str]]:
    """Server-Sent Events: the format streaming LLM APIs use.

        event: response.output_text.delta          <- the event's name
        data: {"type": "...", "delta": "Hel"}      <- its payload (may span lines)
                                                    <- a blank line ends the event

    Lines starting with ":" are comments (keep-alives). An event cut off
    before its blank line is dropped, as the SSE standard says: half an
    event is not an event.
    """
    name, data = "", []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield name or "message", "\n".join(data)
            name, data = "", []
        elif line.startswith(":"):
            continue
        else:
            key, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if key == "event":
                name = value
            elif key == "data":
                data.append(value)


# ---------------------------------------------------------------- retries

def with_retries(call: Callable[[], Reply], retries: int = 2, base_wait: float = 0.5,
                 sleep: Callable[[float], None] | None = None,
                 log: Callable[[str], None] | None = None) -> Reply:
    """Retry only what can succeed next time, with exponential backoff.

    Never retry once text has been shown: the caller already saw (or, in a
    voice agent, heard) the first half, and a retry would start over.
    """
    sleep = sleep or time.sleep  # looked up per call, so tests can patch it
    for attempt in range(retries + 1):
        try:
            result = call()
            result.attempts = attempt + 1
            return result
        except LlmError as e:
            if not e.retryable or e.partial_text or attempt == retries:
                raise
            wait = min(MAX_RETRY_WAIT_S, e.retry_after if e.retry_after is not None
                       else base_wait * 2 ** attempt)
            if log:
                log(f"attempt {attempt + 1} failed: {e}. Retrying in {wait:.1f} s")
            sleep(wait)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- registry

PROVIDERS = ("openai", "fake")


def make_client(name: str, settings: Settings | None = None) -> LlmClient:
    """Build a client by name. Imports lazily so one broken adapter can't break the others."""
    settings = (settings or Settings()).check()
    if name == "openai":
        from openai_llm import OpenAIResponses
        return OpenAIResponses(settings)
    if name == "fake":
        return FakeLlm(settings=settings)
    raise ValueError(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")


def estimate_tokens(text: str) -> int:
    """Rough rule for English: ~4 characters per token. Only an estimate:
    the API's usage.input_tokens is the real count."""
    return max(1, round(len(text) / 4)) if text else 0


class FakeLlm:
    """Answers instantly from a script: for tests and offline demos. No network.

    Each scripted reply is a string (answered, streamed word by word), an
    LlmError (raised, after streaming its partial_text), or a Reply. When
    the script runs out it says what it received, so you can watch history grow.
    """
    name = "fake"

    def __init__(self, *replies: str | LlmError | Reply, settings: Settings | None = None) -> None:
        self.replies = list(replies)
        self.settings = settings or Settings(model="fake-1")
        self.calls: list[tuple[str, list[Message]]] = []

    def _next(self, instructions: str, messages: list[Message]) -> str | LlmError | Reply:
        self.calls.append((instructions, list(messages)))
        if self.replies:
            return self.replies.pop(0)
        return (f"(fake) I got {len(messages)} message(s). "
                f"You said: {messages[-1].text if messages else ''}")

    def _reply(self, text: str, instructions: str, messages: list[Message], first: float) -> Reply:
        sent = instructions + "".join(m.text for m in messages)
        out = estimate_tokens(text)
        usage = Usage(estimate_tokens(sent), out, 0, 0, estimate_tokens(sent) + out)
        return Reply(text, self.name, "fake-1",      # never pretend a real model answered
                     timings=Timings(0.0, 0.05, first, 0.2), usage=usage)

    def complete(self, instructions: str, messages: list[Message]) -> Reply:
        r = self._next(instructions, messages)
        if isinstance(r, LlmError):
            raise r
        return r if isinstance(r, Reply) else self._reply(r, instructions, messages, 0.2)

    def stream(self, instructions: str, messages: list[Message], on_text: OnText) -> Reply:
        r = self._next(instructions, messages)
        if isinstance(r, LlmError):
            if r.partial_text:
                on_text(r.partial_text)
            raise r
        if isinstance(r, Reply):
            on_text(r.text)
            return r
        for i, word in enumerate(r.split(" ")):
            on_text(word if i == 0 else " " + word)
        return self._reply(r, instructions, messages, 0.06)

    def describe_request(self, instructions: str, messages: list[Message], stream: bool) -> str:
        return (f"fake request: {len(instructions)} chars of instructions, "
                f"{len(messages)} message(s), stream={stream}")
