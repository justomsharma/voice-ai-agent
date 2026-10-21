"""The STT interface every provider plugs into, plus what they all share:
API keys, one timed HTTPS POST, error kinds and retries.

    audio_prep.prepare()  ->  provider.transcribe(prepared)  ->  Transcript
                               (openai_stt / deepgram_stt / fake)

Provider-specific code (URLs, field names, JSON shapes) lives in ONE file
per provider. Everything else (CLI, evaluation, the live loop) only talks
to this interface, so comparing providers is a flag, not a rewrite.

Why http.client and not `requests`?
-----------------------------------
`requests.post()` is one opaque call. Doing the steps by hand lets us time
them separately, which is the point of this stage:

    connect   DNS + TCP + TLS handshake      network distance, paid every new connection
    upload    hand the body to the OS        grows with file size (WAV = 32 KB/s at 16 kHz)
    wait      until the first response byte  the model actually transcribing (+ queueing)
    download  read the JSON                  tiny

A voice agent pays every one of these per utterance, which is one reason
real agents stream audio over a socket that is already open.
"""

import http.client
import json
import os
import socket
import ssl
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from audio_prep import AudioContract, Prepared

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
MAX_RETRY_WAIT_S = 10.0


class SttError(Exception):
    """One error type with a `kind`, so callers can react without parsing text.

    kind: auth | rate_limit | bad_request | server | timeout | network | bad_response
    retryable: worth trying again (a busy server), or not (a wrong key)
    """

    def __init__(self, kind: str, message: str, retryable: bool = False,
                 status: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after

    def __str__(self) -> str:
        code = f" HTTP {self.status}" if self.status else ""
        return f"[{self.kind}{code}] {super().__str__()}"


@dataclass(frozen=True)
class Timings:
    connect_s: float = 0.0
    upload_s: float = 0.0
    wait_s: float = 0.0
    download_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.connect_s + self.upload_s + self.wait_s + self.download_s


@dataclass
class Transcript:
    text: str
    provider: str
    model: str
    audio_s: float                  # how much audio was sent
    timings: Timings = field(default_factory=Timings)  # the attempt that succeeded
    attempts: int = 1
    language: str | None = None     # detected or requested, if the API says
    confidence: float | None = None
    request_id: str | None = None   # quote this when asking the provider for help
    extra: dict = field(default_factory=dict)  # anything else worth printing (usage, ...)

    @property
    def rtf(self) -> float:
        """Real-time factor: request time / audio time. 0.2 = 1 s of speech
        took 0.2 s. Batch STT needs the WHOLE recording first, so the user
        waits (speech duration + request time). That's the batch penalty."""
        return self.timings.total_s / self.audio_s if self.audio_s else 0.0


class BatchProvider(Protocol):
    """All a batch STT provider must offer."""
    name: str
    model: str
    contract: AudioContract

    def transcribe(self, audio: Prepared) -> Transcript: ...

    def describe_request(self, audio: Prepared) -> str:
        """What would be sent (no key, no network): for --dry-run."""
        ...


# ---------------------------------------------------------------- keys

def load_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Parse KEY=VALUE lines. Comments, blanks and quotes are handled; nothing else."""
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def api_key(var: str, env: dict | None = None, env_file: Path = ENV_FILE) -> str:
    """Read a key from the environment, else from .env. Never printed or logged."""
    env = os.environ if env is None else env
    value = (env.get(var) or load_env_file(env_file).get(var) or "").strip()
    if not value:
        raise SttError("auth", f"{var} is not set. Copy .env.example to .env and paste "
                               "your key there (or set the environment variable).")
    # A stray newline or quote from copy-paste would make the HTTP library
    # raise an error that QUOTES the header, key included. Refuse it here,
    # without ever echoing the value.
    if not all(33 <= ord(c) <= 126 for c in value):
        raise SttError("auth", f"{var} contains spaces, line breaks or non-ASCII characters; "
                               "re-paste the key on one line")
    return value


# ---------------------------------------------------------------- HTTP

@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]   # lower-case names
    body: bytes
    timings: Timings


Post = Callable[[str, dict[str, str], bytes, float], HttpResponse]


def http_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> HttpResponse:
    """One HTTPS POST on a fresh connection, timed step by step."""
    u = urlsplit(url)
    path = u.path + (f"?{u.query}" if u.query else "")
    if u.scheme == "http":   # only for tests against a local server
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    else:
        conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=timeout,
                                           context=ssl.create_default_context())
    try:
        t0 = time.perf_counter()
        conn.connect()
        t1 = time.perf_counter()
        conn.putrequest("POST", path)
        for name, value in {**headers, "Content-Length": str(len(body))}.items():
            conn.putheader(name, value)
        conn.endheaders()
        conn.send(body)
        t2 = time.perf_counter()
        resp = conn.getresponse()            # blocks until the first byte comes back
        t3 = time.perf_counter()
        data = resp.read()
        t4 = time.perf_counter()
    except (TimeoutError, socket.timeout) as e:
        raise SttError("timeout", f"no answer within {timeout:g} s", retryable=True) from e
    except ValueError as e:
        # http.client's "Invalid header value b'Bearer sk-...'" would print
        # the key, so don't chain or repeat its message.
        raise SttError("bad_request", "invalid request header (check the API key for "
                                      "stray characters)") from None
    except (OSError, http.client.HTTPException) as e:
        # DNS failure, refused connection, Wi-Fi dropped mid-request, ...
        raise SttError("network", f"{type(e).__name__}: {e}", retryable=True) from e
    finally:
        conn.close()
    return HttpResponse(resp.status, {k.lower(): v for k, v in resp.getheaders()}, data,
                        Timings(t1 - t0, t2 - t1, t3 - t2, t4 - t3))


def check_status(resp: HttpResponse) -> None:
    """Turn an HTTP error status into an SttError with the API's own message."""
    if resp.status < 400:
        return
    message = resp.body[:300].decode("utf-8", "replace")
    code = ""
    try:
        err = json.loads(resp.body)
        if isinstance(err.get("error"), dict):            # OpenAI style
            message, code = err["error"].get("message", message), err["error"].get("code") or ""
        elif "err_msg" in err:                              # Deepgram style
            message, code = err["err_msg"], err.get("err_code", "")
    except (ValueError, AttributeError):
        pass
    s = resp.status
    retry_after = None
    if "retry-after" in resp.headers:
        try:
            retry_after = float(resp.headers["retry-after"])
        except ValueError:
            pass
    if s in (401, 403):
        raise SttError("auth", f"key rejected: {message}", status=s)
    if s == 429:
        # Out of credit also comes back as 429, and retrying can't fix that.
        quota = "quota" in str(code) or "quota" in message.lower()
        raise SttError("rate_limit", message, retryable=not quota, status=s, retry_after=retry_after)
    if s == 408:
        raise SttError("timeout", message, retryable=True, status=s)
    if s >= 500:
        raise SttError("server", message, retryable=True, status=s, retry_after=retry_after)
    raise SttError("bad_request", message, status=s)  # 400 / 413 / 415 ...: our fault


def with_retries(call: Callable[[], Transcript], retries: int = 2, base_wait: float = 0.5,
                 sleep: Callable[[float], None] | None = None,
                 log: Callable[[str], None] | None = None) -> Transcript:
    """Retry only what can succeed next time, with exponential backoff."""
    sleep = sleep or time.sleep  # looked up per call, so tests can patch it
    for attempt in range(retries + 1):
        try:
            result = call()
            result.attempts = attempt + 1
            return result
        except SttError as e:
            if not e.retryable or attempt == retries:
                raise
            wait = min(MAX_RETRY_WAIT_S, e.retry_after if e.retry_after is not None
                       else base_wait * 2 ** attempt)
            if log:
                log(f"attempt {attempt + 1} failed: {e}. Retrying in {wait:.1f} s")
            sleep(wait)
    raise AssertionError("unreachable")


def parse_json(resp: HttpResponse) -> dict:
    try:
        data = json.loads(resp.body)
    except ValueError as e:
        raise SttError("bad_response", f"not JSON: {resp.body[:100]!r}") from e
    if not isinstance(data, dict):
        raise SttError("bad_response", f"expected a JSON object, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------- registry

PROVIDERS = ("openai", "deepgram")


def make_provider(name: str, model: str | None = None, languages: list[str] | None = None,
                  prompt: str | None = None, keywords: list[str] | None = None,
                  timeout: float = 30.0) -> BatchProvider:
    """Build a provider by name. Imports lazily so one broken adapter can't break the others."""
    if name == "openai":
        from openai_stt import OpenAIBatch
        return OpenAIBatch(model=model, languages=languages, prompt=prompt,
                           keywords=keywords, timeout=timeout)
    if name == "deepgram":
        from deepgram_stt import DeepgramBatch  # no free-text prompt in this API
        return DeepgramBatch(model=model, languages=languages, keywords=keywords, timeout=timeout)
    raise ValueError(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")


class FakeProvider:
    """Answers instantly from a script: for tests and offline demos. No network."""
    name = "fake"
    model = "fake-1"

    def __init__(self, text: str = "hello world", contract: AudioContract | None = None,
                 error: SttError | None = None) -> None:
        self.text = text
        self.contract = contract or AudioContract(16000)
        self.error = error
        self.received: list[Prepared] = []

    def transcribe(self, audio: Prepared) -> Transcript:
        self.received.append(audio)
        if self.error:
            raise self.error
        return Transcript(self.text, self.name, self.model, audio.seconds,
                          Timings(0.01, 0.0, 0.1, 0.0))

    def describe_request(self, audio: Prepared) -> str:
        return f"fake request: {audio.seconds:.2f} s of audio"
