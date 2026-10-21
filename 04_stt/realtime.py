"""Live transcription over a WebSocket: OpenAI `gpt-live-transcribe`.

Batch is "record everything, then upload". Streaming keeps ONE connection
open and sends audio while the person is still talking, so text can come
back before they finish:

    client                                   server
      | --- WebSocket handshake (Bearer key) -->|   once, ~100-300 ms
      | --- session.update (format, model) ---->|
      |<-- session.updated ---------------------|
      | --- input_audio_buffer.append (20 ms) ->|   x many, while speaking
      | --- input_audio_buffer.append ... ----->|
      |<-- ...transcription.delta "I want" -----|   PARTIAL: may still change
      | --- input_audio_buffer.commit --------->|   "this turn is over"
      |<-- input_audio_buffer.committed (id) ---|
      |<-- ...transcription.completed "I want to book a ticket." --|   FINAL

The contract (docs checked 2026-10-21):
  * audio: PCM16, mono, 24000 Hz, little-endian, base64 inside JSON
    (not 16 kHz! So the mic is opened at 24 kHz, or the file is resampled)
  * turn_detection must be null: this model has NO server-side VAD. The
    client decides when a turn ends and sends `commit`. That's our
    Stage 03 VAD's job (see live.py).
  * `delay` (minimal / low / medium / high / xhigh) trades latency for
    accuracy: more delay = the model hears more context before emitting.
  * finals from different turns can arrive out of order: match by item_id.

Connection URL: wss://api.openai.com/v1/realtime?intent=transcription. The
guide doesn't spell this URL out (it links to the WebSocket guide), so it's
a constant you can override with --url if OpenAI moves it.
"""

import base64
import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from audio_prep import AudioContract
from stt import SttError, api_key

URL = "wss://api.openai.com/v1/realtime?intent=transcription"
MODEL = "gpt-live-transcribe"
KEY_VAR = "OPENAI_API_KEY"
CONTRACT = AudioContract(sample_rate=24000, channels=1)
DELAYS = ("minimal", "low", "medium", "high", "xhigh")

READY, COMMITTED, PARTIAL, FINAL, ERROR, CLOSED, OTHER = (
    "ready", "committed", "partial", "final", "error", "closed", "other")


# ---------------------------------------------------------------- events we send

def session_update(model: str = MODEL, languages: list[str] | None = None,
                   delay: str | None = None, prompt: str | None = None,
                   keywords: list[str] | None = None) -> dict:
    transcription: dict = {"model": model}
    if languages:
        transcription["languages"] = languages  # a list; never also send "language"
    if delay:
        transcription["delay"] = delay
    if prompt:
        transcription["prompt"] = prompt
    if keywords:
        transcription["keywords"] = keywords
    return {"type": "session.update",
            "session": {"type": "transcription",
                        "audio": {"input": {
                            "format": {"type": "audio/pcm", "rate": CONTRACT.sample_rate},
                            "transcription": transcription,
                            "turn_detection": None}}}}


def append_event(pcm: bytes) -> dict:
    # JSON can't carry raw bytes, so audio is base64 text: 4 chars per 3 bytes,
    # +33% over the wire. (Deepgram's socket takes raw binary frames instead.)
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")}


COMMIT = {"type": "input_audio_buffer.commit"}
CLEAR = {"type": "input_audio_buffer.clear"}  # throw away uncommitted audio (a false start)


# ---------------------------------------------------------------- events we receive

@dataclass(frozen=True)
class Update:
    kind: str                  # ready | committed | partial | final | error | closed | other
    item_id: str | None = None
    text: str = ""             # partial: the NEW piece only; final: the whole turn
    at: float = 0.0            # when it arrived (perf_counter seconds)
    raw_type: str = ""


def parse_event(event: dict, at: float = 0.0) -> Update:
    t = event.get("type", "")
    item = event.get("item_id")
    if t in ("session.updated", "transcription_session.updated"):
        return Update(READY, at=at, raw_type=t)
    if t == "input_audio_buffer.committed":
        return Update(COMMITTED, item, at=at, raw_type=t)
    if t == "conversation.item.input_audio_transcription.delta":
        return Update(PARTIAL, item, event.get("delta", ""), at, t)
    if t == "conversation.item.input_audio_transcription.completed":
        return Update(FINAL, item, event.get("transcript", ""), at, t)
    if t == "conversation.item.input_audio_transcription.failed":
        msg = (event.get("error") or {}).get("message", "transcription failed")
        return Update(ERROR, item, msg, at, t)
    if t == "error":
        err = event.get("error") or {}
        return Update(ERROR, None, err.get("message") or err.get("code") or str(err), at, t)
    return Update(OTHER, item, at=at, raw_type=t)


# ---------------------------------------------------------------- the connection

def default_connect(url: str, headers: dict[str, str], timeout: float):
    from websockets.sync.client import connect  # lazy: tests use a fake socket
    return connect(url, additional_headers=headers, open_timeout=timeout, max_size=None)


class RealtimeSession:
    """One live transcription connection. Sending happens on the caller's
    thread; a background thread receives, so a slow network never blocks
    audio capture. Received updates queue up until poll() collects them."""

    def __init__(self, key: str | None = None, model: str = MODEL,
                 languages: list[str] | None = None, delay: str | None = None,
                 prompt: str | None = None, keywords: list[str] | None = None,
                 url: str = URL, connect: Callable = default_connect,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.url = url
        self.config = session_update(model, languages, delay, prompt, keywords)
        self._key = key
        self._connect = connect
        self._clock = clock
        self._ws = None
        self._updates: queue.Queue[Update] = queue.Queue()
        self._closing = False
        self.bytes_sent = 0

    def open(self, timeout: float = 10.0) -> float:
        """Connect, configure, wait until the server confirms. Returns the setup time."""
        key = self._key or api_key(KEY_VAR)
        t0 = self._clock()
        try:
            self._ws = self._connect(self.url, {"Authorization": f"Bearer {key}"}, timeout)
        except Exception as e:  # handshake refused, DNS, TLS, timeout...
            raise _connect_error(e) from e
        self.send(self.config)
        deadline = t0 + timeout
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise SttError("timeout", f"no session.updated within {timeout:g} s")
            try:
                update = parse_event(json.loads(self._ws.recv(timeout=remaining)), self._clock())
            except TimeoutError:
                continue
            except Exception as e:
                raise SttError("network", f"connection lost during setup: {e}") from e
            if update.kind == ERROR:
                raise SttError("bad_request", f"session rejected: {update.text}")
            if update.kind == READY:
                break
        threading.Thread(target=self._reader, daemon=True).start()
        return self._clock() - t0

    def _reader(self) -> None:
        while True:
            try:
                message = self._ws.recv()
            except Exception as e:  # ConnectionClosed, network drop
                if not self._closing:
                    self._updates.put(Update(CLOSED, text=str(e) or type(e).__name__, at=self._clock()))
                return
            try:
                event = json.loads(message)
            except ValueError:
                continue
            self._updates.put(parse_event(event, self._clock()))

    def send(self, event: dict) -> None:
        data = json.dumps(event)
        try:
            self._ws.send(data)
        except Exception as e:
            raise SttError("network", f"send failed, connection lost: {e}", retryable=True) from e
        self.bytes_sent += len(data)

    def append(self, pcm: bytes) -> None:
        self.send(append_event(pcm))

    def commit(self) -> None:
        self.send(COMMIT)

    def clear(self) -> None:
        self.send(CLEAR)

    def poll(self, timeout: float = 0.0) -> list[Update]:
        """Everything received so far. With a timeout, wait that long for the first one."""
        out = []
        try:
            out.append(self._updates.get(timeout=timeout) if timeout > 0 else self._updates.get_nowait())
            while True:
                out.append(self._updates.get_nowait())
        except queue.Empty:
            pass
        return out

    def close(self) -> None:
        self._closing = True
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass


def _connect_error(e: Exception) -> SttError:
    status = getattr(getattr(e, "response", None), "status_code", None)  # websockets InvalidStatus
    if status in (401, 403):
        return SttError("auth", "key rejected during WebSocket handshake", status=status)
    if status == 429:
        return SttError("rate_limit", "too many connections / no quota", status=status)
    if status is not None:
        return SttError("server" if status >= 500 else "bad_request",
                        f"handshake refused: {e}", retryable=status >= 500, status=status)
    if isinstance(e, TimeoutError):
        return SttError("timeout", "WebSocket handshake timed out", retryable=True)
    return SttError("network", f"can't connect: {type(e).__name__}: {e}", retryable=True)


# ---------------------------------------------------------------- partial vs final

@dataclass
class Turn:
    """One utterance, from the first audio we sent to its final text."""
    number: int
    speech: tuple[float, float] | None = None  # where it was in the audio (VAD), seconds
    item_id: str | None = None
    partial: str = ""                 # provisional: can still change
    final: str | None = None          # committed: won't change
    error: str | None = None
    first_audio_at: float | None = None
    first_text_at: float | None = None
    last_loud_at: float | None = None  # the moment you actually stopped speaking
    commit_at: float | None = None
    final_at: float | None = None

    @property
    def text(self) -> str:
        return self.final if self.final is not None else self.partial

    @property
    def done(self) -> bool:
        return self.final is not None or self.error is not None


@dataclass
class Board:
    """Keeps partial and final text apart, per turn.

    A partial ("I want to") is a guess the model may still revise; a final
    is committed. Anything downstream (an LLM in Stage 05) should only act on
    finals, or it'll answer a sentence that's about to change.
    """
    turns: list[Turn] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def start_turn(self, at: float) -> Turn:
        turn = Turn(len(self.turns) + 1, first_audio_at=at)
        self.turns.append(turn)
        return turn

    def drop_last(self) -> None:
        """A false start (a click the VAD rejected): forget the turn."""
        if self.turns and self.turns[-1].commit_at is None:
            self.turns.pop()

    def committed(self, at: float, last_loud_at: float | None,
                  speech: tuple[float, float] | None) -> None:
        turn = self.turns[-1]
        turn.commit_at, turn.last_loud_at, turn.speech = at, last_loud_at, speech

    def apply(self, u: Update) -> Turn | None:
        """Route a server update to its turn. Returns that turn, if any."""
        if u.kind == ERROR and u.item_id is None:
            self.errors.append(u.text)
            return None
        if u.kind not in (COMMITTED, PARTIAL, FINAL, ERROR):
            return None
        turn = self._turn_for(u)
        if turn is None:
            return None
        if u.kind == PARTIAL:
            turn.partial += u.text
            if turn.first_text_at is None:
                turn.first_text_at = u.at
        elif u.kind == FINAL:
            turn.final = u.text.strip()
            turn.final_at = u.at
            if turn.first_text_at is None:
                turn.first_text_at = u.at
        elif u.kind == ERROR:
            turn.error = u.text
        return turn

    def _turn_for(self, u: Update) -> Turn | None:
        for turn in self.turns:
            if u.item_id is not None and turn.item_id == u.item_id:
                return turn
        # Unknown id: the server's commits come back in the order we sent
        # them, so it belongs to the oldest turn that has no id yet.
        for turn in self.turns:
            if turn.item_id is None and (u.kind != COMMITTED or turn.commit_at is not None):
                turn.item_id = u.item_id
                return turn
        return None

    def pending(self) -> list[Turn]:
        return [t for t in self.turns if t.commit_at is not None and not t.done]
