import base64
import json
import queue

import pytest

from realtime import (CLOSED, COMMITTED, ERROR, FINAL, OTHER, PARTIAL, READY, Board, RealtimeSession,
                      Update, append_event, parse_event, session_update)
from stt import SttError


class FakeWs:
    """A WebSocket whose 'server' is a Python function reacting to what we send."""

    def __init__(self, server=None, break_after: int | None = None) -> None:
        self.sent: list[dict] = []
        self.inbox: queue.Queue = queue.Queue()
        self.server = server or (lambda event, ws: None)
        self.break_after = break_after
        self.closed = False

    def push(self, event: dict) -> None:
        self.inbox.put(json.dumps(event))

    def send(self, data: str) -> None:
        if self.break_after is not None and len(self.sent) >= self.break_after:
            raise ConnectionError("connection reset")
        event = json.loads(data)
        self.sent.append(event)
        self.server(event, self)

    def recv(self, timeout: float | None = None):
        try:
            item = self.inbox.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True
        self.inbox.put(ConnectionError("closed by us"))


class FakeServer:
    """Behaves like gpt-live-transcribe: a partial while audio arrives, a final per commit."""

    def __init__(self, finals: list[str], reject: str | None = None) -> None:
        self.finals = finals
        self.reject = reject
        self.turn = 0
        self.spoke = False

    def __call__(self, event: dict, ws: FakeWs) -> None:
        t = event["type"]
        item = f"item_{self.turn}"
        if t == "session.update":
            ws.push({"type": "session.created"})
            ws.push({"type": "error", "error": {"message": self.reject}} if self.reject
                    else {"type": "session.updated"})
        elif t == "input_audio_buffer.append" and not self.spoke:
            self.spoke = True
            ws.push({"type": "conversation.item.input_audio_transcription.delta",
                     "item_id": item, "delta": self.finals[self.turn].split()[0]})
        elif t == "input_audio_buffer.clear":
            self.spoke = False
        elif t == "input_audio_buffer.commit":
            ws.push({"type": "input_audio_buffer.committed", "item_id": item})
            if self.turn < len(self.finals):
                ws.push({"type": "conversation.item.input_audio_transcription.completed",
                         "item_id": item, "transcript": self.finals[self.turn]})
            self.turn += 1
            self.spoke = False


def connect_to(ws):
    calls = []

    def connect(url, headers, timeout):
        calls.append((url, headers))
        return ws

    connect.calls = calls
    return connect


def test_session_update_shape():
    e = session_update(languages=["en", "hi"], delay="low", keywords=["Pune"])
    audio_in = e["session"]["audio"]["input"]
    assert e["type"] == "session.update" and e["session"]["type"] == "transcription"
    assert audio_in["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_in["turn_detection"] is None  # no server VAD: we commit
    assert audio_in["transcription"] == {"model": "gpt-live-transcribe", "languages": ["en", "hi"],
                                         "delay": "low", "keywords": ["Pune"]}
    assert "language" not in audio_in["transcription"]


def test_append_event_is_base64_pcm():
    e = append_event(b"\x01\x00\xfe\xff")
    assert e["type"] == "input_audio_buffer.append"
    assert base64.b64decode(e["audio"]) == b"\x01\x00\xfe\xff"


@pytest.mark.parametrize("event,kind,item,text", [
    ({"type": "session.updated"}, READY, None, ""),
    ({"type": "input_audio_buffer.committed", "item_id": "a"}, COMMITTED, "a", ""),
    ({"type": "conversation.item.input_audio_transcription.delta", "item_id": "a", "delta": "I w"},
     PARTIAL, "a", "I w"),
    ({"type": "conversation.item.input_audio_transcription.completed", "item_id": "a",
      "transcript": "I want."}, FINAL, "a", "I want."),
    ({"type": "conversation.item.input_audio_transcription.failed", "item_id": "a",
      "error": {"message": "bad audio"}}, ERROR, "a", "bad audio"),
    ({"type": "error", "error": {"message": "Invalid value"}}, ERROR, None, "Invalid value"),
    ({"type": "rate_limits.updated"}, OTHER, None, ""),
])
def test_parse_event(event, kind, item, text):
    u = parse_event(event, at=1.5)
    assert (u.kind, u.item_id, u.text, u.at) == (kind, item, text, 1.5)


def test_open_configures_and_waits_for_ready():
    ws = FakeWs(FakeServer(["hi"]))
    connect = connect_to(ws)
    s = RealtimeSession(key="sk-test", languages=["en"], connect=connect)
    assert s.open() >= 0
    assert connect.calls[0][1] == {"Authorization": "Bearer sk-test"}
    assert connect.calls[0][0].startswith("wss://api.openai.com/v1/realtime")
    assert ws.sent[0]["type"] == "session.update"
    s.close()
    assert ws.closed


def test_session_round_trip_partial_then_final():
    ws = FakeWs(FakeServer(["book a ticket"]))
    s = RealtimeSession(key="k", connect=connect_to(ws))
    s.open()
    s.append(b"\x00\x00" * 480)
    s.commit()
    updates = []
    while len(updates) < 3:
        updates += s.poll(timeout=1.0)
    assert [u.kind for u in updates] == [PARTIAL, COMMITTED, FINAL]
    assert updates[-1].text == "book a ticket"
    assert s.bytes_sent > 0
    s.close()


def test_session_rejected():
    ws = FakeWs(FakeServer([], reject="Invalid model"))
    with pytest.raises(SttError, match="Invalid model") as e:
        RealtimeSession(key="k", connect=connect_to(ws)).open()
    assert e.value.kind == "bad_request"


def test_session_never_ready_times_out():
    ws = FakeWs()  # a server that never answers
    with pytest.raises(SttError) as e:
        RealtimeSession(key="k", connect=connect_to(ws)).open(timeout=0.2)
    assert e.value.kind == "timeout"


class HandshakeRefused(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = type("R", (), {"status_code": status})()


@pytest.mark.parametrize("error,kind", [
    (HandshakeRefused(401), "auth"), (HandshakeRefused(429), "rate_limit"),
    (HandshakeRefused(503), "server"), (HandshakeRefused(404), "bad_request"),
    (TimeoutError(), "timeout"), (OSError("getaddrinfo failed"), "network")])
def test_handshake_errors(error, kind):
    def connect(url, headers, timeout):
        raise error

    with pytest.raises(SttError) as e:
        RealtimeSession(key="k", connect=connect).open()
    assert e.value.kind == kind


def test_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("stt.ENV_FILE", tmp_path / "none")
    with pytest.raises(SttError) as e:
        RealtimeSession(connect=connect_to(FakeWs())).open()
    assert e.value.kind == "auth"


def test_send_on_dead_socket_is_network_error():
    ws = FakeWs(FakeServer(["x"]), break_after=1)
    s = RealtimeSession(key="k", connect=connect_to(ws))
    s.open()
    with pytest.raises(SttError) as e:
        s.append(b"\x00\x00")
    assert e.value.kind == "network"


def test_reader_reports_a_dropped_connection():
    ws = FakeWs(FakeServer(["x"]))
    s = RealtimeSession(key="k", connect=connect_to(ws))
    s.open()
    ws.inbox.put(ConnectionError("network unreachable"))
    [u] = s.poll(timeout=1.0)
    assert u.kind == CLOSED and "unreachable" in u.text


# ------------------------------------------------------------- Board

def test_board_partial_then_final():
    b = Board()
    b.start_turn(at=0.0)
    b.apply(Update(PARTIAL, "a", "I want", at=0.5))
    t = b.apply(Update(PARTIAL, "a", " to", at=0.6))
    assert (t.partial, t.final, t.text, t.first_text_at) == ("I want to", None, "I want to", 0.5)
    b.committed(at=1.0, last_loud_at=0.4, speech=(0.1, 0.9))
    assert b.pending() == [t]
    b.apply(Update(FINAL, "a", " I want to book. ", at=1.3))
    assert (t.final, t.text, t.final_at, t.done) == ("I want to book.", "I want to book.", 1.3, True)
    assert b.pending() == []


def test_board_out_of_order_finals_match_by_item_id():
    b = Board()
    for n in range(2):
        b.start_turn(at=n)
        b.committed(at=n + 0.5, last_loud_at=None, speech=None)
    b.apply(Update(COMMITTED, "first"))
    b.apply(Update(COMMITTED, "second"))
    b.apply(Update(FINAL, "second", "two"))  # arrives first
    b.apply(Update(FINAL, "first", "one"))
    assert [t.final for t in b.turns] == ["one", "two"]


def test_board_errors():
    b = Board()
    b.start_turn(at=0)
    b.committed(at=1, last_loud_at=None, speech=None)
    b.apply(Update(COMMITTED, "a"))
    b.apply(Update(ERROR, "a", "audio too short"))
    b.apply(Update(ERROR, None, "rate limited"))
    assert b.turns[0].error == "audio too short" and b.turns[0].done
    assert b.errors == ["rate limited"]


def test_board_drop_last_only_uncommitted():
    b = Board()
    b.start_turn(at=0)
    b.drop_last()
    assert b.turns == []
    b.start_turn(at=0)
    b.committed(at=1, last_loud_at=None, speech=None)
    b.drop_last()
    assert len(b.turns) == 1


def test_board_ignores_updates_with_no_turn():
    assert Board().apply(Update(FINAL, "x", "hello")) is None


def test_connect_error_redacts_the_key():
    def connect(url, headers, timeout):
        raise ValueError(f"invalid Authorization header: {headers['Authorization']}")

    with pytest.raises(SttError) as e:
        RealtimeSession(key="sk-SECRET", connect=connect).open()
    assert "SECRET" not in str(e.value) and "***" in str(e.value)


def test_failed_setup_closes_the_socket():
    ws = FakeWs(FakeServer([], reject="nope"))
    with pytest.raises(SttError):
        RealtimeSession(key="k", connect=connect_to(ws)).open()
    assert ws.closed


def test_board_late_partial_of_a_cleared_click_does_not_steal_the_next_turn():
    from realtime import CLEARED
    b = Board()
    b.start_turn(at=0)
    b.apply(Update(PARTIAL, "click", "m"))
    b.drop_last()                              # gate sent clear
    b.start_turn(at=1)                         # real speech starts
    b.apply(Update(PARTIAL, "click", "mm"))    # late partial from the click
    b.apply(Update(PARTIAL, "click2", "x"))    # unknown id before the clear is confirmed
    b.apply(Update(CLEARED))
    b.apply(Update(PARTIAL, "real", "book"))
    b.committed(at=2, last_loud_at=None, speech=None)
    b.apply(Update(COMMITTED, "real"))
    b.apply(Update(FINAL, "real", "book a ticket"))
    [t] = b.turns
    assert (t.item_id, t.partial, t.final) == ("real", "book", "book a ticket")
    assert b.pending() == []


def test_board_without_cleared_event_still_gets_the_final():
    b = Board()
    b.start_turn(at=0)
    b.drop_last()
    b.start_turn(at=1)
    b.apply(Update(PARTIAL, "real", "book"))   # skipped: can't tell whose it is
    b.committed(at=2, last_loud_at=None, speech=None)
    b.apply(Update(COMMITTED, "real"))
    b.apply(Update(FINAL, "real", "book a ticket"))
    assert b.turns[0].final == "book a ticket"


def test_board_clearing_ends_at_next_commit_even_without_cleared():
    # If the server never confirms the clear, partials must come back after
    # the next commit instead of vanishing for the rest of the session.
    b = Board()
    b.start_turn(at=0)
    b.drop_last()                        # clear sent, no "cleared" ever arrives
    for n, item in enumerate(["t1", "t2", "t3"], start=1):
        b.start_turn(at=n)
        b.apply(Update(PARTIAL, item, "word"))
        b.committed(at=n + 0.5, last_loud_at=None, speech=None)
        b.apply(Update(COMMITTED, item))
        b.apply(Update(FINAL, item, f"turn {n}"))
    assert [t.final for t in b.turns] == ["turn 1", "turn 2", "turn 3"]
    assert [t.partial for t in b.turns] == ["", "word", "word"]  # only turn 1 is unsure
    assert not b.clearing


def test_board_error_reply_to_clear_ends_clearing():
    b = Board()
    b.start_turn(at=0)
    b.drop_last()
    b.apply(Update(ERROR, None, "Unknown event input_audio_buffer.clear"))
    assert not b.clearing and b.errors
