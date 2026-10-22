import pytest

from config import Settings
from llm import (FakeLlm, LlmError, Message, Reply, api_key, error_from_status, estimate_tokens,
                 iter_sse, make_client, parse_json, with_retries)
from llm_fakes import CutAfter, response


# ---------------------------------------------------------------- SSE

def test_sse_events_split_on_blank_lines():
    lines = [b"event: a\n", b'data: {"x": 1}\n', b"\n", b"event: b\n", b"data: 2\n", b"\n"]
    assert list(iter_sse(lines)) == [("a", '{"x": 1}'), ("b", "2")]


def test_sse_crlf_comments_and_multiline_data():
    lines = [b": keep-alive\r\n", b"data: line one\r\n", b"data: line two\r\n", b"\r\n"]
    assert list(iter_sse(lines)) == [("message", "line one\nline two")]


def test_sse_half_event_at_the_end_is_dropped():
    lines = [b"event: a\n", b"data: 1\n", b"\n", b"event: b\n", b'data: {"cut']
    assert list(iter_sse(lines)) == [("a", "1")]


def test_http_stream_maps_a_dropped_connection():
    resp = response(body=CutAfter(b"data: 1\n\n", 1))
    with pytest.raises(LlmError) as e:
        list(resp.lines())
    assert e.value.kind == "network" and e.value.retryable


def test_http_stream_maps_a_read_timeout():
    resp = response(body=CutAfter(b"", 0, TimeoutError()))
    with pytest.raises(LlmError, match="quiet") as e:
        list(resp.lines())
    assert e.value.kind == "timeout"


# ---------------------------------------------------------------- errors

@pytest.mark.parametrize("status,body,kind,retryable", [
    (401, {"error": {"message": "bad key"}}, "auth", False),
    (404, {"error": {"message": "no model", "code": "model_not_found"}}, "bad_request", False),
    (400, {"error": {"message": "Unsupported parameter"}}, "bad_request", False),
    (429, {"error": {"message": "slow down", "code": "rate_limit_exceeded"}}, "rate_limit", True),
    (429, {"error": {"message": "You exceeded your current quota", "code": "insufficient_quota"}},
     "rate_limit", False),
    (500, b"<html>oops</html>", "server", True),
    (408, {}, "timeout", True),
])
def test_status_to_error_kind(status, body, kind, retryable):
    import json
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    e = error_from_status(status, {}, raw)
    assert (e.kind, e.retryable, e.status) == (kind, retryable, status)


def test_retry_after_header_is_read():
    assert error_from_status(429, {"retry-after": "3"}, b"{}").retry_after == 3.0


def test_parse_json_rejects_garbage():
    with pytest.raises(LlmError) as e:
        parse_json(b"<html>")
    assert e.value.kind == "bad_response"
    with pytest.raises(LlmError):
        parse_json(b"[1, 2]")


# ---------------------------------------------------------------- keys

def test_missing_key_names_the_variable():
    with pytest.raises(LlmError, match="OPENAI_API_KEY") as e:
        api_key("OPENAI_API_KEY")
    assert e.value.kind == "auth"


def test_key_from_env_file_and_env_wins(tmp_path):
    f = tmp_path / ".env"
    f.write_text("OPENAI_API_KEY=sk-file\n", encoding="utf-8")
    assert api_key("OPENAI_API_KEY", env={}, env_file=f) == "sk-file"
    assert api_key("OPENAI_API_KEY", env={"OPENAI_API_KEY": "sk-env"}, env_file=f) == "sk-env"


def test_key_with_a_line_break_is_refused_without_echoing_it():
    with pytest.raises(LlmError) as e:
        api_key("K", env={"K": "sk-se cret"})
    assert "se cret" not in str(e.value)


# ---------------------------------------------------------------- retries

def test_retry_then_success():
    script = [LlmError("server", "busy", retryable=True), Reply("hi", "fake", "m")]
    waits = []

    def call():
        r = script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    reply = with_retries(call, retries=2, sleep=waits.append)
    assert reply.text == "hi" and reply.attempts == 2 and waits == [0.5]


def test_no_retry_once_text_was_shown():
    calls = []

    def call():
        calls.append(1)
        raise LlmError("stream_cut", "dropped", retryable=True, partial_text="Sure, the")
    with pytest.raises(LlmError):
        with_retries(call, retries=3, sleep=lambda s: None)
    assert len(calls) == 1


def test_no_retry_for_a_wrong_key():
    calls = []

    def call():
        calls.append(1)
        raise LlmError("auth", "bad key")
    with pytest.raises(LlmError):
        with_retries(call, retries=3, sleep=lambda s: None)
    assert len(calls) == 1


def test_retry_wait_is_capped():
    waits = []
    script = [LlmError("rate_limit", "x", retryable=True, retry_after=999), Reply("ok", "f", "m")]

    def call():
        r = script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    with_retries(call, sleep=waits.append)
    assert waits == [10.0]


# ---------------------------------------------------------------- fake + registry

def test_fake_streams_word_by_word_and_records_calls():
    fake = FakeLlm("one two three")
    pieces = []
    reply = fake.stream("be brief", [Message("user", "count")], pieces.append)
    assert pieces == ["one", " two", " three"] and "".join(pieces) == reply.text
    assert fake.calls == [("be brief", [Message("user", "count")])]


def test_fake_error_streams_partial_text_then_raises():
    fake = FakeLlm(LlmError("stream_cut", "dropped", partial_text="Half an"))
    pieces = []
    with pytest.raises(LlmError):
        fake.stream("", [Message("user", "hi")], pieces.append)
    assert pieces == ["Half an"]


def test_unknown_role_and_provider_refused():
    with pytest.raises(ValueError):
        Message("system", "hi")      # instructions travel separately
    with pytest.raises(ValueError, match="openai"):
        make_client("nope")


def test_make_client_checks_settings():
    with pytest.raises(ValueError):
        make_client("fake", Settings(effort="low", temperature=0.5))
    assert make_client("openai").name == "openai"


def test_token_estimate_is_rough_but_sane():
    assert estimate_tokens("") == 0
    assert estimate_tokens("What's the weather like?") == 6
