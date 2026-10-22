"""The OpenAI adapter against recorded shapes of real responses (2026-10-22)."""

import pytest

from config import Settings
from llm import LlmError, Message
from llm_fakes import CutAfter, FakeOpen, response, sse
from openai_llm import URL, OpenAIResponses

MSG = [Message("user", "What's the weather in Pune?"),
       Message("assistant", "I can't check live weather."),
       Message("user", "What about tomorrow?")]
USAGE = {"input_tokens": 55, "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
         "output_tokens": 23, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 78}
SSE = {"content-type": "text/event-stream; charset=utf-8", "x-request-id": "req_1"}


def done(text="Hi there.", status="completed", reason=None, model="gpt-6-luna"):
    return {"id": "resp_1", "object": "response", "status": status, "model": model, "error": None,
            "incomplete_details": {"reason": reason} if reason else None,
            "output": [{"id": "msg_1", "type": "message", "status": status, "role": "assistant",
                        "content": [{"type": "output_text", "annotations": [], "text": text}]}],
            "usage": USAGE}


def deltas(*pieces):
    return [{"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0,
             "content_index": 0, "delta": p, "sequence_number": 4 + i} for i, p in enumerate(pieces)]


def stream_body(*pieces, end="response.completed", status="completed", reason=None):
    text = "".join(pieces)
    head = [{"type": "response.created", "response": {**done(""), "status": "in_progress", "output": []},
             "sequence_number": 0},
            {"type": "response.output_item.added", "output_index": 0, "sequence_number": 2,
             "item": {"id": "msg_1", "type": "message", "content": []}}]
    tail = [{"type": "response.output_text.done", "text": text, "sequence_number": 90},
            {"type": end, "response": done(text, status, reason), "sequence_number": 99}]
    return sse(*head, *deltas(*pieces), *tail)


def client(*replies, **settings):
    fake = FakeOpen(*replies)
    return OpenAIResponses(Settings(**settings), key="sk-test", open_=fake,
                           sleep=lambda s: None), fake


# ---------------------------------------------------------------- request

def test_request_body_matches_the_api_contract():
    c, fake = client(response(body=done()))
    c.complete("Be brief.", MSG)
    url, headers, _, timeout = fake.calls[0]
    assert url == URL and timeout == 30.0
    assert headers["Authorization"] == "Bearer sk-test"
    assert fake.body() == {
        "model": "gpt-6-luna", "instructions": "Be brief.",
        "input": [{"role": "user", "content": "What's the weather in Pune?"},
                  {"role": "assistant", "content": "I can't check live weather."},
                  {"role": "user", "content": "What about tomorrow?"}],
        "reasoning": {"effort": "none"}, "max_output_tokens": 400, "store": False}


def test_optional_settings_are_sent_only_when_set():
    c, fake = client(response(body=stream_body("x"), headers=SSE),
                     effort=None, max_output_tokens=None, temperature=None, verbosity="low")
    c.stream("", MSG[:1], lambda t: None)
    body = fake.body()
    assert "reasoning" not in body and "max_output_tokens" not in body
    assert "instructions" not in body and "temperature" not in body
    assert body["text"] == {"verbosity": "low"} and body["stream"] is True


def test_dry_run_never_shows_the_key():
    c, _ = client()
    text = c.describe_request("Be brief.", MSG, stream=True)
    assert "$OPENAI_API_KEY" in text and "sk-test" not in text
    assert '"stream": true' in text and "What about tomorrow?" in text


def test_missing_key_fails_before_any_request():
    fake = FakeOpen()
    with pytest.raises(LlmError) as e:
        OpenAIResponses(open_=fake).complete("", MSG)
    assert e.value.kind == "auth" and fake.calls == []


# ---------------------------------------------------------------- complete

def test_complete_reply_with_usage_and_timings():
    c, _ = client(response(body=done("  I can't check that.  "), headers={"x-request-id": "req_9"}))
    r = c.complete("", MSG)
    assert r.text == "I can't check that." and r.complete and r.reason is None
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.reasoning_tokens,
            r.usage.cached_tokens, r.usage.total_tokens) == (55, 23, 0, 0, 78)
    assert r.request_id == "req_9" and r.response_id == "resp_1"
    # Not streaming: the first word is only usable when the whole reply is.
    assert r.timings.first_text_s == r.timings.total_s


def test_incomplete_reply_is_flagged_not_hidden():
    c, _ = client(response(body=done("## The Cartographer of", "incomplete", "max_output_tokens")))
    r = c.complete("", MSG)
    assert not r.complete and r.reason == "max_output_tokens"


def test_failed_status_raises():
    body = {**done(""), "status": "failed", "error": {"code": "server_error", "message": "boom"}}
    c, _ = client(response(body=body), response(body=body), response(body=body), retries=0)
    with pytest.raises(LlmError, match="boom") as e:
        c.complete("", MSG)
    assert e.value.kind == "server"


def test_refusal_text_is_returned():
    body = done()
    body["output"][0]["content"] = [{"type": "refusal", "refusal": "I can't help with that."}]
    c, _ = client(response(body=body))
    assert c.complete("", MSG).text == "I can't help with that."


@pytest.mark.parametrize("body", [b"<html>502</html>", {"status": "completed"},
                                  {"status": "queued", "output": []}])
def test_malformed_responses_are_bad_response(body):
    c, _ = client(response(body=body))
    with pytest.raises(LlmError) as e:
        c.complete("", MSG)
    assert e.value.kind == "bad_response"


def test_http_errors_are_mapped_and_retried():
    busy = response(503, {"error": {"message": "overloaded"}})
    c, fake = client(busy, response(body=done("ok")))
    r = c.complete("", MSG)
    assert r.text == "ok" and r.attempts == 2 and len(fake.calls) == 2


def test_unknown_model_is_not_retried():
    c, fake = client(response(404, {"error": {"message": "The model `x` does not exist",
                                              "code": "model_not_found"}}))
    with pytest.raises(LlmError, match="does not exist") as e:
        c.complete("", MSG)
    assert e.value.kind == "bad_request" and len(fake.calls) == 1


# ---------------------------------------------------------------- stream

def test_stream_shows_deltas_in_order_and_returns_a_complete_reply():
    c, _ = client(response(body=stream_body("1", ",", " 2", ".", ), headers=SSE))
    shown = []
    r = c.stream("", MSG, shown.append)
    assert shown == ["1", ",", " 2", "."] and r.text == "1, 2."
    assert r.complete and r.usage.output_tokens == 23 and r.request_id == "req_1"
    assert 0 < r.timings.first_text_s <= r.timings.total_s


def test_stream_incomplete_is_flagged():
    body = stream_body("Once upon", end="response.incomplete", status="incomplete",
                       reason="max_output_tokens")
    c, _ = client(response(body=body, headers=SSE))
    r = c.stream("", MSG, lambda t: None)
    assert not r.complete and r.reason == "max_output_tokens" and r.text == "Once upon"


def test_stream_that_stops_before_completed_is_not_an_answer():
    body = stream_body("Sure, the", " forecast")
    cut = body[:body.index(b"response.output_text.done") - len("event: ")]
    c, _ = client(response(body=cut, headers=SSE), retries=0)
    with pytest.raises(LlmError) as e:
        c.stream("", MSG, lambda t: None)
    assert e.value.kind == "stream_cut" and e.value.partial_text == "Sure, the forecast"


def test_connection_drop_mid_stream_keeps_partial_and_is_not_retried():
    body = stream_body("Sure,", " the", " forecast")
    lines_before_cut = body[:body.index(b'" the"')].count(b"\n") + 2  # after the " the" event
    c, fake = client(response(body=CutAfter(body, lines_before_cut), headers=SSE),
                     response(body=stream_body("never"), headers=SSE))
    shown = []
    with pytest.raises(LlmError) as e:
        c.stream("", MSG, shown.append)
    assert e.value.kind == "network" and e.value.partial_text == "Sure, the"
    assert len(fake.calls) == 1        # text was already shown: no retry


def test_drop_before_any_text_is_retried():
    body = stream_body("ok")
    c, fake = client(response(body=CutAfter(body, 2), headers=SSE),
                     response(body=body, headers=SSE))
    r = c.stream("", MSG, lambda t: None)
    assert r.text == "ok" and r.attempts == 2


def test_failed_and_error_events_raise_with_partial_text():
    failed = sse(*deltas("Half"), {"type": "response.failed", "response": {
        "status": "failed", "error": {"code": "server_error", "message": "boom"}}})
    error = sse(*deltas("Half"), {"type": "error", "code": "invalid_prompt", "message": "nope"})
    for body, kind in ((failed, "server"), (error, "failed")):
        c, _ = client(response(body=body, headers=SSE))
        with pytest.raises(LlmError) as e:
            c.stream("", MSG, lambda t: None)
        assert e.value.kind == kind and e.value.partial_text == "Half"


def test_http_error_on_a_stream_request_is_json():
    c, _ = client(response(400, {"error": {"message": "Unsupported parameter: 'temperature'"}},
                           headers={"content-type": "application/json"}))
    with pytest.raises(LlmError, match="temperature") as e:
        c.stream("", MSG, lambda t: None)
    assert e.value.kind == "bad_request"


def test_200_without_an_event_stream_is_bad_response():
    c, _ = client(response(body=done(), headers={"content-type": "application/json"}))
    with pytest.raises(LlmError, match="event stream"):
        c.stream("", MSG, lambda t: None)


def test_garbage_event_is_bad_response():
    c, _ = client(response(body=b"event: x\ndata: {not json\n\n", headers=SSE))
    with pytest.raises(LlmError) as e:
        c.stream("", MSG, lambda t: None)
    assert e.value.kind == "bad_response"
