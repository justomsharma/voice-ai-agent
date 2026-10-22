"""Edge cases found in code review: each one used to crash, leak or mislead."""

import ssl

import pytest

import llm
from config import Settings
from history import Conversation
from llm import FakeLlm, LlmError, Reply, Timings, error_from_status, iter_sse
from llm_eval import model_specs
from llm_fakes import FakeOpen, response
from openai_llm import OpenAIResponses, failed_error, output_text
from stream import Trace, main as stream_main


class FakeConnection:
    """Stands in for HTTPSConnection; connect() raises what we tell it to."""
    instances: list["FakeConnection"] = []

    def __init__(self, *args, error: BaseException, **kwargs) -> None:
        self.error = error
        self.closed = False
        FakeConnection.instances.append(self)

    def connect(self) -> None:
        raise self.error

    def close(self) -> None:
        self.closed = True


def open_with(monkeypatch, error: BaseException):
    FakeConnection.instances.clear()
    monkeypatch.setattr(llm.http.client, "HTTPSConnection",
                        lambda *a, **k: FakeConnection(error=error))
    return lambda: llm.http_open("https://api.example.com/v1/x", {}, b"{}", 5.0)


def test_certificate_error_is_a_tls_problem_not_a_bad_key(monkeypatch):
    call = open_with(monkeypatch, ssl.SSLCertVerificationError("certificate verify failed"))
    with pytest.raises(LlmError, match="TLS problem") as e:
        call()
    assert e.value.kind == "network" and "API key" not in str(e.value)
    assert FakeConnection.instances[0].closed


def test_ctrl_c_while_waiting_closes_the_connection(monkeypatch):
    call = open_with(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        call()
    assert FakeConnection.instances[0].closed


def test_non_utf8_event_stream_is_bad_response():
    with pytest.raises(LlmError) as e:
        list(iter_sse([b"data: \xff\xfe\n", b"\n"]))
    assert e.value.kind == "bad_response"


def test_odd_shapes_in_a_response_do_not_crash():
    assert "boom" in str(failed_error({"status": "failed", "error": "boom"}))
    assert str(failed_error("not a dict"))
    assert output_text({"output": [{"type": "message", "content": ["oops", {"type": "output_text",
                                                                           "text": "ok"}]}]}) == "ok"
    c = OpenAIResponses(key="sk-test", open_=FakeOpen(response(body={
        "status": "incomplete", "incomplete_details": "max_output_tokens", "output": []})))
    assert c.complete("", [llm.Message("user", "hi")]).reason is None


def test_key_fragments_in_error_messages_are_redacted():
    body = b'{"error": {"message": "Incorrect API key provided: sk-proj-AbCd****wxyz. See ..."}}'
    e = error_from_status(401, {}, body)
    assert "AbCd" not in str(e) and "wxyz" not in str(e) and "sk-[redacted]" in str(e)


def test_negative_retries_refused():
    with pytest.raises(ValueError, match="retries"):
        Settings(retries=-1).check()


def test_trace_is_aligned_to_the_attempt_that_produced_the_text():
    trace = Trace(start=0.0)
    trace.pieces = [(30.5, "Hi"), (30.7, ".")]       # 30 s timeout, then a good retry
    reply = Reply("Hi.", "fake", "m", timings=Timings(0.1, 0.3, 0.5, 0.8), attempts=2)
    trace.align(reply)
    assert trace.first_s == pytest.approx(0.5) and trace.last_s == pytest.approx(0.7)
    assert trace.first_sentence()[0] <= reply.timings.total_s


def test_runs_with_empty_replies_do_not_crash(capsys):
    fake = FakeLlm(Reply("", "fake", "m"), Reply("", "fake", "m"))
    assert stream_main(["hi", "--provider", "fake", "--runs", "2"], make=lambda n, s: fake) == 0
    assert "2 runs:" in capsys.readouterr().out


def test_reset_also_resets_the_forgotten_count():
    convo = Conversation(max_turns=1)
    convo.commit("q1", Reply("a1", "fake", "m"))
    convo.commit("q2", Reply("a2", "fake", "m"))
    convo.clear()
    assert convo.forgotten == 0


def test_model_spec_default_effort_means_not_sent():
    assert model_specs(["gpt-6.1-sol:default"], "x", "none") == [("gpt-6.1-sol", None)]
