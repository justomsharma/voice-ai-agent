import pytest

from audio_prep import AudioContract
from stt import (FakeProvider, SttError, Timings, Transcript, api_key, check_status,
                 load_env_file, make_provider, parse_json, with_retries)
from stt_fakes import prepared, reply


def test_load_env_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text('# comment\n\nOPENAI_API_KEY="sk-abc"\nEMPTY=\nnot a line\nX = y \n')
    assert load_env_file(f) == {"OPENAI_API_KEY": "sk-abc", "EMPTY": "", "X": "y"}
    assert load_env_file(tmp_path / "missing") == {}


def test_api_key_env_wins_over_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text("K=from-file\n")
    assert api_key("K", env={"K": "from-env"}, env_file=f) == "from-env"
    assert api_key("K", env={}, env_file=f) == "from-file"


def test_api_key_missing_names_the_variable(tmp_path):
    with pytest.raises(SttError) as e:
        api_key("OPENAI_API_KEY", env={}, env_file=tmp_path / "none")
    assert e.value.kind == "auth" and "OPENAI_API_KEY" in str(e.value)
    assert ".env.example" in str(e.value)


@pytest.mark.parametrize("status,kind,retryable", [
    (400, "bad_request", False), (401, "auth", False), (403, "auth", False),
    (408, "timeout", True), (413, "bad_request", False), (429, "rate_limit", True),
    (500, "server", True), (503, "server", True)])
def test_check_status_kinds(status, kind, retryable):
    with pytest.raises(SttError) as e:
        check_status(reply(status, {"error": {"message": "nope"}}))
    assert (e.value.kind, e.value.retryable, e.value.status) == (kind, retryable, status)
    assert "nope" in str(e.value)


def test_check_status_ok():
    check_status(reply(200))


def test_out_of_quota_429_is_not_retryable():
    with pytest.raises(SttError) as e:
        check_status(reply(429, {"error": {"message": "You exceeded your current quota",
                                           "code": "insufficient_quota"}}))
    assert e.value.kind == "rate_limit" and not e.value.retryable


def test_check_status_deepgram_style_and_retry_after():
    with pytest.raises(SttError) as e:
        check_status(reply(503, {"err_code": "X", "err_msg": "busy"}, {"retry-after": "3"}))
    assert "busy" in str(e.value) and e.value.retry_after == 3.0


def test_check_status_non_json_body():
    with pytest.raises(SttError, match="Bad Gateway"):
        check_status(reply(502, "<html>Bad Gateway</html>"))


def test_parse_json_errors():
    with pytest.raises(SttError) as e:
        parse_json(reply(200, "not json"))
    assert e.value.kind == "bad_response"
    with pytest.raises(SttError):
        parse_json(reply(200, "[1, 2]"))


def ok() -> Transcript:
    return Transcript("hi", "fake", "f", 1.0)


def test_retries_then_succeeds():
    calls, waits = [], []

    def call():
        calls.append(1)
        if len(calls) < 3:
            raise SttError("server", "busy", retryable=True)
        return ok()

    t = with_retries(call, retries=2, sleep=waits.append)
    assert t.attempts == 3 and waits == [0.5, 1.0]  # exponential backoff


def test_does_not_retry_permanent_errors():
    calls = []

    def call():
        calls.append(1)
        raise SttError("auth", "bad key")

    with pytest.raises(SttError):
        with_retries(call, sleep=lambda s: None)
    assert len(calls) == 1


def test_gives_up_after_retries_and_caps_retry_after():
    waits = []

    def call():
        raise SttError("rate_limit", "slow down", retryable=True, retry_after=60)

    with pytest.raises(SttError, match="slow down"):
        with_retries(call, retries=1, sleep=waits.append)
    assert waits == [10.0]


def test_transcript_rtf():
    t = Transcript("x", "p", "m", 2.0, Timings(0.1, 0.1, 0.3, 0.0))
    assert t.rtf == pytest.approx(0.25)
    assert Transcript("x", "p", "m", 0.0).rtf == 0.0


def test_make_provider():
    assert make_provider("openai").model == "gpt-transcribe"
    assert make_provider("deepgram", model="nova-2").model == "nova-2"
    with pytest.raises(ValueError, match="unknown provider"):
        make_provider("nope")


def test_fake_provider_records_and_fails_on_demand():
    p = FakeProvider("hello")
    audio = prepared()
    assert p.transcribe(audio).text == "hello" and p.received == [audio]
    bad = FakeProvider(error=SttError("network", "down", retryable=True))
    with pytest.raises(SttError):
        bad.transcribe(audio)
    assert FakeProvider(contract=AudioContract(8000)).contract.sample_rate == 8000


def test_api_key_is_stripped_and_validated_without_echo(tmp_path):
    assert api_key("K", env={"K": "  sk-abc\n"}, env_file=tmp_path / "x") == "sk-abc"
    with pytest.raises(SttError) as e:
        api_key("K", env={"K": "sk-ab c"}, env_file=tmp_path / "x")
    assert e.value.kind == "auth" and "sk-ab" not in str(e.value)
