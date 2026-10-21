"""OpenAI and Deepgram adapters, against a fake network."""

import pytest

from audio_prep import wav_bytes
from deepgram_stt import DeepgramBatch
from openai_stt import URL, OpenAIBatch, multipart
from stt import SttError
from stt_fakes import FakePost, prepared, reply


def test_multipart_layout_is_exact():
    body = multipart([("model", "m1"), ("languages[]", "en")], "a.wav", "audio/wav", b"RIFF", "B")
    assert body == (b'--B\r\nContent-Disposition: form-data; name="model"\r\n\r\nm1\r\n'
                    b'--B\r\nContent-Disposition: form-data; name="languages[]"\r\n\r\nen\r\n'
                    b'--B\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                    b"Content-Type: audio/wav\r\n\r\nRIFF\r\n--B--\r\n")


def test_openai_fields():
    p = OpenAIBatch(languages=["en", "hi"], prompt="train tickets", keywords=["Pune", "IRCTC"])
    assert p.fields() == [("model", "gpt-transcribe"), ("languages[]", "en"),
                          ("languages[]", "hi"), ("prompt", "train tickets"),
                          ("keywords[]", "Pune"), ("keywords[]", "IRCTC")]
    # Older models take a single `language`.
    assert OpenAIBatch(model="whisper-1", languages=["en", "hi"]).fields() == [
        ("model", "whisper-1"), ("language", "en")]


def test_openai_request_and_response():
    post = FakePost(reply(200, {"text": " Book a ticket. ", "languages": ["en"],
                                "usage": {"type": "duration", "seconds": 1}},
                          {"x-request-id": "req_1"}))
    p = OpenAIBatch(key="sk-test", post=post)
    audio = prepared(1.0)
    t = p.transcribe(audio)
    url, headers, body, timeout = post.calls[0]
    assert url == URL and timeout == 30.0
    assert headers["Authorization"] == "Bearer sk-test"
    assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert wav_bytes(audio.fmt, audio.samples) in body  # the WAV travels untouched
    assert (t.text, t.provider, t.model, t.language, t.request_id) == (
        "Book a ticket.", "openai", "gpt-transcribe", "en", "req_1")
    assert t.audio_s == pytest.approx(1.0) and t.timings.wait_s == 0.4
    assert t.extra == {"usage": {"type": "duration", "seconds": 1}}


def test_openai_languages_as_objects():
    post = FakePost(reply(200, {"text": "x", "languages": [{"language": "hi"}, {"language": "en"}]}))
    assert OpenAIBatch(key="k", post=post).transcribe(prepared()).language == "hi, en"


def test_openai_empty_languages():
    post = FakePost(reply(200, {"text": "x", "languages": []}))
    assert OpenAIBatch(key="k", post=post).transcribe(prepared()).language is None


@pytest.mark.parametrize("resp,kind", [
    (reply(401, {"error": {"message": "Incorrect API key"}}), "auth"),
    (reply(429, {"error": {"message": "Rate limit"}}), "rate_limit"),
    (reply(500, {"error": {"message": "oops"}}), "server"),
    (reply(200, "not json"), "bad_response"),
    (reply(200, {"nothing": 1}), "bad_response")])
def test_openai_errors(resp, kind):
    with pytest.raises(SttError) as e:
        OpenAIBatch(key="k", post=FakePost(resp)).transcribe(prepared())
    assert e.value.kind == kind


def test_openai_network_error_passes_through():
    post = FakePost(SttError("timeout", "no answer", retryable=True))
    with pytest.raises(SttError, match="no answer"):
        OpenAIBatch(key="k", post=post).transcribe(prepared())


def test_openai_describe_request_never_contains_key():
    p = OpenAIBatch(key="sk-secret", languages=["en"])
    text = p.describe_request(prepared(3.0))
    assert "sk-secret" not in text
    assert "$OPENAI_API_KEY" in text
    assert "languages[] = en" in text
    assert "96,044 bytes = 44 header + 96,000 PCM (3.00 s x 16000 Hz x 1 ch x 2 B)" in text


def test_deepgram_url_params():
    assert DeepgramBatch().url() == "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true"
    assert "language=en" in DeepgramBatch(languages=["en"]).url()
    assert "language=multi" in DeepgramBatch(languages=["en", "hi"]).url()
    assert "keyterm=Pune&keyterm=IRCTC" in DeepgramBatch(keywords=["Pune", "IRCTC"]).url()


DG_OK = {"metadata": {"request_id": "dg-1", "model_info": {"abc": {"name": "general-nova-3"}}},
         "results": {"channels": [{"alternatives": [
             {"transcript": "book a ticket", "confidence": 0.97, "words": []}]}]}}


def test_deepgram_request_and_response():
    post = FakePost(reply(200, DG_OK))
    audio = prepared(1.0)
    t = DeepgramBatch(key="dg-test", languages=["en"], post=post).transcribe(audio)
    url, headers, body, _ = post.calls[0]
    assert headers == {"Authorization": "Token dg-test", "Content-Type": "audio/wav"}
    assert body == wav_bytes(audio.fmt, audio.samples)  # raw body, no multipart
    assert (t.text, t.confidence, t.request_id, t.language) == ("book a ticket", 0.97, "dg-1", "en")
    assert t.extra == {"model_info": {"name": "general-nova-3"}}


def test_deepgram_bad_shape():
    post = FakePost(reply(200, {"results": {"channels": []}}))
    with pytest.raises(SttError) as e:
        DeepgramBatch(key="k", post=post).transcribe(prepared())
    assert e.value.kind == "bad_response"


def test_deepgram_auth_error():
    post = FakePost(reply(401, {"err_code": "INVALID_AUTH", "err_msg": "Invalid credentials."}))
    with pytest.raises(SttError, match="Invalid credentials"):
        DeepgramBatch(key="k", post=post).transcribe(prepared())


def test_deepgram_describe_request():
    text = DeepgramBatch(key="dg-secret").describe_request(prepared(1.0))
    assert "dg-secret" not in text and "Token $DEEPGRAM_API_KEY" in text
    assert "32,044 bytes" in text
