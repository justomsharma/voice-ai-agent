"""OpenAI batch transcription: one WAV in, one transcript out.

Model: `gpt-transcribe`, OpenAI's recommended file-transcription model
(docs checked 2026-10-21). `whisper-1` and `gpt-4o-(mini-)transcribe` are
deprecated and shut down on 2027-02-26.

The API contract (POST https://api.openai.com/v1/audio/transcriptions):

    Authorization: Bearer <key>
    Content-Type: multipart/form-data   <- like an HTML form with a file upload
      file          the audio file (wav, mp3, m4a, webm, ...), <= 25 MB
      model         gpt-transcribe
      languages[]   optional hints, ISO codes ("en", "hi"); repeat for several
      prompt        optional context ("a call about train tickets to Pune")
      keywords[]    optional words you expect: names, product terms
    -> {"text": "...", "languages": [...], "usage": {...}}

The file can be any rate. We still send 16 kHz mono: speech models listen
at 16 kHz internally, so more is just a bigger upload (48 kHz stereo = 6x
the bytes, same words).

Multipart is built by hand below so you can see (and --dry-run print)
exactly what goes over the wire. The SDK does the same thing for you.
"""

import uuid

from audio_prep import AudioContract, Prepared, wav_bytes
from stt import (Post, SttError, Transcript, api_key, check_keywords, check_status, http_post,
                 parse_json)

URL = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_MODEL = "gpt-transcribe"
KEY_VAR = "OPENAI_API_KEY"
CONTRACT = AudioContract(sample_rate=16000, channels=1, max_bytes=25 * 1024 * 1024)


def multipart(fields: list[tuple[str, str]], filename: str, content_type: str, data: bytes,
              boundary: str) -> bytes:
    """multipart/form-data: each part is headers + blank line + value, separated by
    a boundary string that must not appear inside any value (hence random)."""
    out = bytearray()
    for name, value in fields:
        out += (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n").encode()
    out += (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n").encode()
    out += data
    out += f"\r\n--{boundary}--\r\n".encode()
    return bytes(out)


class OpenAIBatch:
    name = "openai"
    contract = CONTRACT

    def __init__(self, model: str | None = None, languages: list[str] | None = None,
                 prompt: str | None = None, keywords: list[str] | None = None,
                 timeout: float = 30.0, key: str | None = None, post: Post = http_post) -> None:
        self.model = model or DEFAULT_MODEL
        self.languages = languages or []
        self.prompt = prompt
        self.keywords = check_keywords(keywords)
        self.timeout = timeout
        self._key = key          # tests pass one; normally read from env at call time
        self._post = post

    def fields(self) -> list[tuple[str, str]]:
        fields = [("model", self.model)]
        # gpt-transcribe takes a list of languages; the older models take one.
        if self.model.startswith("gpt-transcribe"):
            fields += [("languages[]", lang) for lang in self.languages]
        elif self.languages:
            fields.append(("language", self.languages[0]))
        if self.prompt:
            fields.append(("prompt", self.prompt))
        fields += [("keywords[]", k) for k in self.keywords]
        return fields

    def build(self, audio: Prepared, boundary: str | None = None) -> tuple[dict[str, str], bytes]:
        boundary = boundary or uuid.uuid4().hex
        body = multipart(self.fields(), "audio.wav", "audio/wav",
                         wav_bytes(audio.fmt, audio.samples), boundary)
        return {"Content-Type": f"multipart/form-data; boundary={boundary}"}, body

    def transcribe(self, audio: Prepared) -> Transcript:
        key = self._key or api_key(KEY_VAR)
        headers, body = self.build(audio)
        resp = self._post(URL, {"Authorization": f"Bearer {key}", **headers}, body, self.timeout)
        check_status(resp)
        data = parse_json(resp)
        text = data.get("text")
        if not isinstance(text, str):
            raise SttError("bad_response", f"no 'text' in response: {list(data)}")
        languages = data.get("languages") or []
        # Real responses look like [{"code": "en"}]; accept plain strings too.
        language = ", ".join(str(x.get("code") or x.get("language") or x) if isinstance(x, dict) else str(x)
                             for x in languages) or None
        extra = {"usage": data["usage"]} if "usage" in data else {}
        return Transcript(text.strip(), self.name, self.model, audio.seconds, resp.timings,
                          language=language, request_id=resp.headers.get("x-request-id"),
                          extra=extra)

    def describe_request(self, audio: Prepared) -> str:
        headers, body = self.build(audio, boundary="BOUNDARY")
        wav = len(wav_bytes(audio.fmt, audio.samples))
        f = audio.fmt
        lines = [f"POST {URL}",
                 f"  Authorization: Bearer ${KEY_VAR}      (read from env/.env, never printed)",
                 f"  Content-Type: {headers['Content-Type']}",
                 f"  body: {len(body):,} bytes, multipart parts:"]
        lines += [f"    {name} = {value}" for name, value in self.fields()]
        lines.append(f"    file = audio.wav (audio/wav) {wav:,} bytes = 44 header + "
                     f"{wav - 44:,} PCM ({audio.seconds:.2f} s x {f.sample_rate} Hz x "
                     f"{f.channels} ch x {f.sample_width} B)")
        lines.append(f"  overhead: {len(body) - wav} bytes of multipart framing")
        return "\n".join(lines)
