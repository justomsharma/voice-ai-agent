"""Deepgram batch transcription: the second provider behind the same interface.

Model: `nova-3`, Deepgram's current general model (docs checked 2026-10-21).

Compare its contract with OpenAI's. Same job, different shape:

    POST https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&language=en
    Authorization: Token <key>          <- "Token", not "Bearer"
    Content-Type: audio/wav
    <the WAV bytes ARE the body>        <- no multipart form; options go in the URL

    -> {"metadata": {"request_id", "duration", "model_info", ...},
        "results": {"channels": [{"alternatives": [{"transcript", "confidence",
                                                    "words": [...]}]}]}}

Note `channels` and `alternatives`: Deepgram transcribes each channel
separately (a stereo call recording = caller and agent) and can return
several guesses. We send mono and take the best guess.
"""

from urllib.parse import urlencode

from audio_prep import AudioContract, Prepared, wav_bytes
from stt import Post, SttError, Transcript, api_key, check_status, http_post, parse_json

BASE_URL = "https://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-3"
KEY_VAR = "DEEPGRAM_API_KEY"
CONTRACT = AudioContract(sample_rate=16000, channels=1)


class DeepgramBatch:
    name = "deepgram"
    contract = CONTRACT

    def __init__(self, model: str | None = None, languages: list[str] | None = None,
                 keywords: list[str] | None = None, timeout: float = 30.0,
                 key: str | None = None, post: Post = http_post) -> None:
        self.model = model or DEFAULT_MODEL
        self.languages = languages or []
        self.keywords = keywords or []
        self.timeout = timeout
        self._key = key
        self._post = post

    def url(self) -> str:
        params = [("model", self.model), ("smart_format", "true")]
        if len(self.languages) == 1:
            params.append(("language", self.languages[0]))
        elif self.languages:
            params.append(("language", "multi"))  # Deepgram's code-switching mode
        # Nova-3 "keyterm prompting": boost names and jargon.
        params += [("keyterm", k) for k in self.keywords]
        return f"{BASE_URL}?{urlencode(params)}"

    def transcribe(self, audio: Prepared) -> Transcript:
        key = self._key or api_key(KEY_VAR)
        body = wav_bytes(audio.fmt, audio.samples)
        resp = self._post(self.url(), {"Authorization": f"Token {key}",
                                       "Content-Type": "audio/wav"}, body, self.timeout)
        check_status(resp)
        data = parse_json(resp)
        try:
            channel = data["results"]["channels"][0]
            best = channel["alternatives"][0]
            text = best["transcript"]
        except (KeyError, IndexError, TypeError) as e:
            raise SttError("bad_response", f"unexpected JSON shape: {e!r}") from e
        meta = data.get("metadata") or {}
        models = meta.get("model_info") or {}
        extra = {"model_info": next(iter(models.values()))} if models else {}
        return Transcript(text.strip(), self.name, self.model, audio.seconds, resp.timings,
                          language=channel.get("detected_language") or
                          (self.languages[0] if len(self.languages) == 1 else None),
                          confidence=best.get("confidence"),
                          request_id=meta.get("request_id"), extra=extra)

    def describe_request(self, audio: Prepared) -> str:
        wav = len(wav_bytes(audio.fmt, audio.samples))
        return "\n".join([
            f"POST {self.url()}",
            f"  Authorization: Token ${KEY_VAR}      (read from env/.env, never printed)",
            "  Content-Type: audio/wav",
            f"  body: the WAV itself, {wav:,} bytes = 44 header + {wav - 44:,} PCM "
            f"({audio.seconds:.2f} s x {audio.fmt.sample_rate} Hz x {audio.fmt.channels} ch x 2 B)",
        ])
