"""OpenAI Responses API: instructions + messages in, a reply out.

API contract (POST https://api.openai.com/v1/responses, docs checked and
wire format verified with real calls on 2026-10-22):

    Authorization: Bearer <key>
    Content-Type: application/json
    {
      "model": "gpt-6-luna",
      "instructions": "You are a voice assistant ...",   <- the system prompt
      "input": [                                          <- the conversation, oldest first
        {"role": "user", "content": "What's the weather in Pune?"},
        {"role": "assistant", "content": "I can't check live weather ..."},
        {"role": "user", "content": "What about tomorrow?"}
      ],
      "reasoning": {"effort": "none"},
      "max_output_tokens": 400,
      "store": false,                                     <- we keep the history ourselves
      "stream": true                                      <- only for stream()
    }

Without "stream" -> one JSON object:
    {"id": "resp_...", "status": "completed" | "incomplete" | "failed",
     "incomplete_details": {"reason": "max_output_tokens"} | null,
     "output": [{"type": "message", "content": [{"type": "output_text", "text": "..."}]}],
     "usage": {"input_tokens": 55, "output_tokens": 23,
               "input_tokens_details": {"cached_tokens": 0},
               "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 78},
     "error": null}

With "stream": true -> Server-Sent Events, in this order:
    response.created, response.in_progress, response.output_item.added,
    response.content_part.added,
    response.output_text.delta  {"delta": "1"}   <- many of these: the text
    response.output_text.done, response.content_part.done, response.output_item.done,
    response.completed | response.incomplete | response.failed   <- the end
    (or an `error` event). There is no "[DONE]" line.

Errors (bad key, unknown model, bad parameter) come back as a normal JSON
error with an HTTP status, even when you asked for a stream.
"""

import json

from config import Settings
from llm import (LlmError, Message, OnText, Open, Reply, Timings, Usage, api_key,
                 error_from_status, estimate_tokens, http_open, iter_sse, parse_json,
                 with_retries)

URL = "https://api.openai.com/v1/responses"
KEY_VAR = "OPENAI_API_KEY"
TEXT_EVENTS = ("response.output_text.delta", "response.refusal.delta")
END_EVENTS = ("response.completed", "response.incomplete")


class OpenAIResponses:
    name = "openai"

    def __init__(self, settings: Settings | None = None, key: str | None = None,
                 open_: Open = http_open, sleep=None, log=None) -> None:
        self.settings = (settings or Settings()).check()
        self._key = key          # tests pass one; normally read from env at call time
        self._open = open_
        self._sleep = sleep
        self._log = log

    # ------------------------------------------------------------ request

    def body(self, instructions: str, messages: list[Message], stream: bool = False) -> dict:
        s = self.settings
        body: dict = {"model": s.model}
        if instructions:
            body["instructions"] = instructions
        body["input"] = [{"role": m.role, "content": m.text} for m in messages]
        if s.effort is not None:
            body["reasoning"] = {"effort": s.effort}
        if s.max_output_tokens is not None:
            body["max_output_tokens"] = s.max_output_tokens
        if s.temperature is not None:
            body["temperature"] = s.temperature
        if s.verbosity is not None:
            body["text"] = {"verbosity": s.verbosity}
        body["store"] = False
        if stream:
            body["stream"] = True
        return body

    def _send(self, instructions: str, messages: list[Message], stream: bool):
        key = self._key or api_key(KEY_VAR)
        body = json.dumps(self.body(instructions, messages, stream)).encode()
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                   "Accept": "text/event-stream" if stream else "application/json"}
        resp = self._open(URL, headers, body, self.settings.timeout_s)
        if resp.status >= 400:
            try:
                raw = resp.read()
            finally:
                resp.close()
            raise error_from_status(resp.status, resp.headers, raw)
        return resp

    # ------------------------------------------------------------ complete

    def complete(self, instructions: str, messages: list[Message]) -> Reply:
        return with_retries(lambda: self._complete_once(instructions, messages),
                            self.settings.retries, sleep=self._sleep, log=self._log)

    def _complete_once(self, instructions: str, messages: list[Message]) -> Reply:
        resp = self._send(instructions, messages, stream=False)
        try:
            raw = resp.read()
        finally:
            resp.close()
        total = resp.elapsed()
        data = parse_json(raw)
        text = output_text(data)
        # Without streaming, the first word is usable only when the last one is.
        timings = Timings(resp.connect_s, resp.wait_s, total if text else None, total)
        return self._reply(data, resp.headers, timings, text)

    # ------------------------------------------------------------ stream

    def stream(self, instructions: str, messages: list[Message], on_text: OnText) -> Reply:
        return with_retries(lambda: self._stream_once(instructions, messages, on_text),
                            self.settings.retries, sleep=self._sleep, log=self._log)

    def _stream_once(self, instructions: str, messages: list[Message], on_text: OnText) -> Reply:
        resp = self._send(instructions, messages, stream=True)
        shown: list[str] = []
        first = None
        try:
            ctype = resp.headers.get("content-type", "")
            if "text/event-stream" not in ctype:
                raise LlmError("bad_response", f"expected an event stream, got {ctype or 'nothing'}")
            for name, data in iter_sse(resp.lines()):
                event = parse_json(data, f"event {name!r}")
                kind = event.get("type", name)
                if kind in TEXT_EVENTS:
                    delta = event.get("delta")
                    if not isinstance(delta, str):
                        raise LlmError("bad_response", f"{kind} without a text delta")
                    if delta:
                        if first is None:
                            first = resp.elapsed()
                        shown.append(delta)
                        on_text(delta)
                elif kind in END_EVENTS:
                    final = event.get("response")
                    if not isinstance(final, dict):
                        raise LlmError("bad_response", f"{kind} without a response object")
                    timings = Timings(resp.connect_s, resp.wait_s, first, resp.elapsed())
                    # The text is what we showed, piece by piece. The end
                    # event adds the status and the token counts.
                    return self._reply(final, resp.headers, timings, "".join(shown))
                elif kind == "response.failed":
                    raise failed_error(event.get("response") or {})
                elif kind == "error":
                    raise event_error(event)
                # Everything else (created, in_progress, *.added, *.done) is bookkeeping.
            # The socket closed cleanly, but the server never said "completed".
            raise LlmError("stream_cut", "the stream ended before the response was complete",
                           retryable=True)
        except LlmError as e:
            e.partial_text = e.partial_text or "".join(shown)
            raise
        finally:
            resp.close()

    # ------------------------------------------------------------ shared

    def _reply(self, data: dict, headers: dict, timings: Timings, text: str) -> Reply:
        status = data.get("status")
        if status == "failed":
            raise failed_error(data)
        if status not in ("completed", "incomplete"):
            raise LlmError("bad_response", f"unexpected status {status!r}")
        reason = (data.get("incomplete_details") or {}).get("reason")
        return Reply(text.strip(), self.name, data.get("model") or self.settings.model,
                     status, reason, timings, parse_usage(data.get("usage")),
                     request_id=headers.get("x-request-id"), response_id=data.get("id"))

    def describe_request(self, instructions: str, messages: list[Message], stream: bool) -> str:
        body = self.body(instructions, messages, stream)
        raw = json.dumps(body).encode()
        sent = instructions + "".join(m.text for m in messages)
        return "\n".join([
            f"POST {URL}",
            f"  Authorization: Bearer ${KEY_VAR}      (read from env/.env, never printed)",
            "  Content-Type: application/json",
            f"  body: {len(raw):,} bytes, ~{estimate_tokens(sent)} input tokens (estimate)",
            json.dumps(body, indent=2, ensure_ascii=False)])


def output_text(data: dict) -> str:
    """Join the text parts of every message in `output`."""
    output = data.get("output")
    if not isinstance(output, list):
        raise LlmError("bad_response", f"no 'output' list in response: {list(data)}")
    parts = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue        # reasoning items, tool calls (Stage 09), ...
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                parts.append(part.get("text") or "")
            elif part.get("type") == "refusal":
                parts.append(part.get("refusal") or "")
    return "".join(parts)


def parse_usage(u) -> Usage:
    if not isinstance(u, dict):
        return Usage()
    return Usage(u.get("input_tokens"), u.get("output_tokens"),
                 (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
                 (u.get("input_tokens_details") or {}).get("cached_tokens"),
                 u.get("total_tokens"))


def failed_error(data: dict) -> LlmError:
    """status "failed": the request was accepted, then the model run broke."""
    err = data.get("error") or {}
    code = err.get("code") or "unknown"
    return LlmError("server" if code == "server_error" else "failed",
                    f"{code}: {err.get('message') or 'the response failed'}",
                    retryable=code == "server_error")


def event_error(event: dict) -> LlmError:
    """An `error` event inside the stream."""
    err = event.get("error") if isinstance(event.get("error"), dict) else event
    code = err.get("code") or err.get("type") or "unknown"
    message = f"{code}: {err.get('message') or 'stream error'}"
    if code == "rate_limit_exceeded":
        return LlmError("rate_limit", message, retryable=True)
    if code == "server_error":
        return LlmError("server", message, retryable=True)
    return LlmError("failed", message)
