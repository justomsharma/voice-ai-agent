"""How a reply will sound, not how it reads.

A chat UI rewards long, formatted answers. A phone call punishes them:
every word costs the caller time, and TTS reads markdown literally
("asterisk asterisk Step one"). These numbers make that visible.
"""

import re

WORDS_PER_SECOND = 2.5   # ~150 words per minute, a typical speaking pace

# **bold**, __bold__, `code`, [links](...), "# heading", "- bullet", "1. item"
MARKDOWN = re.compile(r"\*\*|__|`|\[[^\]]+\]\([^)]+\)|^\s*#{1,6}\s|^\s*[-*•]\s|^\s*\d+[.)]\s", re.M)
SENTENCE_END = re.compile(r"[.!?]\s")
SENTENCE_END_AT_EOF = re.compile(r"[.!?]\s*$")


def word_count(text: str) -> int:
    return len(text.split())


def speak_seconds(text: str) -> float:
    """Roughly how long TTS will take to say it."""
    return word_count(text) / WORDS_PER_SECOND


def has_markdown(text: str) -> bool:
    return bool(MARKDOWN.search(text))


def first_sentence_end(text: str, done: bool = False) -> int | None:
    """Index just past the first sentence, or None if no sentence has ended
    yet. Stage 06 TTS can start speaking at this point, not at the last word.

    Mid-stream, "It costs 1." may continue with "5 dollars", so a sentence
    has only ended once a space follows, or the stream is `done`."""
    m = SENTENCE_END.search(text) or (SENTENCE_END_AT_EOF.search(text) if done else None)
    return m.start() + 1 if m else None


def describe(text: str) -> str:
    md = ", MARKDOWN (TTS would read the symbols)" if has_markdown(text) else ""
    return f"{word_count(text)} words, ~{speak_seconds(text):.1f} s to say{md}"
