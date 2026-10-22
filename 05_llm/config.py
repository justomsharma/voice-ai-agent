"""Settings for one LLM call, and the prompt files.

Every knob that changes what the model says or how fast it says it, in one
place, with the defaults chosen for a voice agent:

    model              gpt-6-luna: OpenAI's fastest, cheapest model (docs, 2026-10-22)
    effort             "none": no hidden reasoning before the first word.
                       Reasoning tokens are generated (and billed) BEFORE any
                       visible text, so they add straight to time-to-first-token.
                       Luna's own default is "medium", so we always send this.
    max_output_tokens  a safety cap, not a way to make answers short. Hitting it
                       cuts the reply mid-sentence (status "incomplete"). The
                       prompt is what keeps replies short. Reasoning tokens count
                       against this cap too.
    temperature        randomness. Only accepted when effort is "none".
    verbosity          "low" | "medium" | "high": OpenAI's length hint.
"""

from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT = "assistant"
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
VERBOSITIES = ("low", "medium", "high")


@dataclass(frozen=True)
class Settings:
    model: str = "gpt-6-luna"
    effort: str | None = "none"          # None = don't send it (the model's default)
    max_output_tokens: int | None = 400
    temperature: float | None = None     # None = don't send it
    verbosity: str | None = None
    timeout_s: float = 30.0              # longest wait for any byte
    retries: int = 2

    def check(self) -> "Settings":
        """Refuse what the API would refuse, before paying for a round trip."""
        if self.effort is not None and self.effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}")
        if self.verbosity is not None and self.verbosity not in VERBOSITIES:
            raise ValueError(f"verbosity must be one of {', '.join(VERBOSITIES)}")
        if self.max_output_tokens is not None and self.max_output_tokens < 16:
            raise ValueError("max_output_tokens must be at least 16 (the API minimum)")
        if self.temperature is not None:
            if not 0 <= self.temperature <= 2:
                raise ValueError("temperature must be between 0 and 2")
            # Reasoning models only take sampling settings when they don't
            # reason (verified: HTTP 400 "Unsupported parameter: 'temperature'").
            if self.effort != "none":
                raise ValueError("temperature only works with --effort none")
        if self.timeout_s <= 0:
            raise ValueError("timeout must be positive")
        return self


def prompt_names() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.txt"))


def load_prompt(name_or_path: str = DEFAULT_PROMPT) -> str:
    """A prompt by name (prompts/<name>.txt) or by file path."""
    path = Path(name_or_path)
    if not path.is_file():
        path = PROMPTS_DIR / f"{name_or_path}.txt"
    if not path.is_file():
        raise ValueError(f"no prompt {name_or_path!r}; choose from {', '.join(prompt_names())} "
                         "or give a file path")
    return path.read_text(encoding="utf-8").strip()
