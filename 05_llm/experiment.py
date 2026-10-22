"""Milestone 2: same questions, different instructions. What changes?

    python 05_llm/experiment.py                                    # every prompt x 4 questions
    python 05_llm/experiment.py --prompt assistant --prompt detailed
    python 05_llm/experiment.py -q "How do vaccines work?" -q "Book me a cab"
    python 05_llm/experiment.py -v                                 # print every full reply

The instructions are the cheapest, most powerful knob you have. "You are a
concise phone assistant" and "Explain everything in detail" give the SAME
model a 3-second answer or a 2-minute one, with or without markdown that
a TTS voice would read out loud.
"""

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass

from chat import add_llm_args, ms, settings_from
from config import load_prompt, prompt_names
from llm import LlmClient, LlmError, Reply, make_client
from spoken import has_markdown, speak_seconds, word_count
from stream import stream_once

QUESTIONS = [
    "What's the weather like?",                    # missing info: ask, or admit you can't check
    "Explain how a refrigerator works.",           # how long is "explain"?
    "Give me three tips for sleeping better.",     # a list: markdown or spoken?
    "What's my account balance?",                  # no data: invent, or say so?
]


@dataclass
class Cell:
    prompt: str
    question: str
    reply: Reply | None = None
    first_s: float | None = None
    error: str = ""

    @property
    def text(self) -> str:
        return self.reply.text if self.reply else ""


def run(client: LlmClient, prompts: list[str], questions: list[str],
        progress: Callable[[str], None] = lambda s: None) -> list[Cell]:
    cells = []
    for q in questions:
        for name in prompts:
            cell = Cell(name, q)
            progress(f"  {name:<10} {q}")
            try:
                cell.reply, trace = stream_once(client, load_prompt(name), q)
                cell.first_s = trace.first_s
            except LlmError as e:
                cell.error = str(e)
            cells.append(cell)
    return cells


def row(c: Cell, width: int) -> str:
    if c.error:
        return f"  {c.prompt:<{width}}  ERROR {c.error}"
    t = c.text
    preview = " ".join(t.split())
    preview = preview if len(preview) <= 60 else preview[:57] + "..."
    cut = "" if c.reply.complete else f" [incomplete: {c.reply.reason}]"
    return (f"  {c.prompt:<{width}}  {word_count(t):>5}  {speak_seconds(t):>6.1f}s  "
            f"{'yes' if has_markdown(t) else '-':>3}  {'yes' if '?' in t else '-':>3}  "
            f"{ms(c.first_s):>8}  {ms(c.reply.timings.total_s):>8}  {preview!r}{cut}")


def table(cells: list[Cell], prompts: list[str], verbose: bool = False) -> str:
    width = max(len(p) for p in prompts + ["prompt"])
    head = (f"  {'prompt':<{width}}  {'words':>5}  {'~speak':>7}  {'md':>3}  {'ask':>3}  "
            f"{'first':>8}  {'total':>8}  reply")
    lines = []
    for q in dict.fromkeys(c.question for c in cells):
        lines += ["", f"Q: {q!r}", head]
        for c in (c for c in cells if c.question == q):
            lines.append(row(c, width))
            if verbose and c.text:
                lines += [f"      | {line}" for line in c.text.splitlines()]
    lines += ["", f"Per prompt (averages over {len(set(c.question for c in cells))} questions)",
              f"  {'prompt':<{width}}  {'words':>5}  {'~speak':>7}  {'markdown':>8}  "
              f"{'first':>8}  {'total':>8}  {'out tokens':>10}"]
    for p in prompts:
        ok = [c for c in cells if c.prompt == p and c.reply]
        if not ok:
            lines.append(f"  {p:<{width}}  (all failed)")
            continue
        n = len(ok)
        firsts = [c.first_s for c in ok if c.first_s is not None]
        lines.append(
            f"  {p:<{width}}  {sum(word_count(c.text) for c in ok) / n:>5.0f}  "
            f"{sum(speak_seconds(c.text) for c in ok) / n:>6.1f}s  "
            f"{sum(has_markdown(c.text) for c in ok):>4} of {n}  "
            f"{ms(sum(firsts) / len(firsts) if firsts else None):>8}  "
            f"{ms(sum(c.reply.timings.total_s for c in ok) / n):>8}  "
            f"{sum(c.reply.usage.output_tokens or 0 for c in ok) / n:>10.0f}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare prompts on the same questions.")
    add_llm_args(p, prompt=False)
    p.add_argument("--prompt", action="append", default=None,
                   help="prompt name or file, repeatable (default: every prompt in 05_llm/prompts)")
    p.add_argument("-q", "--question", action="append", default=None,
                   help="a question, repeatable (default: 4 built-in questions)")
    p.add_argument("-v", "--verbose", action="store_true", help="print every full reply")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, make: Callable[..., LlmClient] = make_client) -> int:
    args = parse_args(argv)
    prompts = args.prompt or prompt_names()
    questions = args.question or QUESTIONS
    try:
        client = make(args.provider, settings_from(args))
        for p in prompts:
            load_prompt(p)          # fail now on a typo, not after 10 paid calls
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{len(prompts)} prompt(s) x {len(questions)} question(s) = "
          f"{len(prompts) * len(questions)} calls to {client.settings.model}")
    cells = run(client, prompts, questions, progress=print)
    print(table(cells, prompts, args.verbose))
    print("\nmd = markdown a TTS voice would read aloud; ask = the reply asks a question")
    return 1 if all(c.error for c in cells) else 0


if __name__ == "__main__":
    sys.exit(main())
