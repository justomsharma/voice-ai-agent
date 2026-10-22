"""Milestone 4: a small, repeatable eval set, plus a latency benchmark.

    python 05_llm/llm_eval.py                                  # every case in eval_cases.json
    python 05_llm/llm_eval.py -v                               # show every reply
    python 05_llm/llm_eval.py --category followup
    python 05_llm/llm_eval.py --repeat 3                       # LLMs vary run to run
    python 05_llm/llm_eval.py --prompt plain                   # what is the prompt worth?
    python 05_llm/llm_eval.py --no-history                     # follow-ups without their history
    python 05_llm/llm_eval.py --model gpt-6-luna --model gpt-6.1-sol:low   # compare models

Each case is a transcript (plus earlier turns, if any) and a few string
checks. They're deliberately crude: "asks a question" means "contains a ?".
They catch regressions quickly and repeatably. They don't judge quality.
Read the replies (-v) too. Stage 12 adds LLM-as-judge and proper statistics.

Categories:
    instruction   follows the system instructions even when asked not to
    concise       short enough to say on a call, no markdown
    followup      uses the earlier turns ("What about tomorrow?")
    ambiguity     asks when it can't know what you mean
    restraint     says "I can't check" instead of inventing facts
"""

import argparse
import json
import math
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from chat import add_llm_args, instructions_from, ms, safe_console, settings_from
from config import Settings
from llm import LlmClient, LlmError, Message, Reply, make_client
from spoken import has_markdown, word_count
from stream import Trace

CASES_FILE = Path(__file__).resolve().parent / "eval_cases.json"
CHECKS = ("max_words", "no_markdown", "asks_question", "must_include_any",
          "must_not_include", "no_digits")
CATEGORIES = ("instruction", "concise", "followup", "ambiguity", "restraint")


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    input: str
    history: tuple[Message, ...] = ()
    checks: dict = field(default_factory=dict)
    why: str = ""


def load_cases(path: Path = CASES_FILE) -> list[Case]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for c in data["cases"]:
        unknown = set(c.get("checks", {})) - set(CHECKS)
        if unknown:
            raise ValueError(f"case {c['id']}: unknown check(s) {sorted(unknown)}; "
                             f"use {', '.join(CHECKS)}")
        cases.append(Case(c["id"], c["category"], c["input"],
                          tuple(Message(role, text) for role, text in c.get("history", [])),
                          c.get("checks", {}), c.get("why", "")))
    return cases


def normalize(text: str) -> str:
    """Lower-case, and straighten curly quotes: models write can’t, checks say can't."""
    return text.lower().replace("’", "'").replace("‘", "'").replace(
        "“", '"').replace("”", '"')


def run_checks(checks: dict, text: str) -> list[tuple[str, bool, str]]:
    """(check, passed, detail) for every check of a case."""
    t = normalize(text)
    out = []
    for name, want in checks.items():
        if name == "max_words":
            n = word_count(text)
            out.append((name, n <= want, f"{n} words, max {want}"))
        elif name == "no_markdown":
            md = has_markdown(text)
            out.append((name, not (want and md), "markdown found" if md else ""))
        elif name == "asks_question":
            asks = "?" in text
            out.append((name, asks == want, "asks a question" if asks else "no question"))
        elif name == "must_include_any":
            hit = next((w for w in want if normalize(w) in t), None)
            out.append((name, hit is not None, f"found {hit!r}" if hit else f"none of {want}"))
        elif name == "must_not_include":
            hit = next((w for w in want if normalize(w) in t), None)
            out.append((name, hit is None, f"found {hit!r}" if hit else ""))
        elif name == "no_digits":
            digits = re.findall(r"\d+", text)
            out.append((name, not (want and digits), f"digits {digits}" if digits else ""))
    return out


@dataclass
class Result:
    case: Case
    model: str
    reply: Reply | None = None
    first_s: float | None = None
    first_sentence_s: float | None = None
    error: str = ""
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (not self.error and self.reply is not None and self.reply.complete
                and all(ok for _, ok, _ in self.checks))


def run_case(client: LlmClient, instructions: str, case: Case, history: bool = True) -> Result:
    result = Result(case, client.settings.model)
    messages = (list(case.history) if history else []) + [Message("user", case.input)]
    trace = Trace()
    try:
        result.reply = client.stream(instructions, messages, trace)
        trace.align(result.reply)
    except LlmError as e:
        result.error = str(e)
        return result
    result.first_s = trace.first_s
    sentence = trace.first_sentence()
    result.first_sentence_s = sentence[0] if sentence else None
    result.checks = run_checks(case.checks, result.reply.text)
    return result


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile: p50 = median, p90 = 9 in 10 were at least this fast."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def row(r: Result) -> str:
    c = r.case
    if r.error:
        return f"  {c.id:<17} {c.category:<11} ERROR {r.error}"
    failed = [f"{name} ({detail})" for name, ok, detail in r.checks if not ok]
    if not r.reply.complete:
        failed.insert(0, f"incomplete: {r.reply.reason}")
    return (f"  {c.id:<17} {c.category:<11} {'pass' if r.passed else 'FAIL':<5} "
            f"{ms(r.first_s):>8} {ms(r.reply.timings.total_s):>8} {word_count(r.reply.text):>5}"
            f"   {'; '.join(failed)}")


def summary(results: list[Result]) -> str:
    lines = []
    for model in dict.fromkeys(r.model for r in results):
        rs = [r for r in results if r.model == model]
        ok = [r for r in rs if r.reply]
        passed = sum(r.passed for r in rs)
        lines += ["", f"{model}: {passed}/{len(rs)} passed ({passed / len(rs):.0%})"]
        for cat in CATEGORIES:
            cs = [r for r in rs if r.case.category == cat]
            if cs:
                lines.append(f"  {cat:<12} {sum(r.passed for r in cs)}/{len(cs)}")
        for label, values in (
                ("first text", [r.first_s for r in ok if r.first_s is not None]),
                ("first sentence", [r.first_sentence_s for r in ok if r.first_sentence_s is not None]),
                ("total", [r.reply.timings.total_s for r in ok])):
            if values:
                lines.append(f"  {label:<15} p50 {ms(percentile(values, 50)):>8}   "
                             f"p90 {ms(percentile(values, 90)):>8}")
        if ok:
            tin = sum(r.reply.usage.input_tokens or 0 for r in ok)
            tout = sum(r.reply.usage.output_tokens or 0 for r in ok)
            words = sum(word_count(r.reply.text) for r in ok) / len(ok)
            lines.append(f"  tokens          in {tin:,}  out {tout:,}   "
                         f"avg reply {words:.0f} words")
        if len(ok) < len(rs):
            lines.append(f"  errors          {len(rs) - len(ok)}")
    return "\n".join(lines)


def model_specs(specs: list[str] | None, default_model: str, default_effort: str | None):
    """'gpt-6.1-sol:low' -> ('gpt-6.1-sol', 'low'). Plain names keep --effort."""
    out = []
    for spec in specs or [default_model]:
        name, _, effort = spec.partition(":")
        if effort == "default":
            out.append((name, None))      # don't send it: the model's own default
        else:
            out.append((name, effort or default_effort))
    return out


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Stage 05 eval set and benchmark latency.")
    add_llm_args(p, model=False)
    p.add_argument("--model", action="append", default=None,
                   help="model[:effort], repeatable to compare (default gpt-6-luna)")
    p.add_argument("--cases", default=str(CASES_FILE), help="eval set JSON file")
    p.add_argument("--category", action="append", choices=CATEGORIES, default=None)
    p.add_argument("--repeat", type=int, default=1, help="run every case N times")
    p.add_argument("--no-history", action="store_true",
                   help="drop each case's earlier turns (watch follow-ups fail)")
    p.add_argument("-v", "--verbose", action="store_true", help="print every reply")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, make: Callable[..., LlmClient] = make_client) -> int:
    args = parse_args(argv)
    safe_console()
    try:
        cases = [c for c in load_cases(Path(args.cases))
                 if not args.category or c.category in args.category]
        instructions = instructions_from(args)
        base = settings_from(args, model=Settings().model)
        clients = [make(args.provider, replace(base, model=m, effort=e).check())
                   for m, e in model_specs(args.model, base.model, base.effort)]
        if args.repeat < 1:
            raise ValueError("--repeat must be at least 1")
        if not cases:
            raise ValueError("no cases selected")
    except (ValueError, OSError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    results = []
    for client in clients:
        s = client.settings
        print(f"\n{s.model} (effort {s.effort or 'default'}), prompt "
              f"{'--instructions' if args.instructions is not None else args.prompt}, "
              f"{len(cases)} case(s) x {args.repeat}{', NO HISTORY' if args.no_history else ''}")
        print(f"  {'case':<17} {'category':<11} {'':<5} {'first':>8} {'total':>8} {'words':>5}"
              f"   failed checks")
        for _ in range(args.repeat):
            for case in cases:
                r = run_case(client, instructions, case, history=not args.no_history)
                results.append(r)
                print(row(r), flush=True)
                if args.verbose and r.reply:
                    print(f"      | {r.reply.text}")
    print(summary(results))
    return 1 if all(r.error for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
