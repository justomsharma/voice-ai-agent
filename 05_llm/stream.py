"""Milestone 4: watch a reply stream in, and measure when each part arrives.

    python 05_llm/stream.py "Explain how a refrigerator works."
    python 05_llm/stream.py "Explain how a refrigerator works." --timeline   # every piece, timed
    python 05_llm/stream.py "Explain how a refrigerator works." --compare    # streamed vs not
    python 05_llm/stream.py "What's the weather like?" --runs 5              # spread over runs
    python 05_llm/stream.py "Explain DNS" --effort low                       # cost of reasoning

The numbers that matter for a voice agent:

    first text       the first word exists. Nothing to SAY yet: one word is not a sentence.
    first sentence   the earliest moment Stage 06 TTS can start speaking
    total            the last word. Without streaming, the caller waits this long for anything.
"""

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from chat import (add_llm_args, instructions_from, ms, print_error, safe_console,
                  settings_from, stats)
from llm import LlmClient, LlmError, Message, Reply, make_client
from spoken import first_sentence_end


@dataclass
class Trace:
    """Collects the streamed pieces with their arrival times (s since start)."""
    start: float = field(default_factory=time.perf_counter)
    pieces: list[tuple[float, str]] = field(default_factory=list)
    echo: Callable[[str], None] | None = None

    def __call__(self, piece: str) -> None:
        self.pieces.append((time.perf_counter() - self.start, piece))
        if self.echo:
            self.echo(piece)

    @property
    def text(self) -> str:
        return "".join(p for _, p in self.pieces)

    @property
    def first_s(self) -> float | None:
        return self.pieces[0][0] if self.pieces else None

    @property
    def last_s(self) -> float | None:
        return self.pieces[-1][0] if self.pieces else None

    def align(self, reply: Reply) -> "Trace":
        """Put the pieces on the clock of the attempt that produced them.

        Our clock started before the FIRST attempt; reply.timings only cover
        the LAST one. Retries happen only before any text, so every piece came
        from the last attempt: shift them so the first piece lands at that
        attempt's first_text_s. Then first text, first sentence and total
        share one clock. (reply.attempts says a retry happened.)"""
        if self.pieces and reply.timings.first_text_s is not None:
            shift = self.pieces[0][0] - reply.timings.first_text_s
            self.pieces = [(t - shift, p) for t, p in self.pieces]
        return self

    def first_sentence(self) -> tuple[float, str] | None:
        """(time, sentence): when the first sentence was complete."""
        text = ""
        for i, (t, piece) in enumerate(self.pieces):
            text += piece
            end = first_sentence_end(text, done=i == len(self.pieces) - 1)
            if end is not None:
                return t, text[:end]
        return None


def stream_once(client: LlmClient, instructions: str, text: str,
                echo: Callable[[str], None] | None = None) -> tuple[Reply, Trace]:
    trace = Trace(echo=echo)
    reply = client.stream(instructions, [Message("user", text)], trace)
    return reply, trace.align(reply)


def timeline(trace: Trace) -> str:
    lines = ["  time      piece"]
    lines += [f"  {t * 1000:6.0f} ms  {piece!r}" for t, piece in trace.pieces]
    return "\n".join(lines)


def summary(reply: Reply, trace: Trace) -> str:
    n = len(trace.pieces)
    sentence = trace.first_sentence()
    lines = [f"  first text      {ms(trace.first_s)}   one piece; nothing worth saying yet"]
    if sentence:
        lines.append(f"  first sentence  {ms(sentence[0])}   <- Stage 06 TTS could start "
                     f"speaking here: {sentence[1]!r}")
    lines.append(f"  total           {ms(reply.timings.total_s)}   "
                 f"<- without streaming, silence until here")
    if n > 1 and trace.last_s > trace.first_s:
        rate = (n - 1) / (trace.last_s - trace.first_s)
        lines.append(f"  {n} pieces, ~{rate:.0f} pieces/s while writing")
    return "\n".join(lines)


def compare(client: LlmClient, instructions: str, text: str) -> str:
    """The same question without and with streaming."""
    full = client.complete(instructions, [Message("user", text)])
    streamed, trace = stream_once(client, instructions, text)
    sentence = trace.first_sentence()
    gain = full.timings.total_s - (sentence[0] if sentence else streamed.timings.total_s)
    return "\n".join([
        "                  first text   first sentence   total",
        f"  not streamed    {ms(full.timings.first_text_s):>10}   "
        f"{ms(full.timings.total_s):>14}   {ms(full.timings.total_s):>5}",
        f"  streamed        {ms(trace.first_s):>10}   {ms(sentence[0] if sentence else None):>14}   "
        f"{ms(streamed.timings.total_s):>5}",
        f"  -> with streaming, speech could start ~{gain * 1000:.0f} ms sooner "
        f"(one run each; network noise is real, try --runs)"])


def spread(values: list[float]) -> str:
    return (f"median {ms(statistics.median(values))}  "
            f"min {ms(min(values))}  max {ms(max(values))}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream a reply and time its arrival.")
    p.add_argument("text", help="the user's message (a transcript)")
    add_llm_args(p)
    p.add_argument("--timeline", action="store_true", help="list every piece with its time")
    p.add_argument("--compare", action="store_true", help="also run it without streaming")
    p.add_argument("--runs", type=int, default=1, help="repeat and show the spread")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, make: Callable[..., LlmClient] = make_client) -> int:
    args = parse_args(argv)
    safe_console()
    try:
        client = make(args.provider, settings_from(args))
        instructions = instructions_from(args)
        if args.runs < 1:
            raise ValueError("--runs must be at least 1")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    try:
        print("agent> ", end="", flush=True)
        reply, trace = stream_once(client, instructions, args.text,
                                   echo=lambda t: print(t, end="", flush=True))
        print()
        if not reply.complete:
            print(f"  [incomplete: {reply.reason}] cut off: not a complete answer")
        if args.timeline:
            print(timeline(trace))
        print(summary(reply, trace))
        print(stats(reply))
        if args.runs > 1:
            runs = [(reply, trace)] + [stream_once(client, instructions, args.text)
                                       for _ in range(args.runs - 1)]
            firsts = [t.first_s for _, t in runs if t.first_s is not None]
            sentences = [s[0] for _, t in runs if (s := t.first_sentence())]
            print(f"\n{args.runs} runs:")
            if firsts:
                print(f"  first text      {spread(firsts)}")
            if sentences:
                print(f"  first sentence  {spread(sentences)}")
            print(f"  total           {spread([r.timings.total_s for r, _ in runs])}")
        if args.compare:
            print("\nSame question, without and with streaming:")
            print(compare(client, instructions, args.text))
    except LlmError as e:
        print_error(e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
