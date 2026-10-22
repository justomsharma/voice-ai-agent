"""Milestones 1 and 3: a transcript in, a reply out; then a conversation.

    python 05_llm/chat.py "What's the weather like?"             # one call, whole reply
    python 05_llm/chat.py "What's the weather like?" --dry-run   # the exact request, no key
    python 05_llm/chat.py "How does DNS work?" --prompt detailed # same question, other instructions
    python 05_llm/chat.py "hi" --instructions "Answer like a pirate."
    python 05_llm/chat.py                                        # multi-turn: /history /reset /quit
    python 05_llm/chat.py --no-history                           # watch follow-ups fail
    python 05_llm/chat.py --stream                               # words appear as they're written
    python 05_llm/chat.py --provider fake                        # offline, no key, no cost

    transcript (Stage 04) -> instructions + history + new message -> LLM -> reply
                                                                       -> latency, tokens, how it sounds
"""

import argparse
import sys
from collections.abc import Callable

from config import DEFAULT_PROMPT, Settings, load_prompt
from history import Conversation
from llm import PROVIDERS, LlmClient, LlmError, Reply, make_client
from spoken import describe

HINTS = {
    "auth": "check the key in .env (copy .env.example) or your environment",
    "rate_limit": "wait a bit, or check your plan's credit / limits",
    "bad_request": "the API rejected the request: check --model / --effort / --temperature",
    "timeout": "slow network or a slow model: try --timeout 60",
    "network": "are you online? (DNS, Wi-Fi, proxy, firewall)",
    "server": "the provider is having trouble; it was retried already",
    "stream_cut": "the connection ended early; try again",
}


# ---------------------------------------------------------------- shared CLI bits

def add_llm_args(p: argparse.ArgumentParser, prompt: bool = True, model: bool = True) -> None:
    """The options every Stage 05 script shares."""
    d = Settings()
    p.add_argument("--provider", choices=PROVIDERS, default="openai")
    if model:
        p.add_argument("--model", default=d.model, help=f"default {d.model}")
    p.add_argument("--effort", default=d.effort,
                   help=f"reasoning effort: none|low|medium|high|xhigh|max (default {d.effort})")
    p.add_argument("--max-output-tokens", type=int, default=d.max_output_tokens,
                   help=f"safety cap on reply length (default {d.max_output_tokens})")
    p.add_argument("--temperature", type=float, default=None,
                   help="randomness 0-2, only with --effort none (default: not sent)")
    p.add_argument("--verbosity", default=None, help="low|medium|high (default: not sent)")
    p.add_argument("--timeout", type=float, default=d.timeout_s,
                   help=f"max seconds of silence per socket wait (default {d.timeout_s:g})")
    p.add_argument("--retries", type=int, default=d.retries,
                   help=f"retries on transient errors, before any text (default {d.retries})")
    if prompt:
        p.add_argument("--prompt", default=DEFAULT_PROMPT,
                       help=f"prompt name in 05_llm/prompts or a file path (default {DEFAULT_PROMPT})")
        p.add_argument("--instructions", default=None, help="instructions text (overrides --prompt)")


def settings_from(args: argparse.Namespace, model: str | None = None) -> Settings:
    effort = None if args.effort in (None, "", "default") else args.effort
    return Settings(model=model or args.model, effort=effort,
                    max_output_tokens=args.max_output_tokens, temperature=args.temperature,
                    verbosity=args.verbosity, timeout_s=args.timeout, retries=args.retries).check()


def instructions_from(args: argparse.Namespace) -> str:
    return args.instructions if args.instructions is not None else load_prompt(args.prompt)


def ms(seconds: float | None) -> str:
    return "-" if seconds is None else f"{seconds * 1000:.0f} ms"


def stats(reply: Reply) -> str:
    t, u = reply.timings, reply.usage
    retried = f"  ({reply.attempts} attempts)" if reply.attempts > 1 else ""
    return "\n".join([
        f"  time:   first text {ms(t.first_text_s)} | total {ms(t.total_s)}   "
        f"(connect {ms(t.connect_s)}, then {ms(t.wait_s)} until the server answered){retried}",
        f"  tokens: in {u.input_tokens} (cached {u.cached_tokens}) | "
        f"out {u.output_tokens} (reasoning {u.reasoning_tokens})   [{reply.provider} {reply.model}]",
        f"  spoken: {describe(reply.text)}"])


def print_error(e: LlmError) -> None:
    if e.partial_text:
        print("\n  [cut off] the text above is NOT a complete answer", file=sys.stderr)
    print(f"error: {e}", file=sys.stderr)
    if e.kind in HINTS:
        print(f"hint: {HINTS[e.kind]}", file=sys.stderr)


def ask(client: LlmClient, instructions: str, messages, stream: bool) -> Reply:
    """One call. With stream=True, text is printed as it arrives."""
    if not stream:
        reply = client.complete(instructions, messages)
        print(f"agent> {reply.text}")
        return reply
    print("agent> ", end="", flush=True)
    reply = client.stream(instructions, messages, lambda t: print(t, end="", flush=True))
    print()
    return reply


# ---------------------------------------------------------------- chat

def turn(client: LlmClient, convo: Conversation, text: str, stream: bool) -> Reply | None:
    """Send one user message with the history, print the reply, save it if complete."""
    messages = convo.messages_for(text)
    try:
        reply = ask(client, convo.instructions, messages, stream)
    except LlmError as e:
        print_error(e)
        print("  memory: this turn was not saved", file=sys.stderr)
        return None
    if not reply.complete:
        print(f"  [incomplete: {reply.reason}] the reply was cut off, so it's not saved to memory")
    saved = convo.commit(text, reply)
    print(stats(reply))
    earlier = len(messages) - 1
    forgot = f", forgot {convo.forgotten} old turn(s)" if convo.forgotten else ""
    print(f"  memory: sent {earlier // 2} earlier turn(s), ~{convo.tokens(messages)} tokens "
          f"(estimate){forgot}; this turn {'saved' if saved else 'NOT saved'}")
    return reply


def interactive(client: LlmClient, convo: Conversation, stream: bool,
                read: Callable[[str], str]) -> int:
    print("Type a message and press Enter. /history shows what is sent, /reset forgets, "
          "/quit ends.\n")
    while True:
        try:
            text = read("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return 0
        if text == "/reset":
            convo.clear()
            print("  memory cleared\n")
            continue
        if text == "/history":
            print(f"  instructions: {convo.instructions[:80]!r}...")
            for m in convo.history():
                print(f"  {m.role:>9}: {m.text}")
            print(f"  ({len(convo.turns)} turn(s), ~{convo.tokens(convo.history())} tokens)\n")
            continue
        turn(client, convo, text, stream)
        print()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Talk to an LLM: one message or a conversation.")
    p.add_argument("text", nargs="?", help="one message (leave out for a conversation)")
    add_llm_args(p)
    p.add_argument("--stream", action="store_true", help="print the reply as it is written")
    p.add_argument("--no-history", action="store_true", help="send each message alone")
    p.add_argument("--max-turns", type=int, default=None, help="keep only the last N turns")
    p.add_argument("--max-history-tokens", type=int, default=None,
                   help="forget old turns above this many (estimated) input tokens")
    p.add_argument("--dry-run", action="store_true", help="show the request, don't send it")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, make: Callable[..., LlmClient] = make_client,
         read: Callable[[str], str] = input) -> int:
    args = parse_args(argv)
    try:
        client = make(args.provider, settings_from(args))
        convo = Conversation(instructions_from(args),
                             max_turns=0 if args.no_history else args.max_turns,
                             max_tokens=args.max_history_tokens)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        text = args.text or "What's the weather like?"
        print("Dry run, nothing sent. The request would be:\n")
        print(client.describe_request(convo.instructions, convo.messages_for(text), args.stream))
        return 0
    if args.text is None:
        return interactive(client, convo, args.stream, read)
    reply = turn(client, convo, args.text, args.stream)
    return 0 if reply is not None else 1


if __name__ == "__main__":
    sys.exit(main())
