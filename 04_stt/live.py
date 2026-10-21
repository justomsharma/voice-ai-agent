"""Step B: live transcription. Partial text while you speak, final text when you stop.

    python 04_stt/live.py --dry-run                   # see the events, no key needed
    python 04_stt/live.py                             # microphone, 15 s (Ctrl+C stops)
    python 04_stt/live.py --wav 01_audio/recordings/pause_16000hz.wav   # file, played in real time
    python 04_stt/live.py --wav X.wav --delay high    # slower, more accurate partials
    python 04_stt/live.py --wav X.wav --no-gate       # send silence too (compare cost)

    mic/WAV @ 24 kHz -> 20 ms frames -> Stage 03 VAD -> gate -> WebSocket -> partials/finals

Where Stage 03 plugs in
-----------------------
gpt-live-transcribe has no server-side VAD: WE decide when a turn ends.
The gate turns VAD states into protocol events:

    QUIET     keep the last --preroll-ms in a ring buffer, send nothing
    STARTING  send the pre-roll + this frame: the quiet start of the first
              word ("h" in "hello") is in the pre-roll, so it isn't cut off
    (false start: STARTING -> QUIET)  `clear` the server buffer
    SPEAKING / STOPPING  keep sending (pauses inside a sentence too)
    segment ends (min silence passed)  `commit`: the server finalises the turn

So "final after you stopped" = VAD min silence (500 ms) + the model's
finalisation time. That's the same trade-off as Stage 03, now with words.
"""

import argparse
import collections
import json
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "03_vad"))
sys.path.insert(0, str(ROOT / "01_audio"))
from audio_format import AudioFormat  # noqa: E402
from wav_io import read_wav  # noqa: E402

from detect import frame_samples, mic_frames, split_frames  # noqa: E402
from energy import frame_dbfs  # noqa: E402
from vad import STARTING, EnergyVad, Segment, VadConfig  # noqa: E402

from audio_prep import pcm_bytes, prepare  # noqa: E402
from realtime import (CLOSED, CONTRACT, DELAYS, ERROR, FINAL, PARTIAL, URL, Board, RealtimeSession,  # noqa: E402
                      Turn, append_event, session_update)
from stt import SttError  # noqa: E402
from transcribe import HINTS  # noqa: E402

APPEND, COMMIT, CLEAR = "append", "commit", "clear"


class Gate:
    """Turns per-frame VAD decisions into append / commit / clear actions.

    Pure logic (no socket, no clock), so tests can drive it frame by frame.
    """

    def __init__(self, config: VadConfig, preroll_ms: int = 300, gated: bool = True) -> None:
        self.vad = EnergyVad(config)
        self.gated = gated
        self._ring: collections.deque = collections.deque(
            maxlen=max(1, round(preroll_ms / config.frame_ms)) + 1)
        self.sending = False      # are we inside a turn (audio going to the server)?

    def push(self, frame: np.ndarray, db: float, frame_s: float | None = None) -> list[tuple]:
        before = self.vad.state
        ended = self.vad.push(db, frame_s)
        after = self.vad.state
        if not self.gated:
            return self._ungated(frame, ended)
        actions: list[tuple] = []
        # The ring always holds the latest frames, even while sending. After a
        # false start we `clear` the server, so the next start must re-send
        # them: they may hold the real onset (a soft "k" can dip for one frame).
        self._ring.append(frame)
        if not self.sending:
            if after == STARTING or self.vad.in_speech:
                self.sending = True
                actions += [(APPEND, f) for f in self._ring]
        else:
            actions.append((APPEND, frame))
            if ended:
                actions.append((COMMIT, ended[0]))
                self.sending = False
                self._ring.clear()  # committed audio must never be sent twice
            elif before == STARTING and not (after == STARTING or self.vad.in_speech):
                actions.append((CLEAR,))  # it was a click, not speech
                self.sending = False
        return actions

    def _ungated(self, frame: np.ndarray, ended: list[Segment]) -> list[tuple]:
        actions: list[tuple] = [(APPEND, frame)]
        if ended:
            actions.append((COMMIT, ended[0]))
        return actions

    def flush(self) -> list[tuple]:
        """End of audio: close an open turn."""
        ended = self.vad.flush()
        if ended and (self.sending or not self.gated):
            self.sending = False
            return [(COMMIT, ended[0])]
        if self.sending:
            self.sending = False
            return [(CLEAR,)]
        return []


def wav_frames(path: str, n_ms: int) -> tuple[AudioFormat, Iterator[np.ndarray]]:
    """A WAV resampled to the live contract (24 kHz mono), cut into frames."""
    fmt, samples = read_wav(path)
    audio = prepare(fmt, samples, CONTRACT)
    return audio.fmt, split_frames(audio.samples, frame_samples(audio.fmt, n_ms))


def show(turn: Turn, kind: str, at: float) -> str:
    if kind == FINAL:
        return f"[{at:6.2f}s] turn {turn.number} FINAL    \"{turn.final}\""
    return f"[{at:6.2f}s] turn {turn.number} partial  \"{turn.partial}...\""


def committed(board: Board) -> list[Turn]:
    """Turns we sent a commit for. (Ungated mode can end with uncommitted silence.)"""
    return [t for t in board.turns if t.commit_at is not None]


def summary(board: Board, setup_s: float, bytes_sent: int, audio_s: float,
            gated: bool = True) -> str:
    def gap(a, b):
        return f"{b - a:+.2f} s" if a is not None and b is not None else "  -    "

    # Ungated, a turn's "first audio" is the silence after the last commit,
    # so "first text" would mostly measure how long you were quiet. Hide it.

    lines = [f"\nSetup (connect + session.updated): {setup_s * 1000:.0f} ms   "
             f"sent {bytes_sent / 1024:.0f} KB of JSON for {audio_s:.2f} s of audio",
             "\nturn  speech (audio)   first text   final after   final after    text",
             "                        (after 1st   commit        you stopped",
             "                         audio sent)               (incl. VAD)"]
    for t in committed(board):
        spoken = f"{t.speech[0]:5.2f}-{t.speech[1]:5.2f} s" if t.speech else "      ?      "
        text = (f'"{t.final}"' if t.final is not None else
                f"ERROR: {t.error}" if t.error else
                f'NO FINAL (late or lost); last partial "{t.partial}"')
        first = gap(t.first_audio_at, t.first_text_at) if gated else "  -    "
        lines.append(f"{t.number:>4}  {spoken}   {first:>9}   "
                     f"{gap(t.commit_at, t.final_at):>9}     {gap(t.last_loud_at, t.final_at):>9}"
                     f"     {text}")
    if not committed(board):
        lines.append("   (no speech detected: check the mic level with 03_vad/detect.py)")
    for e in board.errors:
        lines.append(f"server error: {e}")
    finals = [t.final for t in committed(board) if t.final]
    lines.append(f"\nFull transcript (finals only): {' '.join(finals) or '(none)'}")
    return "\n".join(lines)


def run(frames: Iterator[np.ndarray], fmt: AudioFormat, session: RealtimeSession, gate: Gate,
        paced: bool, final_timeout: float, clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print) -> Board:
    board = Board()
    setup_s = session.open()
    log(f"Connected in {setup_s * 1000:.0f} ms. Listening...\n")
    t0 = clock()
    sent_s = 0.0
    last_loud_at = None
    stop_db = gate.vad.config.stop_db

    def act(actions: list[tuple], now: float) -> None:
        for action in actions:
            if action[0] == APPEND:
                if not board.turns or board.turns[-1].commit_at is not None:
                    board.start_turn(now)
                session.append(pcm_bytes(action[1]))
            elif action[0] == COMMIT:
                seg: Segment = action[1]
                session.commit()
                board.committed(now, last_loud_at, (seg.start_s, seg.end_s))
                log(f"[{now - t0:6.2f}s] turn {board.turns[-1].number} COMMIT   "
                    f"(VAD: speech {seg.start_s:.2f}-{seg.end_s:.2f} s ended)")
            elif action[0] == CLEAR:
                session.clear()
                board.drop_last()

    def drain(timeout: float = 0.0) -> bool:
        for u in session.poll(timeout):
            if u.kind == CLOSED:
                log(f"error: connection closed: {u.text}")
                return False
            turn = board.apply(u)
            if turn is not None and u.kind in (PARTIAL, FINAL):
                log(show(turn, u.kind, u.at - t0))
            elif u.kind == ERROR and turn is None:
                log(f"[{u.at - t0:6.2f}s] server error: {u.text}")
        return True

    alive = True
    try:
        for i, frame in enumerate(frames):
            now = clock()
            db = frame_dbfs(frame)
            if db >= stop_db:
                last_loud_at = now
            act(gate.push(frame, db, fmt.duration_seconds(len(frame))), now)
            sent_s += fmt.duration_seconds(len(frame))
            if not (alive := drain()):
                break
            if paced:  # play the file in real time, like a mic would deliver it
                sleep(max(0.0, t0 + sent_s - clock()))
        if alive:
            act(gate.flush(), clock())
            # Wait for finals still in flight: a final can be late, or never come.
            deadline = clock() + final_timeout
            while alive and board.pending() and clock() < deadline:
                alive = drain(timeout=0.1)
    except SttError as e:
        log(f"error: {e}")
        drain()  # keep text that had already arrived before the connection broke
    except KeyboardInterrupt:  # still show what we got so far
        log("\nStopped (Ctrl+C).")
    finally:
        session.close()
    log(summary(board, setup_s, session.bytes_sent, sent_s, gated=gate.gated))
    return board


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    d = VadConfig()
    p = argparse.ArgumentParser(description="Live transcription with partial and final text.")
    p.add_argument("--wav", default=None, help="play this WAV in real time instead of the mic")
    p.add_argument("--fast", action="store_true", help="with --wav: send as fast as possible")
    p.add_argument("--seconds", type=float, default=15.0, help="mic: how long to listen (default 15)")
    p.add_argument("--device", default=None, help="mic device index or name substring")
    p.add_argument("--language", "-l", action="append", default=None, help="expected language, repeatable")
    p.add_argument("--keyword", "-k", action="append", default=None, help="expected name/term, repeatable")
    p.add_argument("--prompt", default=None, help="context for the model")
    p.add_argument("--delay", choices=DELAYS, default=None, help="latency vs accuracy (server default if unset)")
    p.add_argument("--model", default="gpt-live-transcribe")
    p.add_argument("--url", default=URL, help="WebSocket URL override")
    p.add_argument("--start-db", type=float, default=d.start_db)
    p.add_argument("--stop-db", type=float, default=d.stop_db)
    p.add_argument("--min-silence-ms", type=int, default=d.min_silence_ms,
                   help="pause that ends a turn = commit (default 500)")
    p.add_argument("--preroll-ms", type=int, default=300, help="audio kept from before speech starts")
    p.add_argument("--no-gate", action="store_true", help="send every frame, even silence")
    p.add_argument("--final-timeout", type=float, default=10.0,
                   help="how long to wait for late finals at the end (default 10 s)")
    p.add_argument("--dry-run", action="store_true", help="print the protocol events, send nothing")
    args = p.parse_args(argv)
    if args.device is not None and args.device.isdigit():
        args.device = int(args.device)
    return args


def dry_run(session: RealtimeSession) -> None:
    frame = np.zeros((480, 1), np.int16)  # 20 ms at 24 kHz
    example = append_event(pcm_bytes(frame))
    example["audio"] = example["audio"][:24] + f"... ({len(example['audio'])} base64 chars)"
    print(f"Dry run, nothing sent.\n\nconnect {session.url}\n"
          "  header Authorization: Bearer $OPENAI_API_KEY   (never printed)\n")
    print("send (once):\n" + json.dumps(session.config, indent=2))
    print("\nsend (every 20 ms of speech: 480 samples x 2 B = 960 B of PCM):\n"
          + json.dumps(example, indent=2))
    print('\nsend (when the VAD says the turn ended):\n{"type": "input_audio_buffer.commit"}')
    print("\nreceive: conversation.item.input_audio_transcription.delta (partial) ... "
          ".completed (final)")


def main(argv: list[str] | None = None, make_session: Callable[..., RealtimeSession] = RealtimeSession,
         sleep: Callable[[float], None] = time.sleep) -> int:
    args = parse_args(argv)
    try:  # bad thresholds or keywords: one error line, not a traceback
        config = VadConfig(start_db=args.start_db, stop_db=args.stop_db,
                           min_silence_ms=args.min_silence_ms)
        session = make_session(model=args.model, languages=args.language, delay=args.delay,
                               prompt=args.prompt, keywords=args.keyword, url=args.url)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.dry_run:
        dry_run(session)
        return 0
    try:
        if args.wav:
            fmt, frames = wav_frames(args.wav, config.frame_ms)
            source = f"{args.wav} (resampled to 24 kHz, {'as fast as possible' if args.fast else 'real time'})"
        else:
            fmt = AudioFormat(CONTRACT.sample_rate, 1)  # ask the OS for 24 kHz directly
            frames = mic_frames(args, fmt, frame_samples(fmt, config.frame_ms))
            source = f"microphone at 24 kHz, {args.seconds:g} s (Ctrl+C stops)"
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Source: {source}")
    print(f"Gate:   {'off: every frame is sent' if args.no_gate else f'on: {args.preroll_ms} ms pre-roll'}, "
          f"commit after {config.min_silence_ms} ms of silence")
    gate = Gate(config, args.preroll_ms, gated=not args.no_gate)
    try:
        board = run(frames, fmt, session, gate, paced=bool(args.wav) and not args.fast,
                    final_timeout=args.final_timeout, sleep=sleep)
    except SttError as e:  # failed before streaming started (key, handshake)
        print(f"error: {e}", file=sys.stderr)
        if e.kind in HINTS:
            print(f"hint: {HINTS[e.kind]}", file=sys.stderr)
        return 1
    except RuntimeError as e:  # mic problems from Stage 03's mic_frames
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0 if all(t.final is not None for t in committed(board)) else 1


if __name__ == "__main__":
    sys.exit(main())
