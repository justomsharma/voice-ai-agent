"""Find the speech in a recording (or the live mic) with an energy VAD.

    python 03_vad/detect.py --wav 01_audio/recordings/take_16000hz.wav
    python 03_vad/detect.py --wav X.wav --noise -35        # a noisy room
    python 03_vad/detect.py --wav X.wav --plot             # see dB vs thresholds
    python 03_vad/detect.py --wav X.wav --out speech.wav   # hear what was kept
    python 03_vad/detect.py                                # live mic, 10 s

    audio -> 20 ms frames -> loudness (dBFS) -> EnergyVad -> speech segments

VAD answers "is there speech sound right now?". It does NOT know whether
you've finished your sentence. "I want to... [pause] ...book a ticket"
looks like silence in the middle. Deciding "they're done, reply now" is
turn detection (Stage 08), which also looks at the words.
"""

import argparse
import queue
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Stage folders start with a digit, so they can't be imported as packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_audio"))
from audio_format import AudioFormat  # noqa: E402
from wav_io import read_wav, write_wav  # noqa: E402

from energy import FULL_SCALE, frame_dbfs  # noqa: E402
from vad import EnergyVad, Segment, VadConfig  # noqa: E402

CHAR_MS = 100   # one timeline character = 100 ms
ROW_CHARS = 60  # 6 s of audio per printed row


@dataclass
class Analysis:
    dbs: list[float]         # loudness of every frame
    raw: list[bool]          # every frame judged alone: >= start_db?
    segments: list[Segment]  # after the state machine
    samples: np.ndarray      # all the audio that was analysed (frames, channels)


def split_frames(samples: np.ndarray, n: int) -> Iterator[np.ndarray]:
    """Cut audio into n-frame blocks. The last one may be shorter.

    Unlike Stage 02 we don't pad it: zero padding would make the last
    frame look quieter than it really is.
    """
    for off in range(0, len(samples), n):
        yield samples[off:off + n]


def analyze(frames: Iterable[np.ndarray], channels: int, config: VadConfig,
            log: Callable[[str], None] | None = None) -> Analysis:
    """Run the VAD over frames as they arrive (file or live mic alike).

    `log` gets a line the moment the VAD *decides* something. Compare that
    time with the segment's own start/end: the gap is the VAD's built-in
    delay (min speech to start, min silence to stop). In a voice agent,
    the min-silence gap adds directly to response latency.
    """
    vad = EnergyVad(config)
    dbs: list[float] = []
    segments: list[Segment] = []
    kept: list[np.ndarray] = []
    was_speaking = False
    for frame in frames:
        kept.append(frame)
        db = frame_dbfs(frame)
        dbs.append(db)
        done = vad.push(db)
        segments += done
        if log:
            if vad.in_speech and not was_speaking:
                log(f"[{vad.elapsed_s:6.2f}s] speech START  (began at {vad.speech_start_s:.2f}s)")
            for s in done:
                log(f"[{vad.elapsed_s:6.2f}s] speech END    ({s.start_s:.2f}s - {s.end_s:.2f}s)")
        was_speaking = vad.in_speech
    for s in vad.flush():
        segments.append(s)
        if log:
            log(f"[{vad.elapsed_s:6.2f}s] speech END    ({s.start_s:.2f}s - {s.end_s:.2f}s)"
                "  <- audio ended")
    samples = np.concatenate(kept) if kept else np.zeros((0, channels), np.int16)
    return Analysis(dbs, vad.raw, segments, samples)


def add_noise(samples: np.ndarray, noise_db: float, rng: np.random.Generator) -> np.ndarray:
    """Mix in white noise with an RMS of `noise_db` dBFS (a fan, a busy room).

    For Gaussian noise the RMS *is* the standard deviation, so sigma comes
    straight from the dB level. Clip, don't wrap: int16 overflow would turn
    +32768 into -32768, a loud click.
    """
    sigma = 10 ** (noise_db / 20) * FULL_SCALE
    noisy = samples.astype(np.float64) + rng.normal(0.0, sigma, samples.shape)
    return np.clip(np.round(noisy), -32768, 32767).astype(np.int16)


def segments_to_flags(segments: list[Segment], n_frames: int, frame_ms: int) -> list[bool]:
    """Per-frame speech yes/no after the VAD (frame centre inside a segment)."""
    frame_s = frame_ms / 1000
    centres = [(i + 0.5) * frame_s for i in range(n_frames)]
    return [any(s.start_s <= t < s.end_s for s in segments) for t in centres]


def timeline(flags: list[bool], frame_ms: int, char_ms: int = CHAR_MS) -> str:
    """One char per `char_ms`: '#' all speech, '.' none, '+' mixed (flicker)."""
    per = max(1, round(char_ms / frame_ms))
    out = []
    for i in range(0, len(flags), per):
        chunk = flags[i:i + per]
        hits = sum(chunk)
        out.append("#" if hits == len(chunk) else "." if hits == 0 else "+")
    return "".join(out)


def keep_speech(fmt: AudioFormat, samples: np.ndarray, segments: list[Segment]) -> np.ndarray:
    """Only the speech parts, back to back: what a VAD would pass on to STT."""
    rate = fmt.sample_rate
    parts = [samples[round(s.start_s * rate):round(s.end_s * rate)] for s in segments]
    return np.concatenate(parts) if parts else samples[:0]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    d = VadConfig()  # one source of truth for the defaults
    p = argparse.ArgumentParser(description="Energy-based voice activity detection.")
    p.add_argument("--wav", default=None, help="16-bit WAV to analyse (default: live mic)")
    p.add_argument("--frame-ms", type=int, default=d.frame_ms, help="frame size (default 20)")
    p.add_argument("--start-db", type=float, default=d.start_db,
                   help="a frame this loud (dBFS) can START speech (default -40)")
    p.add_argument("--stop-db", type=float, default=d.stop_db,
                   help="speech CONTINUES while frames stay above this (default -45)")
    p.add_argument("--min-speech-ms", type=int, default=d.min_speech_ms,
                   help="ignore sounds shorter than this (default 200)")
    p.add_argument("--min-silence-ms", type=int, default=d.min_silence_ms,
                   help="a pause must last this long to end speech (default 500)")
    p.add_argument("--pad-ms", type=int, default=d.pad_ms,
                   help="extra audio kept before/after each segment (default 100)")
    p.add_argument("--noise", type=float, default=None, metavar="DB",
                   help="mix in white noise at this level, e.g. -35 (a noisy room)")
    p.add_argument("--seed", type=int, default=0, help="noise seed (default 0)")
    p.add_argument("--plot", action="store_true", help="show waveform and dB vs thresholds")
    p.add_argument("--out", default=None, help="write only the speech parts to this WAV")
    p.add_argument("--rate", type=int, default=16000, help="mic sample rate (default 16000)")
    p.add_argument("--channels", type=int, default=1, help="mic channels (default 1)")
    p.add_argument("--seconds", type=float, default=10.0, help="mic: how long to listen (default 10)")
    p.add_argument("--device", default=None, help="mic device index or name substring")
    args = p.parse_args(argv)
    if args.device is not None and args.device.isdigit():
        args.device = int(args.device)
    return args


def mic_frames(args: argparse.Namespace, fmt: AudioFormat, n: int) -> Iterator[np.ndarray]:
    # Imported here so tests and --wav runs never need an audio device.
    import sounddevice as sd

    try:
        sd.check_input_settings(device=args.device, channels=fmt.channels,
                                dtype="int16", samplerate=fmt.sample_rate)
    except (sd.PortAudioError, ValueError) as e:
        raise ValueError(f"mic can't record {fmt.sample_rate} Hz / {fmt.channels} ch: {e}\n"
                         "hint: python 01_audio/record.py --list-devices, then --device") from e
    total = round(args.seconds * fmt.sample_rate)

    def live() -> Iterator[np.ndarray]:
        blocks: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status) -> None:
            blocks.put(indata.copy())  # audio thread: copy and enqueue only

        with sd.InputStream(samplerate=fmt.sample_rate, channels=fmt.channels, dtype="int16",
                            blocksize=n, device=args.device, callback=callback):
            got = 0
            try:
                while got < total:
                    try:
                        block = blocks.get(timeout=2.0)
                    except queue.Empty:
                        raise RuntimeError("no audio arrived for 2 s -- is the mic connected/allowed?")
                    got += len(block)
                    yield block
            except KeyboardInterrupt:
                # Ctrl+C just ends the recording early; we still report
                # what was heard so far.
                print("\nStopped.")

    return live()


def format_timelines(raw_line: str, vad_line: str, char_ms: int = CHAR_MS) -> str:
    rows = []
    for i in range(0, max(len(raw_line), 1), ROW_CHARS):
        rows.append(f"  {i * char_ms / 1000:5.1f}s  raw |{raw_line[i:i + ROW_CHARS]}|")
        rows.append(f"          vad |{vad_line[i:i + ROW_CHARS]}|")
    return "\n".join(rows)


def make_figure(fmt: AudioFormat, a: Analysis, config: VadConfig):
    """Waveform on top, per-frame dB below with both thresholds: the view
    you tune thresholds with. Green = what the VAD called speech."""
    import matplotlib.pyplot as plt  # lazy: ~0.5 s import, only for --plot

    t = np.arange(len(a.samples)) / fmt.sample_rate
    frame_t = (np.arange(len(a.dbs)) + 0.5) * config.frame_ms / 1000
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(14, 6))
    top.plot(t, a.samples[:, 0] / FULL_SCALE, lw=0.5)
    top.set(ylabel="amplitude", ylim=(-1, 1), title="Waveform (green = speech segments)")
    bottom.plot(frame_t, a.dbs, lw=1, label="frame loudness")
    bottom.axhline(config.start_db, color="red", ls="--", label=f"start {config.start_db:g} dBFS")
    bottom.axhline(config.stop_db, color="orange", ls="--", label=f"stop {config.stop_db:g} dBFS")
    bottom.set(xlabel="time (s)", ylabel="dBFS", ylim=(-100, 0))
    bottom.legend(loc="upper right")
    for s in a.segments:
        for ax in (top, bottom):
            ax.axvspan(s.start_s, s.end_s, color="green", alpha=0.15)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = VadConfig(args.frame_ms, args.start_db, args.stop_db,
                           args.min_speech_ms, args.min_silence_ms, args.pad_ms)
        rng = np.random.default_rng(args.seed)
        if args.wav:
            fmt, samples = read_wav(args.wav)
            if args.noise is not None:
                samples = add_noise(samples, args.noise, rng)
            frames = split_frames(samples, fmt.frames_per_block(config.frame_ms))
        else:
            fmt = AudioFormat(args.rate, args.channels)
            frames = mic_frames(args, fmt, fmt.frames_per_block(config.frame_ms))
            if args.noise is not None:
                frames = (add_noise(f, args.noise, rng) for f in frames)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    noise = f"  + white noise at {args.noise:g} dBFS" if args.noise is not None else ""
    print(f"Source: {args.wav or 'microphone (speak now)'}  "
          f"{fmt.sample_rate} Hz, {fmt.channels} ch{noise}")
    print(f"VAD:    {config.frame_ms} ms frames | start >= {config.start_db:g} dBFS, "
          f"stop < {config.stop_db:g} dBFS | min speech {config.min_speech_ms} ms, "
          f"min silence {config.min_silence_ms} ms, pad {config.pad_ms} ms\n")
    try:
        a = analyze(frames, fmt.channels, config, log=print)
    except (RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    total_s = fmt.duration_seconds(len(a.samples))
    speech_s = sum(s.end_s - s.start_s for s in a.segments)
    print(f"\n{len(a.segments)} speech segment(s): {speech_s:.2f} s of {total_s:.2f} s audio")
    for s in a.segments:
        print(f"  {s.start_s:6.2f}s - {s.end_s:6.2f}s  ({s.end_s - s.start_s:.2f} s)")

    vad_flags = segments_to_flags(a.segments, len(a.dbs), config.frame_ms)
    print(f"\nTimeline: 1 char = {CHAR_MS} ms   # speech   + flicker   . silence")
    print(format_timelines(timeline(a.raw, config.frame_ms), timeline(vad_flags, config.frame_ms)))

    if args.out:
        if a.segments:
            write_wav(args.out, fmt, keep_speech(fmt, a.samples, a.segments))
            print(f"\nWrote speech only to {args.out}")
        else:
            print(f"\nNo speech found: nothing written to {args.out}")
    if args.plot:
        import matplotlib.pyplot as plt
        make_figure(fmt, a, config)
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
