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
