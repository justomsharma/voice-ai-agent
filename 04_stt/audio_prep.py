"""Get audio into the exact shape an STT API expects: its "audio contract".

Every STT engine documents what it accepts. For the ones we use:

    OpenAI gpt-transcribe       a file (wav/mp3/m4a/...), <= 25 MB
    Deepgram nova-3             raw file bytes as the request body
    OpenAI gpt-live-transcribe  raw PCM16, mono, 24000 Hz, base64 in JSON

Our recordings are int16 PCM at whatever rate the mic gave (16 kHz, 48 kHz,
stereo...). So before sending we may need to:

  1. Downmix to mono. Speech models want one voice signal; stereo just
     doubles the upload.
  2. Resample to the rate the engine wants.
  3. Wrap the PCM in a WAV header (batch) or send it bare (streaming).

Resample, never relabel
-----------------------
Changing the number in the WAV header ("this is 16 kHz") without changing
the samples doesn't convert anything. 48,000 samples labelled 16 kHz play
3x slower and an octave and a half lower, and the model hears a monster
voice. Try it: `transcribe.py --relabel-rate 16000` on a 48 kHz file.

Real resampling computes NEW samples at the new instants. Going down
(48k -> 16k) it must first remove everything above the new Nyquist (8 kHz).
Otherwise those frequencies don't vanish, they *alias* into fake lower
tones. Taking every 3rd sample (`x[::3]`) skips that filter, and a 10 kHz
hiss turns into a 6 kHz whistle.

We use the textbook band-limited interpolator: each output sample is a
weighted sum of nearby input samples, the weights being a sinc (the ideal
low-pass filter) tapered by a Kaiser window so it can be finite. It's what
libsamplerate / soxr do, minus their speed tricks.
"""

import io
import time
import wave
from dataclasses import dataclass, field

import numpy as np

from audio_format import AudioFormat

FULL_SCALE = 32768.0
SILENT_PEAK_DB = -60.0   # quieter than this everywhere = probably no speech
TAPS_PER_SIDE = 16       # filter length: more = sharper cut, slower
KAISER_BETA = 8.6        # window shape: ~ -80 dB stopband
ROLLOFF = 0.94           # start cutting at 94% of Nyquist: room for the filter's slope
CHUNK = 4096             # output samples computed per step (bounds memory)


@dataclass(frozen=True)
class AudioContract:
    """What one STT engine accepts. Provider adapters declare one."""
    sample_rate: int
    channels: int = 1
    max_bytes: int | None = None   # e.g. OpenAI's 25 MB upload limit
    min_seconds: float = 0.1       # shorter than this can't hold a word


@dataclass(frozen=True)
class Prepared:
    """Audio after it was made to fit the contract, plus what we did to it."""
    fmt: AudioFormat
    samples: np.ndarray            # int16, (frames, channels)
    source_fmt: AudioFormat
    steps: list[str] = field(default_factory=list)     # what changed, for printing
    warnings: list[str] = field(default_factory=list)  # suspicious, but we still send
    prep_s: float = 0.0            # how long preparing took (wall clock)

    @property
    def seconds(self) -> float:
        return self.fmt.duration_seconds(len(self.samples))


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average the channels. (Summing could overflow int16; averaging can't.)"""
    if samples.shape[1] == 1:
        return samples
    mean = samples.astype(np.float64).mean(axis=1, keepdims=True)
    return np.round(mean).astype(np.int16)


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Band-limited resampling of int16 (frames, channels) audio."""
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError(f"sample rates must be > 0, got {src_rate} -> {dst_rate}")
    if src_rate == dst_rate:
        return samples.copy()
    n, channels = samples.shape
    # Output sample m sits at input position t = m * src / dst. Reduce the
    # ratio to up/down = L/M: then t's fractional part only takes L values
    # ("phases"), so the weights can be computed once per phase. That is
    # the "polyphase" trick, and the difference between 7 s and 0.2 s here.
    g = np.gcd(src_rate, dst_rate)
    up, down = dst_rate // g, src_rate // g
    m_total = n * dst_rate // src_rate
    # Cutoff as a fraction of the INPUT Nyquist. Downsampling 48k -> 16k must
    # cut at 1/3 of it (8 kHz); upsampling keeps everything (1.0).
    cutoff = min(1.0, up / down) * ROLLOFF
    # A lower cutoff = a wider sinc, so it needs more input samples per side.
    half = int(np.ceil(TAPS_PER_SIDE / cutoff))
    offsets = np.arange(-half + 1, half + 1)
    d = (np.arange(up) / up)[:, None] - offsets[None, :]   # distance to each neighbour
    table = cutoff * np.sinc(cutoff * d) * _kaiser(d / half)   # (phases, taps)
    # Zeros around the edges: the filter sees silence beyond the audio.
    x = np.pad(samples.astype(np.float64), ((half, half), (0, 0)))
    out = np.empty((m_total, channels))
    for start in range(0, m_total, CHUNK):
        m = np.arange(start, min(m_total, start + CHUNK))
        base = (m * down) // up                    # whole part of t
        phase = (m * down) % up                    # which weight row
        idx = base[:, None] + offsets[None, :] + half
        out[m] = np.einsum("mk,mkc->mc", table[phase], x[idx])
    # Clip, don't wrap: a filtered peak can overshoot int16 slightly.
    return np.clip(np.round(out), -32768, 32767).astype(np.int16)


def _kaiser(u: np.ndarray) -> np.ndarray:
    """Kaiser window at positions u in [-1, 1] (0 outside)."""
    inside = np.abs(u) < 1
    arg = KAISER_BETA * np.sqrt(np.where(inside, 1 - u * u, 0.0))
    return np.where(inside, np.i0(arg) / np.i0(KAISER_BETA), 0.0)


def peak_dbfs(samples: np.ndarray) -> float:
    peak = int(np.abs(samples.astype(np.int32)).max()) if samples.size else 0
    return -120.0 if peak == 0 else float(20 * np.log10(peak / FULL_SCALE))


def prepare(fmt: AudioFormat, samples: np.ndarray, contract: AudioContract) -> Prepared:
    """Validate, downmix and resample. Raises ValueError for audio we refuse to send."""
    t0 = time.perf_counter()
    if fmt.sample_width != 2 or samples.dtype != np.int16:
        raise ValueError("expected 16-bit PCM (int16) audio")
    if len(samples) == 0:
        raise ValueError("audio is empty: nothing to transcribe")
    seconds = fmt.duration_seconds(len(samples))
    if seconds < contract.min_seconds:
        raise ValueError(f"audio is {seconds * 1000:.0f} ms; need at least "
                         f"{contract.min_seconds * 1000:.0f} ms to hold a word")
    steps, warnings = [], []
    out = samples
    if fmt.channels != contract.channels:
        if contract.channels != 1:
            raise ValueError(f"can't convert {fmt.channels} ch to {contract.channels} ch")
        out = to_mono(out)
        steps.append(f"{fmt.channels} ch -> mono (average)")
    if fmt.sample_rate != contract.sample_rate:
        out = resample(out, fmt.sample_rate, contract.sample_rate)
        steps.append(f"{fmt.sample_rate} Hz -> {contract.sample_rate} Hz (resampled)")
    new_fmt = AudioFormat(contract.sample_rate, contract.channels)
    size = len(out) * new_fmt.bytes_per_frame + 44
    if contract.max_bytes is not None and size > contract.max_bytes:
        raise ValueError(f"payload {size / 1e6:.1f} MB > limit {contract.max_bytes / 1e6:.0f} MB; "
                         "split the audio (e.g. with --vad) or shorten it")
    peak = peak_dbfs(out)
    if peak < SILENT_PEAK_DB:
        warnings.append(f"audio is nearly silent (peak {peak:.0f} dBFS). STT models can "
                        "hallucinate words from silence, so don't trust the text")
    clipped = np.mean(np.abs(out.astype(np.int32)) >= 32767)
    if clipped > 0.001:
        warnings.append(f"{clipped:.1%} of samples are clipped (too loud): accuracy suffers")
    return Prepared(new_fmt, out, fmt, steps, warnings, time.perf_counter() - t0)


def pcm_bytes(samples: np.ndarray) -> bytes:
    """Bare little-endian int16: what a streaming API receives."""
    return np.ascontiguousarray(samples, dtype="<i2").tobytes()


def wav_bytes(fmt: AudioFormat, samples: np.ndarray) -> bytes:
    """The same PCM with a 44-byte WAV header in front: what a batch API receives."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(fmt.channels)
        w.setsampwidth(fmt.sample_width)
        w.setframerate(fmt.sample_rate)
        w.writeframes(pcm_bytes(samples))
    return buf.getvalue()
