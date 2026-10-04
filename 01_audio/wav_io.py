"""Read and write 16-bit PCM as .wav (with header) and .raw (no header).

Why both formats?

    .raw  is literally the PCM bytes and nothing else. It's what flows over
          a WebSocket or WebRTC track in a realtime voice system. Open it
          and you can't tell if it's 8 kHz mono or 48 kHz stereo -- that
          knowledge has to travel *separately* (which is exactly why
          LiveKit's AudioFrame carries sample_rate and num_channels
          alongside the data).
    .wav  is the same bytes with a 44-byte RIFF header in front that records
          the format. Compare the two file sizes: they differ by exactly 44.

We use the stdlib `wave` module instead of scipy/soundfile so the format
stays transparent: no hidden float conversion, no dependency, and what you
write is byte-for-byte what you read back.

Array convention (used across this repo): int16 numpy array shaped
(frames, channels). Even mono is 2-D, (N, 1), so code never has to branch
on "is this mono?" -- and it's the shape sounddevice hands us.
"""

import wave
from pathlib import Path

import numpy as np

from audio_format import AudioFormat

# '<i2' = little-endian ('<') signed integer ('i') of 2 bytes. WAV is
# little-endian by definition, so we spell it out instead of relying on
# the machine's native byte order (`np.int16`), which only happens to match
# on x86/ARM.
PCM16 = np.dtype("<i2")


def _as_pcm16_bytes(samples: np.ndarray) -> bytes:
    """Interleaved little-endian bytes: frame 0 [L, R], frame 1 [L, R], ...

    A C-ordered (frames, channels) array laid out in memory *is* the
    interleaved layout, so tobytes() gives exactly the PCM byte stream.
    """
    if samples.dtype != np.int16:
        # Refuse rather than silently cast: float audio is usually in
        # [-1.0, 1.0] and astype(int16) would turn it all into 0s and 1s.
        raise TypeError(f"expected int16 samples, got {samples.dtype}")
    return np.ascontiguousarray(samples, dtype=PCM16).tobytes()


def write_wav(path: str | Path, fmt: AudioFormat, samples: np.ndarray) -> None:
    path = Path(path)
    # Create recordings/ on demand so a fresh clone "just works".
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(fmt.channels)
        w.setsampwidth(fmt.sample_width)
        w.setframerate(fmt.sample_rate)
        w.writeframes(_as_pcm16_bytes(samples))


def write_raw(path: str | Path, samples: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_as_pcm16_bytes(samples))


def read_wav(path: str | Path) -> tuple[AudioFormat, np.ndarray]:
    try:
        return _read_wav(path)
    except (wave.Error, EOFError) as e:
        # The stdlib raises wave.Error for float WAVs ("unknown format: 3",
        # common from TTS engines) or non-WAV files, and EOFError for a
        # 0-byte file. Normalise them to ValueError so callers have one
        # "this isn't a file we can read" exception to handle.
        raise ValueError(f"{path}: not a readable 16-bit PCM WAV ({str(e) or type(e).__name__})") from e


def _read_wav(path: str | Path) -> tuple[AudioFormat, np.ndarray]:
    with wave.open(str(Path(path)), "rb") as w:
        if w.getsampwidth() != 2:
            # Everything downstream (int16 min/max, dBFS reference 32768)
            # assumes 16-bit. Say so clearly instead of mis-decoding.
            raise ValueError(
                f"{path}: {w.getsampwidth() * 8}-bit WAV not supported; "
                "this project uses 16-bit PCM only"
            )
        fmt = AudioFormat(w.getframerate(), w.getnchannels(), w.getsampwidth())
        data = w.readframes(w.getnframes())
    return fmt, _decode(data, fmt)


def read_raw(path: str | Path, fmt: AudioFormat) -> np.ndarray:
    # The caller must supply the format: a .raw file can't tell us.
    return _decode(Path(path).read_bytes(), fmt)


def _decode(data: bytes, fmt: AudioFormat) -> np.ndarray:
    # A trailing partial frame (e.g. a truncated file) can't be a valid
    # instant across all channels, so drop it rather than misalign channels.
    usable = len(data) - len(data) % fmt.bytes_per_frame
    # frombuffer is zero-copy and read-only; astype(np.int16) gives us a
    # normal, writable, native-endian array that callers can freely modify.
    flat = np.frombuffer(data[:usable], dtype=PCM16).astype(np.int16)
    return flat.reshape(-1, fmt.channels)
