import wave

import numpy as np
import pytest

from audio_format import AudioFormat
from wav_io import read_raw, read_wav, write_raw, write_wav


def _tone(fmt, n=1600):
    return (np.arange(n * fmt.channels, dtype=np.int16) % 1000).reshape(n, fmt.channels)


@pytest.mark.parametrize("fmt", [AudioFormat(16000, 1), AudioFormat(48000, 2)])
def test_wav_roundtrip(tmp_path, fmt):
    x = _tone(fmt)
    p = tmp_path / "a.wav"
    write_wav(p, fmt, x)
    fmt2, y = read_wav(p)
    assert fmt2 == fmt and np.array_equal(x, y)
    assert p.stat().st_size == 44 + x.nbytes  # header + raw PCM


def test_raw_is_headerless(tmp_path):
    fmt = AudioFormat(8000, 1)
    x = _tone(fmt)
    p = tmp_path / "a.raw"
    write_raw(p, x)
    assert p.stat().st_size == x.nbytes
    assert np.array_equal(read_raw(p, fmt), x)


def test_rejects_8bit_wav(tmp_path):
    p = tmp_path / "8bit.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(b"\x80" * 10)
    with pytest.raises(ValueError):
        read_wav(p)


def test_rejects_float_samples(tmp_path):
    # float audio in [-1, 1] cast to int16 would silently become all zeros
    with pytest.raises(TypeError):
        write_wav(tmp_path / "f.wav", AudioFormat(16000, 1), np.zeros((10, 1), np.float32))


def test_raw_drops_trailing_partial_frame(tmp_path):
    p = tmp_path / "t.raw"
    p.write_bytes(b"\x01\x00\x02\x00\x03\x00" + b"\x04")  # 3 stereo samples + 1 stray byte
    y = read_raw(p, AudioFormat(8000, 2))
    assert y.shape == (1, 2) and y.tolist() == [[1, 2]]


@pytest.mark.parametrize("content", [
    b"",                                   # 0-byte file left by an aborted run
    b"not a wav file at all, just text",   # wrong format entirely
    # 32-bit float WAV header (format tag 3), as many TTS engines produce
    b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x03\x00\x01\x00\x80\x3e\x00\x00"
    b"\x00\xfa\x00\x00\x04\x00\x20\x00data\x00\x00\x00\x00",
])
def test_unreadable_wav_raises_valueerror(tmp_path, content):
    p = tmp_path / "bad.wav"
    p.write_bytes(content)
    with pytest.raises(ValueError):
        read_wav(p)
