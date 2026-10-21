import io
import wave

import numpy as np
import pytest

from audio_format import AudioFormat
from audio_prep import (AudioContract, pcm_bytes, peak_dbfs, prepare, resample, to_mono,
                        wav_bytes)

C16 = AudioContract(16000)


def sine(freq: float, seconds: float, rate: int, amplitude: float = 10000.0) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    return np.round(amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16).reshape(-1, 1)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def peak_freq(x: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(x[:, 0].astype(np.float64)))
    return float(np.fft.rfftfreq(len(x), 1 / rate)[np.argmax(spectrum)])


def middle(x: np.ndarray) -> np.ndarray:
    return x[len(x) // 10: -len(x) // 10]  # skip the edges (filter ramp-in)


def test_to_mono_averages_channels():
    x = np.array([[100, 300], [-100, -101]], np.int16)
    assert to_mono(x).tolist() == [[200], [-100]]


def test_to_mono_extremes_dont_overflow():
    x = np.array([[32767, 32767], [-32768, -32768]], np.int16)
    assert to_mono(x).tolist() == [[32767], [-32768]]


@pytest.mark.parametrize("src,dst", [(48000, 16000), (44100, 16000), (16000, 24000), (8000, 16000)])
def test_resample_keeps_frequency_level_and_duration(src, dst):
    x = sine(1000, 1.0, src)
    y = resample(x, src, dst)
    assert len(y) == dst  # 1 s in, 1 s out
    assert peak_freq(y, dst) == pytest.approx(1000, abs=2)
    assert rms(middle(y)) == pytest.approx(rms(middle(x)), rel=0.01)


def test_resample_removes_content_above_new_nyquist():
    # 10 kHz can't exist at 16 kHz (Nyquist 8 kHz). The filter must remove it...
    x = sine(10000, 1.0, 48000)
    good = resample(x, 48000, 16000)
    assert rms(middle(good)) < rms(x) * 0.01          # >= 40 dB quieter
    # ...whereas naive decimation folds it down to a fake 6 kHz tone.
    naive = x[::3]
    assert peak_freq(naive, 16000) == pytest.approx(6000, abs=2)
    assert rms(naive) == pytest.approx(rms(x), rel=0.05)


def test_resample_same_rate_is_a_copy():
    x = sine(440, 0.1, 16000)
    y = resample(x, 16000, 16000)
    assert np.array_equal(x, y) and y is not x


def test_resample_stereo_keeps_channels():
    x = np.repeat(sine(500, 0.5, 48000), 2, axis=1)
    assert resample(x, 48000, 16000).shape == (8000, 2)


def test_prepare_downmixes_and_resamples():
    x = np.repeat(sine(500, 1.0, 48000), 2, axis=1)
    p = prepare(AudioFormat(48000, 2), x, C16)
    assert p.fmt == AudioFormat(16000, 1)
    assert p.samples.shape == (16000, 1)
    assert p.seconds == pytest.approx(1.0)
    assert p.steps == ["2 ch -> mono (average)", "48000 Hz -> 16000 Hz (resampled)"]
    assert p.warnings == []


def test_prepare_already_matching_does_nothing():
    p = prepare(AudioFormat(16000, 1), sine(500, 1.0, 16000), C16)
    assert p.steps == []


def test_prepare_rejects_empty_and_too_short():
    with pytest.raises(ValueError, match="empty"):
        prepare(AudioFormat(16000, 1), np.zeros((0, 1), np.int16), C16)
    with pytest.raises(ValueError, match="50 ms"):
        prepare(AudioFormat(16000, 1), sine(500, 0.05, 16000), C16)


def test_prepare_rejects_too_big():
    contract = AudioContract(16000, max_bytes=10_000)
    with pytest.raises(ValueError, match="limit"):
        prepare(AudioFormat(16000, 1), sine(500, 1.0, 16000), contract)


def test_prepare_rejects_non_int16():
    with pytest.raises(ValueError, match="16-bit"):
        prepare(AudioFormat(16000, 1), np.zeros((16000, 1), np.float32), C16)


def test_prepare_warns_on_silence_and_clipping():
    quiet = prepare(AudioFormat(16000, 1), sine(500, 1.0, 16000, amplitude=10), C16)
    assert "nearly silent" in quiet.warnings[0]
    loud = prepare(AudioFormat(16000, 1), np.full((16000, 1), 32767, np.int16), C16)
    assert "clipped" in loud.warnings[0]


def test_peak_dbfs():
    assert peak_dbfs(np.zeros((10, 1), np.int16)) == -120.0
    assert peak_dbfs(np.array([[16384]], np.int16)) == pytest.approx(-6.02, abs=0.01)


def test_wav_bytes_round_trip():
    x = sine(500, 0.5, 16000)
    data = wav_bytes(AudioFormat(16000, 1), x)
    assert len(data) == 44 + len(x) * 2  # header + PCM, nothing else
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        back = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 1)
    assert np.array_equal(back, x)


def test_pcm_bytes_is_little_endian():
    assert pcm_bytes(np.array([[1], [-2]], np.int16)) == b"\x01\x00\xfe\xff"


def test_relabel_is_not_resampling():
    # The bug this stage warns about: same samples, new label = new duration.
    x = sine(1000, 1.0, 48000)
    assert AudioFormat(16000, 1).duration_seconds(len(x)) == 3.0
    assert peak_freq(x, 16000) == pytest.approx(1000 / 3, abs=2)  # pitch drops 3x
