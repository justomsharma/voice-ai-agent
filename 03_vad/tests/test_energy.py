import numpy as np
import pytest

from energy import FLOOR_DB, frame_dbfs


def sine(amplitude: float, n: int = 320, rate: int = 16000, hz: float = 1000.0) -> np.ndarray:
    t = np.arange(n) / rate
    return np.round(amplitude * np.sin(2 * np.pi * hz * t)).astype(np.int16).reshape(-1, 1)


def test_digital_silence_is_floor():
    # log10(0) is -inf; a VAD needs a real number it can compare.
    assert frame_dbfs(np.zeros((320, 1), np.int16)) == FLOOR_DB


def test_empty_frame_is_floor():
    assert frame_dbfs(np.zeros((0, 1), np.int16)) == FLOOR_DB


def test_full_scale_sine_is_minus_3db():
    # A sine's RMS is peak / sqrt(2) -> 20*log10(0.707) = -3.01 dB.
    assert frame_dbfs(sine(32767)) == pytest.approx(-3.01, abs=0.05)


def test_known_level():
    # Constant +/-3277 (a square wave) has RMS 3277 -> 0.1 of full scale = -20 dB.
    x = np.where(np.arange(320) % 2 == 0, 3277, -3277).astype(np.int16).reshape(-1, 1)
    assert frame_dbfs(x) == pytest.approx(-20.0, abs=0.01)


def test_no_int16_overflow():
    # -32768**2 overflows int16. We must square in float64.
    assert frame_dbfs(np.full((320, 1), -32768, np.int16)) == pytest.approx(0.0, abs=1e-9)


def test_stereo_uses_all_channels():
    mono = sine(10000)
    assert frame_dbfs(np.hstack([mono, mono])) == pytest.approx(frame_dbfs(mono))
