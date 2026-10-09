import numpy as np
import pytest

from audio_format import AudioFormat
from detect import (add_noise, analyze, keep_speech, segments_to_flags, split_frames,
                    timeline)
from energy import frame_dbfs
from vad import Segment, VadConfig

FMT = AudioFormat(16000, 1)


def tone(seconds: float, amplitude: int = 10000, rate: int = 16000, channels: int = 1) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    x = np.round(amplitude * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
    return np.repeat(x.reshape(-1, 1), channels, axis=1)


def silence(seconds: float, rate: int = 16000, channels: int = 1) -> np.ndarray:
    return np.zeros((round(seconds * rate), channels), np.int16)


def run(samples, fmt=FMT, config=VadConfig()):
    return analyze(split_frames(samples, fmt.frames_per_block(config.frame_ms)),
                   fmt.channels, config)


def test_split_frames_keeps_short_last_frame():
    blocks = list(split_frames(np.arange(10, dtype=np.int16).reshape(-1, 1), 4))
    assert [len(b) for b in blocks] == [4, 4, 2]


def test_analyze_finds_tone():
    a = run(np.concatenate([silence(1), tone(1), silence(1)]))
    assert len(a.dbs) == 150 and len(a.raw) == 150
    assert [(s.start_s, s.end_s) for s in a.segments] == [
        (pytest.approx(0.9), pytest.approx(2.1))]
    assert len(a.samples) == 3 * 16000


def test_analyze_speech_until_end():
    a = run(np.concatenate([silence(0.5), tone(1)]))
    assert [(s.start_s, s.end_s) for s in a.segments] == [
        (pytest.approx(0.4), pytest.approx(1.5))]


def test_analyze_stereo_48k():
    fmt = AudioFormat(48000, 2)
    x = np.concatenate([silence(1, 48000, 2), tone(1, rate=48000, channels=2), silence(1, 48000, 2)])
    a = run(x, fmt)
    assert len(a.dbs) == 150
    assert len(a.segments) == 1
    assert a.samples.shape == (3 * 48000, 2)


def test_analyze_empty():
    a = run(silence(0))
    assert a.dbs == [] and a.segments == []
    assert a.samples.shape == (0, 1)


def test_analyze_logs_start_and_end():
    lines = []
    analyze(split_frames(np.concatenate([silence(1), tone(1), silence(1)]), 320),
            1, VadConfig(), log=lines.append)
    assert len(lines) == 2
    assert "START" in lines[0] and "END" in lines[1]


def test_add_noise_level_and_repeatable():
    rng1, rng2 = np.random.default_rng(1), np.random.default_rng(1)
    a = add_noise(silence(1), -30, rng1)
    assert a.dtype == np.int16
    assert frame_dbfs(a) == pytest.approx(-30, abs=0.3)
    assert np.array_equal(a, add_noise(silence(1), -30, rng2))


def test_add_noise_clips_not_wraps():
    loud = np.full((16000, 1), 32000, np.int16)
    out = add_noise(loud, 0, np.random.default_rng(0))
    assert out.dtype == np.int16
    assert out.max() == 32767  # many samples clipped at the top...
    # ...and stayed positive. Wrap-around would flip ~half of them negative
    # (P(>0) would drop from ~0.83 to ~0.34).
    assert np.mean(out > 0) > 0.7


def test_noise_causes_false_positives():
    # The core lesson: on pure silence, enough noise looks like speech.
    quiet = silence(3)
    assert run(quiet).segments == []
    assert len(run(add_noise(quiet, -30, np.random.default_rng(0))).segments) >= 1


def test_segments_to_flags():
    flags = segments_to_flags([Segment(0.1, 0.2)], 15, 20)
    assert flags == [False] * 5 + [True] * 5 + [False] * 5


def test_timeline_chars():
    # 100 ms per char = 5 frames: all speech '#', none '.', some '+'.
    flags = [True] * 5 + [False] * 5 + [True, False, True, False, False]
    assert timeline(flags, 20) == "#.+"


def test_keep_speech():
    x = np.arange(16000, dtype=np.int16).reshape(-1, 1)
    out = keep_speech(FMT, x, [Segment(0.1, 0.2), Segment(0.5, 0.6)])
    assert len(out) == 3200
    assert out[0, 0] == 1600 and out[1600, 0] == 8000


def test_keep_speech_nothing():
    assert keep_speech(FMT, silence(1), []).shape == (0, 1)
