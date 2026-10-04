import numpy as np
import pytest

from audio_format import AudioFormat
from inspect_audio import compute_stats, magnitude_spectrum, main, zoom_window
from wav_io import write_wav


def sine(fmt, hz=440, secs=1.0, amp=32767):
    t = np.arange(int(fmt.sample_rate * secs)) / fmt.sample_rate
    x = (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)
    return np.repeat(x[:, None], fmt.channels, axis=1)


def test_full_scale_sine_stats():
    fmt = AudioFormat(16000, 1)
    s = compute_stats(fmt, sine(fmt), 44 + 32000)
    assert s["frames"] == 16000 and s["duration_s"] == 1.0
    assert s["peak_dbfs"] == pytest.approx(0, abs=0.1)
    assert s["rms_dbfs"] == pytest.approx(-3.01, abs=0.1)
    assert s["header_bytes"] == 44


def test_silence_is_minus_inf(recwarn):
    fmt = AudioFormat(8000, 1)
    s = compute_stats(fmt, np.zeros((800, 1), np.int16), 1644)
    assert s["rms_dbfs"] == -np.inf and s["peak_dbfs"] == -np.inf
    assert len(recwarn) == 0


def test_stereo_counts():
    fmt = AudioFormat(48000, 2)
    s = compute_stats(fmt, sine(fmt, secs=0.5), 44 + 96000)
    assert (s["frames"], s["total_samples"]) == (24000, 48000)


@pytest.mark.parametrize("rate", [8000, 16000, 48000])
def test_spectrum_peak_at_tone(rate):
    fmt = AudioFormat(rate, 1)
    f, db = magnitude_spectrum(fmt, sine(fmt, hz=1000))
    assert abs(f[np.argmax(db)] - 1000) < 5 and f[-1] == fmt.nyquist_hz


def test_zoom_clamps_short_file():
    fmt = AudioFormat(16000, 1)
    assert len(zoom_window(fmt, sine(fmt, secs=0.005))) == 80  # 5 ms < 20 ms
    assert len(zoom_window(fmt, sine(fmt))) == 320


def test_raw_without_rate_exits_2(tmp_path):
    p = tmp_path / "a.raw"
    p.write_bytes(b"\0\0" * 10)
    assert main([str(p), "--no-plot"]) == 2


def test_cli_wav_no_plot(tmp_path, capsys):
    fmt = AudioFormat(16000, 1)
    p = tmp_path / "a.wav"
    write_wav(p, fmt, sine(fmt))
    assert main([str(p), "--no-plot"]) == 0
    assert "16000" in capsys.readouterr().out


def test_empty_recording_does_not_crash():
    fmt = AudioFormat(16000, 1)
    s = compute_stats(fmt, np.zeros((0, 1), np.int16), 44)
    assert (s["frames"], s["duration_s"], s["min"], s["max"]) == (0, 0.0, 0, 0)
    assert s["rms_dbfs"] == -np.inf


def test_zoom_centres_on_loudest_point():
    fmt = AudioFormat(16000, 1)
    x = np.zeros((16000, 1), np.int16)
    x[12000] = 30000  # a click 0.75 s in; the file's middle is silent
    w = zoom_window(fmt, x)
    assert len(w) == 320 and w.max() == 30000


def test_int16_min_peak_does_not_overflow():
    s = compute_stats(AudioFormat(8000, 1), np.array([[-32768]], np.int16), 46)
    assert s["peak_dbfs"] == 0.0 and s["clipped"] == 1


def test_missing_file_exits_1(tmp_path):
    assert main([str(tmp_path / "nope.wav"), "--no-plot"]) == 1


def test_comparison_table_for_multiple_files(tmp_path, capsys):
    for rate in (8000, 48000):
        fmt = AudioFormat(rate, 1)
        write_wav(tmp_path / f"t_{rate}.wav", fmt, sine(fmt))
    assert main([str(tmp_path / "t_8000.wav"), str(tmp_path / "t_48000.wav"), "--no-plot"]) == 0
    out = capsys.readouterr().out
    assert "== comparison ==" in out and "48000" in out
