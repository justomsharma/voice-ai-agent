import pytest
from audio_format import AudioFormat


def test_16k_mono_20ms_block():
    f = AudioFormat(16000, 1)
    assert f.frames_per_block(20) == 320
    assert f.frames_per_block(20) * f.bytes_per_frame == 640
    assert (f.bit_depth, f.bytes_per_second, f.nyquist_hz) == (16, 32000, 8000)


def test_48k_stereo():
    f = AudioFormat(48000, 2)
    assert f.bytes_per_frame == 4
    assert f.frames_per_block(20) == 960
    assert f.duration_seconds(96000) == 2.0


@pytest.mark.parametrize("rate,ch", [(0, 1), (16000, 0)])
def test_rejects_invalid(rate, ch):
    with pytest.raises(ValueError):
        AudioFormat(rate, ch)
