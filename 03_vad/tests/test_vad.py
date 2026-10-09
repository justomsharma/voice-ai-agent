import pytest

from vad import EnergyVad, Segment, VadConfig

LOUD, QUIET, BETWEEN = -20.0, -90.0, -42.0  # BETWEEN: under start (-40), over stop (-45)


def run(dbs, **cfg) -> list[Segment]:
    vad = EnergyVad(VadConfig(**cfg))
    out = []
    for db in dbs:
        out += vad.push(db)
    return out + vad.flush()


def approx(segs):
    return [(pytest.approx(s.start_s), pytest.approx(s.end_s)) for s in segs]


def test_silence_has_no_speech():
    assert run([QUIET] * 100) == []


def test_one_second_tone_is_one_padded_segment():
    # Loud frames 50..99 = 1.00 s..2.00 s; 100 ms of padding each side.
    segs = run([QUIET] * 50 + [LOUD] * 50 + [QUIET] * 50)
    assert [(s.start_s, s.end_s) for s in segs] == approx([Segment(0.9, 2.1)])


def test_short_burst_is_ignored():
    # 60 ms door slam < 200 ms min speech.
    assert run([QUIET] * 50 + [LOUD] * 3 + [QUIET] * 50) == []


def test_short_burst_counts_when_min_speech_is_zero():
    # The lesson: without min speech, any click becomes "speech".
    assert len(run([QUIET] * 50 + [LOUD] * 3 + [QUIET] * 50, min_speech_ms=0)) == 1


def test_short_pause_stays_one_segment():
    # 300 ms pause < 500 ms min silence: "I want to... book a ticket".
    segs = run([QUIET] * 20 + [LOUD] * 30 + [QUIET] * 15 + [LOUD] * 30 + [QUIET] * 50)
    assert [(s.start_s, s.end_s) for s in segs] == approx([Segment(0.3, 2.0)])


def test_long_pause_splits_in_two():
    segs = run([QUIET] * 20 + [LOUD] * 30 + [QUIET] * 40 + [LOUD] * 30 + [QUIET] * 50)
    assert [(s.start_s, s.end_s) for s in segs] == approx([Segment(0.3, 1.1), Segment(1.7, 2.5)])


def test_between_thresholds_cannot_start_speech():
    # Hysteresis, part 1: -42 dB is not loud enough to START.
    assert run([QUIET] * 10 + [BETWEEN] * 50) == []


def test_between_thresholds_keeps_speech_going():
    # Hysteresis, part 2: once speaking, -42 dB is loud enough to CONTINUE.
    segs = run([LOUD] * 20 + [BETWEEN] * 30 + [QUIET] * 40)
    assert [(s.start_s, s.end_s) for s in segs] == approx([Segment(0.0, 1.1)])


def test_flush_closes_open_speech():
    # Speech until the very end: end padding is clamped to the audio we have.
    vad = EnergyVad(VadConfig())
    for _ in range(30):
        assert vad.push(LOUD) == []
    assert vad.in_speech
    assert [(s.start_s, s.end_s) for s in vad.flush()] == approx([Segment(0.0, 0.6)])
    assert vad.state == "QUIET"


def test_flush_drops_too_short_start():
    vad = EnergyVad(VadConfig())
    for _ in range(5):
        vad.push(LOUD)
    assert vad.state == "STARTING"
    assert vad.flush() == []


def test_padding_never_overlaps():
    segs = run([LOUD] * 20 + [QUIET] * 6 + [LOUD] * 20 + [QUIET] * 20,
               min_silence_ms=100, pad_ms=200)
    assert len(segs) == 2
    assert segs[1].start_s >= segs[0].end_s


def test_raw_is_per_frame_start_decision():
    vad = EnergyVad(VadConfig())
    for db in [QUIET, LOUD, BETWEEN, LOUD]:
        vad.push(db)
    assert vad.raw == [False, True, False, True]


def test_live_properties():
    vad = EnergyVad(VadConfig())
    for _ in range(25):
        vad.push(QUIET)
    for _ in range(10):  # 10 x 20 ms = min speech -> SPEAKING on the 10th
        vad.push(LOUD)
    assert vad.in_speech
    assert vad.elapsed_s == pytest.approx(0.7)
    assert vad.speech_start_s == pytest.approx(0.4)  # 0.5 s - 100 ms pad


@pytest.mark.parametrize("bad", [
    dict(start_db=-50, stop_db=-40),  # stop louder than start
    dict(frame_ms=0),
    dict(min_speech_ms=-1),
    dict(min_silence_ms=-1),
    dict(pad_ms=-1),
])
def test_bad_config_rejected(bad):
    with pytest.raises(ValueError):
        VadConfig(**bad)


def test_short_last_frame_advances_clock_by_its_real_length():
    vad = EnergyVad(VadConfig())
    for _ in range(30):
        vad.push(LOUD)
    vad.push(LOUD, frame_s=0.01)  # a 10 ms leftover at the end of a file
    assert vad.elapsed_s == pytest.approx(0.61)
    assert [(s.start_s, s.end_s) for s in vad.flush()] == approx([Segment(0.0, 0.61)])
