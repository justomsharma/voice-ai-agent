import pytest

from jitter_buffer import JitterBuffer, frames_for_ms


def push(jb, *seqs):
    # payload = the seq as one byte, send time = 1000 + seq: easy to check.
    return [jb.push(s, bytes([s]), 1000 + s) for s in seqs]


def test_frames_for_ms_rounds_up():
    assert frames_for_ms(60, 20) == 3
    assert frames_for_ms(50, 20) == 3  # 2 frames would be only 40 ms
    assert frames_for_ms(0, 20) == 0
    assert frames_for_ms(60, 1000 * 320 / 16000) == 3  # frame_ms computed as a float


def test_preroll_waits_for_target():
    jb = JitterBuffer(3)
    push(jb, 0, 1)
    assert jb.pop() is None and not jb.started
    push(jb, 2)
    f = jb.pop()
    assert jb.started and f.seq == 0 and f.payload == b"\x00" and f.send_time_ns == 1000


def test_zero_target_starts_on_first_packet():
    jb = JitterBuffer(0)
    push(jb, 5)
    assert jb.pop().seq == 5


def test_reordered_packets_play_in_order():
    jb = JitterBuffer(2)
    push(jb, 1, 0)
    assert [jb.pop().seq, jb.pop().seq] == [0, 1]
    assert jb.reordered == 1


def test_gap_plays_silence_and_counts_missed():
    jb = JitterBuffer(1)
    push(jb, 0, 2)
    frames = [jb.pop() for _ in range(3)]
    assert [f.payload is None for f in frames] == [False, True, False]
    assert jb.missed_count() == 1 and jb.played == 2


def test_late_packet_rejected():
    jb = JitterBuffer(1)
    push(jb, 0)
    jb.pop()
    jb.pop()  # slot 1 passes with nothing to play
    assert push(jb, 1) == ["late"]
    assert jb.late == 1


def test_duplicate_ignored():
    jb = JitterBuffer(5)
    assert push(jb, 0, 0) == ["ok", "duplicate"]
    assert jb.received == 1 and jb.duplicate == 1


def test_end_of_stream_not_missed():
    jb = JitterBuffer(1)
    push(jb, 0)
    for _ in range(5):  # the playout clock keeps ticking after the last packet
        jb.pop()
    assert jb.missed_count() == 0


def test_negative_target_rejected():
    with pytest.raises(ValueError):
        JitterBuffer(-1)
