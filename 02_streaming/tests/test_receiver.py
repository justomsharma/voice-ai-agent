from packet import Packet
from receiver import IDLE_STOP_S, Receiver, check_output_device


def _feed(rx, n):
    for i in range(n):
        rx.on_packet(Packet(i, 0, 16000, 1, bytes(640)), 0)


def test_stops_when_stream_is_shorter_than_the_buffer():
    # 5 s buffer, 10 packets: pre-roll never completes, nothing ever plays.
    # The receiver must still notice the stream ended instead of hanging.
    rx = Receiver(buffer_ms=5000)
    rx.speaker_ready = True
    _feed(rx, 10)
    assert not rx.jb.started
    assert not rx.should_stop(rx.last_arrival + 0.1)
    assert rx.should_stop(rx.last_arrival + IDLE_STOP_S + 0.1)


def test_waits_for_buffer_to_drain_before_stopping():
    rx = Receiver(buffer_ms=20)
    rx.speaker_ready = True
    _feed(rx, 3)  # started, 3 frames still waiting to be played
    assert rx.jb.started and rx.jb.depth == 3
    assert not rx.should_stop(rx.last_arrival + IDLE_STOP_S + 0.1)


def test_unknown_output_device_is_a_clear_error():
    err = check_output_device("no-such-speaker-xyz")
    assert err is not None and "--device" in err
    assert check_output_device(None) is None  # the default device
