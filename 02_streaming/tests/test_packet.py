import pytest

from packet import HEADER_SIZE, Packet, decode, encode


def test_roundtrip():
    p = Packet(7, 1_700_000_000_123_456_789, 16000, 1, b"\x01\x02" * 320)
    data = encode(p)
    assert len(data) == HEADER_SIZE + 640  # 18-byte header + 20 ms of 16 kHz mono
    assert decode(data) == p


def test_header_is_18_network_order_bytes():
    assert HEADER_SIZE == 18
    # seq=1 in big-endian ("network order") is 00 00 00 01.
    assert encode(Packet(1, 0, 16000, 1, b""))[:4] == b"\x00\x00\x00\x01"


def test_short_packet_rejected():
    with pytest.raises(ValueError):
        decode(b"\x00" * 10)
