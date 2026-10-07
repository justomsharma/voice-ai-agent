"""One audio frame on the wire: an 18-byte header followed by raw PCM.

In 01_audio each 20 ms block went into a list. Here each block becomes a
*packet* and crosses a network. The receiver gets nothing but these bytes,
so everything it needs to know has to be inside them:

    offset  size  field          why it's there
    0       4     seq            0, 1, 2, ... A gap means a packet was lost;
                                 a lower number arriving after a higher one
                                 means the network reordered them.
    4       8     send_time_ns   when the sender stamped it. Sender and
                                 receiver run on one machine and share a
                                 clock, so arrival - send_time = network delay.
    12      4     sample_rate    raw PCM can't describe itself (the 01_audio
    16      2     channels       lesson), so the format travels with the bytes.
    18      ...   pcm            int16 little-endian samples, interleaved.

This is a simplified RTP header (RFC 3550), the format real VoIP and WebRTC
calls use. RTP also carries a sequence number, but its timestamp is
different from ours: it counts *samples* (+320 per 20 ms frame at 16 kHz),
not wall-clock time, so it says where the audio belongs in the stream
but can't measure one-way delay. RTP also doesn't put the format in every
packet: a "payload type" number refers to a format agreed during call
setup, which saves bytes. We spend 6 extra bytes per packet to keep things
self-explanatory.
"""

import struct
from dataclasses import dataclass

# "!" = network byte order (big-endian, the convention for protocol headers)
# and no padding between fields. I = uint32, Q = uint64, H = uint16.
# Spelling the layout out means sender and receiver agree on every byte no
# matter what machine either one runs on.
_HEADER = struct.Struct("!IQIH")
HEADER_SIZE = _HEADER.size  # 4 + 8 + 4 + 2 = 18


@dataclass(frozen=True)
class Packet:
    seq: int  # uint32: wraps after 2^32 packets = 2.7 years of 20 ms frames
    send_time_ns: int
    sample_rate: int
    channels: int
    pcm: bytes


def encode(p: Packet) -> bytes:
    return _HEADER.pack(p.seq, p.send_time_ns, p.sample_rate, p.channels) + p.pcm


def decode(data: bytes) -> Packet:
    # UDP delivers a whole datagram or nothing, but anyone can send anything
    # to our port. Reject what can't be one of ours instead of misreading it.
    if len(data) < HEADER_SIZE:
        raise ValueError(f"packet is {len(data)} bytes, shorter than the {HEADER_SIZE}-byte header")
    seq, send_time_ns, sample_rate, channels = _HEADER.unpack_from(data)
    return Packet(seq, send_time_ns, sample_rate, channels, data[HEADER_SIZE:])
