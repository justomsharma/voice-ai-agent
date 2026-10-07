"""The jitter buffer: turn an uneven stream of packets back into steady audio.

The problem
-----------
The sender produces one frame every 20 ms, like a metronome. After the
network, they arrive unevenly: 3 ms apart, then 45 ms, sometimes out of
order, sometimes not at all. The speaker, though, needs exactly one frame
every 20 ms. If the next frame isn't there when the speaker asks, you hear
a gap.

The fix: wait a little before you start
---------------------------------------
Hold back the first few packets (the "pre-roll") before starting playback.
After that, play one frame per tick, in sequence-number order, no matter
what:

    frame is here         -> play it
    frame isn't here yet  -> play silence; it's now *missed*
    it shows up later     -> too late, its moment has passed: discard ("late")

Every frame held back gives later packets one frame's worth of extra time
to arrive. That's the whole trade-off:

    small buffer -> low delay,  but every late packet becomes a gap (choppy)
    big buffer   -> few gaps,   but everything is heard later (laggy)

This is a *fixed* jitter buffer: its size is set once. Real ones (WebRTC's
NetEQ) are *adaptive*. They watch how much jitter the network has right now
and grow or shrink the buffer, and they hide gaps by stretching the
neighbouring audio instead of playing silence. Same idea, more cleverness.

This class is pure logic: no threads, sockets or clocks. The receiver calls
push() when a packet arrives and pop() once per playout tick, which keeps it
easy to test.
"""

import math
from dataclasses import dataclass


def frames_for_ms(buffer_ms: float, frame_ms: float) -> int:
    """How many frames to hold to cover `buffer_ms`. Rounds *up*: a 50 ms
    buffer of 20 ms frames holds 3 (60 ms), because 2 (40 ms) is less than
    what was asked for. round() first so 60 / 19.999999 doesn't become 4."""
    return math.ceil(round(buffer_ms / frame_ms, 6))


@dataclass(frozen=True)
class Frame:
    """One playout slot. payload None = nothing to play (missed)."""
    seq: int
    payload: bytes | None
    send_time_ns: int | None


class JitterBuffer:
    def __init__(self, target_frames: int):
        if target_frames < 0:
            raise ValueError(f"target_frames must be >= 0, got {target_frames}")
        self.target_frames = target_frames
        # A dict, not a queue: packets arrive in any order, and we look them
        # up by sequence number when their turn comes.
        self._held: dict[int, tuple[bytes, int]] = {}
        self._next_seq: int | None = None  # None while pre-rolling
        self.highest_seq: int | None = None
        self._missed: list[int] = []
        self.late_seqs: set[int] = set()
        self.received = self.late = self.duplicate = self.reordered = self.played = 0

    @property
    def started(self) -> bool:
        return self._next_seq is not None

    @property
    def depth(self) -> int:
        """Frames waiting right now. Times frame_ms, it's the extra delay the
        buffer is adding at this moment."""
        return len(self._held)

    def push(self, seq: int, payload: bytes, send_time_ns: int) -> str:
        if seq in self._held:
            self.duplicate += 1
            return "duplicate"
        if self.highest_seq is not None and seq < self.highest_seq:
            # A later packet overtook this one. Counted before the "late"
            # check: reordering is what the network did, whether or not the
            # packet still makes it in time.
            self.reordered += 1
        # Every arrival counts toward "newest seen", late ones included,
        # otherwise reordering between two late packets would go unnoticed.
        self.highest_seq = seq if self.highest_seq is None else max(self.highest_seq, seq)
        if self._next_seq is not None and seq < self._next_seq:
            # Its slot has already been played (as silence). Playing it now
            # would put the audio out of order, so drop it.
            self.late += 1
            self.late_seqs.add(seq)
            return "late"
        self.received += 1
        self._held[seq] = (payload, send_time_ns)
        # Pre-roll: start only once enough is held. target 0 still needs one
        # packet; there's nothing to play before that.
        if self._next_seq is None and len(self._held) >= max(1, self.target_frames):
            # Start from the lowest seq held: with jitter, the first packet
            # to *arrive* isn't necessarily the first one *sent*.
            self._next_seq = min(self._held)
        return "ok"

    def pop(self) -> Frame | None:
        """Called once per playout tick. None = still pre-rolling."""
        if self._next_seq is None:
            return None
        seq = self._next_seq
        # The playout clock always moves forward, even over a gap. Waiting
        # for a missing packet would delay every frame after it, forever.
        self._next_seq += 1
        item = self._held.pop(seq, None)
        if item is None:
            self._missed.append(seq)
            return Frame(seq, None, None)
        self.played += 1
        return Frame(seq, *item)

    def missed_count(self) -> int:
        """Slots played as silence. Slots past the newest packet we've seen
        don't count: that's just the stream having ended, or not having
        reached us yet."""
        if self.highest_seq is None:
            return 0
        return sum(1 for s in self._missed if s <= self.highest_seq)
