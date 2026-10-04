"""The shape of a stream of PCM audio, and the arithmetic that follows from it.

PCM (Pulse-Code Modulation) is the simplest possible digital audio: measure
the microphone's voltage at a fixed rate and store each measurement as an
integer. Nothing else -- no compression, no header, no metadata. That's why
three numbers fully describe it:

    sample_rate   how many measurements per second (Hz)
    channels      how many independent signals (1 = mono, 2 = stereo)
    sample_width  bytes per measurement (2 bytes = 16-bit signed int)

Vocabulary used everywhere in this repo (the terms get mixed up a lot online):

    sample  one integer, for one channel, at one instant.
    frame   one sample *per channel* at one instant. Mono: 1 sample per frame.
            Stereo: 2 samples per frame (L, R), stored interleaved: L R L R ...
    block   N consecutive frames handed over together. Also called a buffer,
            chunk, or -- in LiveKit -- an AudioFrame (confusingly, a LiveKit
            "AudioFrame" is a *block* of frames, with `samples_per_channel`
            telling you N). Realtime systems always move audio in blocks
            (typically 10-20 ms) because you can't process a sound that hasn't
            finished arriving, and per-sample handling would be far too slow.

This module is pure math with no I/O on purpose: everything here can be
unit-tested without a microphone, and both record.py and inspect_audio.py
share one definition of "what the bytes mean".
"""

from dataclasses import dataclass


# frozen=True: an AudioFormat describes data that already exists. If you
# could mutate sample_rate after recording, the same bytes would suddenly
# "mean" a different duration -- a whole class of bugs we make impossible.
# Frozen also gives us __eq__ and __hash__, handy for comparing files.
@dataclass(frozen=True)
class AudioFormat:
    sample_rate: int
    channels: int
    # Default 2 bytes = 16-bit signed PCM ("int16", "s16le"). It's what
    # telephony, most STT APIs and LiveKit's AudioFrame use: 65,536 levels
    # (~96 dB dynamic range) is plenty for speech, and integers are exact,
    # compact and trivially portable. 24/32-bit is for music production.
    sample_width: int = 2

    def __post_init__(self) -> None:
        # Fail at construction, not later as a ZeroDivisionError deep inside
        # some duration calculation where the cause is hard to see.
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")
        if self.channels < 1:
            raise ValueError(f"channels must be >= 1, got {self.channels}")
        if self.sample_width < 1:
            raise ValueError(f"sample_width must be >= 1, got {self.sample_width}")

    @property
    def bit_depth(self) -> int:
        """Bits per sample. Sets the amplitude resolution (loud vs quiet)."""
        return self.sample_width * 8

    @property
    def bytes_per_frame(self) -> int:
        """One instant across all channels. Multiply by frames to get bytes."""
        return self.sample_width * self.channels

    @property
    def bytes_per_second(self) -> int:
        """The raw bandwidth of the stream: what you'd push over a network
        socket per second if you sent uncompressed PCM. 16 kHz mono int16 is
        32,000 B/s (256 kbit/s); 48 kHz is 3x that."""
        return self.bytes_per_frame * self.sample_rate

    @property
    def nyquist_hz(self) -> float:
        """The highest frequency this sample rate can represent: rate / 2.
        To capture a wave you need at least two samples per cycle (one up,
        one down). Anything above Nyquist is lost -- the OS/driver filters
        it out before sampling, otherwise it would "alias" into a fake lower
        tone. Speech intelligibility lives mostly below ~4 kHz, but consonants
        like "s"/"f" have energy up to ~8 kHz and beyond -- which is exactly
        why 8 kHz telephone audio sounds muffled and 16 kHz is the usual
        STT default."""
        return self.sample_rate / 2

    def frames_per_block(self, block_ms: int) -> int:
        """How many frames fit in one block of `block_ms` milliseconds.

        Integer math (multiply first, then floor-divide) instead of
        `rate * block_ms / 1000` so we never get float artifacts like
        319.99999 -- block sizes must be exact whole numbers of frames.
        """
        return self.sample_rate * block_ms // 1000

    def duration_seconds(self, frames: int) -> float:
        """Duration depends only on frames and rate -- NOT on channels or
        bit depth. Stereo doubles the bytes, not the length."""
        return frames / self.sample_rate
