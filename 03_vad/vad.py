"""Energy-based Voice Activity Detection: loud frames -> clean speech segments.

The naive rule "frame louder than X = speech" flickers: a word's quiet
middle drops out, a cough counts as a sentence. Real VADs (Silero, WebRTC,
LiveKit, Pipecat) wrap the per-frame decision in a small state machine.
This is Pipecat's, with Silero's two thresholds:

           loud frame                  loud for min_speech_ms
    QUIET ------------> STARTING ---------------------------> SPEAKING
      ^                    |                                    |    ^
      |   dropped quiet    |                        quiet frame |    | loud again
      +--------------------+                                    v    | (just a pause)
      |                                                       STOPPING
      +-------------------------------------------------------------+
                     quiet for min_silence_ms -> segment ends

The four knobs:
  start_db / stop_db  Hysteresis. Starting needs `start_db`; staying in
                      speech only needs the lower `stop_db`. A voice hovering
                      near one threshold would otherwise flip on/off every
                      frame. (Silero: 0.5 to start, 0.35 to stop.)
  min_speech_ms       A door slam or click is loud but short. Speech lasts.
  min_silence_ms      People pause mid-sentence. Don't end the segment on
                      the first quiet frame. This is also *latency*: the VAD
                      can only say "speech ended" min_silence_ms after it
                      did. (LiveKit default 550 ms.)
  pad_ms              Word edges like "s", "f", "h" are quiet and fall under
                      the threshold. Keep a little audio on each side so
                      they aren't chopped off.

Pure logic: it takes one dB number per frame, so it works the same for a
file or a live mic, and tests need no audio at all.
"""

from dataclasses import dataclass

QUIET, STARTING, SPEAKING, STOPPING = "QUIET", "STARTING", "SPEAKING", "STOPPING"


@dataclass(frozen=True)
class VadConfig:
    frame_ms: int = 20
    start_db: float = -40.0
    stop_db: float = -45.0
    min_speech_ms: int = 200
    min_silence_ms: int = 500
    pad_ms: int = 100

    def __post_init__(self) -> None:
        if self.frame_ms <= 0:
            raise ValueError(f"frame_ms must be > 0, got {self.frame_ms}")
        if self.stop_db > self.start_db:
            raise ValueError(f"stop_db ({self.stop_db}) must be <= start_db ({self.start_db}): "
                             "staying in speech should be easier than starting it")
        for name in ("min_speech_ms", "min_silence_ms", "pad_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float


class EnergyVad:
    def __init__(self, config: VadConfig) -> None:
        self.config = config
        self._frame_s = config.frame_ms / 1000
        self._pad_s = config.pad_ms / 1000
        # Durations -> frame counts. At least 1: "0 ms" means "decide on
        # this frame", not "decide before any frame exists".
        self._speech_frames = max(1, round(config.min_speech_ms / config.frame_ms))
        self._silence_frames = max(1, round(config.min_silence_ms / config.frame_ms))
        self.state = QUIET
        self.raw: list[bool] = []  # per-frame "db >= start_db", for showing the flicker
        self._index = 0            # frames pushed so far
        self._elapsed_s = 0.0      # audio seen so far (the last frame may be short)
        self._first_loud = 0       # frame index where the current speech began
        self._last_loud = 0        # last frame index that was over stop_db
        self._count = 0            # frames in a row in STARTING / STOPPING
        self._prev_end_s = 0.0     # end of the last segment, so padding never overlaps

    @property
    def in_speech(self) -> bool:
        return self.state in (SPEAKING, STOPPING)

    @property
    def elapsed_s(self) -> float:
        return self._elapsed_s

    @property
    def speech_start_s(self) -> float:
        return max(self._prev_end_s, self._first_loud * self._frame_s - self._pad_s)

    def push(self, db: float, frame_s: float | None = None) -> list[Segment]:
        """Feed one frame's loudness. Returns a segment if one just ended.

        `frame_s` is this frame's real length. Leave it out for a full frame;
        pass it for the short leftover at the end of a file, so the clock
        (and the last segment's end) doesn't run past the real audio.
        """
        c = self.config
        i = self._index
        self._index += 1
        self._elapsed_s += self._frame_s if frame_s is None else frame_s
        self.raw.append(db >= c.start_db)

        if self.state == QUIET:
            if db >= c.start_db:
                self._first_loud = self._last_loud = i
                self._count = 1
                self.state = STARTING
                self._maybe_confirm()
        elif self.state == STARTING:
            if db >= c.stop_db:
                self._last_loud = i
                self._count += 1
                self._maybe_confirm()
            else:
                self.state = QUIET  # too short: a click, not speech
        elif self.state == SPEAKING:
            if db >= c.stop_db:
                self._last_loud = i
            else:
                self.state = STOPPING
                self._count = 1
                return self._maybe_end()
        elif self.state == STOPPING:
            if db >= c.stop_db:
                self._last_loud = i
                self.state = SPEAKING  # it was only a pause
            else:
                self._count += 1
                return self._maybe_end()
        return []

    def flush(self) -> list[Segment]:
        """End of audio: close speech that is still open."""
        if self.in_speech:
            return [self._end_segment()]
        self.state = QUIET  # an unconfirmed STARTING is dropped, like a click
        return []

    def _maybe_confirm(self) -> None:
        if self._count >= self._speech_frames:
            self.state = SPEAKING

    def _maybe_end(self) -> list[Segment]:
        if self._count >= self._silence_frames:
            return [self._end_segment()]
        return []

    def _end_segment(self) -> Segment:
        start = self.speech_start_s
        # Clamp the end padding to the audio we've actually seen.
        end = min(self.elapsed_s, (self._last_loud + 1) * self._frame_s + self._pad_s)
        self._prev_end_s = end
        self.state = QUIET
        return Segment(start, end)
