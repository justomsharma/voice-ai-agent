"""How loud is one frame? The single number an energy VAD decides on.

RMS ("root mean square") = sqrt(mean(sample^2)). It is the *average*
loudness of the frame. Peak is a bad choice here: one click sets the peak
of a whole frame. RMS asks how much energy there is overall.

We report it in dBFS (decibels relative to full scale), exactly like
01_audio/inspect_audio.py, because loudness is heard on a log scale:

      0 dBFS   the loudest int16 can go
    -20 dBFS   loud, close speech
    -40 dBFS   quiet speech / a noisy room          <- our default start
    -60 dBFS   a quiet room's background hiss
   -120 dBFS   our floor for digital silence (real math says -inf)

Why frames of ~20 ms? Speech changes sound every ~10-30 ms (one vowel,
one consonant), so a 20 ms frame is "one sound". Much shorter, and RMS
jumps around within a single vowel cycle. Much longer, and the start of a
word gets blurred together with the silence before it. WebRTC's VAD uses
10/20/30 ms for the same reason, and it's the frame size Stage 02 sends.
"""

import numpy as np

# Same convention as 01_audio: int16's most negative value maps to -1.0.
FULL_SCALE = 32768.0

# log10(0) = -inf, and -inf is awkward in comparisons, averages and plots.
# -120 dB is far below any real microphone's noise (~-90 dB at best), so
# no real frame lands here, but it's still a normal float.
FLOOR_DB = -120.0


def frame_dbfs(frame: np.ndarray) -> float:
    """RMS loudness of one int16 frame (all channels together), in dBFS."""
    if frame.size == 0:
        return FLOOR_DB
    # float64 BEFORE squaring: (-32768)**2 doesn't fit in int16 and would
    # silently wrap around to a wrong, small number.
    x = frame.astype(np.float64) / FULL_SCALE
    rms = float(np.sqrt(np.mean(x * x)))
    if rms == 0.0:
        return FLOOR_DB
    return max(FLOOR_DB, float(20 * np.log10(rms)))
