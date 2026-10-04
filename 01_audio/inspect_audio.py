"""Inspect 16-bit PCM recordings: print what the numbers are, then plot them.

    python 01_audio/inspect_audio.py 01_audio/recordings/take_16000hz.wav
    python 01_audio/inspect_audio.py rec/take_8000hz.wav rec/take_16000hz.wav rec/take_48000hz.wav
    python 01_audio/inspect_audio.py take.raw --rate 16000 --channels 1

Pass several files to compare them side by side -- that's the 8 / 16 / 48 kHz
experiment from the README.

(Named inspect_audio.py, not inspect.py: Python puts a script's own folder
first on sys.path, so a file called inspect.py would hijack the stdlib
`inspect` module that numpy and matplotlib import, and crash them.)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from audio_format import AudioFormat
from wav_io import read_raw, read_wav

# dBFS = "decibels relative to Full Scale". 0 dBFS is the loudest value
# int16 can hold; everything real is negative. We use 32768 (= 2**15) as
# full scale, the magnitude of int16's most negative value, so the full
# range maps to [-1.0, +1.0). That's the same convention as the float audio
# most ML/STT libraries expect.
FULL_SCALE = 32768.0

INT16_MIN, INT16_MAX = -32768, 32767

ZOOM_MS = 20  # Same as record.py's block size: "this is one block".


def _dbfs(linear: float) -> float:
    """20*log10 because amplitude (not power) ratios; log10(0) -> -inf.

    -inf is the honest answer for digital silence, so we keep it and just
    silence numpy's divide-by-zero warning instead of clamping to a fake
    number like -999.
    """
    with np.errstate(divide="ignore"):
        return float(20 * np.log10(linear))


def compute_stats(fmt: AudioFormat, samples: np.ndarray, file_bytes: int) -> dict:
    frames = samples.shape[0]
    if frames == 0:
        lo = hi = 0
        peak = rms = 0.0
    else:
        lo, hi = int(samples.min()), int(samples.max())
        # Widen before abs/square: np.abs(int16(-32768)) overflows back to
        # -32768, and squaring int16 overflows immediately. float64 is exact
        # for every int16 value.
        x = samples.astype(np.float64)
        peak = float(np.max(np.abs(x))) / FULL_SCALE
        # RMS ("root mean square") is the *average* loudness, a better
        # measure of how loud something sounds than the single peak sample.
        # A full-scale sine has RMS = peak/sqrt(2) = -3.01 dBFS.
        rms = float(np.sqrt(np.mean(x**2))) / FULL_SCALE

    data_bytes = samples.size * fmt.sample_width
    return {
        "sample_rate": fmt.sample_rate,
        "channels": fmt.channels,
        "bit_depth": fmt.bit_depth,
        "frames": frames,
        "total_samples": samples.size,  # frames * channels
        "duration_s": fmt.duration_seconds(frames),
        "min": lo,
        "max": hi,
        "peak_dbfs": _dbfs(peak),
        "rms_dbfs": _dbfs(rms),
        # Samples pinned at the int16 limits almost always mean the mic
        # input was too hot and the waveform's tops got flattened (clipping).
        # Clipping adds harsh distortion that no later stage can undo.
        "clipped": int(np.count_nonzero((samples == INT16_MIN) | (samples == INT16_MAX))),
        "data_bytes": data_bytes,
        "file_bytes": file_bytes,
        "header_bytes": file_bytes - data_bytes,
    }


def magnitude_spectrum(fmt: AudioFormat, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How much energy the recording has at each frequency, in dBFS.

    This is where sample rate differences become visible: an FFT of audio
    sampled at R Hz can only contain frequencies from 0 to R/2 (Nyquist).
    """
    if samples.shape[0] < 2:
        return np.array([]), np.array([])
    # Channel 0 only: for a mic recording, channels are near-identical, and
    # one clean line per file keeps the comparison plot readable.
    x = samples[:, 0].astype(np.float64) / FULL_SCALE
    # A Hann window tapers the start/end to zero. Without it the FFT treats
    # the abrupt cut at the edges as a real sound and smears energy across
    # all frequencies ("spectral leakage"), hiding the real shape.
    window = np.hanning(len(x))
    # rfft: input is real, so the negative-frequency half is a mirror image;
    # skip it. Bins run 0 .. rate/2 in steps of rate/N.
    spectrum = np.fft.rfft(x * window)
    freqs = np.fft.rfftfreq(len(x), d=1 / fmt.sample_rate)
    # Scale so a full-scale sine reads ~0 dBFS: x2 for the discarded
    # mirror half, divide by the window's sum to undo its attenuation.
    magnitude = np.abs(spectrum) * 2 / window.sum()
    # Floor at 1e-10 (-200 dB) instead of -inf, purely so the log-scale
    # plot has something finite to draw for empty bins.
    return freqs, 20 * np.log10(np.maximum(magnitude, 1e-10))


def zoom_range(fmt: AudioFormat, samples: np.ndarray, ms: int = ZOOM_MS) -> tuple[int, int]:
    """[start, stop) of an `ms`-long window centred on the loudest frame.

    Centred on the loudest moment rather than the middle of the file,
    because the middle of a speech recording may well be a pause -- and a
    20 ms window of silence teaches nothing.
    """
    frames = samples.shape[0]
    n = fmt.frames_per_block(ms)
    if frames <= n:  # Clamp: a file shorter than the window is shown whole.
        return 0, frames
    loudest = int(np.argmax(np.abs(samples[:, 0].astype(np.int32))))
    start = min(max(loudest - n // 2, 0), frames - n)
    return start, start + n


def zoom_window(fmt: AudioFormat, samples: np.ndarray, ms: int = ZOOM_MS) -> np.ndarray:
    start, stop = zoom_range(fmt, samples, ms)
    return samples[start:stop]


def plot(files: list[tuple[str, AudioFormat, np.ndarray]]):
    """One row per file: [full waveform | 20 ms zoom | spectrum]. Returns the Figure."""
    # Imported here, not at the top, so `--no-plot` and the tests don't pay
    # matplotlib's ~0.5 s import cost or need a display.
    import matplotlib.pyplot as plt

    rows = len(files)
    fig, axes = plt.subplots(rows, 3, figsize=(16, 3.4 * rows), squeeze=False)
    # The same x-limits on every spectrum, up to the *highest* Nyquist among
    # the files, so a lower sample rate visibly "runs out" of frequencies.
    max_nyquist = max(fmt.nyquist_hz for _, fmt, _ in files)

    for row, (name, fmt, samples) in zip(axes, files):
        ax_wave, ax_zoom, ax_spec = row
        label = f"{name}\n{fmt.sample_rate} Hz · {fmt.channels} ch · {fmt.bit_depth}-bit"

        # 1) Full waveform. The y-axis is pinned to the full int16 range on
        # every row so loudness is comparable between files and you can see
        # how much of the available range ("headroom") the recording uses.
        t = np.arange(samples.shape[0]) / fmt.sample_rate
        ax_wave.plot(t, samples, linewidth=0.5)
        ax_wave.set(title=label, xlabel="time (s)", ylabel="int16 value",
                    ylim=(INT16_MIN, INT16_MAX))

        # 2) One 20 ms block, every sample drawn as a dot. Count the dots:
        # 160 at 8 kHz, 320 at 16 kHz, 960 at 48 kHz. Same 20 ms of sound,
        # same shape -- just measured more or less often.
        start, stop = zoom_range(fmt, samples)
        tz = np.arange(start, stop) / fmt.sample_rate * 1000
        ax_zoom.plot(tz, samples[start:stop, 0], ".-", markersize=3, linewidth=0.6)
        ax_zoom.set(title=f"{ZOOM_MS} ms around the loudest point: {stop - start} samples",
                    xlabel="time (ms)", ylabel="int16 value")

        # 3) Spectrum on a log-frequency axis (that's how we hear pitch:
        # each octave is a doubling). The dashed line is Nyquist -- nothing
        # can exist to its right.
        freqs, db = magnitude_spectrum(fmt, samples)
        ax_spec.semilogx(freqs[1:], db[1:], linewidth=0.6)  # [1:] skips 0 Hz (log axis)
        ax_spec.axvline(fmt.nyquist_hz, color="red", linestyle="--",
                        label=f"Nyquist {fmt.nyquist_hz:g} Hz")
        ax_spec.set(title="Magnitude spectrum (channel 0)", xlabel="frequency (Hz)",
                    ylabel="dBFS", xlim=(20, max_nyquist), ylim=(-140, 0))
        ax_spec.legend(loc="lower left")

    fig.tight_layout()
    return fig


def _print_stats(path: Path, s: dict, fmt: AudioFormat) -> None:
    block = fmt.frames_per_block(ZOOM_MS)
    print(f"\n== {path} ==")
    print(f"  sample rate      {s['sample_rate']} Hz   (Nyquist {fmt.nyquist_hz:g} Hz)")
    print(f"  channels         {s['channels']}")
    print(f"  bit depth        {s['bit_depth']}-bit signed PCM")
    print(f"  frames           {s['frames']}   (samples per channel)")
    print(f"  total samples    {s['total_samples']}   (frames x channels)")
    print(f"  duration         {s['duration_s']:.3f} s   (frames / sample rate)")
    print(f"  min / max        {s['min']} / {s['max']}   (int16 range {INT16_MIN}..{INT16_MAX})")
    print(f"  peak             {s['peak_dbfs']:.1f} dBFS")
    print(f"  RMS              {s['rms_dbfs']:.1f} dBFS")
    print(f"  clipped samples  {s['clipped']}")
    print(f"  data bytes       {s['data_bytes']}   (frames x {fmt.bytes_per_frame} bytes/frame)")
    print(f"  file bytes       {s['file_bytes']}   (header {s['header_bytes']} bytes)")
    print(f"  bytes / second   {fmt.bytes_per_second}")
    print(f"  {ZOOM_MS} ms block      {block} frames = {block * fmt.bytes_per_frame} bytes")


def _print_comparison(rows: list[tuple[Path, dict]]) -> None:
    """When comparing files, one table makes 'what changed' obvious."""
    print("\n== comparison ==")
    print(f"  {'file':<28}{'rate':>8}{'frames':>10}{'duration':>10}{'bytes':>10}{'Nyquist':>9}")
    for path, s in rows:
        print(f"  {path.name:<28}{s['sample_rate']:>8}{s['frames']:>10}"
              f"{s['duration_s']:>9.3f}s{s['data_bytes']:>10}{s['sample_rate'] // 2:>9}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help=".wav or .raw files")
    parser.add_argument("--rate", type=int, help="sample rate for .raw files (they have no header)")
    parser.add_argument("--channels", type=int, default=1, help="channel count for .raw files")
    parser.add_argument("--no-plot", action="store_true", help="print stats only")
    args = parser.parse_args(argv)

    loaded: list[tuple[Path, AudioFormat, np.ndarray]] = []
    for path in args.paths:
        try:
            if path.suffix.lower() == ".raw":
                if args.rate is None:
                    # A .raw file is just numbers; guessing the rate would
                    # silently give wrong durations. Make the user say it.
                    print(f"error: {path} is headerless raw PCM; pass --rate "
                          "(and --channels if not mono)", file=sys.stderr)
                    return 2
                fmt = AudioFormat(args.rate, args.channels)
                samples = read_raw(path, fmt)
            else:
                fmt, samples = read_wav(path)
        except (OSError, ValueError) as e:  # missing file, bad/unsupported WAV
            print(f"error: {e}", file=sys.stderr)
            return 1
        loaded.append((path, fmt, samples))

    stats = []
    for path, fmt, samples in loaded:
        s = compute_stats(fmt, samples, path.stat().st_size)
        _print_stats(path, s, fmt)
        stats.append((path, s))
    if len(stats) > 1:
        _print_comparison(stats)

    if not args.no_plot:
        import matplotlib.pyplot as plt

        plot([(p.name, fmt, x) for p, fmt, x in loaded])
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
