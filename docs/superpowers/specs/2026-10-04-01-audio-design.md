# 01_audio — Design Spec

Date: 2026-10-04
Status: Approved in conversation; pending written-spec review

## Purpose

A first-principles learning repo for building a voice AI agent, one stage per
folder. Stage 01 makes raw audio concrete: microphone input, PCM, sample rate,
bit depth, channels, frames, buffers, duration, waveform, why realtime systems
chunk audio, and where latency begins. It deliberately mirrors the shape of a
realtime "AudioFrame" (16-bit signed PCM + sample rate + channel count +
samples-per-channel) without using any framework.

## Constraints

- No LiveKit, Pipecat, STT, LLM, or TTS.
- Every important implementation decision is explained in code comments.
- Windows 11, Python 3.14, plain `venv` + `requirements.txt` (no `uv`).
- Dependencies: `sounddevice` (PortAudio binding), `numpy`, `matplotlib`,
  `pytest`. WAV I/O uses stdlib `wave` so the file format stays transparent.

## Success criteria

1. `python 01_audio/record.py --rate R` records the mic in fixed-size blocks,
   logs each block, and saves `.wav` + `.raw` for R in {8000, 16000, 48000}.
2. `python 01_audio/inspect_audio.py f1.wav f2.wav f3.wav` prints stats for each
   file and shows a stacked comparison plot.
3. Observable differences: frame count and bytes scale linearly with rate;
   duration is constant; zoomed 20 ms window shows 160 / 320 / 960 samples;
   spectrum is cut off at Nyquist (4 / 8 / 24 kHz).
4. `pytest` passes without a microphone.

## Repository layout

```
voice-ai-agent/
├── README.md              roadmap of the 12 stages, setup, how to run
├── requirements.txt
├── pyproject.toml         pytest config only (pythonpath = 01_audio)
├── .gitignore             .venv/, recordings/, caches
├── 02_streaming/ … 12_evals/
│   └── README.md          one-line stage goal (git does not track empty dirs)
└── 01_audio/
    ├── audio_format.py
    ├── wav_io.py
    ├── record.py
    ├── inspect_audio.py
    ├── tests/
    └── README.md
```

Note: the requested `inspect.py` is named `inspect_audio.py` because a script's
own directory is first on `sys.path`, so `inspect.py` would shadow the stdlib
`inspect` module that numpy, matplotlib and `dataclasses` import.

## Components

### `audio_format.py` (pure, no I/O)

`AudioFormat` frozen dataclass: `sample_rate: int`, `channels: int`,
`sample_width: int = 2` (bytes; 2 = int16). Validates on construction.
Derived values:

- `bit_depth` → `sample_width * 8`
- `bytes_per_frame` → `sample_width * channels`
- `bytes_per_second` → `bytes_per_frame * sample_rate`
- `nyquist_hz` → `sample_rate / 2`
- `frames_per_block(block_ms)` → `sample_rate * block_ms // 1000`
- `duration_seconds(frames)` → `frames / sample_rate`

Terminology fixed here and used everywhere: a **sample** is one number for one
channel; a **frame** is one sample per channel at one instant; a **block**
(a.k.a. buffer / chunk / AudioFrame in LiveKit) is N frames delivered together.

### `wav_io.py` (file I/O only)

`write_wav(path, fmt, samples)`, `write_raw(path, samples)`,
`read_wav(path) -> (AudioFormat, ndarray)`, `read_raw(path, fmt)`.
Arrays are int16 shaped `(frames, channels)`. `read_wav` rejects non-16-bit
files. Kept separate so both scripts and the tests share one implementation of
the file format, and tests never import `sounddevice`.

### `record.py`

CLI: `--rate` (default 16000), `--channels` (1), `--seconds` (3),
`--block-ms` (20), `--device`, `--name` (default `take`), `--list-devices`.

Flow:
1. `sd.check_input_settings(...)` with dtype `int16`; on failure print a clear
   error and hint at `--list-devices`.
2. Print device name and its native `default_samplerate` (reveals OS resampling).
3. Open `sd.InputStream(samplerate, channels, dtype='int16', blocksize, callback)`.
4. Callback (PortAudio audio thread): copy `indata`, capture
   `time.inputBufferAdcTime` and overflow flag, `queue.put(...)`. Nothing else.
5. Main thread drains queue; per block logs index, frames, ms of audio, and
   capture-to-Python delay (`stream.time - adc_time`). Stops once target frames
   collected.
6. Concatenate, trim to exact `seconds * rate` frames, save
   `01_audio/recordings/<name>_<rate>hz.wav` and `.raw`; print summary.
7. Ctrl+C saves what was captured so far. Overflows counted and reported.

### `inspect_audio.py`

CLI: one or more paths; `.raw` files require `--rate` and `--channels`.

Functions (importable for tests): `compute_stats(fmt, samples, file_bytes) -> dict`,
`magnitude_spectrum(fmt, samples) -> (freqs, dbfs)`, `plot(files)`.

Stats: sample rate, channels, bit depth, frames, total samples, duration,
min/max (int16), peak dBFS, RMS dBFS, clipped samples (|x| == 32767/−32768),
data bytes, file bytes, header bytes.

Plot: one row per file, three columns — full waveform (time axis in seconds),
zoomed 20 ms window with sample markers, magnitude spectrum (dBFS vs Hz, log
frequency axis) with a vertical Nyquist line. Shared y-limits for comparison.

## Error handling

- Unsupported device settings → message + `--list-devices` hint, exit code 1.
- Input overflow → counted, reported at end (not silently ignored).
- KeyboardInterrupt during recording → save partial capture.
- Unsupported WAV sample width → message, exit code 1.

## Testing

pytest, no microphone required:
- `AudioFormat` math (16 kHz mono 20 ms → 320 frames, 640 bytes; validation).
- WAV write → read round-trip preserves format and samples.
- Stats on synthetic full-scale sine: peak ≈ 0 dBFS, RMS ≈ −3 dBFS; min/max.
- Spectrum peak lands at the sine's frequency.

Manual: short recording at 8/16/48 kHz, then inspect all three.

## Out of scope

Streaming/networking, VAD, resampling code of our own, playback, any model.
