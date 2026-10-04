# 01 — Audio: what a microphone actually gives you

Before a voice agent can hear anything, it receives a stream of integers.
This stage records those integers, saves them, and shows what they mean.

```
record.py         mic → 20 ms blocks of int16 PCM → .wav + .raw
inspect_audio.py  file → stats + plot (waveform | 20 ms zoom | spectrum)
audio_format.py   the arithmetic: rate × channels × bytes → size, duration
wav_io.py         read/write .wav (44-byte header + PCM) and .raw (PCM only)
```

## Concepts in 60 seconds

| Term | Meaning | Example (16 kHz mono) |
|---|---|---|
| **PCM** | Measure the mic voltage at regular intervals; store each measurement as an integer. No compression. | — |
| **Sample rate** | Measurements per second | 16,000 Hz |
| **Bit depth** | Size of each measurement. 16-bit = int16 = −32768…32767 | 16 bits = 2 bytes |
| **Channels** | Independent signals. Voice AI is almost always mono. | 1 |
| **Sample** | One integer, for one channel | 2 bytes |
| **Frame** | One sample per channel at one instant. Stereo stores frames interleaved: `L R L R …` | 2 bytes |
| **Block** (buffer, chunk) | N frames delivered together | 20 ms = 320 frames = 640 bytes |
| **Nyquist** | Highest frequency a rate can hold: rate ÷ 2 | 8 kHz |
| **dBFS** | Loudness relative to the int16 maximum. 0 = loudest possible; everything real is negative. | speech ≈ −30 to −10 |

The arithmetic you'll use forever:

```
bytes/second  = sample_rate × channels × bytes_per_sample      16000 × 1 × 2 = 32,000
duration      = frames ÷ sample_rate                           48000 ÷ 16000 = 3.0 s
frames/block  = sample_rate × block_ms ÷ 1000                  16000 × 20 ÷ 1000 = 320
```

### So what's an "AudioFrame"?

In realtime frameworks, an `AudioFrame` is **one block**, not one frame in
the sense above. LiveKit's
[`AudioFrame`](https://docs.livekit.io/reference/python/livekit/rtc/audio_frame.html)
is exactly what `record.py` produces on every callback:

```
data                 int16 PCM, interleaved by channel   ← indata.copy()
sample_rate          16000                              ← --rate
num_channels         1                                  ← --channels
samples_per_channel  320                                ← blocksize (20 ms)
```

The rate and channel count must travel *with* the bytes, because raw PCM
can't describe itself. Open a `.raw` file and nothing tells you whether it's
8 kHz mono or 48 kHz stereo. That's the difference between our `.raw` and
`.wav` files: the WAV has 44 bytes in front that record the format.

## Run it

```bash
.venv/Scripts/python 01_audio/record.py --list-devices      # find your mic
.venv/Scripts/python 01_audio/record.py                     # 3 s, 16 kHz, mono
.venv/Scripts/python 01_audio/inspect_audio.py 01_audio/recordings/take_16000hz.wav
```

While recording you'll see one line per block:

```
  block  frames      ms   since prev   peak
      3     320    20.0      41.8 ms   4210 ######
      4     320    20.0       0.1 ms   5120 ###
```

## The experiment: 8 kHz → 16 kHz → 48 kHz

Say the **same phrase** each time. "Sixty-six sisters see the sea" works
well, because "s" sounds carry energy at 4–10 kHz.

```bash
.venv/Scripts/python 01_audio/record.py --rate 8000
.venv/Scripts/python 01_audio/record.py --rate 16000
.venv/Scripts/python 01_audio/record.py --rate 48000
.venv/Scripts/python 01_audio/inspect_audio.py 01_audio/recordings/take_8000hz.wav 01_audio/recordings/take_16000hz.wav 01_audio/recordings/take_48000hz.wav
```

### What changes (3 s, mono)

| | 8 kHz | 16 kHz | 48 kHz |
|---|---|---|---|
| frames | 24,000 | 48,000 | 144,000 |
| data bytes | 48,000 | 96,000 | 288,000 |
| bytes / second | 16,000 | 32,000 | 96,000 |
| frames per 20 ms block | 160 | 320 | 960 |
| Nyquist (highest frequency) | 4 kHz | 8 kHz | 24 kHz |
| sounds like | telephone, muffled "s" | clear speech (STT standard) | full fidelity |

### What does *not* change

- **Duration.** 3 s is 3 s. Rate changes how often you measure, not how long.
- **Waveform shape.** The 20 ms zoom panel shows the same wave drawn with
  160, 320 or 960 dots.
- **Amplitude / loudness.** That's bit depth's job, not sample rate's.
- **The latency floor.** A 20 ms block is 20 ms at any rate, just with more
  samples in it.

### What to look at in the plot

1. **Zoom panel:** count the dots. More samples per millisecond, same curve.
2. **Spectrum panel:** each curve stops dead at its red Nyquist line. At
   8 kHz everything above 4 kHz is gone, and that's where the "s" lives.
3. **Waveform panel:** the y-axis is the full int16 range. Speech that only
   reaches ±3000 is using about 10% of the range, i.e. a lot of headroom.
   Samples pinned at the edges (`clipped samples > 0`) mean the input was
   too loud and is now distorted for good.

## Two things you'll notice on Windows

**Blocks arrive in bursts.** Through MME (sounddevice's default host API),
the "since prev" column alternates between ~0 ms and ~40 ms. The driver
hands over two 20 ms blocks at once. Audio *production* is steady, but
*delivery* jitters. This is why realtime systems put buffers between stages.

**Your mic probably doesn't run at 8 kHz at all.** Run `--list-devices`.
The same mic appears under several host APIs:

- **MME** says 44100 Hz for everything, and silently resamples to whatever
  you ask for.
- **WASAPI** reports the true hardware rate (usually 48000 Hz) and *refuses*
  other rates (`Invalid sample rate`).

So an "8 kHz recording" is really 48 kHz audio that the OS low-pass filtered
and decimated. The filter is why you won't see aliasing in the spectrum.

## Why realtime systems chunk audio, and where latency begins

You can't process a sound that hasn't finished arriving. If the agent waited
for the whole utterance, it couldn't react until you stopped talking.
Instead, audio moves in small fixed blocks, and every stage (VAD, STT, …)
works on each block as it arrives.

The block size is a trade-off:

- **Smaller blocks** mean lower latency, but more per-block overhead (calls,
  thread hops, network packets).
- **Larger blocks** are more efficient, but every block adds its own
  duration to the delay.

10–30 ms is the industry sweet spot.

Latency begins before any of our code runs:

```
sound hits mic ─┬─ ≥ block_ms        a block is only delivered once it's full (20 ms)
                ├─ + driver buffer    printed at start: "driver-reported input buffering"
                └─ + delivery jitter  the bursts in "since prev"
                      → our callback finally sees the block
```

`record.py` prints this floor when it starts. On the test machine it was
20 ms + 40 ms. Every later stage of the agent adds to it; nothing can
subtract from it.

## Design notes

- **Callback + queue, not `sd.rec()`.** The callback runs on PortAudio's
  realtime thread and only copies the block into a queue. Printing or
  writing files there would cause dropped audio ("input overflow", which
  `record.py` counts and reports).
- **`indata.copy()`.** PortAudio reuses its buffer for the next block, so
  storing the view without copying would corrupt every block.
- **int16 end to end.** No float conversion hides what PCM is.
- **stdlib `wave`.** Keeps the file format transparent. `.wav` size is
  exactly `.raw` size + 44.
- **No mic timestamps.** PortAudio's `inputBufferAdcTime` returned garbage
  on Windows (0, or a value on an unrelated clock), so the recorder doesn't
  show it. Measure latency end to end instead.
- **The file is `inspect_audio.py`, not `inspect.py`.** Python puts a
  script's folder first on `sys.path`, so `inspect.py` would replace the
  stdlib `inspect` module that numpy and matplotlib import, and crash them.

## Tests

```bash
.venv/Scripts/python -m pytest      # synthetic sine waves; no mic needed
```

## Exercises

1. Record with `--channels 2` and check that `bytes/s` doubles while
   `duration` doesn't.
2. Try `--block-ms 10` and `--block-ms 100` and watch the "since prev" column.
3. Open `take_16000hz.raw` with `--rate 8000` and see what wrong metadata
   does to duration (the bytes are the same; the meaning isn't).
4. Whisper, then shout. Compare RMS dBFS and look for clipped samples.
