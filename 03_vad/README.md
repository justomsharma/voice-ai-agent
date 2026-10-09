# 03 — VAD: is someone speaking right now?

In 02_streaming, frames arrived and got played. Now the program **listens**
to each frame and decides: speech, or not speech?

```
audio → 20 ms frames → loudness (RMS, dBFS) → state machine → speech segments
                                                            → timeline, plot, speech-only WAV
```

| File | Job |
|---|---|
| `energy.py` | One frame → one loudness number in dBFS |
| `vad.py` | The state machine: turns flickery per-frame guesses into clean segments |
| `detect.py` | Runs it on a WAV or the mic: prints a timeline, plots, saves the speech |

## VAD is the ear, not the brain

VAD only hears **sound**. It doesn't understand **words**.

| | VAD (this stage) | Turn detection (Stage 08) |
|---|---|---|
| Question | "Is speech sound happening right now?" | "Has the person *finished*?" |
| Looks at | Loudness / sound patterns | Words, meaning, pauses, tone |
| "I want to… [pause] …book a ticket" | Pause = silence ✅ (correct!) | Sentence unfinished, so wait ⏳ |

If the agent replied the moment VAD said "silence", it would interrupt you.
Real agents (LiveKit, Pipecat) run both: Silero VAD as the fast ear, plus a
separate turn-detector model as the brain.

## Run it

```bash
.venv/Scripts/python 03_vad/detect.py --wav 01_audio/recordings/take_16000hz.wav
.venv/Scripts/python 03_vad/detect.py --wav 01_audio/recordings/take_16000hz.wav --plot   # see it
.venv/Scripts/python 03_vad/detect.py --wav 01_audio/recordings/take_16000hz.wav --noise -35
.venv/Scripts/python 03_vad/detect.py --wav 01_audio/recordings/take_16000hz.wav --out 03_vad/recordings/speech.wav
.venv/Scripts/python 03_vad/detect.py                     # live mic, 10 s (Ctrl+C stops early)
```

Want a better test file? Record 10 s with a pause in the middle:
`.venv/Scripts/python 01_audio/record.py --seconds 10 --name pause`.

## How it works

**1. Loudness per frame.** RMS = the average energy of the frame's samples,
in dBFS (same scale as Stage 01). Measured on this machine:

| Sound | Frame loudness |
|---|---|
| Your room, nobody talking | about **-94 dBFS** |
| Your voice | **-30 to -45 dBFS** |
| Our start threshold | **-40 dBFS** |

**Why 20 ms frames?** Speech changes sound every 10–30 ms (one vowel, one
consonant), so 20 ms is roughly "one sound". Shorter frames make the RMS
jumpy, and longer ones blur a word's start into the silence before it.
WebRTC's VAD uses 10/20/30 ms for the same reason.

**2. The state machine.** Pipecat's 4 states, with Silero's two thresholds:

```
QUIET ──loud──► STARTING ──loud for min_speech──► SPEAKING ──quiet──► STOPPING
  ▲                │ (too short: a click)            ▲                    │
  └────────────────┘                                 └──loud again────────┤ (just a pause)
  ▲                                                                       │
  └──────────────────── quiet for min_silence: segment ends ──────────────┘
```

**3. The four knobs**, compared with what the pros ship:

| Knob | What it stops | Ours | Silero | LiveKit | Pipecat |
|---|---|---|---|---|---|
| start / stop threshold | Flicker when a voice hovers near one line (hysteresis) | -40 / -45 dB | 0.5 / 0.35 | 0.5 / 0.35 | 0.7 |
| min speech | Clicks and door slams counting as speech | 200 ms | 250 ms | 50 ms | 200 ms |
| min silence | A breath ending your sentence | 500 ms | 100 ms | 550 ms | 200 ms |
| padding | Quiet word edges ("s", "f", "h") getting chopped | 100 ms | 30 ms | 500 ms (before) | — |

(Their thresholds are model probabilities from 0 to 1, and ours are loudness. Same idea, different scale.)

## The experiments (measured on your 3 s `take_16000hz.wav`)

```
# 1. defaults
raw |........++.##....+++++..+#+.+.|     ← per frame: flickers ('+')
vad |..........####################|     ← 1 clean segment, 1.00 s – 3.00 s
```
The 400 ms dip at ~1.4 s (down to -71 dB) is shorter than min silence
(500 ms), so it stays one segment. That's the "pause mid-sentence" rule
working.

| # | Run | Result | Lesson |
|---|---|---|---|
| 2 | `--start-db -25 --stop-db -30` (too strict) | **0 segments**: your whole sentence lost | **False negative**: quiet speech missed |
| 3 | `--start-db -60 --stop-db -65` (too loose) | Starts at 0.58 s instead of 1.00 s: breaths and lip noise now count | **False positive** risk: anything above -60 dB counts |
| 4 | `--noise -50` (quiet fan) | Same as defaults | Noise below the threshold is harmless |
| 5 | `--noise -35` (noisy room) | **Everything is speech**: 0.00 s – 3.00 s | **Energy VAD is useless once noise is louder than the threshold** |
| 6 | `--min-speech-ms 0 --min-silence-ms 20 --stop-db -40` (no state machine) | **7 segments** in one sentence | Why the state machine exists |
| 7 | `--pad-ms 0 --out 03_vad/recordings/nopad.wav` | Segment 1.10 – 2.90 s instead of 1.00 – 3.00 s | 100 ms gone at each edge. Listen to `nopad.wav` vs a default `--out` run |

## Why energy VAD fails in noise

**Loudness is not speech.** A fan, music, traffic or a door slam can be loud,
and a whisper is quiet. One fixed threshold can't separate them:

- Raise it to ignore the noise, and quiet words disappear (**false negatives**).
- Lower it to catch quiet words, and noise becomes "speech" (**false positives**).

How real systems fix this:

| Fix | Idea |
|---|---|
| Adaptive threshold | Track the room's background level and set the threshold a few dB above it |
| Spectral features | Speech has a pitch (~85–255 Hz) and formants, and a fan doesn't |
| ML model (Silero) | A small neural net trained on thousands of hours of speech vs noise |

Even Pipecat still requires a minimum **volume** on top of Silero's verdict,
so energy checks remain in use, just not on their own.

## What this means for your agent's latency

The VAD can only say "speech ended" **after** `min_silence` of quiet has
passed. With 500 ms, the agent loses half a second before STT or the LLM
even starts. That's one slice of your 2–4 s response time. Lower it and the
agent interrupts you mid-pause. Turn detection (Stage 08) is how real
agents get out of that trade-off.

## Check yourself

1. Why can a loud noise fool an energy-based detector?
2. Why should one silent frame not end a speech segment?
3. Why doesn't silence mean the speaker has finished their turn?

If you can answer these without looking back, Stage 03 is done.

## Limitations

A learning baseline, not production VAD: a fixed threshold, no noise
tracking, no spectral features, no ML. Next step: run Silero on the same
files and compare.

## Tests

```bash
.venv/Scripts/python -m pytest 03_vad -q     # no mic needed
```
Covered: silence, tones, short bursts, pauses inside speech, hysteresis,
padding, noise, stereo / 48 kHz, empty files, and the CLI.
