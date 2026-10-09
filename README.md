# Voice AI Agent — from first principles

Building a realtime voice agent one layer at a time, starting from raw
microphone samples. Each stage is a small, runnable folder that teaches one
idea before the next stage builds on it.

**Rule:** no voice frameworks (LiveKit, Pipecat, …). Every layer is built by
hand first so it's clear what those frameworks do for you.

| Stage | Folder | What it teaches | Status |
|---|---|---|---|
| 01 | [01_audio](01_audio/) | PCM, sample rate, bit depth, frames, blocks, latency floor | ✅ |
| 02 | [02_streaming](02_streaming/) | UDP packets, delay, jitter, loss, jitter buffers | ✅ |
| 03 | [03_vad](03_vad/) | Voice activity detection: RMS energy, thresholds, hysteresis, VAD vs turn detection | ✅ |
| 04 | [04_stt](04_stt/) | Speech-to-text | — |
| 05 | [05_llm](05_llm/) | Language model reply | — |
| 06 | [06_tts](06_tts/) | Text-to-speech | — |
| 07 | [07_realtime_pipeline](07_realtime_pipeline/) | mic → VAD → STT → LLM → TTS → speaker | — |
| 08 | [08_turn_taking](08_turn_taking/) | End-of-turn detection, barge-in | — |
| 09 | [09_tools](09_tools/) | Tool / function calling | — |
| 10 | [10_guardrails](10_guardrails/) | Safety and staying on-task | — |
| 11 | [11_observability](11_observability/) | Per-stage latency, logs, traces | — |
| 12 | [12_evals](12_evals/) | Measuring quality | — |

## Setup (Windows, Python 3.11+)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
```

`sounddevice` ships its own PortAudio library on Windows and macOS, so
there's nothing else to install. On Linux, run `apt install libportaudio2`.

## Run

```bash
.venv/Scripts/python 01_audio/record.py --rate 16000
.venv/Scripts/python 01_audio/inspect_audio.py 01_audio/recordings/take_16000hz.wav
.venv/Scripts/python -m pytest          # no microphone needed

# 02: two terminals
.venv/Scripts/python 02_streaming/receiver.py
.venv/Scripts/python 02_streaming/sender.py --wav 01_audio/recordings/take_16000hz.wav --jitter-ms 40 --delay-ms 80

# 03: find the speech
.venv/Scripts/python 03_vad/detect.py --wav 01_audio/recordings/take_16000hz.wav --plot
```

Start with [01_audio/README.md](01_audio/README.md), then [02_streaming/README.md](02_streaming/README.md), then [03_vad/README.md](03_vad/README.md).
