# 02 — Streaming: how audio moves, and why buffering costs latency

In 01_audio, the mic's 20 ms blocks went into a list. Now each block becomes
a **packet**, crosses a network, and has to be rebuilt into smooth audio on
the other side, live.

```
SENDER (terminal 2)                              RECEIVER (terminal 1)
mic or .wav                                      UDP socket
  → 20 ms frame (320 samples, 640 bytes)           → jitter buffer: put in order, hold --buffer-ms
  → packet: [seq | send time | rate | ch | PCM]    → speaker pulls 1 frame every 20 ms
  → fake bad network: delay, jitter, loss          → missing frame? play silence
  → UDP  ─────────── 127.0.0.1:5005 ─────────────→ → stats + recordings/received.wav
```

| File | Job |
|---|---|
| `packet.py` | 18-byte header + PCM. A simplified RTP, the format real calls use |
| `netsim.py` | The bad network: drops packets, delays each one, and the random delays reorder them |
| `jitter_buffer.py` | Puts packets back in order, waits before starting, then plays one frame per tick no matter what |
| `sender.py` | mic / wav → packets → bad network → UDP |
| `receiver.py` | UDP → jitter buffer → speaker, and prints what went wrong |

## Four words

| Word | What happens | What you hear |
|---|---|---|
| **Delay** | Every packet takes a while to arrive | Everything is late, but clean |
| **Packet loss** | Some packets never arrive | Clicks and gaps |
| **Jitter** | Each packet takes a *different* time, so they arrive bunched up or out of order | Choppy, robotic audio |
| **Jitter buffer** | The receiver waits a bit before playing, so late packets can catch up | Smooth, but later |

## Run it (two terminals)

```bash
# terminal 1: the listener, start it first
.venv/Scripts/python 02_streaming/receiver.py

# terminal 2: the talker (replays your 01_audio recording)
.venv/Scripts/python 02_streaming/sender.py --wav 01_audio/recordings/take_16000hz.wav
```

The receiver prints one line per second, then a summary when the stream ends:

```
      t   recv  late  missed  reord   in buffer         net delay   play delay
    1.0     48     0       0      0     2 fr (  40 ms)       1.1 ms      56.0 ms
```

- **net delay**: from when the sender stamped the packet to when it arrived.
- **play delay**: from stamp until the frame was handed to the speaker,
  i.e. the network time plus the time it waited in the jitter buffer.
- **late**: the packet arrived after its moment had passed, so it was discarded.
- **missed**: slots that were played as silence (lost packets + late ones).

Use the live mic by leaving out `--wav` (10 s by default). **Wear headphones.**
Otherwise the speaker feeds back into the mic.

| Sender flag | Default | Meaning |
|---|---|---|
| `--frame-ms` | 20 | Audio per packet |
| `--delay-ms` | 0 | Network delay |
| `--jitter-ms` | 0 | Each packet's delay varies by ± this much |
| `--loss` | 0 | Fraction dropped (0.05 = 5%) |
| `--seed` | random | Same seed = the same bad network every run |

Receiver: `--buffer-ms` (default 60).

## The experiments (measured on the test machine, 3 s recording)

```bash
# 1. clean network
sender.py --wav take_16000hz.wav
# 2. far away
sender.py --wav take_16000hz.wav --delay-ms 200
# 3. lossy
sender.py --wav take_16000hz.wav --loss 0.05 --seed 1
# 4. jittery: run the SAME network against different buffers
receiver.py --buffer-ms 0     then   sender.py --wav take_16000hz.wav --delay-ms 80 --jitter-ms 40 --seed 1
receiver.py --buffer-ms 40    then   (same sender command)
receiver.py --buffer-ms 80    then   (same sender command)
receiver.py --buffer-ms 200   then   (same sender command)
```

| # | Network | Buffer | Played as silence | Delay at playout | What you hear |
|---|---|---|---|---|---|
| 1 | clean | 60 ms | 0% | ~60 ms | perfect |
| 2 | 200 ms delay | 60 ms | 0% | ~245 ms | perfect, just late |
| 3 | 5% loss | 60 ms | 8% (12 packets)* | ~56 ms | clicks and holes |
| 4a | 80 ± 40 ms | **0** | **76–92%** | ~45–60 ms | broken, barely recognisable |
| 4b | 80 ± 40 ms | 40 | 1–15 frames | ~110–150 ms | mostly fine, some clicks |
| 4c | 80 ± 40 ms | 80 | 0% | ~158 ms | perfect |
| 4d | 80 ± 40 ms | 200 | 0% | ~255 ms | perfect, but 100 ms later than it needed to be |

\* Loss is random. Seed 1 happens to drop 12 of 150 packets, more than the
5% average. Try other seeds.

**Read row 4 top to bottom. It's the whole lesson.** The network is
identical in all four runs. With no buffer, 9 out of 10 frames are thrown
away, because they arrived after their moment. The packets weren't lost;
they were *late*. 80 ms of buffer fixes everything. 200 ms fixes nothing
more and just adds delay.

Why 80? The network delay ranged from 41 to 120 ms, a spread of about 80 ms.
The buffer has to cover that spread. **The right buffer size = how much the
delay varies, not how big it is.** Row 2 has a huge delay and no variation,
and needs no extra buffer.

Why does 4b vary between runs? A fixed buffer's cushion depends on how
early or late the packets it started with happened to be. Real systems
(WebRTC's NetEQ) use an **adaptive** buffer that keeps measuring jitter
and resizes itself, so it doesn't depend on that luck.

Listen to the results: each run saves what was played. Compare them with
01_audio's tool, which shows the silent holes in the waveform:

```bash
.venv/Scripts/python 01_audio/inspect_audio.py 02_streaming/recordings/received.wav
```

## What this means for a voice agent's latency

Add up the "transport" part of one direction, from your mouth to the agent's ear:

```
mic block (20 ms) + mic driver (~40 ms)          ~60 ms   (01_audio)
network                                    10-100 ms   (internet, Wi-Fi)
jitter buffer                               40-80 ms   (this stage)
--------------------------------------------------------
                                          ~100-250 ms  before any AI runs
```

The agent's reply comes back through the same pipe, including the speaker
(about 100 ms on this machine, printed in the summary). So **0.2–0.5 s of a
voice agent's delay is pure transport and buffering**. Phone-quality
conversation aims for under 150 ms one way (ITU-T G.114).

Your 2–4 s is mostly *not* this. The rest is waiting to decide you've
stopped talking, then speech-to-text, the LLM, and text-to-speech. Later
stages measure each one. But this stage shows the pattern behind all of
them: **every buffer adds latency.** That includes any buffer that waits
for "enough" of something before passing it on, such as a whole sentence
before TTS or a whole audio clip before STT.

## Why UDP, not TCP

TCP guarantees every byte arrives in order. When a packet is lost it
re-sends it and **holds back everything after it** until the gap is filled
("head-of-line blocking"). That's perfect for downloading a file, and
terrible for live audio: one lost packet freezes the stream for a full
round-trip. UDP just sends; the receiver decides what to do about gaps
(here: play silence and move on). That's why RTP/WebRTC audio runs over
UDP.

(Many voice-agent platforms do stream audio over WebSockets, which run
over TCP. It works on good networks, and it's exactly why they stutter on
bad ones.)

## Design notes

- **The format travels in every packet** (rate, channels), so the receiver
  needs no flags to match the sender. Real RTP sends a 1-byte "payload type"
  instead, and agrees the format during call setup.
- **The sound card drives playout.** The speaker's callback pulls one frame
  every 20 ms. It's the steadiest clock in the system and the one that
  actually consumes audio.
- **Speaker warm-up.** On Windows the speaker asks for its first frame about
  55–80 ms after it opens. Packets arriving before that would pile up and
  quietly make the buffer bigger than `--buffer-ms`, so the receiver drops
  them and reports the count. Real apps have the speaker running before the
  call's audio starts.
- **`latency="low"`** for the speaker: the OS's own output buffer is ~100 ms
  instead of ~200 ms. Yet another buffer, yet more latency.
- **Delay is measured with one clock.** Sender and receiver are on the same
  machine, so `arrival − send_time` is meaningful. Across two machines it
  isn't (their clocks disagree by more than the delay). Real systems
  estimate jitter from the *differences* between arrival gaps (RFC 3550).
- **Bad network on the send side.** That's simpler than a relay process in
  the middle, and the receiver can't tell the difference.

## Tests

```bash
.venv/Scripts/python -m pytest      # no mic, speaker or network needed
```

## Exercises

1. `--frame-ms 10` vs `--frame-ms 60`: watch packets/s and kbit/s in the
   sender output. Smaller frames mean lower latency but more packets and
   more header overhead (18 B on a 640 B frame is 2.7%; on a 10 ms frame, 5.3%).
2. Find the smallest `--buffer-ms` that gives 0% silence for
   `--jitter-ms 20`, then for `--jitter-ms 60`.
3. `--loss 0.2`: at what loss rate does speech stop being understandable?
4. Stream `take_48000hz.wav`: same frames per second, 3× the bytes.
5. Start the sender without a receiver. Nothing breaks: UDP doesn't know
   or care whether anyone is listening.
