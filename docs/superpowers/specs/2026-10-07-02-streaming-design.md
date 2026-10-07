# 02_streaming — Design Spec

Date: 2026-10-07
Status: Approved in conversation

## Purpose

Show how audio moves *continuously* from one place to another: the mic is
cut into small frames, each frame becomes a UDP packet, the packets cross a
(simulated) bad network, and a receiver rebuilds the stream and plays it.
The learner should hear and measure delay, packet loss and jitter, and see
why a jitter buffer fixes choppiness by adding latency.

## Constraints

- No LiveKit, Pipecat, Twilio, STT, LLM, TTS. Python + sounddevice + numpy.
- Localhost only. Two processes: `receiver.py` and `sender.py`.
- Same style as 01_audio: teaching comments on every "why", int16 PCM,
  pure logic separated from I/O so it can be tested without a mic or socket.
- Reuses `01_audio/audio_format.py` and `01_audio/wav_io.py`.

## Success criteria

1. `receiver.py` then `sender.py` streams the mic (or a `--wav` file) and the
   audio plays on the speaker.
2. Sender flags `--frame-ms`, `--delay-ms`, `--jitter-ms`, `--loss`, `--seed`
   change what the receiver hears and reports.
3. Receiver flag `--buffer-ms` sets the jitter buffer; the printed stats show
   the trade-off: small buffer → many late frames, big buffer → higher delay.
4. Receiver saves what it played to `02_streaming/recordings/received.wav`.
5. `pytest` passes without mic, speaker or sockets.

## Layout

```
02_streaming/
├── packet.py         header pack/unpack
├── netsim.py         NetSim (drop/delay decisions) + DelayLine (timed sender)
├── jitter_buffer.py  JitterBuffer (reorder, pre-roll, fixed playout)
├── sender.py         mic | --wav → frames → NetSim → UDP
├── receiver.py       UDP → JitterBuffer → speaker, stats, received.wav
├── tests/            test_packet.py, test_netsim.py, test_jitter_buffer.py
└── README.md
```

## Components

### `packet.py`
Header, network byte order, 18 bytes: `seq u32 | send_time_ns u64 |
sample_rate u32 | channels u16`, then int16 PCM. Like RTP: the sequence number
reveals loss and reordering; the timestamp lets the receiver measure delay;
rate/channels travel with the bytes (the 01_audio lesson).
`encode(Packet) -> bytes`, `decode(bytes) -> Packet`; `decode` rejects data
shorter than the header with `ValueError`.

### `netsim.py`
`NetSim(delay_ms, jitter_ms, loss, seed)`. `decide() -> float | None`:
`None` = dropped (probability `loss`), otherwise delay in seconds =
`delay_ms + uniform(-jitter_ms, +jitter_ms)`, clamped at 0. Because each
packet gets its own delay, jitter larger than a frame reorders packets — as
on a real network.
`DelayLine(send_fn)`: one background thread + a heap ordered by due time;
`put(delay_s, data)` schedules, `close()` waits until everything is sent.
The impairment lives on the send side; to the receiver it is
indistinguishable from a bad network in the middle.

### `jitter_buffer.py`
`JitterBuffer(target_frames)`. Pure, no threads (receiver wraps it in a lock).
- `push(seq, payload, send_time_ns) -> "ok" | "late" | "duplicate"`.
- Pre-roll: playback starts once `target_frames` packets are held
  (0 → on the first packet). `next_seq` starts at the lowest held seq.
- `pop() -> (payload, send_time_ns) | None`, called once per playout tick.
  Before start → `None` (pre-roll silence). After start the playout clock
  always advances: present → returned; missing → `None` and the seq is
  recorded as missed. A packet arriving for a seq already passed is "late"
  and discarded.
- Counters: received, late, duplicate, reordered (arrived with seq lower than
  one already seen), played, `missed_count()` (ignores missed seqs above the
  highest received, i.e. the end of the stream).

### `sender.py`
Flags: `--host 127.0.0.1 --port 5005 --frame-ms 20 --delay-ms 0 --jitter-ms 0
--loss 0 --seed`, input `--wav PATH` or mic (`--rate 16000 --channels 1
--seconds 10 --device`). Mic: InputStream callback → queue → main thread.
WAV: frames paced on an absolute clock (`start + i * frame_s`) so timing
doesn't drift. Each frame → `Packet` → `NetSim.decide()` → `DelayLine`.
Prints one stats line per second (sent, dropped). Waits for the DelayLine to
drain before exiting.

### `receiver.py`
Flags: `--host --port --buffer-ms 60 --out`. A socket thread decodes packets,
records network delay (`arrival - send_time`, same machine clock), pushes into
the JitterBuffer. The first packet fixes the format and frame size; an
`sd.OutputStream` with `blocksize = frames per packet` then pulls one frame per
callback (the sound card is the playout clock); missing → zeros. Each played
frame records its end-to-end delay (`play_time - send_time`). One stats line
per second: received, missed, late, reordered, buffer depth, avg network delay,
avg delay at playout. Stops 1.5 s after the last packet (or Ctrl+C), prints a
summary, writes `received.wav` (every slot from first to highest seq, missing
slots as silence).

## Error handling

- Port in use / bad device → clear message, exit 1.
- Malformed packet → counted and ignored.
- Format change mid-stream → ignored with a warning (one stream per run).

## Testing

pytest, no mic/sockets: packet round-trip and short-packet rejection; NetSim
loss rate ≈ `loss` over many draws, delays within `[delay-jitter,
delay+jitter]`, same seed → same decisions; JitterBuffer pre-roll, in-order
play, reorder fix-up, gap → `None` + missed, late packet rejected, duplicate,
end-of-stream misses not counted.

Manual: four README experiments (clean, delay, loss, jitter × buffer size).

## Out of scope

Codecs (Opus), adaptive jitter buffers, real packet-loss concealment,
clock-drift correction, NAT/real network, echo cancellation (use headphones
or `--wav` for mic loops).
