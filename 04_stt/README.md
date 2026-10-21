# 04 — STT: turning speech into words

Stage 03 found **when** someone speaks. Now we find out **what** they said.
We do it without hiding the steps in between: the audio format, the bytes
on the wire, the time each part takes, and how wrong the words are.

```
WAV / mic ──► prepare ──────────────► STT engine ──────────────► transcript
              mono, resample,         HTTPS (batch)               partial text (may change)
              WAV header or bare      WebSocket (streaming)       final text (committed)
              PCM                                                 + latency, + WER
```

| File | Job |
|---|---|
| `audio_prep.py` | The **audio contract**: mono, resampling (properly), WAV bytes, sanity checks |
| `stt.py` | The provider **interface**, API keys, one timed HTTPS POST, error kinds, retries |
| `openai_stt.py` | OpenAI `gpt-transcribe` (batch): multipart upload, built by hand |
| `deepgram_stt.py` | Deepgram `nova-3` (batch): raw-bytes upload. Same interface, different contract |
| `wer.py` | **Word error rate**: align, count S/D/I, plus CER |
| `transcribe.py` | **Step A**: one WAV → one request → transcript + latency + WER |
| `realtime.py` | WebSocket session for `gpt-live-transcribe`, plus the partial/final board |
| `live.py` | **Step B**: mic or WAV → Stage 03 VAD → stream → partial and final text |
| `evaluate.py` | **Step C**: a folder of recordings → WER and latency per provider |

## Setup: your API key

```bash
copy .env.example .env          # macOS/Linux: cp .env.example .env
# open .env and paste: OPENAI_API_KEY=sk-...
.venv/Scripts/python -m pip install -r requirements.txt   # adds `websockets`
```

`.env` is git-ignored. Keys are read when a request is made and are never
printed (even `--dry-run` shows `$OPENAI_API_KEY`, not the key). Deepgram
is optional: add `DEEPGRAM_API_KEY` only if you want to compare.

## Run it

```bash
# Step A: batch. Start with --dry-run: no key, no cost, shows the exact request.
.venv/Scripts/python 04_stt/transcribe.py 01_audio/recordings/take_16000hz.wav --dry-run
.venv/Scripts/python 04_stt/transcribe.py 01_audio/recordings/take_16000hz.wav --ref "what you said"
.venv/Scripts/python 04_stt/transcribe.py 01_audio/recordings/pause_16000hz.wav --vad
.venv/Scripts/python 04_stt/transcribe.py X.wav -l hi -l en                 # Hinglish hint
.venv/Scripts/python 04_stt/transcribe.py X.wav -k Pune -k IRCTC            # names it should know

# Step B: live
.venv/Scripts/python 04_stt/live.py --dry-run                               # the protocol, no key
.venv/Scripts/python 04_stt/live.py --wav 01_audio/recordings/pause_16000hz.wav
.venv/Scripts/python 04_stt/live.py                                         # mic, 15 s

# Step C: evaluate (see "Build your test set" below)
.venv/Scripts/python 04_stt/evaluate.py 04_stt/eval -v

# Any two texts
.venv/Scripts/python 04_stt/wer.py "I want to book a ticket to Pune" "i want to book the ticket in pune please"
```

## 1. The audio contract

Every engine documents what it accepts. Read that page before sending a single byte.

| | OpenAI `gpt-transcribe` | Deepgram `nova-3` | OpenAI `gpt-live-transcribe` |
|---|---|---|---|
| Transport | HTTPS POST, multipart form | HTTPS POST, the WAV *is* the body | WebSocket, JSON events |
| Audio | a file: wav/mp3/m4a/webm…, ≤ 25 MB | a file | bare PCM16, mono, **24 kHz**, base64 |
| Options | form fields (`model`, `languages[]`, `prompt`, `keywords[]`) | URL query (`model`, `language`, `keyterm`) | `session.update` event |
| Auth | `Bearer <key>` | `Token <key>` | `Bearer <key>` in the handshake |
| We send | 16 kHz mono WAV | 16 kHz mono WAV | 24 kHz mono PCM |

Why 16 kHz for batch? Speech models listen at 16 kHz internally. 48 kHz
stereo is 6× the bytes and gives the model nothing extra. Run `--dry-run`
on `take_48000hz.wav`: it's downmixed and resampled to 96,044 bytes.

**Resample, never relabel.** Changing the header from 48000 to 16000
without changing the samples doesn't convert anything. The audio becomes
3× slower and much deeper:

```
transcribe.py 01_audio/recordings/take_48000hz.wav --relabel-rate 16000 --dry-run
RELABEL:  pretending it's 16000 Hz without resampling: it now 'lasts' 9.00 s and sounds wrong
```

Real resampling computes new samples. Going **down**, it must first remove
everything above the new Nyquist frequency (8 kHz at 16 kHz). Otherwise
those frequencies fold back as fake tones. The tests show it:
taking every 3rd sample turns a 10 kHz tone into a full-volume 6 kHz
whistle, and `resample()` turns it into near-silence (−40 dB or more). Our
resampler is a Kaiser-windowed sinc, which is how libsamplerate and soxr
work. The polyphase trick takes it from 7 s to 0.27 s for 10 s of audio.

Before paying for a request, we check:

| Check | Why |
|---|---|
| empty, or < 100 ms | nothing to transcribe. Refused |
| > 25 MB (OpenAI) | the API would reject it. Refused, with a hint to split or use `--vad` |
| peak < −60 dBFS | **warning**: models can *hallucinate* words from silence ("Thank you for watching") |
| > 0.1 % clipped samples | **warning**: distortion costs accuracy |

## 2. Batch vs streaming

```
Batch (Step A)
you speak  ████████████ 3 s
                        upload ─► model ─► text          user waits 3 s + request time
Streaming (Step B)
you speak  ████████████
socket     ▲▲▲▲▲▲▲▲▲▲▲▲ 20 ms chunks while you talk
text          "I want" "I want to book" … ─► final "I want to book a ticket."
                                          ▲ min silence + finalising, a few hundred ms
```

| | Batch | Streaming |
|---|---|---|
| When text arrives | after you finish **and** the request ends | while you speak |
| Connection | new HTTPS request per utterance (TCP + TLS each time) | one socket, opened once |
| Complexity | one function call | threads, partial vs final, turns, reconnects |
| Accuracy | hears the whole sentence at once | must guess early; `--delay` trades speed for accuracy |
| Good for | voicemail, uploads, evaluation | voice agents |

`transcribe.py` breaks a batch request down step by step:

```
Request:  connect 180 ms | upload 2 ms | wait 640 ms | download 0 ms | total 822 ms
          RTF 0.27: 3.00 s of speech took 0.82 s. Batch user waits 3.82 s from first word to text
```

(Example numbers. Run it to get yours.)

| Number | What it is |
|---|---|
| connect | DNS + TCP + TLS handshake. Paid on every new connection, and it grows with distance to the server |
| upload | handing the body to the OS. Grows with file size |
| wait | until the first response byte: upload finishing + queue + **the model working** |
| download | reading the JSON. Tiny |
| RTF | real-time factor = request time ÷ audio time |

## 3. Partial vs final

```
[  0.84s] turn 1 partial  "I want..."
[  1.31s] turn 1 partial  "I want to book a..."
[  2.42s] turn 1 COMMIT   (VAD: speech 0.52-1.98 s ended)
[  2.71s] turn 1 FINAL    "I want to book a ticket."
```

A **partial** is a guess, and the model may still revise it. A **final**
is committed. The board (`realtime.Board`) keeps them apart per turn.
The rule for Stage 05: **only send finals to the LLM**, or it will start
answering a sentence that's still changing. Finals from different turns
can arrive out of order, so they're matched by `item_id`, not by arrival
order.

## 4. Where Stage 03's VAD plugs in

`gpt-live-transcribe` has **no server-side VAD**. The client must say "this
turn is over" (`commit`). So our energy VAD gets a real job:

| VAD state | Gate does | Why |
|---|---|---|
| QUIET | keep the last 300 ms in a ring buffer, send nothing | don't pay to stream silence |
| → STARTING | send the 300 ms **pre-roll** + this frame | the quiet start of the first word ("h"ello) is in the pre-roll, so it isn't cut off |
| STARTING → QUIET | `input_audio_buffer.clear` | it was a click, so drop it server-side. The ring keeps rolling, so if real speech follows right away, its onset is re-sent |
| SPEAKING / STOPPING | keep sending | short pauses stay inside the turn |
| segment ends | `input_audio_buffer.commit` | the server finalises the turn |

The summary table splits the latency apart:

```
turn  speech (audio)   first text   final after   final after    text
                        (after 1st   commit        you stopped
                         audio sent)               (incl. VAD)
   1   0.52- 1.98 s     +0.35 s      +0.29 s       +0.79 s       "I want to book a ticket."
```

"Final after you stopped" = **VAD min silence (500 ms)** + the model
finalising. Lower `--min-silence-ms` and finals come sooner, but a
mid-sentence pause gets committed as a finished turn. That's Stage 03's
trade-off again, and Stage 08's turn detection is the real fix.

Batch too: `transcribe.py --vad` sends only the speech segments. On
`pause_16000hz.wav` it uploads **2.12 s instead of 10 s (79% less)**.
Check with `--ref` that it didn't cost words (as deletions).

## 5. Word error rate

```
REF: i want to book a   ticket to pune ******
HYP: i want to book the ticket in pune please
                    S          S       I
WER 37.5%  (3 errors / 8 words: S=2 D=0 I=1)
```

**WER = (S + D + I) / reference words.** It can go above 100% when a
model invents words. Text is normalised first (lowercase, no punctuation,
Unicode-safe for Devanagari), so "Ticket." = "ticket". Not normalised:
"10" vs "ten", "e-mail" vs "email". Keep your references written the way
the model writes, or those differences count as errors.

**CER** (characters) is gentler on spelling variants in the same script:
"kya" vs "kia" is 1 error in 3 letters instead of a whole wrong word. It
**can't** bridge scripts: "kya" vs "क्या" share no letters (~100% CER).
Hinglish may come back romanised or in Devanagari, so write your
reference in the script the model actually outputs. (Also, "क्या" is 4
Unicode code points, so Hindi CER isn't comparable to English CER.)

**Corpus WER** sums the errors over all files, *then* divides. Averaging
per-file WERs would let a 2-word clip count as much as a 200-word one.

## 6. When things fail

| Failure | What happens |
|---|---|
| No key | `[auth] OPENAI_API_KEY is not set. Copy .env.example to .env…` before anything is sent |
| Wrong key (401) | `[auth HTTP 401] key rejected…`, not retried |
| Busy (429 / 5xx) | retried 2× with backoff (0.5 s, 1 s), honouring `Retry-After` (max 10 s) |
| Out of credit (429 `insufficient_quota`) | **not** retried: waiting won't fix it |
| Timeout / network down | `[timeout]` / `[network]`, retried |
| Unsupported WAV (float, 24-bit) | refused before sending |
| Socket drops mid-stream | error line, then the summary with whatever text arrived |
| Final never arrives | waits `--final-timeout` (10 s), then `NO FINAL (late or lost)` and exit code 1 |
| evaluate: auth / quota error | stops calling that provider instead of failing 50 more times |
| evaluate: any failed file | counted as every word missed, so a provider can't look better by failing on hard files |

## Why OpenAI (and Deepgram second)?

Checked against the official docs on 2026-10-21:

- **OpenAI** is the primary. One key covers batch (`gpt-transcribe`, the
  recommended model; `whisper-1` and `gpt-4o-*-transcribe` shut down on
  2027-02-26) and live (`gpt-live-transcribe`). Its live model leaves turn
  detection to the client, so it **uses our Stage 03 VAD**. It also needs
  24 kHz, so the resampling lesson is a real step, not an exercise.
- **Deepgram `nova-3`** is the second batch adapter. It proves the interface
  works for two providers, and its contract (raw body, query options) is a
  useful contrast. Step C compares them on *your* audio.
- **Later, worth benchmarking:** Deepgram Flux (streaming with turn
  detection built in, which overlaps Stage 08), Sarvam (Indian languages and
  Hinglish), AssemblyAI, and local Whisper (no network, no per-minute cost).

Adding a provider means one file with `contract`, `transcribe()` and
`describe_request()`, plus one line in `stt.make_provider`.

## Build your test set (Step C)

Record each clip with Stage 01, then copy it in with a category prefix and
write down **exactly** what you said:

```bash
mkdir 04_stt\eval
.venv/Scripts/python 01_audio/record.py --seconds 5 --name quiet_01
copy 01_audio\recordings\quiet_01_16000hz.wav 04_stt\eval\quiet_01.wav
```

Then create `04_stt/eval/quiet_01.txt` in your editor with the sentence,
e.g. `book a ticket to pune for tomorrow`. UTF-8 or UTF-16 both work.

Aim for 2–3 clips of each of these (`04_stt/eval/` is git-ignored, since it's your voice):

| Category | Say something like | What it tests |
|---|---|---|
| `clean` | normal voice, quiet room | the baseline |
| `quiet` | the same sentences, softly, from 1 m away | deletions, and the VAD cutting words |
| `accent` | Indian-accented English, naturally | the model's training data |
| `hinglish` | "mujhe kal subah Pune ka ticket chahiye" | code-switching; compare `-l hi -l en` vs none. Check which script comes back, match your reference to it, then compare CER |
| `names` | "call Ravi Kulkarni at Infosys" | proper nouns; then try `-k` keywords |
| `numbers` | "PNR 4521 9873 06, 10:45 am" | digits vs words (a normalisation trap) |
| `pause` | "I want to… [2 s] …book a ticket" | VAD splitting one thought in two |
| `noise` | fan/TV on, or `--noise -35` on clean clips | insertions and deletions |

Then:

```bash
.venv/Scripts/python 04_stt/evaluate.py 04_stt/eval -v
.venv/Scripts/python 04_stt/evaluate.py 04_stt/eval -p openai -p deepgram --csv 04_stt/eval/results.csv
.venv/Scripts/python 04_stt/evaluate.py 04_stt/eval --noise -35
.venv/Scripts/python 04_stt/evaluate.py 04_stt/eval --vad          # does trimming cost words?
.venv/Scripts/python 04_stt/evaluate.py 04_stt/eval -k Ravi -k Infosys   # do keywords fix names?
```

The latency columns time only the attempt that succeeded. If a request
was retried, the user actually waited longer.

Look for patterns, not one number. Which category is worst? Are the errors
mostly D (missed), S (misheard) or I (invented)? Is the median request time
stable, and what's the p90?

## Experiments

| # | Run | Look for |
|---|---|---|
| 1 | `transcribe.py take_16000hz.wav --ref "…"` | your baseline WER and the 4 latency parts |
| 2 | the same on `take_48000hz.wav` and `take_8000hz.wav` | 48k → same words, bigger source. 8k → muffled telephone audio: does WER rise? |
| 3 | `--relabel-rate 16000` on the 48 kHz file | garbage or empty text: the cost of lying about the rate |
| 4 | `pause_16000hz.wav` with and without `--vad` | upload size, latency, and whether WER changed |
| 5 | run #1 three times | how much `connect` and `wait` vary between runs |
| 6 | `live.py --wav pause_16000hz.wav` then add `--delay high` | first-text time vs accuracy |
| 7 | `live.py --wav … --min-silence-ms 200` | faster finals, but does a pause split your sentence into 2 turns? |
| 8 | `live.py --wav … --no-gate` | KB sent with and without the gate (that's your bill) |

## What this means for your agent's latency

Your 2–4 s response time gets its STT share from: **VAD min silence
(~0.5 s)** + **finalising (~0.3 s)**, and with batch also **the whole
utterance + a new HTTPS connection** per turn. Streaming over an
already-open socket removes the biggest part. That's why every realtime
agent streams.

## Check yourself

1. Why does writing "16000" in the WAV header of a 48 kHz recording break transcription?
2. In a voice agent, why should the LLM only ever see final text?
3. Our live model has no server VAD. Who ends a turn, and what does it cost in latency?
4. A model returns 12 words for a 10-word reference with no mistakes otherwise. What's the WER?
5. Why sum errors across files instead of averaging the per-file WERs?

## Limitations

- **Not yet run against the real APIs.** It was built without a key, so the
  tests use fake HTTP and WebSocket servers. Your first real run is the
  real test. If `live.py` can't connect, the WebSocket URL is the
  likeliest culprit (`--url` overrides it). The guide doesn't spell it out.
  Also unverified for this model: `input_audio_buffer.clear` and its
  `cleared` reply. If the server never confirms a clear, the next turn
  shows only its final (no partials), and later turns are fine. On your
  first run, tap the desk once before speaking and check that partials
  still show.
- No file-response streaming (`stream=true` on `gpt-transcribe`, which
  streams the *text* of a finished file). Streaming the audio is the
  lesson here.
- Mono only. No diarisation (who spoke), no word timestamps, no
  reconnecting a dropped socket.
- The live mic is opened at 24 kHz and the OS resamples. If your mic
  refuses 24 kHz, try `--device`.
- WER normalisation is basic (see section 5).

## Tests

```bash
.venv/Scripts/python -m pytest 04_stt -q      # no key, no network, no mic
```

Covered: resampling (frequency, level, length, anti-aliasing), contract
checks, multipart bytes, both providers' request and response shapes,
every error kind, retries, a real local HTTP server (timings, refused,
timeout), the WebSocket protocol, partial/final/out-of-order turns, the
VAD gate (pre-roll, clear, commit), live runs with a fake server (a dropped
socket, a lost final), and evaluation (corpus WER, CSV).
