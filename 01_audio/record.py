"""Record the microphone as raw 16-bit PCM, block by block, and save it.

    python 01_audio/record.py                       # 3 s, 16 kHz, mono
    python 01_audio/record.py --rate 8000
    python 01_audio/record.py --rate 48000 --seconds 5 --name hello
    python 01_audio/record.py --list-devices

Saves 01_audio/recordings/<name>_<rate>hz.wav and .raw (same PCM bytes,
with and without the 44-byte WAV header).

How audio actually reaches us
-----------------------------
  mic -> ADC (analog-to-digital converter) samples the voltage
      -> OS audio driver fills a buffer
      -> PortAudio (C library; `sounddevice` wraps it) calls our callback
         with one *block* of frames, on its own audio thread
      -> we hand the block to the main thread through a queue

We deliberately record in small fixed blocks (20 ms by default) with a
callback, instead of one big `sd.rec()` call, because that's how every
realtime voice system works: audio must be processed while it's still
arriving. The per-block log this script prints shows you that stream.
"""

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from audio_format import AudioFormat
from wav_io import write_raw, write_wav

RECORDINGS_DIR = Path(__file__).parent / "recordings"

# int16 = 16-bit signed PCM. We ask PortAudio for int16 directly (instead
# of its default float32) so what we store is exactly what an AudioFrame
# carries, with no conversion step hiding what PCM is.
DTYPE = "int16"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record the mic as raw 16-bit PCM.")
    # 16 kHz default: the de-facto standard for speech recognition. It keeps
    # everything up to 8 kHz (enough for consonants) at a third of 48 kHz's
    # bandwidth.
    p.add_argument("--rate", type=int, default=16000, help="sample rate in Hz (default 16000)")
    p.add_argument("--channels", type=int, default=1,
                   help="1 = mono (default; voice AI almost always uses mono)")
    p.add_argument("--seconds", type=float, default=3.0, help="how long to record (default 3)")
    # 20 ms blocks: the common unit in realtime voice (WebRTC, Opus, most VAD
    # models use 10-30 ms). Short enough to keep latency low; long enough
    # that per-block overhead (function calls, thread hops) stays small.
    p.add_argument("--block-ms", type=int, default=20, help="block size in ms (default 20)")
    p.add_argument("--device", default=None,
                   help="input device index or name substring (see --list-devices)")
    p.add_argument("--name", default="take", help="output file prefix (default 'take')")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    args = p.parse_args(argv)
    # sounddevice accepts an int index or a name substring; argparse gives a
    # str, so convert pure digits to an int index.
    if args.device is not None and args.device.isdigit():
        args.device = int(args.device)
    return args


def check_settings(args: argparse.Namespace) -> str | None:
    """Return an error message if this device can't do what we asked.

    Checking up front turns a cryptic failure mid-stream into a clear
    message before anything starts.
    """
    if args.seconds <= 0 or args.block_ms <= 0:
        return "--seconds and --block-ms must be positive"
    try:
        sd.check_input_settings(device=args.device, channels=args.channels,
                                dtype=DTYPE, samplerate=args.rate)
    except (sd.PortAudioError, ValueError) as e:
        return (f"this input device can't record {args.rate} Hz / {args.channels} ch / "
                f"{DTYPE}: {e}\nhint: run with --list-devices and pick one via --device")
    return None


def record(args: argparse.Namespace) -> tuple[AudioFormat, np.ndarray, int]:
    """Capture audio block by block. Returns (format, samples, overflow_count)."""
    fmt = AudioFormat(args.rate, args.channels)
    blocksize = fmt.frames_per_block(args.block_ms)
    target_frames = round(args.seconds * fmt.sample_rate)

    # queue.Queue is thread-safe: the PortAudio thread puts blocks in, the
    # main thread takes them out. This is the classic producer/consumer
    # hand-off that every realtime audio app uses.
    blocks: queue.Queue = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags) -> None:
        # Runs on PortAudio's audio thread, once per block. Rules for this
        # function: be fast, never block, no printing or file I/O. If it's
        # slow, the driver's buffer fills up and audio is lost (an
        # "input overflow"). So: copy, enqueue, return.
        #
        # .copy() is essential: `indata` is a view onto PortAudio's own
        # buffer, which gets reused for the next block. Keeping the view
        # would leave us with N references to the same, constantly
        # overwritten memory.
        #
        # (We don't use time_info.inputBufferAdcTime, "when the ADC captured
        # this block": on Windows MME it alternates between 0 and a value on
        # an unrelated clock, and WASAPI reports ~0.2 ms -- impossible, since
        # a 20 ms block is at least 20 ms old when it arrives. Driver
        # timestamps are often unreliable; measure latency end-to-end instead.)
        blocks.put((indata.copy(), bool(status.input_overflow)))

    collected: list[np.ndarray] = []
    got = 0
    overflows = 0
    print(f"Recording {args.seconds:g} s at {fmt.sample_rate} Hz, {fmt.channels} ch, "
          f"{fmt.bit_depth}-bit, in {args.block_ms} ms blocks of {blocksize} frames "
          f"({blocksize * fmt.bytes_per_frame} bytes)... speak now. Ctrl+C stops early.\n")
    print("  block  frames      ms   since prev   peak")

    with sd.InputStream(samplerate=fmt.sample_rate, channels=fmt.channels, dtype=DTYPE,
                        blocksize=blocksize, device=args.device, callback=callback) as stream:
        # Latency starts here, before a line of our code runs:
        #   1. block size: a block can't be delivered until its last frame
        #      has been captured, so its oldest frame is >= block_ms old.
        #   2. driver/OS buffering. stream.latency is PortAudio's estimate of
        #      the stream's whole input latency; whether it already counts
        #      our block depends on the host API, so we show both numbers
        #      rather than adding them.
        # Every later stage (VAD, STT, ...) can only add to this.
        print(f"  (latency: a {args.block_ms} ms block can't arrive before it's full; "
              f"PortAudio estimates {stream.latency * 1000:.1f} ms total input latency)")
        last_arrival = None
        try:
            while got < target_frames:
                try:
                    # Timeout so a dead/blocked mic produces an error instead
                    # of hanging forever.
                    block, overflow = blocks.get(timeout=2.0)
                except queue.Empty:
                    raise RuntimeError("no audio arrived for 2 s -- is the mic connected/allowed?")
                now = time.perf_counter()
                overflows += overflow
                collected.append(block)
                got += len(block)

                # Blocks should arrive every ~block_ms. Watching the gaps
                # shows "jitter": the OS may deliver in bursts (Windows MME
                # hands over two 20 ms blocks together every ~40 ms), not a
                # perfect metronome -- which is why realtime systems need
                # buffers between stages.
                gap = "" if last_arrival is None else f"{(now - last_arrival) * 1000:6.1f} ms"
                last_arrival = now
                peak = int(np.max(np.abs(block.astype(np.int32))))
                bar = "#" * int(20 * peak / 32768)
                print(f"  {len(collected):5d}  {len(block):6d}  {len(block) / fmt.sample_rate * 1000:6.1f}"
                      f"   {gap:>10}   {peak:5d} {bar}")
        except KeyboardInterrupt:
            print("\nStopped early -- saving what was captured.")

    samples = np.concatenate(collected) if collected else np.zeros((0, fmt.channels), np.int16)
    # The last block may overshoot the requested length (3 s isn't always a
    # whole number of blocks); trim so the file is exactly what was asked.
    return fmt, samples[:target_frames], overflows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_devices:
        print(sd.query_devices())
        print("\n> = default input, < = default output. Host API in use by default:",
              sd.query_hostapis(sd.default.hostapi)["name"])
        return 0

    if (err := check_settings(args)) is not None:
        print(f"error: {err}", file=sys.stderr)
        return 1

    dev = sd.query_devices(args.device, "input")
    hostapi = sd.query_hostapis(dev["hostapi"])["name"]
    # The hardware runs at one *native* rate (usually 48 kHz). If we ask for
    # another, some host APIs resample for us -- low-pass filtering first so
    # frequencies above the new Nyquist don't alias -- and some refuse.
    # On Windows: MME (sounddevice's default) silently resamples and always
    # reports 44100 here regardless of the hardware; WASAPI reports the true
    # rate and rejects anything else ("Invalid sample rate"). So "recording
    # at 8 kHz" via MME really means "48 kHz, filtered and decimated by the OS".
    print(f"Device: {dev['name']}  [{hostapi}]  "
          f"host API's default rate: {dev['default_samplerate']:g} Hz")

    try:
        fmt, samples, overflows = record(args)
    except (RuntimeError, sd.PortAudioError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if len(samples) == 0:
        print("error: nothing was recorded", file=sys.stderr)
        return 1

    # Build both names directly rather than with Path.with_suffix(): a name
    # like "take.v2" would make with_suffix treat ".v2_16000hz" as the
    # extension and replace it, dropping the rate so takes overwrite each other.
    stem = f"{args.name}_{fmt.sample_rate}hz"
    wav_path, raw_path = RECORDINGS_DIR / f"{stem}.wav", RECORDINGS_DIR / f"{stem}.raw"
    write_wav(wav_path, fmt, samples)
    write_raw(raw_path, samples)

    print(f"\nCaptured {len(samples)} frames = {fmt.duration_seconds(len(samples)):.3f} s")
    if overflows:
        # Don't hide data loss: an overflow means blocks were dropped
        # because we (or the system) didn't keep up.
        print(f"WARNING: {overflows} input overflow(s) -- some audio was lost")
    if not samples.any():
        # Windows returns pure zeros (not an error) when mic access is
        # blocked in Settings > Privacy & security > Microphone.
        print("WARNING: the recording is pure digital silence -- check mic privacy settings / mute")
    print(f"Saved {wav_path}  ({wav_path.stat().st_size} bytes)")
    print(f"Saved {raw_path}  ({raw_path.stat().st_size} bytes = 44 fewer: no header)")
    print(f"\nNext: python 01_audio/inspect_audio.py {wav_path.relative_to(Path.cwd()) if wav_path.is_relative_to(Path.cwd()) else wav_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
