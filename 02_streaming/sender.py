"""Stream audio as 20 ms UDP packets through a deliberately bad network.

    python 02_streaming/sender.py --wav 01_audio/recordings/take_16000hz.wav
    python 02_streaming/sender.py                      # live mic, 10 s (use headphones!)
    python 02_streaming/sender.py --wav X.wav --delay-ms 80 --jitter-ms 40 --loss 0.05 --seed 1

Start receiver.py first, in another terminal.

    audio source -> 20 ms frame -> packet (seq, time, format, PCM)
                 -> NetSim: drop it? how long to hold it?
                 -> DelayLine -> UDP socket -> receiver.py

Why UDP and not TCP?
--------------------
TCP guarantees every byte arrives, in order. When a packet is lost, it
re-sends it and holds back *everything after it* until the gap is filled
(called "head-of-line blocking"). For a file download that's exactly right.
For live voice it's wrong: a frame that arrives 300 ms late is useless, and
waiting for it freezes the audio behind it. UDP sends each packet once and
forgets about it, so the receiver decides what to do about gaps. That's
why RTP/WebRTC media runs over UDP.
"""

import argparse
import queue
import socket
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

# Stage folders start with a digit, so they can't be imported as packages.
# Put 01_audio on the path to reuse its format math and WAV reader: this
# stage builds on the last one instead of copying it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_audio"))
from audio_format import AudioFormat  # noqa: E402
from wav_io import PCM16, read_wav  # noqa: E402

from netsim import DelayLine, NetSim  # noqa: E402
from packet import HEADER_SIZE, Packet, encode  # noqa: E402

# Largest UDP payload over IPv4: 65535 - 20 (IP header) - 8 (UDP header).
MAX_UDP_PAYLOAD = 65507


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream audio frames over UDP through a simulated network.")
    p.add_argument("--host", default="127.0.0.1", help="receiver address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=5005, help="receiver UDP port (default 5005)")
    p.add_argument("--frame-ms", type=int, default=20,
                   help="audio per packet in ms (default 20, the VoIP/WebRTC norm)")
    p.add_argument("--delay-ms", type=float, default=0.0, help="simulated one-way network delay")
    p.add_argument("--jitter-ms", type=float, default=0.0,
                   help="each packet's delay varies by up to +/- this much")
    p.add_argument("--loss", type=float, default=0.0,
                   help="fraction of packets dropped, e.g. 0.05 = 5%%")
    p.add_argument("--seed", type=int, default=None,
                   help="fix the random network so runs are repeatable")
    p.add_argument("--wav", default=None, help="stream this 16-bit WAV instead of the mic")
    p.add_argument("--rate", type=int, default=16000, help="mic sample rate (default 16000)")
    p.add_argument("--channels", type=int, default=1, help="mic channels (default 1)")
    p.add_argument("--seconds", type=float, default=10.0, help="mic: how long to stream (default 10)")
    p.add_argument("--device", default=None, help="mic device index or name substring")
    args = p.parse_args(argv)
    if args.device is not None and args.device.isdigit():
        args.device = int(args.device)
    return args


def split_frames(samples: np.ndarray, n: int) -> Iterator[np.ndarray]:
    """Cut (frames, channels) audio into blocks of exactly n frames.

    The last block is padded with silence. Every packet must carry the same
    number of frames, because the receiver's speaker asks for a fixed block
    size each time.
    """
    for off in range(0, len(samples), n):
        block = samples[off:off + n]
        if len(block) < n:
            pad = np.zeros((n - len(block), samples.shape[1]), dtype=np.int16)
            block = np.concatenate([block, pad])
        yield block


def wav_source(path: str, frame_ms: int) -> tuple[AudioFormat, Iterator[np.ndarray]]:
    fmt, samples = read_wav(path)
    n = fmt.frames_per_block(frame_ms)

    def paced() -> Iterator[np.ndarray]:
        # A file is available all at once, but a mic delivers a frame only
        # once it's full. Release frame i at start + (i+1) x frame_duration
        # so the file behaves like a live mic. The times are computed from
        # the *start*, not by sleeping frame_ms after each send: small
        # oversleeps would otherwise add up and the stream would drift slow.
        start = time.perf_counter()
        for i, block in enumerate(split_frames(samples, n)):
            wait = start + (i + 1) * n / fmt.sample_rate - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            yield block

    return fmt, paced()


def mic_source(args: argparse.Namespace) -> tuple[AudioFormat, Iterator[np.ndarray]]:
    # Imported here, not at the top, so the tests (and --wav runs) never
    # need an audio device.
    import sounddevice as sd

    fmt = AudioFormat(args.rate, args.channels)
    n = fmt.frames_per_block(args.frame_ms)
    total = round(args.seconds * fmt.sample_rate)
    try:
        sd.check_input_settings(device=args.device, channels=fmt.channels,
                                dtype="int16", samplerate=fmt.sample_rate)
    except (sd.PortAudioError, ValueError) as e:
        # ValueError is the one error main() reports for bad settings.
        raise ValueError(f"mic can't record {fmt.sample_rate} Hz / {fmt.channels} ch: {e}\n"
                         "hint: python 01_audio/record.py --list-devices, then --device") from e

    def live() -> Iterator[np.ndarray]:
        blocks: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status) -> None:
            # Same rule as 01_audio/record.py: on the audio thread, only
            # copy and enqueue. Never do slow work here.
            blocks.put(indata.copy())

        with sd.InputStream(samplerate=fmt.sample_rate, channels=fmt.channels, dtype="int16",
                            blocksize=n, device=args.device, callback=callback):
            got = 0
            while got < total:
                try:
                    block = blocks.get(timeout=2.0)
                except queue.Empty:
                    raise RuntimeError("no audio arrived for 2 s -- is the mic connected/allowed?")
                got += len(block)
                yield block

    return fmt, live()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.frame_ms <= 0:
        print("error: --frame-ms must be positive", file=sys.stderr)
        return 1
    try:
        net = NetSim(args.delay_ms, args.jitter_ms, args.loss, args.seed)
        fmt, frames = wav_source(args.wav, args.frame_ms) if args.wav else mic_source(args)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    n = fmt.frames_per_block(args.frame_ms)
    payload = n * fmt.bytes_per_frame
    if payload + HEADER_SIZE > MAX_UDP_PAYLOAD:
        print(f"error: a {args.frame_ms} ms frame is {payload} bytes, too big for one UDP packet",
              file=sys.stderr)
        return 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)
    send_errors = 0

    def send(data: bytes) -> None:
        nonlocal send_errors
        try:
            sock.sendto(data, addr)
        except OSError:
            # UDP has no connection, so nothing here notices the receiver is
            # missing. Windows, though, may report an ICMP "port unreachable"
            # from an earlier packet as an error on a later send. Count it
            # and keep going, like a real sender would.
            send_errors += 1

    line = DelayLine(send)
    per_second = max(1, 1000 // args.frame_ms)
    print(f"Source: {args.wav or 'microphone'}  {fmt.sample_rate} Hz, {fmt.channels} ch")
    print(f"Frame:  {args.frame_ms} ms = {n} frames = {payload} B of PCM + {HEADER_SIZE} B header"
          f"  ({per_second} packets/s, {(payload + HEADER_SIZE) * per_second * 8 / 1000:.0f} kbit/s)")
    print(f"Network: delay {args.delay_ms:g} ms, jitter +/-{args.jitter_ms:g} ms, "
          f"loss {args.loss:.0%}  -> udp://{args.host}:{args.port}\n")

    sent = dropped = 0
    try:
        for seq, block in enumerate(frames):
            # Stamp the time *before* the simulated network, so the delay
            # the receiver measures includes everything the network did.
            pkt = Packet(seq, time.time_ns(), fmt.sample_rate, fmt.channels,
                         np.ascontiguousarray(block, dtype=PCM16).tobytes())
            delay = net.decide()
            if delay is None:
                dropped += 1  # the network "lost" it: it is simply never sent
            else:
                line.put(delay, encode(pkt))
            sent += 1
            if sent % per_second == 0:
                print(f"  t={sent * args.frame_ms / 1000:5.1f}s  sent {sent:5d}  "
                      f"dropped {dropped:4d} ({dropped / sent:.1%})")
    except KeyboardInterrupt:
        print("\nStopped.")
    except (RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        line.close()  # let packets still "in the network" arrive
        sock.close()

    print(f"\nDone: {sent} packets, {dropped} dropped by the simulated network"
          + (f", {send_errors} send errors (is receiver.py running?)" if send_errors else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
