"""Receive UDP audio packets, smooth them with a jitter buffer, play them.

    python 02_streaming/receiver.py                  # 60 ms jitter buffer
    python 02_streaming/receiver.py --buffer-ms 0    # no buffer: hear the jitter
    python 02_streaming/receiver.py --buffer-ms 200  # smooth, but late

Start this first, then sender.py in another terminal. Stops by itself
1.5 s after the stream ends (or Ctrl+C), prints a summary, and saves what
it played to 02_streaming/recordings/received.wav.

Two clocks, two threads
-----------------------
    network thread   packets arrive whenever the network delivers them
                     -> jitter_buffer.push()
    speaker thread   the sound card asks for exactly one frame every 20 ms,
                     steady as a metronome -> jitter_buffer.pop()

The jitter buffer sits between the uneven clock and the steady one. We let
the *sound card* drive playout (its callback pulls frames) instead of a
Python timer, because the sound card is what actually consumes audio at
the real rate. That's also how real voice apps are built.
"""

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_audio"))
from audio_format import AudioFormat  # noqa: E402
from wav_io import PCM16, write_wav  # noqa: E402

from jitter_buffer import Frame, JitterBuffer, frames_for_ms  # noqa: E402
from packet import Packet, decode  # noqa: E402

RECORDINGS_DIR = Path(__file__).parent / "recordings"
# No packets for this long = the stream is over. Much longer than any jitter
# we simulate, short enough that you aren't left waiting.
IDLE_STOP_S = 1.5


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Receive, de-jitter and play UDP audio frames.")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=5005, help="UDP port (default 5005)")
    # 60 ms default: typical VoIP jitter buffers sit around 40-80 ms.
    p.add_argument("--buffer-ms", type=float, default=60.0,
                   help="jitter buffer: audio to hold before playing (default 60)")
    p.add_argument("--device", default=None, help="output device index or name substring")
    p.add_argument("--out", default=str(RECORDINGS_DIR / "received.wav"),
                   help="where to save what was played")
    args = p.parse_args(argv)
    if args.device is not None and args.device.isdigit():
        args.device = int(args.device)
    return args


class Receiver:
    """Everything both threads share, behind one lock."""

    def __init__(self, buffer_ms: float):
        self.buffer_ms = buffer_ms
        self.lock = threading.Lock()
        self.first_packet = threading.Event()
        # Unknown until the first packet arrives: the format is in the packets.
        self.fmt: AudioFormat | None = None
        self.frames_per_packet = 0
        self.frame_ms = 0.0
        self.jb: JitterBuffer | None = None
        self.last_arrival = 0.0
        self.ignored = 0
        # The speaker takes ~50-80 ms after opening to ask for its first
        # frame. Packets that arrive before then would pile up in the jitter
        # buffer and silently make it bigger than --buffer-ms. A real app's
        # speaker is already running when the call's audio starts, so we
        # imitate that: drop packets until the playout clock is ticking.
        self.speaker_ready = False
        self.warmup_seqs: set[int] = set()
        self.underflows = 0
        self.net_delays_ms: list[float] = []   # arrival - send, per packet
        self.play_delays_ms: list[float] = []  # handed to speaker - send, per frame
        self.slots: list[Frame] = []           # every playout slot, for the WAV

    def on_packet(self, pkt: Packet, arrival_ns: int) -> None:
        with self.lock:
            if self.fmt is None:
                self.fmt = AudioFormat(pkt.sample_rate, pkt.channels)
                self.frames_per_packet = len(pkt.pcm) // self.fmt.bytes_per_frame
                self.frame_ms = self.frames_per_packet * 1000 / self.fmt.sample_rate
                self.jb = JitterBuffer(frames_for_ms(self.buffer_ms, self.frame_ms))
                self.first_packet.set()
            elif ((pkt.sample_rate, pkt.channels) != (self.fmt.sample_rate, self.fmt.channels)
                  or len(pkt.pcm) != self.frames_per_packet * self.fmt.bytes_per_frame):
                # One stream per run: the speaker was opened for one format
                # and block size. Anything else can't be played as-is.
                self.ignored += 1
                return
            self.last_arrival = time.monotonic()
            if not self.speaker_ready:
                self.warmup_seqs.add(pkt.seq)
                return
            if self.jb.push(pkt.seq, pkt.pcm, pkt.send_time_ns) != "duplicate":
                # Both clocks are this machine's clock, so this subtraction
                # is meaningful. Across two machines it wouldn't be: their
                # clocks differ by more than the delay we're measuring.
                self.net_delays_ms.append((arrival_ns - pkt.send_time_ns) / 1e6)

    def playout(self, outdata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags) -> None:
        # Sound card thread: same rules as the mic callback in 01_audio.
        # Be quick, never block for long, no printing.
        if status.output_underflow:
            self.underflows += 1  # we were too slow: the card played garbage/silence
        with self.lock:
            self.speaker_ready = True
            frame = self.jb.pop()
            if frame is not None:
                self.slots.append(frame)
                if frame.payload is not None:
                    self.play_delays_ms.append((time.time_ns() - frame.send_time_ns) / 1e6)
        if frame is None or frame.payload is None:
            outdata.fill(0)  # pre-roll or a missing frame: silence
        else:
            outdata[:] = np.frombuffer(frame.payload, dtype=PCM16).reshape(-1, self.fmt.channels)


def recv_loop(sock: socket.socket, rx: Receiver, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            data, _ = sock.recvfrom(65536)
        except TimeoutError:
            continue  # the timeout is only there so we notice `stop`
        except OSError:
            return
        arrival_ns = time.time_ns()  # stamp first, before any other work
        try:
            pkt = decode(data)
        except ValueError:
            rx.ignored += 1
            continue
        rx.on_packet(pkt, arrival_ns)


def _avg(xs: list[float]) -> str:
    return f"{sum(xs) / len(xs):7.1f} ms" if xs else "      -   "


def stats_line(rx: Receiver, t: float, net_from: int, play_from: int) -> str:
    jb = rx.jb
    return (f"  {t:5.1f}  {jb.received:5d}  {jb.late:4d}  {jb.missed_count():6d}  {jb.reordered:5d}"
            f"   {jb.depth:3d} fr ({jb.depth * rx.frame_ms:4.0f} ms)"
            f"   {_avg(rx.net_delays_ms[net_from:])}   {_avg(rx.play_delays_ms[play_from:])}")


def summary(rx: Receiver, output_latency_ms: float) -> None:
    jb = rx.jb
    slots = [s for s in rx.slots if s.seq <= jb.highest_seq]
    silent_seqs = {s.seq for s in slots if s.payload is None}
    silent = len(silent_seqs)
    # A silent slot whose packet did show up (after its turn) was late; one
    # whose packet never showed up was lost. Late packets for slots before
    # playback started aren't in `slots` at all, so match by seq, not count.
    # (A packet discarded during speaker warm-up also arrived, just early.)
    late = len(silent_seqs & (jb.late_seqs | rx.warmup_seqs))
    net, play = rx.net_delays_ms, rx.play_delays_ms
    print(f"\n--- summary (jitter buffer {rx.buffer_ms:g} ms = {jb.target_frames} frames) ---")
    print(f"  frames in stream      {len(slots)}")
    print(f"  played                {len(slots) - silent}")
    print(f"  played as silence     {silent}  ({silent / max(1, len(slots)):.1%})"
          f"  = {silent - late} lost in network + {late} arrived too late")
    print(f"  arrived out of order  {jb.reordered}")
    if net:
        print(f"  network delay         avg {sum(net) / len(net):6.1f} ms   "
              f"min {min(net):6.1f}   max {max(net):6.1f}   (spread = jitter)")
    if play:
        print(f"  delay at playout      avg {sum(play) / len(play):6.1f} ms   "
              f"max {max(play):6.1f}   (network + waiting in the jitter buffer)")
    print(f"  + speaker latency     ~{output_latency_ms:.0f} ms more inside the OS/driver (not measured above)")
    if rx.warmup_seqs:
        print(f"  speaker warm-up       {len(rx.warmup_seqs)} packets dropped before the speaker started")
    if rx.ignored:
        print(f"  ignored packets       {rx.ignored} (malformed or a different format)")
    if rx.underflows:
        print(f"  output underflows     {rx.underflows} (the sound card ran dry)")


def save(rx: Receiver, path: Path) -> None:
    slots = [s for s in rx.slots if s.seq <= rx.jb.highest_seq]
    if not slots:
        return
    silence = bytes(rx.frames_per_packet * rx.fmt.bytes_per_frame)
    # Exactly what the listener heard, slot by slot: gaps stay as silence so
    # you can see them in 01_audio/inspect_audio.py.
    pcm = b"".join(s.payload if s.payload is not None else silence for s in slots)
    samples = np.frombuffer(pcm, dtype=PCM16).astype(np.int16).reshape(-1, rx.fmt.channels)
    write_wav(path, rx.fmt, samples)
    print(f"\nSaved what was played to {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.buffer_ms < 0:
        print("error: --buffer-ms must be >= 0", file=sys.stderr)
        return 1
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((args.host, args.port))
    except OSError as e:
        print(f"error: can't listen on udp://{args.host}:{args.port} ({e}). "
              "Is another receiver already running?", file=sys.stderr)
        return 1
    sock.settimeout(0.2)

    rx = Receiver(args.buffer_ms)
    stop = threading.Event()
    thread = threading.Thread(target=recv_loop, args=(sock, rx, stop), daemon=True)
    thread.start()
    print(f"Listening on udp://{args.host}:{args.port}, jitter buffer {args.buffer_ms:g} ms. "
          "Now start sender.py. Ctrl+C to stop.")

    output_latency_ms = 0.0
    try:
        while not rx.first_packet.wait(0.5):
            pass
        print(f"First packet: {rx.fmt.sample_rate} Hz, {rx.fmt.channels} ch, "
              f"{rx.frame_ms:g} ms frames ({rx.frames_per_packet} frames). "
              f"Buffer holds {rx.jb.target_frames} frames before playing.\n")
        with sd.OutputStream(samplerate=rx.fmt.sample_rate, channels=rx.fmt.channels, dtype="int16",
                             blocksize=rx.frames_per_packet, device=args.device,
                             # "low": a smaller OS/driver output buffer. On
                             # Windows MME it's ~100 ms instead of ~200 ms.
                             latency="low",
                             callback=rx.playout) as stream:
            output_latency_ms = stream.latency * 1000
            print("      t   recv  late  missed  reord   in buffer         net delay   play delay")
            start, net_from, play_from = time.monotonic(), 0, 0
            while True:
                time.sleep(1.0)
                with rx.lock:
                    print(stats_line(rx, time.monotonic() - start, net_from, play_from))
                    net_from, play_from = len(rx.net_delays_ms), len(rx.play_delays_ms)
                    if time.monotonic() - rx.last_arrival > IDLE_STOP_S and rx.jb.depth == 0:
                        break
    except KeyboardInterrupt:
        print("\nStopped.")
    except sd.PortAudioError as e:
        print(f"error: can't open the speaker: {e}", file=sys.stderr)
        return 1
    finally:
        stop.set()
        thread.join()
        sock.close()

    if rx.jb is None or rx.jb.highest_seq is None:
        print("No packets received.")
        return 0
    summary(rx, output_latency_ms)
    save(rx, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
