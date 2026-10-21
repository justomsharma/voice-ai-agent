"""Step A: transcribe a WAV file with one batch request, and show every step.

    python 04_stt/transcribe.py 01_audio/recordings/take_16000hz.wav --dry-run   # no key needed
    python 04_stt/transcribe.py 01_audio/recordings/take_16000hz.wav
    python 04_stt/transcribe.py X.wav --ref "i want to book a ticket"           # + WER
    python 04_stt/transcribe.py X.wav --vad                                      # Stage 03 trims silence
    python 04_stt/transcribe.py X.wav --provider deepgram
    python 04_stt/transcribe.py X.wav --language hi --language en               # Hinglish hint
    python 04_stt/transcribe.py take_48000hz.wav --relabel-rate 16000           # the classic bug

    WAV -> prepare (mono, resample) -> [VAD trim] -> HTTPS POST -> transcript
                                                                -> latency breakdown, WER

Batch = the whole recording must exist before the request starts. The user
waits (how long they spoke) + (request time). Stage 04's live.py removes
the first part by sending audio while they're still speaking.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

# Stage folders start with a digit, so they can't be imported as packages.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "03_vad"))
sys.path.insert(0, str(ROOT / "01_audio"))
from audio_format import AudioFormat  # noqa: E402
from wav_io import read_wav, write_wav  # noqa: E402

from audio_prep import Prepared, prepare, wav_bytes  # noqa: E402
from stt import PROVIDERS, BatchProvider, SttError, Transcript, make_provider, with_retries  # noqa: E402
from wer import cer, format_alignment, read_reference, summary, wer  # noqa: E402

HINTS = {
    "auth": "check the key in .env (copy .env.example) or your environment",
    "rate_limit": "wait a bit, or check your plan's credit / limits",
    "bad_request": "the API rejected the request: check --model / --language and the audio",
    "timeout": "slow network or a long file: try --timeout 120",
    "network": "are you online? (DNS, Wi-Fi, proxy, firewall)",
    "server": "the provider is having trouble; it was retried already",
}


def vad_trim(fmt: AudioFormat, samples: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep only Stage 03's speech segments. Returns (audio, number of segments).

    Less audio = smaller upload and less billed time. The risk: a VAD that's
    too strict chops quiet words, and the transcript loses them (WER
    deletions). Compare --vad vs no --vad with --ref to see if it costs words.
    """
    from detect import analyze, frame_samples, keep_speech, split_frames
    from vad import VadConfig

    config = VadConfig()
    a = analyze(split_frames(samples, frame_samples(fmt, config.frame_ms)), fmt, config)
    return keep_speech(fmt, a.samples, a.segments), len(a.segments)


def reference_for(args: argparse.Namespace) -> str | None:
    if args.ref is not None:
        return args.ref
    path = Path(args.ref_file) if args.ref_file else Path(args.wav).with_suffix(".txt")
    if args.ref_file or path.is_file():
        return read_reference(path)
    return None


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms"


def report(t: Transcript, audio: Prepared, reference: str | None) -> str:
    tm = t.timings
    lines = [
        f"Request:  connect {ms(tm.connect_s)} | upload {ms(tm.upload_s)} | "
        f"wait {ms(tm.wait_s)} | download {ms(tm.download_s)} | total {ms(tm.total_s)}"
        f"  ({t.attempts} attempt{'s' * (t.attempts > 1)})",
        f"          RTF {t.rtf:.2f}: {audio.seconds:.2f} s of speech took {tm.total_s:.2f} s. "
        f"Batch user waits {audio.seconds + tm.total_s:.2f} s from first word to text",
    ]
    meta = [f"{t.provider} / {t.model}"]
    if t.language:
        meta.append(f"language: {t.language}")
    if t.confidence is not None:
        meta.append(f"confidence: {t.confidence:.2f}")
    if t.request_id:
        meta.append(f"request id: {t.request_id}")
    lines.append(f"Model:    {'   '.join(meta)}")
    for k, v in t.extra.items():
        lines.append(f"          {k}: {v}")
    lines.append(f'\nTranscript:\n  "{t.text}"' if t.text else
                 "\nTranscript: (empty: no speech recognised)")
    if reference is not None:
        w = wer(reference, t.text)
        lines += ["", format_alignment(w), "",
                  f"WER {summary(w)}",
                  f"CER {summary(cer(reference, t.text)).replace('words', 'chars')}"]
    return "\n".join(lines)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe a WAV file with a batch STT API.")
    p.add_argument("wav", help="16-bit PCM WAV (any rate, mono or stereo)")
    p.add_argument("--provider", choices=PROVIDERS, default="openai")
    p.add_argument("--model", default=None, help="override the provider's default model")
    p.add_argument("--language", "-l", action="append", default=None,
                   help="expected language code, repeatable (en, hi, ...)")
    p.add_argument("--prompt", default=None, help="context for the model (OpenAI only)")
    p.add_argument("--keyword", "-k", action="append", default=None,
                   help="a name or term you expect, repeatable (boosts it)")
    p.add_argument("--ref", default=None, help="what was really said, for WER")
    p.add_argument("--ref-file", default=None,
                   help="reference text file (default: <wav>.txt if it exists)")
    p.add_argument("--vad", action="store_true", help="trim to speech with Stage 03 first")
    p.add_argument("--relabel-rate", type=int, default=None, metavar="HZ",
                   help="EXPERIMENT: lie about the sample rate instead of resampling")
    p.add_argument("--save-sent", default=None, metavar="WAV",
                   help="also write the exact audio that was sent, to listen to it")
    p.add_argument("--dry-run", action="store_true", help="show the request, don't send it")
    p.add_argument("--timeout", type=float, default=30.0, help="seconds per attempt (default 30)")
    p.add_argument("--retries", type=int, default=2, help="retries on transient errors (default 2)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None,
         make: Callable[..., BatchProvider] = make_provider) -> int:
    args = parse_args(argv)
    try:
        provider = make(args.provider, model=args.model, languages=args.language,
                        prompt=args.prompt, keywords=args.keyword, timeout=args.timeout)
        fmt, samples = read_wav(args.wav)
        reference = reference_for(args)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"Input:    {args.wav}   {fmt.sample_rate} Hz, {fmt.channels} ch, {fmt.bit_depth}-bit, "
          f"{fmt.duration_seconds(len(samples)):.2f} s")
    if args.relabel_rate:
        fmt = AudioFormat(args.relabel_rate, fmt.channels)
        print(f"RELABEL:  pretending it's {fmt.sample_rate} Hz without resampling: it now "
              f"'lasts' {fmt.duration_seconds(len(samples)):.2f} s and sounds wrong")
    c = provider.contract
    limit = f", <= {c.max_bytes / 2**20:.0f} MB" if c.max_bytes else ""
    print(f"Contract: {provider.name} {provider.model} gets {c.sample_rate} Hz "
          f"{'mono' if c.channels == 1 else f'{c.channels} ch'} 16-bit WAV{limit}")
    try:
        if args.vad:
            before = fmt.duration_seconds(len(samples))
            samples, n = vad_trim(fmt, samples)
            print(f"VAD:      kept {fmt.duration_seconds(len(samples)):.2f} s of {before:.2f} s "
                  f"({n} speech segment{'s' * (n != 1)})")
        audio = prepare(fmt, samples, c)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Prepared: {'; '.join(audio.steps) or 'already matches, sent as is'}  "
          f"[{ms(audio.prep_s)}]")
    print(f"          {len(wav_bytes(audio.fmt, audio.samples)):,} bytes of WAV, "
          f"{audio.seconds:.2f} s")
    for w in audio.warnings:
        print(f"Warning:  {w}")
    if args.save_sent:
        write_wav(args.save_sent, audio.fmt, audio.samples)
        print(f"Saved:    the audio being sent -> {args.save_sent}")

    if args.dry_run:
        print("\nDry run, nothing sent. The request would be:\n")
        print(provider.describe_request(audio))
        return 0

    print("\nSending...")
    try:
        t = with_retries(lambda: provider.transcribe(audio), retries=args.retries,
                         log=lambda line: print(f"          {line}"))
    except SttError as e:
        print(f"error: {e}", file=sys.stderr)
        if e.kind in HINTS:
            print(f"hint: {HINTS[e.kind]}", file=sys.stderr)
        return 1
    print(report(t, audio, reference))
    return 0


if __name__ == "__main__":
    sys.exit(main())
