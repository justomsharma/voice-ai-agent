"""Step C: measure accuracy and latency on YOUR audio, across providers.

    python 04_stt/evaluate.py 04_stt/eval                         # every X.wav that has an X.txt
    python 04_stt/evaluate.py 04_stt/eval -p openai -p deepgram   # same files, two providers
    python 04_stt/evaluate.py 04_stt/eval --noise -35             # same files in a "noisy room"
    python 04_stt/evaluate.py 04_stt/eval --vad -v                # trimmed; show every mistake
    python 04_stt/evaluate.py 04_stt/eval --csv 04_stt/eval/results.csv

The folder: one WAV per utterance plus a .txt with exactly what was said.
The part of the name before the first "_" is its category, so results
group by it:

    04_stt/eval/quiet_01.wav     quiet_01.txt     "book a ticket to pune"
    04_stt/eval/hinglish_01.wav  hinglish_01.txt  "mujhe kal ka ticket chahiye"
    04_stt/eval/numbers_01.wav   numbers_01.txt   "my pnr is 4 5 2 1 9 8 7 3 0 6"

Don't pick a provider from one demo clip. A provider that's best on clean
US English can be worst on Indian names and numbers. The corpus WER sums
the errors of every file before dividing, so long clips weigh more.
"""

import argparse
import csv
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "03_vad"))
sys.path.insert(0, str(ROOT / "01_audio"))
from wav_io import read_wav  # noqa: E402

from audio_prep import prepare  # noqa: E402
from stt import PROVIDERS, BatchProvider, SttError, make_provider, with_retries  # noqa: E402
from transcribe import vad_trim  # noqa: E402
from wer import Score, format_alignment, read_reference, wer  # noqa: E402


@dataclass
class Row:
    file: str
    category: str
    seconds: float
    provider: str
    text: str = ""
    score: Score | None = None
    request_s: float | None = None
    error: str | None = None


def find_pairs(folder: Path) -> tuple[list[tuple[Path, str]], list[Path]]:
    """(wav, reference) for every WAV with a .txt next to it; plus the WAVs without one."""
    pairs, missing = [], []
    for wav in sorted(folder.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.is_file():
            pairs.append((wav, read_reference(txt)))
        else:
            missing.append(wav)
    return pairs, missing


def category(path: Path) -> str:
    return path.stem.split("_")[0] if "_" in path.stem else "-"


def evaluate(pairs: list[tuple[Path, str]], providers: list[BatchProvider], vad: bool = False,
             noise_db: float | None = None, retries: int = 2,
             log: Callable[[str], None] = print) -> list[Row]:
    rows = []
    dead: dict[str, str] = {}  # provider -> why we stopped calling it
    for wav, reference in pairs:
        fmt, samples = read_wav(wav)
        if noise_db is not None:
            from detect import add_noise
            samples = add_noise(samples, noise_db, np.random.default_rng(0))  # same noise every run
        if vad:
            samples, _ = vad_trim(fmt, samples)
        for p in providers:
            row = Row(wav.name, category(wav), fmt.duration_seconds(len(samples)), p.name)
            rows.append(row)
            if p.name in dead:
                row.error = dead[p.name]
                continue
            try:
                audio = prepare(fmt, samples, p.contract)
                t = with_retries(lambda: p.transcribe(audio), retries=retries)
            except ValueError as e:
                row.error = f"audio: {e}"
                # Nothing was sent. An empty result is still a result: every word missed.
                row.score = wer(reference, "")
                continue
            except SttError as e:
                row.error = f"{e.kind}: {e}"
                if e.kind in ("auth", "rate_limit"):  # it'll fail the same way for every file
                    dead[p.name] = f"skipped after {e.kind} error"
                continue
            row.text, row.request_s = t.text, t.timings.total_s
            row.score = wer(reference, t.text)
        log(f"  {wav.name}: " + "  ".join(
            f"{r.provider} {fmt_score(r.score) if r.score else 'ERROR'}" for r in rows[-len(providers):]))
    return rows


def fmt_score(s: Score) -> str:
    return f"{s.rate:6.1%} ({s.errors}/{s.ref_len})" if s.rate != float("inf") else f"   inf (+{s.ins})"


def total(rows: list[Row]) -> Score:
    out = Score(0)
    for r in rows:
        if r.score:
            out = out + r.score
    return out


def report(rows: list[Row], providers: list[str], verbose: bool = False) -> str:
    lines = ["", f"{'file':<22} {'cat':<10} {'audio':>6}" +
             "".join(f"  {p + ' WER':>16} {'ms':>6}" for p in providers)]
    files = list(dict.fromkeys(r.file for r in rows))
    by = {(r.file, r.provider): r for r in rows}
    for f in files:
        first = by[(f, providers[0])]
        cells = []
        for p in providers:
            r = by[(f, p)]
            ms = f"{r.request_s * 1000:.0f}" if r.request_s is not None else "-"
            cells.append(f"  {fmt_score(r.score) if r.score and not r.error else 'ERROR':>16} {ms:>6}")
        lines.append(f"{f:<22} {first.category:<10} {first.seconds:5.1f}s" + "".join(cells))

    lines.append("\nCorpus (all errors / all reference words):")
    for p in providers:
        mine = [r for r in rows if r.provider == p]
        s = total(mine)
        times = [r.request_s for r in mine if r.request_s is not None]
        audio = sum(r.seconds for r in mine if r.request_s is not None)
        lat = (f"request median {np.median(times) * 1000:.0f} ms, p90 {np.percentile(times, 90) * 1000:.0f} ms, "
               f"RTF {sum(times) / audio:.2f}" if times else "no successful requests")
        failed = sum(r.error is not None for r in mine)
        lines.append(f"  {p:<10} WER {s.rate:6.1%}  S={s.sub} D={s.dele} I={s.ins} of {s.ref_len} words"
                     f"   {lat}" + (f"   {failed} failed" if failed else ""))
        cats = sorted({r.category for r in mine})
        if len(cats) > 1:
            lines.append("             " + "  ".join(
                f"{c} {total([r for r in mine if r.category == c]).rate:.1%}" for c in cats))

    errors = [r for r in rows if r.error]
    if errors:
        lines.append("\nErrors:")
        lines += [f"  {r.file} / {r.provider}: {r.error}" for r in errors]
    if verbose:
        lines.append("\nMistakes:")
        for r in rows:
            if r.score and r.score.errors and not r.error:
                lines.append(f"\n{r.file} / {r.provider}:\n{format_alignment(r.score)}")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "category", "seconds", "provider", "wer", "sub", "del", "ins",
                    "ref_words", "request_ms", "error", "text"])
        for r in rows:
            s = r.score
            w.writerow([r.file, r.category, f"{r.seconds:.2f}", r.provider,
                        f"{s.rate:.4f}" if s else "", s.sub if s else "", s.dele if s else "",
                        s.ins if s else "", s.ref_len if s else "",
                        f"{r.request_s * 1000:.0f}" if r.request_s is not None else "",
                        r.error or "", r.text])


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WER and latency over a folder of wav+txt pairs.")
    p.add_argument("folder", help="folder with X.wav + X.txt pairs")
    p.add_argument("--provider", "-p", action="append", choices=PROVIDERS, default=None,
                   help="repeatable (default: openai)")
    p.add_argument("--language", "-l", action="append", default=None, help="language hint, repeatable")
    p.add_argument("--vad", action="store_true", help="trim with Stage 03 before sending")
    p.add_argument("--noise", type=float, default=None, metavar="DB", help="add white noise, e.g. -35")
    p.add_argument("--csv", default=None, help="also save every row to this CSV")
    p.add_argument("--verbose", "-v", action="store_true", help="print the alignment of every mistake")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, make: Callable[..., BatchProvider] = make_provider) -> int:
    args = parse_args(argv)
    folder = Path(args.folder)
    names = args.provider or ["openai"]
    try:
        if not folder.is_dir():
            raise ValueError(f"{folder} is not a folder (see the docstring for the layout)")
        pairs, missing = find_pairs(folder)
        if not pairs:
            raise ValueError(f"no X.wav + X.txt pairs in {folder}")
        providers = [make(n, languages=args.language) for n in names]
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for m in missing:
        print(f"skipping {m.name}: no {m.with_suffix('.txt').name}")
    extras = (["VAD trim"] if args.vad else []) +         ([f"noise {args.noise:g} dBFS"] if args.noise is not None else [])
    print(f"Evaluating {len(pairs)} files with {', '.join(names)}"
          + (f" ({', '.join(extras)})" if extras else "") + "...")
    try:
        rows = evaluate(pairs, providers, vad=args.vad, noise_db=args.noise)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(report(rows, names, args.verbose))
    if args.csv:
        write_csv(Path(args.csv), rows)
        print(f"\nSaved {len(rows)} rows to {args.csv}")
    return 0 if all(r.error is None for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
