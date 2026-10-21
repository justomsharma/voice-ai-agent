"""Word error rate: how wrong is a transcript, compared to what was really said?

Line the two word lists up with the fewest edits (Levenshtein distance on
words), then count the three kinds of mistakes:

    REF: i want to book a  ticket ****
    HYP: i want to book an ticket now
                        S         I

    S  substitution  a word heard wrong        "a" -> "an"
    D  deletion      a word missed             (quiet words, chopped edges)
    I  insertion     a word that wasn't said   (noise, hallucinated filler)

    WER = (S + D + I) / number of reference words

So WER can be over 100%: a model that invents a whole extra sentence has
more insertions than there were words. 5% means 1 mistake per 20 words.

Normalise first, or you measure formatting, not hearing. "Ticket." vs
"ticket" isn't a hearing error. We lowercase and strip punctuation, and
work on Unicode, so Devanagari (Hindi) text works too. What we don't
fix: "10" vs "ten", "e-mail" vs "email", British vs US spelling. Real
benchmarks use bigger normalisers (Whisper's EnglishTextNormalizer) for
exactly this.

CER (character error rate) is the same idea on letters. It's kinder to
spelling variants in the SAME script: "Pune" vs "Poona", "kya" vs "kia".
It can't help across scripts: "kya" vs "क्या" shares no letters, so it
scores ~100%. Hinglish may come back romanised or in Devanagari, so write
your reference in the script the model outputs (or transliterate one side
first). Also, CER counts Unicode code points, and one Devanagari syllable
like "क्या" is 4 of them (क ् य ा), so Hindi CER isn't comparable to English CER.
"""

import codecs
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

OK, SUB, DEL, INS = "ok", "S", "D", "I"


def normalize(text: str) -> str:
    """Lowercase, punctuation to spaces (keeping in-word apostrophes), squash spaces."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    out = []
    for i, ch in enumerate(text):
        if unicodedata.category(ch).startswith("P"):
            # Keep "don't", drop the quotes in "'hello'".
            inner = ch == "'" and 0 < i < len(text) - 1 and text[i - 1].isalnum() and text[i + 1].isalnum()
            out.append(ch if inner else " ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def read_reference(path: str | Path) -> str:
    """A reference transcript file. UTF-8 normally, but PowerShell's `echo > x.txt`
    writes UTF-16, and Notepad may add a BOM, so accept those too."""
    data = Path(path).read_bytes()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16").strip()
    return data.decode("utf-8-sig").strip()


@dataclass
class Score:
    ref_len: int
    sub: int = 0
    dele: int = 0
    ins: int = 0
    ops: list[tuple[str, str, str]] = field(default_factory=list)  # (op, ref, hyp)

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def hits(self) -> int:
        return self.ref_len - self.sub - self.dele

    @property
    def rate(self) -> float:
        """errors / reference length. With an empty reference (a silence
        test), any output at all is infinitely wrong."""
        if self.ref_len == 0:
            return 0.0 if self.errors == 0 else float("inf")
        return self.errors / self.ref_len

    def __add__(self, other: "Score") -> "Score":
        """Corpus totals: sum the counts, THEN divide. Averaging per-file
        WERs would let a 2-word clip weigh as much as a 200-word one."""
        return Score(self.ref_len + other.ref_len, self.sub + other.sub,
                     self.dele + other.dele, self.ins + other.ins)


def align(ref: list[str], hyp: list[str]) -> Score:
    """Minimum-edit alignment of two token lists (classic dynamic programming).

    cost[i][j] = fewest edits turning ref[:i] into hyp[:j]. Each cell comes
    from: diagonal (match or substitute), up (delete ref[i]), left (insert hyp[j]).
    """
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            cost[i][j] = min(diag, cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    # Walk back from the corner to recover which edits were made.
    score = Score(n)
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            op = OK if ref[i - 1] == hyp[j - 1] else SUB
            score.ops.append((op, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            score.ops.append((DEL, ref[i - 1], ""))
            i -= 1
        else:
            score.ops.append((INS, "", hyp[j - 1]))
            j -= 1
    score.ops.reverse()
    score.sub = sum(op == SUB for op, _, _ in score.ops)
    score.dele = sum(op == DEL for op, _, _ in score.ops)
    score.ins = sum(op == INS for op, _, _ in score.ops)
    return score


def wer(reference: str, hypothesis: str) -> Score:
    return align(normalize(reference).split(), normalize(hypothesis).split())


def cer(reference: str, hypothesis: str) -> Score:
    """Characters, ignoring spaces (word boundaries are a WER question)."""
    return align(list(normalize(reference).replace(" ", "")),
                 list(normalize(hypothesis).replace(" ", "")))


def format_alignment(score: Score) -> str:
    """Three lines: REF, HYP and the error markers under each column."""
    ref, hyp, marks = ["REF:"], ["HYP:"], ["    "]
    for op, r, h in score.ops:
        width = max(len(r), len(h), 1)
        ref.append((r or "*" * width).ljust(width))
        hyp.append((h or "*" * width).ljust(width))
        marks.append(("" if op == OK else op).ljust(width))
    return "\n".join(" ".join(row).rstrip() for row in (ref, hyp, marks))


def summary(score: Score) -> str:
    return (f"{score.rate:.1%}  ({score.errors} errors / {score.ref_len} words: "
            f"S={score.sub} D={score.dele} I={score.ins})")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        sys.exit('usage: python 04_stt/wer.py "reference text" "hypothesis text"')
    s = wer(sys.argv[1], sys.argv[2])
    print(format_alignment(s))
    print(f"\nWER {summary(s)}")
    print(f"CER {summary(cer(sys.argv[1], sys.argv[2])).replace('words', 'chars')}")
