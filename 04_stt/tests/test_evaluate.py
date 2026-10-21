import csv

import numpy as np
import pytest

from audio_format import AudioFormat
from evaluate import category, find_pairs, main
from stt import FakeProvider, SttError
from stt_fakes import silence, tone
from wav_io import write_wav


class ByFile(FakeProvider):
    """Answers per file length, so different files get different transcripts."""

    def __init__(self, name, answers, error=None):
        super().__init__(error=error)
        self.name = name
        self.answers = answers  # seconds (rounded) -> text

    def transcribe(self, audio):
        t = super().transcribe(audio)
        t.text = self.answers[round(audio.seconds)]
        return t


@pytest.fixture
def folder(tmp_path):
    for name, secs, ref in [("clean_01", 1, "book a ticket"),
                            ("names_01", 2, "call ravi in pune"),
                            ("orphan", 3, None)]:
        write_wav(tmp_path / f"{name}.wav", AudioFormat(16000, 1),
                  np.concatenate([tone(secs), silence(0.2)]))
        if ref:
            (tmp_path / f"{name}.txt").write_text(ref, encoding="utf-8")
    return tmp_path


def run(argv, providers, capsys):
    by_name = {p.name: p for p in providers}
    code = main([str(a) for a in argv], make=lambda name, **kw: by_name[name])
    return code, capsys.readouterr().out


def test_find_pairs_and_category(folder):
    pairs, missing = find_pairs(folder)
    assert [(w.name, r) for w, r in pairs] == [("clean_01.wav", "book a ticket"),
                                               ("names_01.wav", "call ravi in pune")]
    assert [m.name for m in missing] == ["orphan.wav"]
    assert category(pairs[1][0]) == "names"
    assert category(folder / "x.wav") == "-"


def test_compares_providers_with_corpus_wer(folder, capsys, tmp_path):
    a = ByFile("openai", {1: "book a ticket", 2: "call ravi in pune"})
    b = ByFile("deepgram", {1: "book the ticket", 2: "call ravi pune"})
    code, out = run([folder, "-p", "openai", "-p", "deepgram", "--csv", tmp_path / "r.csv"], [a, b], capsys)
    assert code == 0
    assert "skipping orphan.wav" in out
    # deepgram: 1 sub of 3 + 1 del of 4 = 2 errors / 7 words (not the mean of 33% and 25%)
    assert "openai     WER   0.0%" in out
    assert "deepgram   WER  28.6%  S=1 D=1 I=0 of 7 words" in out
    assert "clean 33.3%  names 25.0%" in out
    rows = list(csv.DictReader(open(tmp_path / "r.csv", encoding="utf-8")))
    assert len(rows) == 4 and rows[1]["provider"] == "deepgram" and rows[1]["sub"] == "1"


def test_verbose_shows_alignments(folder, capsys):
    p = ByFile("openai", {1: "book the ticket", 2: "call ravi in pune"})
    _, out = run([folder, "-v"], [p], capsys)
    assert "clean_01.wav / openai:\nREF: book a   ticket" in out


def test_auth_error_stops_that_provider(folder, capsys):
    bad = ByFile("openai", {}, error=SttError("auth", "no key"))
    code, out = run([folder], [bad], capsys)
    assert code == 1
    assert "names_01.wav / openai: skipped after auth error" in out
    assert len(bad.received) == 1  # didn't hammer the API with a bad key


def test_noise_and_vad_options(folder, capsys):
    p = ByFile("openai", {1: "book a ticket", 2: "call ravi in pune"})
    code, out = run([folder, "--noise", "-50", "--vad"], [p], capsys)
    assert code == 0 and "(VAD trim, noise -50 dBFS)" in out


def test_bad_folder(tmp_path, capsys):
    code = main([str(tmp_path / "nope")], make=lambda *a, **k: None)
    assert code == 1 and "not a folder" in capsys.readouterr().err
    code = main([str(tmp_path)], make=lambda *a, **k: None)
    assert code == 1 and "no X.wav + X.txt pairs" in capsys.readouterr().err
