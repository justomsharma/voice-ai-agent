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
    with open(tmp_path / "r.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
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


def test_failed_files_count_as_all_missed(folder, capsys, monkeypatch):
    monkeypatch.setattr("stt.time.sleep", lambda s: None)
    class FailsOnNames(ByFile):
        def transcribe(self, audio):
            if round(audio.seconds) == 2:
                raise SttError("server", "boom", retryable=True)
            return super().transcribe(audio)

    p = FailsOnNames("openai", {1: "book a ticket"})
    code, out = run([folder], [p], capsys)
    # 0 errors on 3 words + 4 missed words = 4/7, not 0/3 = 0%
    assert code == 1 and "WER  57.1%" in out and "counted as all words missed" in out


def test_everything_failed_shows_na(folder, capsys):
    _, out = run([folder], [ByFile("openai", {}, error=SttError("network", "down"))], capsys)
    assert "WER    n/a" in out


def test_temporary_429_does_not_stop_the_provider(folder, capsys, monkeypatch):
    monkeypatch.setattr("stt.time.sleep", lambda s: None)  # skip the backoff waits
    p = ByFile("openai", {}, error=SttError("rate_limit", "slow down", retryable=True))
    run([folder], [p], capsys)
    assert len(p.received) == 2 * 3  # every file tried, each with 2 retries


def test_duplicate_provider_runs_once(folder, capsys):
    p = ByFile("openai", {1: "book a ticket", 2: "call ravi in pune"})
    code, out = run([folder, "-p", "openai", "-p", "openai"], [p], capsys)
    assert code == 0 and len(p.received) == 2


def test_keywords_reach_the_provider(folder, capsys):
    seen = {}
    p = ByFile("openai", {1: "book a ticket", 2: "call ravi in pune"})

    def make(name, **kw):
        seen.update(kw)
        return p

    assert main([str(folder), "-k", "Ravi", "--prompt", "train booking", "--timeout", "5"],
                make=make) == 0
    assert seen["keywords"] == ["Ravi"] and seen["prompt"] == "train booking" and seen["timeout"] == 5
    assert "keywords Ravi" in capsys.readouterr().out
