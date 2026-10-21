import numpy as np
import pytest

from audio_format import AudioFormat
from audio_prep import AudioContract
from stt import FakeProvider, SttError
from stt_fakes import silence, tone
from transcribe import main
from wav_io import write_wav


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "speech.wav"
    write_wav(path, AudioFormat(48000, 2),
              np.repeat(np.concatenate([silence(1, 48000), tone(1, 48000), silence(1, 48000)]), 2, axis=1))
    return path


def run(argv, provider, capsys):
    code = main([str(a) for a in argv], make=lambda *a, **kw: provider)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_transcribes_and_reports(wav, capsys):
    fake = FakeProvider("book a ticket")
    code, out, _ = run([wav], fake, capsys)
    assert code == 0
    assert "48000 Hz, 2 ch, 16-bit, 3.00 s" in out
    assert "2 ch -> mono (average); 48000 Hz -> 16000 Hz (resampled)" in out
    assert '"book a ticket"' in out
    assert "connect 10 ms | upload 0 ms | wait 100 ms" in out
    assert "RTF 0.04" in out
    sent = fake.received[0]
    assert sent.fmt == AudioFormat(16000, 1) and len(sent.samples) == 48000


def test_wer_from_ref_and_sidecar(wav, capsys):
    code, out, _ = run([wav, "--ref", "book a train ticket"], FakeProvider("book a ticket"), capsys)
    assert code == 0 and "WER 25.0%" in out and "D=1" in out
    wav.with_suffix(".txt").write_text("Book a ticket.", encoding="utf-8")
    _, out, _ = run([wav], FakeProvider("book a ticket"), capsys)
    assert "WER 0.0%" in out


def test_vad_trims_before_sending(wav, capsys):
    fake = FakeProvider()
    code, out, _ = run([wav, "--vad"], fake, capsys)
    assert code == 0 and "VAD:      kept 1.20 s of 3.00 s (1 speech segment)" in out
    assert fake.received[0].seconds == pytest.approx(1.2)


def test_vad_finds_nothing_in_silence(tmp_path, capsys):
    path = tmp_path / "quiet.wav"
    write_wav(path, AudioFormat(16000, 1), silence(1))
    code, _, err = run([path, "--vad"], FakeProvider(), capsys)
    assert code == 1 and "empty" in err


def test_relabel_experiment_skips_resampling(wav, capsys):
    fake = FakeProvider()
    code, out, _ = run([wav, "--relabel-rate", "16000"], fake, capsys)
    assert code == 0 and "RELABEL" in out and "'lasts' 9.00 s" in out
    assert fake.received[0].seconds == pytest.approx(9.0)  # 3x too long


def test_dry_run_sends_nothing(wav, capsys):
    fake = FakeProvider()
    code, out, _ = run([wav, "--dry-run"], fake, capsys)
    assert code == 0 and "Dry run" in out and "fake request: 3.00 s" in out
    assert fake.received == []


def test_dry_run_real_provider_needs_no_key(wav, capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main([str(wav), "--dry-run", "-l", "en"])
    out = capsys.readouterr().out
    assert code == 0 and "POST https://api.openai.com/v1/audio/transcriptions" in out
    assert "languages[] = en" in out


def test_save_sent(wav, tmp_path, capsys):
    out_path = tmp_path / "sent.wav"
    run([wav, "--save-sent", out_path], FakeProvider(), capsys)
    assert out_path.stat().st_size == 44 + 48000 * 2


def test_api_error_prints_one_line_and_hint(wav, capsys):
    fake = FakeProvider(error=SttError("auth", "key rejected", status=401))
    code, _, err = run([wav], fake, capsys)
    assert code == 1
    assert "error: [auth HTTP 401] key rejected" in err and "hint:" in err


def test_bad_wav(tmp_path, capsys):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav")
    code, _, err = run([bad], FakeProvider(), capsys)
    assert code == 1 and err.startswith("error:")


def test_too_short(tmp_path, capsys):
    path = tmp_path / "blip.wav"
    write_wav(path, AudioFormat(16000, 1), tone(0.05))
    code, _, err = run([path], FakeProvider(contract=AudioContract(16000)), capsys)
    assert code == 1 and "at least 100 ms" in err


def test_empty_transcript_is_reported(wav, capsys):
    _, out, _ = run([wav], FakeProvider(""), capsys)
    assert "(empty: no speech recognised)" in out


@pytest.mark.parametrize("rate", ["0", "-16000"])
def test_relabel_rate_must_be_positive(wav, capsys, rate):
    code, _, err = run([wav, "--relabel-rate", rate], FakeProvider(), capsys)
    assert code == 1 and "--relabel-rate must be > 0" in err
