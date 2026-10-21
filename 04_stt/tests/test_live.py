import numpy as np
import pytest

from audio_format import AudioFormat
from live import APPEND, CLEAR, COMMIT, Gate, main, run, wav_frames
from realtime import RealtimeSession
from stt_fakes import silence, tone
from test_realtime import FakeServer, FakeWs, connect_to
from vad import VadConfig
from wav_io import write_wav

LOUD, QUIET = -20.0, -90.0
FRAME = np.zeros((480, 1), np.int16)


def drive(gate, dbs):
    """Push dB values, return (frame index, action kind) for every action, then flush."""
    out = []
    for i, db in enumerate(dbs):
        out += [(i, a[0]) for a in gate.push(FRAME, db)]
    out += [("flush", a[0]) for a in gate.flush()]
    return out


def test_gate_silence_sends_nothing():
    assert drive(Gate(VadConfig()), [QUIET] * 100) == []


def test_gate_sends_preroll_then_commits_after_min_silence():
    acts = drive(Gate(VadConfig(), preroll_ms=300), [QUIET] * 50 + [LOUD] * 50 + [QUIET] * 50)
    appends = [i for i, kind in acts if kind == APPEND]
    # At frame 50 (first loud): 300 ms pre-roll = 15 old frames + this one.
    assert appends[:16] == [50] * 16
    # Loud ends at 99; 500 ms (25 frames) later, at 124, the turn is committed.
    assert (124, COMMIT) in acts and appends[-1] == 124
    assert len(appends) == 16 + (124 - 50)
    assert [k for _, k in acts].count(COMMIT) == 1


def test_gate_commit_carries_the_segment():
    gate = Gate(VadConfig())
    actions = []
    for db in [QUIET] * 50 + [LOUD] * 50 + [QUIET] * 50:
        actions += gate.push(FRAME, db)
    [(_, seg)] = [a for a in actions if a[0] == COMMIT]
    assert (seg.start_s, seg.end_s) == (pytest.approx(0.9), pytest.approx(2.1))


def test_gate_clears_a_false_start():
    acts = drive(Gate(VadConfig()), [QUIET] * 50 + [LOUD] * 3 + [QUIET] * 50)
    assert (53, CLEAR) in acts and COMMIT not in [k for _, k in acts]
    assert all(i <= 53 for i, k in acts)  # nothing sent after the clear


def test_gate_flush_commits_open_speech():
    acts = drive(Gate(VadConfig()), [QUIET] * 10 + [LOUD] * 30)
    assert acts[-1] == ("flush", COMMIT)


def test_gate_flush_clears_unconfirmed_start():
    acts = drive(Gate(VadConfig()), [QUIET] * 10 + [LOUD] * 3)
    assert acts[-1] == ("flush", CLEAR)


def test_ungated_sends_everything():
    acts = drive(Gate(VadConfig(), gated=False), [QUIET] * 50 + [LOUD] * 50 + [QUIET] * 50)
    assert [i for i, k in acts if k == APPEND] == list(range(150))
    assert (124, COMMIT) in acts


def test_pause_inside_a_sentence_is_one_turn():
    dbs = [QUIET] * 20 + [LOUD] * 30 + [QUIET] * 10 + [LOUD] * 30 + [QUIET] * 40  # 200 ms pause
    assert [k for _, k in drive(Gate(VadConfig()), dbs)].count(COMMIT) == 1


@pytest.fixture
def two_turns_wav(tmp_path):
    path = tmp_path / "two.wav"
    x = np.concatenate([silence(0.5), tone(1), silence(1), tone(1), silence(1)])
    write_wav(path, AudioFormat(16000, 1), x)
    return path


def test_wav_frames_resamples_to_24k(two_turns_wav):
    fmt, frames = wav_frames(str(two_turns_wav), 20)
    frames = list(frames)
    assert fmt == AudioFormat(24000, 1)
    assert len(frames[0]) == 480 and len(frames) == 225  # 4.5 s / 20 ms


def make_session(ws):
    return lambda **kw: RealtimeSession(key="k", connect=connect_to(ws), **kw)


def test_run_two_turns_partials_and_finals(two_turns_wav):
    ws = FakeWs(FakeServer(["book a ticket", "to pune please"]))
    fmt, frames = wav_frames(str(two_turns_wav), 20)
    lines = []
    board = run(frames, fmt, make_session(ws)(), Gate(VadConfig()), paced=False,
                final_timeout=2.0, log=lines.append)
    assert [t.final for t in board.turns] == ["book a ticket", "to pune please"]
    assert [t.partial for t in board.turns] == ["book", "to"]
    t = board.turns[0]
    assert t.speech == (pytest.approx(0.4), pytest.approx(1.6))
    assert t.first_audio_at <= t.first_text_at and t.last_loud_at <= t.commit_at <= t.final_at
    out = "\n".join(lines)
    assert 'turn 1 partial  "book ..."' in out and 'turn 1 FINAL    "book a ticket"' in out
    assert "Full transcript (finals only): book a ticket to pune please" in out
    sent = [e["type"] for e in ws.sent]
    assert sent.count("input_audio_buffer.commit") == 2 and ws.closed


def test_run_reports_missing_final(two_turns_wav):
    ws = FakeWs(FakeServer(["only one"]))  # the second final never comes
    fmt, frames = wav_frames(str(two_turns_wav), 20)
    lines = []
    board = run(frames, fmt, make_session(ws)(), Gate(VadConfig()), paced=False,
                final_timeout=0.3, log=lines.append)
    assert board.turns[1].final is None
    assert "NO FINAL (late or lost)" in "\n".join(lines)


def test_run_survives_a_dropped_connection(two_turns_wav):
    ws = FakeWs(FakeServer(["a", "b"]), break_after=30)
    fmt, frames = wav_frames(str(two_turns_wav), 20)
    lines = []
    run(frames, fmt, make_session(ws)(), Gate(VadConfig()), paced=False,
        final_timeout=0.3, log=lines.append)
    out = "\n".join(lines)
    assert "error: [network]" in out and "Full transcript" in out


def test_run_paces_a_file_in_real_time(two_turns_wav):
    ws = FakeWs(FakeServer(["a", "b"]))
    fmt, frames = wav_frames(str(two_turns_wav), 20)
    waits = []
    run(frames, fmt, make_session(ws)(), Gate(VadConfig()), paced=True,
        final_timeout=1.0, sleep=waits.append, log=lambda s: None)
    # The fake sleep doesn't advance time, so each wait is "how far ahead of
    # the real-time schedule we are": 20 ms more per frame, 4.5 s by the end.
    assert len(waits) == 225
    assert waits[-1] == pytest.approx(4.5, abs=0.25)
    assert np.diff(waits).mean() == pytest.approx(0.02, abs=0.002)


def test_main_wav(two_turns_wav, capsys):
    ws = FakeWs(FakeServer(["one", "two"]))
    code = main(["--wav", str(two_turns_wav), "--fast"], make_session=make_session(ws))
    assert code == 0 and "one two" in capsys.readouterr().out


def test_main_missing_final_exit_code(two_turns_wav, capsys):
    ws = FakeWs(FakeServer(["one"]))
    code = main(["--wav", str(two_turns_wav), "--fast", "--final-timeout", "0.2"],
                make_session=make_session(ws))
    assert code == 1


def test_main_dry_run_needs_no_key(capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["--dry-run", "-l", "en", "--delay", "low"]) == 0
    out = capsys.readouterr().out
    assert '"turn_detection": null' in out and '"rate": 24000' in out and '"delay": "low"' in out
    assert "input_audio_buffer.commit" in out


def test_main_handshake_error(two_turns_wav, capsys):
    def refuse(url, headers, timeout):
        raise OSError("no route to host")

    code = main(["--wav", str(two_turns_wav), "--fast"],
                make_session=lambda **kw: RealtimeSession(key="k", connect=refuse, **kw))
    err = capsys.readouterr().err
    assert code == 1 and "[network]" in err and "hint:" in err


def test_main_bad_wav(tmp_path, capsys):
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"nope")
    assert main(["--wav", str(bad)], make_session=make_session(FakeWs())) == 1


def test_gate_false_start_keeps_preroll_for_the_real_start():
    # A soft onset dips once (a "k" gap) right after starting: the VAD falls
    # back to QUIET and we clear. The real turn must still get its pre-roll.
    gate = Gate(VadConfig(), preroll_ms=300)
    frames = [np.full((480, 1), i, np.int16) for i in range(54)]
    dbs = [QUIET] * 20 + [LOUD] * 3 + [QUIET] + [LOUD] * 30
    sent: list[int] = []
    for frame, db in zip(frames, dbs):
        for a in gate.push(frame, db):
            if a[0] == APPEND:
                sent.append(int(a[1][0, 0]))
            elif a[0] == CLEAR:
                sent.clear()   # the server forgets everything since the last commit
    # Restart at frame 24: the ring still holds the 16 frames before it,
    # including the real onset at 20, so nothing of the word is lost.
    assert sent == list(range(9, 54))


def test_gate_never_resends_committed_audio():
    gate = Gate(VadConfig(), preroll_ms=300)
    dbs = [QUIET] * 20 + [LOUD] * 30 + [QUIET] * 25 + [LOUD] * 30 + [QUIET] * 30
    sent = []
    for i, db in enumerate(dbs):
        frame = np.full((480, 1), i, np.int16)
        sent += [int(a[1][0, 0]) for a in gate.push(frame, db) if a[0] == APPEND]
    assert len(sent) == len(set(sent))  # no frame twice


def test_main_bad_keyword_is_one_error_line(capsys):
    assert main(["--dry-run", "-k", "<b>"]) == 1
    assert "error: keyword" in capsys.readouterr().err


def test_no_gate_hides_first_text_time(two_turns_wav):
    ws = FakeWs(FakeServer(["one", "two"]))
    fmt, frames = wav_frames(str(two_turns_wav), 20)
    lines = []
    run(frames, fmt, make_session(ws)(), Gate(VadConfig(), gated=False), paced=False,
        final_timeout=2.0, log=lines.append)
    row = next(line for line in lines[-1].splitlines() if line.strip().startswith("1 "))
    assert row.split("s   ")[1].strip().startswith("-")
