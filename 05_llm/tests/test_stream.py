from llm import FakeLlm, LlmError
from stream import Trace, main


def run(argv, *replies):
    fake = FakeLlm(*replies)
    return main(argv + ["--provider", "fake"], make=lambda name, settings: fake), fake


def test_trace_times_pieces_and_finds_the_first_sentence():
    trace = Trace(start=0.0)
    trace.pieces = [(0.4, "Sure"), (0.5, "."), (0.6, " It costs 1."), (0.7, "5 dollars.")]
    assert trace.first_s == 0.4 and trace.last_s == 0.7
    assert trace.first_sentence() == (0.6, "Sure.")        # needs the space after the dot
    assert trace.text == "Sure. It costs 1.5 dollars."


def test_last_piece_can_end_the_first_sentence():
    trace = Trace(start=0.0)
    trace.pieces = [(0.4, "Which"), (0.5, " city?")]
    assert trace.first_sentence() == (0.5, "Which city?")


def test_stream_prints_text_summary_and_timeline(capsys):
    code, fake = run(["Explain DNS", "--timeline"], "It maps names to numbers. Like a phone book.")
    out = capsys.readouterr().out
    assert code == 0
    assert "agent> It maps names to numbers. Like a phone book.\n" in out
    assert "first sentence" in out and "'It maps names to numbers.'" in out
    assert "'It'" in out and "' maps'" in out          # timeline shows each piece
    assert len(fake.calls) == 1


def test_compare_runs_both_ways(capsys):
    code, fake = run(["hi", "--compare"], "One. Two.", "One. Two.", "One. Two.")
    out = capsys.readouterr().out
    assert code == 0 and "not streamed" in out and "streamed" in out and len(fake.calls) == 3


def test_runs_show_a_spread(capsys):
    code, fake = run(["hi", "--runs", "3"], "A. B.", "A. B.", "A. B.")
    out = capsys.readouterr().out
    assert code == 0 and len(fake.calls) == 3 and "3 runs:" in out and "median" in out


def test_error_mid_stream_exits_1(capsys):
    code, _ = run(["hi"], LlmError("network", "reset", partial_text="Half"))
    assert code == 1 and "NOT a complete answer" in capsys.readouterr().err
