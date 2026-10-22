from chat import main
from llm import FakeLlm, LlmError, Message, Reply


def run(argv, *replies, inputs=()):
    fake = FakeLlm(*replies)
    feed = list(inputs)

    def read(prompt):
        if not feed:
            raise EOFError
        return feed.pop(0)
    code = main(argv + ["--provider", "fake"], make=lambda name, settings: fake, read=read)
    return code, fake


def test_one_message_prints_reply_and_stats(capsys):
    code, fake = run(["What's the weather like?"], "Which city are you in?")
    out = capsys.readouterr().out
    assert code == 0
    assert "agent> Which city are you in?" in out
    assert "first text" in out and "tokens: in" in out and "5 words" in out
    instructions, messages = fake.calls[0]
    assert "spoken aloud" in instructions          # prompts/assistant.txt by default
    assert messages == [Message("user", "What's the weather like?")]


def test_instructions_override_the_prompt():
    _, fake = run(["hi", "--instructions", "Answer like a pirate."], "Ahoy!")
    assert fake.calls[0][0] == "Answer like a pirate."


def test_dry_run_shows_the_request_and_sends_nothing(capsys):
    code, fake = run(["hi", "--dry-run"])
    assert code == 0 and fake.calls == []
    assert "fake request" in capsys.readouterr().out


def test_dry_run_with_the_real_adapter_needs_no_key(capsys):
    assert main(["What's the weather like?", "--dry-run", "--stream"]) == 0
    out = capsys.readouterr().out
    assert "$OPENAI_API_KEY" in out and '"stream": true' in out and '"effort": "none"' in out


def test_interactive_follow_up_carries_history(capsys):
    _, fake = run([], "Which city?", "Warm in Pune.", "Similar tomorrow.",
                  inputs=["What's the weather?", "Pune", "What about tomorrow?", "/quit"])
    sent = fake.calls[-1][1]
    assert [m.text for m in sent] == ["What's the weather?", "Which city?", "Pune",
                                     "Warm in Pune.", "What about tomorrow?"]
    assert "sent 2 earlier turn(s)" in capsys.readouterr().out


def test_no_history_sends_each_message_alone():
    _, fake = run(["--no-history"], "a", "b", inputs=["one", "two"])
    assert fake.calls[-1][1] == [Message("user", "two")]


def test_history_and_reset_commands(capsys):
    _, fake = run([], "first answer", "fresh", inputs=["first", "/history", "/reset", "again"])
    out = capsys.readouterr().out
    assert "assistant: first answer" in out and "memory cleared" in out
    assert fake.calls[-1][1] == [Message("user", "again")]


def test_stream_prints_pieces_then_stats(capsys):
    code, _ = run(["count", "--stream"], "one two three")
    out = capsys.readouterr().out
    assert code == 0 and "agent> one two three\n" in out


def test_error_is_one_line_with_a_hint_and_exit_1(capsys):
    code, _ = run(["hi"], LlmError("auth", "OPENAI_API_KEY is not set"))
    err = capsys.readouterr().err
    assert code == 1 and "error: [auth]" in err and "hint:" in err


def test_cut_off_stream_is_labelled_and_not_saved(capsys):
    _, fake = run(["--stream"], LlmError("network", "reset", partial_text="Sure, the"), "ok",
                  inputs=["one", "two"])
    captured = capsys.readouterr()
    assert "Sure, the" in captured.out and "NOT a complete answer" in captured.err
    assert fake.calls[-1][1] == [Message("user", "two")]


def test_incomplete_reply_is_shown_but_not_saved(capsys):
    cut = Reply("Once upon a", "fake", "m", status="incomplete", reason="max_output_tokens")
    _, fake = run([], cut, "ok", inputs=["story", "next"])
    assert "[incomplete: max_output_tokens]" in capsys.readouterr().out
    assert fake.calls[-1][1] == [Message("user", "next")]


def test_bad_settings_are_refused_before_any_call(capsys):
    code, fake = run(["hi", "--effort", "low", "--temperature", "0.5"])
    assert code == 1 and fake.calls == [] and "effort none" in capsys.readouterr().err
