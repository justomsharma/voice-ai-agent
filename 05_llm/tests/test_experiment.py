from experiment import main
from llm import FakeLlm, LlmError


def run(argv, *replies):
    fake = FakeLlm(*replies)
    return main(argv + ["--provider", "fake"], make=lambda name, settings: fake), fake


def test_every_prompt_gets_the_same_question(capsys):
    code, fake = run(["--prompt", "assistant", "--prompt", "detailed", "-q", "Explain DNS"],
                     "It maps names to addresses.", "# DNS\n\n**DNS** is ...")
    out = capsys.readouterr().out
    assert code == 0
    assert [msgs[0].text for _, msgs in fake.calls] == ["Explain DNS", "Explain DNS"]
    assert "spoken aloud" in fake.calls[0][0] and "detail" in fake.calls[1][0]
    assert "Per prompt" in out and "1 of 1" in out        # detailed used markdown


def test_default_runs_all_prompts_on_the_built_in_questions():
    _, fake = run([])
    assert len(fake.calls) == 3 * 4


def test_one_failure_does_not_stop_the_experiment(capsys):
    code, fake = run(["--prompt", "plain", "-q", "a", "-q", "b"], LlmError("server", "busy"), "ok")
    out = capsys.readouterr().out
    assert code == 0 and "ERROR [server]" in out and len(fake.calls) == 2


def test_unknown_prompt_fails_before_any_call(capsys):
    code, fake = run(["--prompt", "nope"])
    assert code == 1 and fake.calls == []
