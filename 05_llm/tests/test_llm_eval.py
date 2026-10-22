import json

import pytest

from llm import FakeLlm, LlmError, Message, Reply
from llm_eval import (Case, load_cases, main, model_specs, percentile, run_case, run_checks)


def checks(spec, text):
    return {name: ok for name, ok, _ in run_checks(spec, text)}


def test_max_words_and_markdown():
    assert checks({"max_words": 3, "no_markdown": True}, "One two three.") == \
        {"max_words": True, "no_markdown": True}
    assert checks({"max_words": 2, "no_markdown": True}, "- one\n- two") == \
        {"max_words": False, "no_markdown": False}


def test_asks_question_both_ways():
    assert checks({"asks_question": True}, "Which city?") == {"asks_question": True}
    assert checks({"asks_question": True}, "It's sunny.") == {"asks_question": False}
    assert checks({"asks_question": False}, "Which city?") == {"asks_question": False}


def test_include_checks_ignore_case_and_curly_quotes():
    spec = {"must_include_any": ["can't"], "must_not_include": ["booked"]}
    assert checks(spec, "Sorry, I can’t do that.") == {"must_include_any": True,
                                                          "must_not_include": True}
    assert checks(spec, "Done, I BOOKED it.") == {"must_include_any": False,
                                                 "must_not_include": False}


def test_no_digits():
    assert checks({"no_digits": True}, "about three hours") == {"no_digits": True}
    assert checks({"no_digits": True}, "Call 020 5555 1234") == {"no_digits": False}


def test_the_real_eval_set_loads_and_covers_every_category():
    cases = load_cases()
    assert len(cases) >= 10
    assert {c.category for c in cases} == {"instruction", "concise", "followup",
                                          "ambiguity", "restraint"}
    followups = [c for c in cases if c.category == "followup"]
    assert all(c.history and c.history[0].role == "user" for c in followups)


def test_unknown_check_is_refused(tmp_path):
    f = tmp_path / "cases.json"
    f.write_text(json.dumps({"cases": [{"id": "x", "category": "concise", "input": "hi",
                                        "checks": {"max_wrds": 3}}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="max_wrds"):
        load_cases(f)


def test_followup_case_sends_its_history_unless_told_not_to():
    case = Case("name", "followup", "What's my name?",
                (Message("user", "I'm Asha."), Message("assistant", "Hi Asha.")),
                {"must_include_any": ["asha"]})
    fake = FakeLlm("Your name is Asha.", "I don't know your name.")
    assert run_case(fake, "", case).passed
    assert not run_case(fake, "", case, history=False).passed
    assert [len(msgs) for _, msgs in fake.calls] == [3, 1]


def test_incomplete_or_failed_reply_never_passes():
    case = Case("c", "concise", "hi", checks={"max_words": 50})
    cut = Reply("Once upon", "fake", "m", status="incomplete", reason="max_output_tokens")
    fake = FakeLlm(cut, LlmError("server", "busy"))
    assert not run_case(fake, "", case).passed
    r = run_case(fake, "", case)
    assert not r.passed and "server" in r.error


def test_percentile_nearest_rank():
    v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert percentile(v, 50) == 0.5 and percentile(v, 90) == 0.9
    assert percentile([], 50) is None


def test_model_specs_take_an_optional_effort():
    assert model_specs(None, "gpt-6-luna", "none") == [("gpt-6-luna", "none")]
    assert model_specs(["gpt-6-luna", "gpt-6.1-sol:low"], "x", "none") == \
        [("gpt-6-luna", "none"), ("gpt-6.1-sol", "low")]


def test_cli_runs_selected_cases_and_summarises(capsys):
    fake = FakeLlm("Your name is Asha.", "It takes about three hours.", "Pune should be similar.")
    code = main(["--provider", "fake", "--category", "followup"], make=lambda n, s: fake)
    out = capsys.readouterr().out
    assert code == 0 and len(fake.calls) == 3
    assert "3/3 passed" in out and "followup     3/3" in out and "p50" in out


def test_cli_compares_models(capsys):
    seen = []

    def make(name, settings):
        seen.append((settings.model, settings.effort))
        return FakeLlm(settings=settings)
    main(["--provider", "fake", "--category", "concise", "--model", "a", "--model", "b:low"],
         make=make)
    assert seen == [("a", "none"), ("b", "low")]
