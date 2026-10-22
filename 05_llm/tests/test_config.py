import pytest

from config import Settings, load_prompt, prompt_names


def test_defaults_are_voice_friendly():
    s = Settings().check()
    assert s.model == "gpt-6-luna"
    assert s.effort == "none"          # no hidden reasoning before the first word
    assert s.temperature is None       # not sent unless asked for


def test_temperature_needs_effort_none():
    Settings(effort="none", temperature=0.3).check()
    with pytest.raises(ValueError, match="effort none"):
        Settings(effort="low", temperature=0.3).check()
    with pytest.raises(ValueError, match="effort none"):
        Settings(effort=None, temperature=0.3).check()   # model default is medium


@pytest.mark.parametrize("bad", [dict(effort="fast"), dict(verbosity="tiny"),
                                 dict(max_output_tokens=5), dict(temperature=3.0),
                                 dict(timeout_s=0)])
def test_bad_settings_refused(bad):
    with pytest.raises(ValueError):
        Settings(**bad).check()


def test_prompts_by_name_and_path(tmp_path):
    assert {"assistant", "detailed", "plain"} <= set(prompt_names())
    assert "spoken aloud" in load_prompt("assistant")
    f = tmp_path / "mine.txt"
    f.write_text("  Talk like a pirate.\n", encoding="utf-8")
    assert load_prompt(str(f)) == "Talk like a pirate."


def test_unknown_prompt_lists_choices():
    with pytest.raises(ValueError, match="assistant"):
        load_prompt("nope")
