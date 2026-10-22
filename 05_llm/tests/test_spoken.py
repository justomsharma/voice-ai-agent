import pytest

from spoken import describe, first_sentence_end, has_markdown, speak_seconds, word_count


def test_words_and_speaking_time():
    assert word_count("I can help with that. Which city?") == 7
    assert speak_seconds("one two three four five") == 2.0


@pytest.mark.parametrize("text", ["**Bold** answer", "# Weather", "- first\n- second",
                                  "Steps:\n1. Open the app", "Use `pip`", "See [docs](http://x)"])
def test_markdown_is_detected(text):
    assert has_markdown(text)


@pytest.mark.parametrize("text", ["It costs 1.5 dollars.", "Count: 1, 2, 3.",
                                  "A well-known place - really.", "Yes! Which city?"])
def test_plain_speech_is_not_markdown(text):
    assert not has_markdown(text)


def test_first_sentence_end():
    assert first_sentence_end("Sure. Which city?") == 5
    assert first_sentence_end("It costs 1.5 dollars") is None   # a decimal point isn't an ending
    assert first_sentence_end("Which city?") == 11


def test_describe_flags_markdown():
    assert "MARKDOWN" in describe("**Hi**") and "MARKDOWN" not in describe("Hi.")
