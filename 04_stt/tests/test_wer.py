import math

import pytest

from wer import Score, align, cer, format_alignment, normalize, read_reference, summary, wer


def test_normalize_case_and_punctuation():
    assert normalize("  Hello,   World! It's  “fine”. ") == "hello world it's fine"


def test_normalize_curly_apostrophe_and_quotes():
    assert normalize("Don’t 'quote' me") == "don't quote me"


def test_normalize_devanagari_keeps_marks_drops_danda():
    assert normalize("मुझे टिकट चाहिए।") == "मुझे टिकट चाहिए"


def test_identical_is_zero():
    s = wer("I want to book a ticket.", "i want to book a ticket")
    assert (s.errors, s.rate, s.hits) == (0, 0.0, 6)


def test_counts_each_error_kind():
    s = wer("i want to book a ticket", "i want book an ticket now")
    assert (s.sub, s.dele, s.ins) == (1, 1, 1)
    assert s.rate == pytest.approx(3 / 6)
    assert s.hits == 4


def test_alignment_lines():
    s = wer("i want to book a ticket", "i want book an ticket now")
    assert format_alignment(s) == (
        "REF: i want to book a  ticket ***\n"
        "HYP: i want ** book an ticket now\n"
        "            D       S         I")


def test_wer_can_exceed_100_percent():
    s = wer("yes", "yes thank you for watching")
    assert s.ins == 4 and s.rate == 4.0


def test_empty_reference():
    assert wer("", "").rate == 0.0
    assert math.isinf(wer("", "thank you").rate)  # hallucination on silence


def test_empty_hypothesis_is_all_deletions():
    s = wer("book a ticket", "")
    assert (s.dele, s.rate) == (3, 1.0)


def test_cer_is_kinder_to_spelling():
    assert wer("pune", "poona").rate == 1.0
    assert cer("pune", "poona").rate == pytest.approx(3 / 4)


def test_corpus_sums_before_dividing():
    short = wer("yes", "no")  # 1/1 = 100%
    long = wer(" ".join(["word"] * 99), " ".join(["word"] * 99))  # 0/99
    total = short + long
    assert total.rate == pytest.approx(1 / 100)  # not (100% + 0%) / 2 = 50%


def test_align_works_on_any_tokens():
    s = align(list("abc"), list("abd"))
    assert (s.sub, s.ref_len) == (1, 3)


def test_summary_text():
    assert summary(Score(4, sub=1)) == "25.0%  (1 errors / 4 words: S=1 D=0 I=0)"


def test_read_reference_any_common_encoding(tmp_path):
    for i, (enc, text) in enumerate([("utf-8", "बुक करो\n"), ("utf-8-sig", "book it"),
                                     ("utf-16", "book it\r\n")]):  # PowerShell echo >
        f = tmp_path / f"{i}.txt"
        f.write_bytes(text.encode(enc))
        assert read_reference(f) == text.strip()


def test_cer_cannot_bridge_scripts():
    # Same word, two scripts: CER is no help. Match the reference's script.
    assert cer("kya", "kia").rate == pytest.approx(1 / 3)
    assert cer("kya", "क्या").rate >= 1.0
