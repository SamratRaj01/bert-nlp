"""Tests for the Assignment 2 default noise rule (a2_default).

Owner: Samrat. Phase 2 step A8.
"""
from src.med.fast import med_fast
from src.med.levenshtein import med
from src.noise.rng import example_rng
from src.noise.rule_v2 import apply_v2, eligible_positions, replay_v2

TEXT = "The M @-@ 82 highway , built in 1994 , runs 12 @.@ 5 km ; Über-route A."
ALPHA = "abcdefghijklmnopqrstuvwxyz"
SPANS = ["@-@", "@,@", "@.@"]
OPS = ("insertion", "deletion", "substitution")


def _run(op, p, seed=11, key="wt2-valid-000001|op|p", text=TEXT):
    return apply_v2(text, op, p, ALPHA, example_rng(seed, key), SPANS)


def test_p_zero_is_identity():
    for op in OPS:
        noisy, log, eligible = _run(op, 0.0)
        assert noisy == TEXT and log == []
        assert eligible == len(eligible_positions(TEXT, SPANS))


def test_same_seed_same_text_and_log():
    for op in OPS:
        assert _run(op, 0.2) == _run(op, 0.2)


def test_different_seed_diverges():
    assert _run("substitution", 0.3, seed=11)[0] != _run("substitution", 0.3, seed=23)[0]


def test_replay_reproduces_output():
    for op in OPS:
        for p in (0.01, 0.05, 0.10, 0.20, 0.6):
            noisy, log, _ = _run(op, p)
            assert replay_v2(TEXT, log) == noisy


def test_only_ascii_letters_are_edited():
    for op in OPS:
        _, log, _ = _run(op, 0.9)
        assert log
        for e in log:
            ch = TEXT[e["index"]]
            assert ch.isascii() and ch.isalpha(), (op, e)


def test_digits_punctuation_spaces_sentinels_preserved():
    noisy, _, _ = _run("deletion", 1.0)          # delete every eligible letter
    assert noisy == "".join(c for c in TEXT if not (c.isascii() and c.isalpha()))
    for span in SPANS[:1] + SPANS[2:]:
        assert span in noisy
    assert "1994" in noisy and "Ü" in noisy      # non-ASCII letters are not eligible


def test_insertion_goes_after_the_eligible_char():
    noisy, log, _ = _run("insertion", 1.0, text="ab")
    assert len(log) == 2
    assert noisy == "a" + log[0]["replacement_char"] + "b" + log[1]["replacement_char"]


def test_inserted_chars_are_not_recorrupted():
    noisy, log, eligible = _run("insertion", 1.0)
    assert len(log) == eligible                  # exactly one draw per ORIGINAL position
    assert len(noisy) == len(TEXT) + eligible


def test_substitution_never_maps_to_same_letter_either_case():
    _, log, _ = _run("substitution", 1.0)
    for e in log:
        assert e["replacement_char"] != e["source_char"].lower()
        assert e["replacement_char"] in ALPHA


def test_empty_output_retained():
    noisy, log, eligible = _run("deletion", 1.0, text="Abc")
    assert noisy == "" and len(log) == eligible == 3
    assert med_fast("Abc", noisy) == 3


def test_med_not_greater_than_edit_count():
    for op in OPS:
        noisy, log, _ = _run(op, 0.3)
        assert med(TEXT, noisy) <= len(log)
