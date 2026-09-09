"""Required known-answer tests for unit-cost Levenshtein distance.

These five cases are mandated by the assignment; do not change the expected values.
"""
import pytest

from src.med.levenshtein import med, normalized_med

KNOWN_ANSWERS = [
    ("same", "same", 0),        # identity
    ("", "nlp", 3),             # empty-string boundary
    ("kitten", "sitting", 3),   # mixed substitutions / insertion
    ("flaw", "lawn", 2),        # deletion / insertion
    ("intention", "execution", 5),  # lecture reference case
]


@pytest.mark.parametrize("s,t,expected", KNOWN_ANSWERS)
def test_known_answers(s, t, expected):
    assert med(s, t) == expected


@pytest.mark.parametrize("s,t,expected", KNOWN_ANSWERS)
def test_symmetry(s, t, expected):
    assert med(t, s) == expected


def test_identity_is_zero():
    assert med("robustness", "robustness") == 0


def test_normalized_med():
    assert normalized_med("kitten", "sitting") == pytest.approx(3 / 6)
    assert normalized_med("", "nlp") == pytest.approx(3 / 1)  # max(1, 0) guard
