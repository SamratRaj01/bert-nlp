"""Determinism and correctness tests for the noise module.

Owner: Student A.
"""
from src.noise.operations import apply
from src.noise.manifest import replay
from src.noise.rng import example_rng, example_seed

TEXT = "Robust NLP under character noise, tested carefully."
ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _run(operation, intensity, seed=620, example_id="ex-1"):
    rng = example_rng(seed, example_id)
    return apply(TEXT, operation, intensity, ALPHA, rng)


def test_same_seed_same_output():
    for op in ("insertion", "deletion", "substitution"):
        a_text, a_log = _run(op, 0.2)
        b_text, b_log = _run(op, 0.2)
        assert a_text == b_text
        assert a_log == b_log


def test_different_example_id_diverges():
    rng1 = example_rng(620, "ex-1")
    rng2 = example_rng(620, "ex-2")
    t1, _ = apply(TEXT, "substitution", 0.3, ALPHA, rng1)
    t2, _ = apply(TEXT, "substitution", 0.3, ALPHA, rng2)
    assert t1 != t2


def test_intensity_zero_is_identity():
    for op in ("insertion", "deletion", "substitution"):
        text, log = _run(op, 0.0)
        assert text == TEXT
        assert log == []


def test_replay_matches_output():
    for op in ("insertion", "deletion", "substitution"):
        for intensity in (0.05, 0.15, 0.4):
            text, log = _run(op, intensity)
            assert replay(TEXT, log) == text


def test_substitution_never_noop():
    _, log = _run("substitution", 0.5)
    assert log, "expected at least one substitution at this intensity"
    for edit in log:
        assert edit["replacement_char"] != edit["source_char"]


def test_deletion_shortens_and_insertion_lengthens():
    del_text, del_log = _run("deletion", 0.3)
    ins_text, ins_log = _run("insertion", 0.3)
    assert len(del_text) == len(TEXT) - len(del_log)
    assert len(ins_text) == len(TEXT) + len(ins_log)


def test_unicode_code_point_indexing():
    text = "café résumé naïve"          # multi-byte characters present
    rng = example_rng(1, "u")
    noisy, log = apply(text, "deletion", 0.3, ALPHA, rng)
    assert replay(text, log) == noisy
    # indices refer to code points, so they never exceed len(text)
    assert all(0 <= e["index"] < len(text) for e in log)


def test_example_seed_is_deterministic():
    assert example_seed(620, "ex-1") == example_seed(620, "ex-1")
    assert example_seed(620, "ex-1") != example_seed(620, "ex-2")
