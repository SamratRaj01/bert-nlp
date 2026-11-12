import numpy as np

from src.geometry.spans import noisy_span, reassemble_record, select_spans, whitespace_spans
from src.noise.rule_v2 import render_v2


def _log(*edits):
    return [dict(zip(("op", "index", "source_char", "replacement_char"), e)) for e in edits]


def test_whitespace_spans_offsets():
    assert whitespace_spans("  ab  c ") == [(2, 4), (6, 7)]


def test_select_spans_skips_letterless_and_respects_cap():
    rec = [{"record_id": "r1", "article_id": "a", "text": "3 @.@ 5 cat , dog ( 24 )"}]
    spans = select_spans(rec, cap=10, protected_spans=["@.@"])
    assert [s.text for s in spans] == ["cat", "dog"]
    assert [s.span_index for s in spans] == [3, 5]       # index counts ALL spans, kept or not
    assert len(select_spans(rec, cap=1)) == 1


def test_fully_deleted_span_is_explicit_empty_string():
    clean = "ab cd"
    log = _log(("deletion", 0, "a", None), ("deletion", 1, "b", None))
    assert noisy_span(clean, log, 0, 2) == ""
    assert reassemble_record(clean, log) == render_v2(clean, log) == " cd"


def test_insertion_goes_after_letter_inside_same_span():
    clean = "ab cd"
    log = _log(("insertion", 1, None, "x"))
    assert noisy_span(clean, log, 0, 2) == "abx"
    assert noisy_span(clean, log, 3, 5) == "cd"


def test_reassemble_matches_full_render_for_mixed_edits():
    clean = "The cat sat , 24 mats ."
    log = _log(("substitution", 0, "T", "q"), ("deletion", 5, "a", None),
               ("insertion", 8, None, "z"), ("deletion", 17, "m", None))
    assert reassemble_record(clean, log) == render_v2(clean, log)


# --- G4 pooling (uses a tiny fake tokenizer and table, no model download) ---------------------
from src.geometry.pooling import pool_span


class _FakeTok:
    vocab = {"a": 0, "##b": 1, "c": 2}
    inv = {v: k for k, v in vocab.items()}

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [self.vocab[p] for p in text.split()]}

    def convert_ids_to_tokens(self, ids):
        return [self.inv[i] for i in ids]


def test_pool_is_plain_mean_of_rows():
    E = np.array([[1., 0.], [3., 4.], [0., 2.]], dtype=np.float32)
    p = pool_span("a ##b", _FakeTok(), E)
    assert p.pieces == ["a", "##b"] and np.allclose(p.vector, [2., 2.])


def test_pool_empty_span_has_no_vector():
    p = pool_span("", _FakeTok(), np.zeros((3, 2), dtype=np.float32))
    assert p.vector is None and p.n_pieces == 0


# --- G5 metrics ---------------------------------------------------------------------------------
import math

from src.geometry.metrics import pair_metrics


def test_metrics_known_values():
    c, d, ok = pair_metrics(np.array([1., 0.]), np.array([0., 1.]))
    assert ok and abs(c) < 1e-12 and abs(d - math.sqrt(2)) < 1e-12
    c, d, ok = pair_metrics(np.array([3., 4.]), np.array([6., 8.]))   # same direction, different length
    assert ok and abs(c - 1) < 1e-12 and abs(d - 5) < 1e-12


def test_identical_vectors_give_cosine_one_distance_zero():
    v = np.random.default_rng(0).normal(size=768).astype(np.float32)
    c, d, ok = pair_metrics(v, v.copy())
    assert ok and abs(c - 1) < 1e-9 and d == 0


def test_undefined_pairs_are_nan_not_zero():
    z = np.zeros(4)
    for u, v in [(None, np.ones(4)), (np.ones(4), None), (z, np.ones(4)), (np.ones(4), z)]:
        c, d, ok = pair_metrics(u, v)
        assert not ok and math.isnan(c) and math.isnan(d)


def test_unit_vectors_squared_distance_is_two_minus_two_cos():
    r = np.random.default_rng(1); u, v = r.normal(size=8), r.normal(size=8)
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    c, d, _ = pair_metrics(u, v)
    assert abs(d**2 - 2 * (1 - c)) < 1e-12


# --- G6/G7 audit and seed summary (synthetic pairs, no model needed) ---------------------------------
import pandas as pd

from src.geometry.drift import audit_table, parse_condition, seed_summary


def _pairs():
    rows = []
    for cid, vals in {"clean": [(1., 0., True), (1., 0., True)],
                      "del-p0.10-s11": [(0.5, 1., True), (float("nan"), float("nan"), False)],
                      "del-p0.10-s23": [(0.7, 0.5, True), (1., 0., True)]}.items():
        for k, (c, d, ok) in enumerate(vals):
            noisy = "x" if (cid != "clean" and c != 1.) else "same"
            rows.append((cid, f"s{k}", "same", "" if not ok else noisy, c, d, ok))
    return pd.DataFrame(rows, columns=["condition_id", "span_id", "clean_span", "noisy_span", "cosine", "dist", "valid"])


def test_parse_condition():
    assert parse_condition("clean") == ("clean", 0.0, None)
    assert parse_condition("sub-p0.05-s47") == ("substitution", 0.05, 47)


def test_audit_counts_and_valid_only_means():
    a = audit_table(_pairs()).set_index("condition_id")
    r = a.loc["del-p0.10-s11"]
    assert (r.selected, r.valid, r.undefined) == (2, 1, 1) and r.coverage == 0.5
    assert r.mean_cosine == 0.5                      # the undefined pair is excluded, not counted as 0
    assert a.loc["clean"].mean_cosine == 1 and a.loc["clean"].mean_distance == 0


def test_seed_summary_min_max_and_clean_excluded():
    s = seed_summary(audit_table(_pairs()))
    assert len(s) == 1 and s.operation[0] == "deletion" and s.n_seeds[0] == 2
    assert abs(s.mean_cosine_min[0] - 0.5) < 1e-12 and abs(s.mean_cosine_max[0] - 0.85) < 1e-12
