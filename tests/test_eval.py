"""Tests for fast MED, the condition grid, diagnostics and aggregation.

Owner: both. Phase 2 (plan §5 A5-A6, §8).
"""
import random

import pandas as pd
import pytest

from src.eval.coverage import pooled, record_diagnostics
from src.eval.summarize import scores, summary
from src.lm.vocab import Vocab
from src.med.fast import med_fast, normalized_med_fast
from src.med.levenshtein import med
from src.noise.grid import condition_id, conditions
from src.phase2_config import load_phase2_config

KNOWN = [("same", "same", 0), ("", "nlp", 3), ("kitten", "sitting", 3),
         ("flaw", "lawn", 2), ("intention", "execution", 5)]


def test_fast_med_known_answers():
    for s, t, expected in KNOWN:
        assert med_fast(s, t) == expected == med(s, t)
    assert normalized_med_fast("kitten", "sitting") == pytest.approx(3 / 6)


def test_fast_med_matches_reference_on_random_pairs():
    rng = random.Random(620)
    for _ in range(1000):
        a = "".join(rng.choice("ab c@") for _ in range(rng.randint(0, 40)))
        b = "".join(rng.choice("ab c@") for _ in range(rng.randint(0, 40)))
        assert med_fast(a, b) == med(a, b), (a, b)


def test_grid_has_37_conditions_with_unique_ids():
    cfg = load_phase2_config()
    conds = conditions(cfg)
    assert len(conds) == 37 and conds[0]["condition_id"] == "clean"
    assert len({c["condition_id"] for c in conds}) == 37
    assert condition_id("deletion", 0.05, 23) == "del-p0.05-s23"
    assert condition_id("insertion", 0.0, 11) == "clean"


def test_zero_token_output_is_flagged_not_dropped():
    v = Vocab("ws", ["the", "<UNK>", "</s>"], unk="<UNK>")
    d = record_diagnostics("the cat", [], clean_n_tokens=2, vocab=v, tokenizer="whitespace")
    assert d["n_tokens"] == 0 and d["rate_undefined"] is True
    assert d["unk_rate"] is None and d["pieces_per_span"] == 0.0
    assert d["token_count_change"] == -2


def test_diagnostics_exclude_boundaries_and_count_pieces_per_span():
    v = Vocab("wp", ["[UNK]", "robust", "##ness", "</s>"], unk="[UNK]")
    d = record_diagnostics("robustness rules", ["robust", "##ness", "[UNK]"], 2, v, "wordpiece")
    assert d["n_tokens"] == 3 and d["n_unk"] == 1 and d["n_continuation"] == 1
    assert d["pieces_per_span"] == pytest.approx(1.5)
    assert pooled([d])["unk_rate"] == pytest.approx(1 / 3)


def test_scores_aggregate_before_dividing_and_summary_counts():
    rows = []
    for cond, op, p, seed, bits in [("clean", "none", 0.0, -1, (2.0, 6.0)),
                                    ("del-p0.05-s11", "deletion", 0.05, 11, (4.0, 4.0)),
                                    ("del-p0.05-s23", "deletion", 0.05, 23, (3.0, 9.0))]:
        for rid, b in zip(("r1", "r2"), bits):
            rows.append({"train_set": "t", "tokenizer": "whitespace", "model": "q1",
                         "condition_id": cond, "operation": op, "p": p, "seed": seed,
                         "record_id": rid, "nll_bits_sum": b, "nll_nats_sum": b * 0.6931471805599453,
                         "T": 2 if rid == "r1" else 4})
    sc = scores(pd.DataFrame(rows))
    clean = sc[sc.condition_id == "clean"].iloc[0]
    assert clean["H_bits"] == pytest.approx((2.0 + 6.0) / 6)       # not mean of record losses
    sm = summary(sc)
    assert len(sm) == 2                                              # 1 clean + 1 noisy cell
    noisy = sm[sm.operation == "deletion"].iloc[0]
    assert noisy["n_seeds"] == 2
    assert noisy["H_bits_min"] == pytest.approx(8.0 / 6) and noisy["H_bits_max"] == pytest.approx(12.0 / 6)


def test_loss_decomposition_sums_to_corpus_loss():
    from src.eval.decompose import loss_decomposition
    from src.lm.counts import NgramCounts
    from src.lm.models import AddOneLM
    from src.lm.score import Engine, corpus_loss, record_losses
    from src.lm.vocab import whitespace_vocab

    class Split:
        name = "whitespace"
        def tokenize_batch(self, texts):
            return [t.split() for t in texts]

    train = [["the", "cat", "sat"], ["the", "dog", "sat"], ["a", "cat"]]
    v, stats = whitespace_vocab(train, min_count=2)          # 'dog', 'a' -> <UNK>
    c = NgramCounts(v).fit(train)
    eng = Engine("toy", "whitespace", Split(), v, stats, c, AddOneLM(c))
    rows = {"clean": [{"noisy_text": "the cat sat", "operation": None, "p": 0.0, "seed": None},
                      {"noisy_text": "a zebra sat", "operation": None, "p": 0.0, "seed": None}]}
    dec = loss_decomposition(eng, (0.5, 0.5, 0.0), rows)
    sums = [record_losses(eng.lm.q_arrays(v.map_tokens(t.split())), "p_mix", (0.5, 0.5, 0.0))
            for t in ("the cat sat", "a zebra sat")]
    H = corpus_loss(sum(s[0] for s in sums), sum(s[1] for s in sums), sum(s[2] for s in sums))["H_bits"]
    assert sum(r["contribution_bits"] for r in dec) == pytest.approx(H, abs=1e-12)
    by_type = {r["token_type"]: r for r in dec}
    assert by_type["unknown"]["targets"] == 2 and by_type["end"]["targets"] == 2   # 'a', 'zebra'; two </s>
    assert sum(r["share"] for r in dec) == pytest.approx(1.0)
