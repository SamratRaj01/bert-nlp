"""Probability and loss tests for the n-gram engine (assignment §11).

Owner: Prazol. Phase 2 step B8.
"""
import math

import numpy as np
import pytest

from src.lm.counts import NgramCounts
from src.lm.interp import lambda_grid, mix_loss_bits, select_lambdas
from src.lm.models import AddOneLM, model_probs, nll_bits, p_mle
from src.lm.score import corpus_loss, record_losses
from src.lm.vocab import Vocab, whitespace_vocab, wordpiece_vocab

TOL = 1e-12


def _hand_case():
    """V = {a, b, </s>}; records 'a', 'a', 'b' -> C(<s>,a)=2, C(<s>,b)=1, C(<s>,</s>)=0."""
    v = Vocab("hand", ["a", "b", "</s>"], unk=None)
    c = NgramCounts(v).fit([["a"], ["a"], ["b"]])
    return v, c, AddOneLM(c)


def test_hand_vocab_add_one_bigram():
    v, c, lm = _hand_case()
    h = (v.start_id,)
    assert c.C(h, v.index["a"]) == 2 and c.C(h, v.index["b"]) == 1 and c.C(h, v.end_id) == 0
    assert lm.q(2, h, v.index["a"]) == pytest.approx(3 / 6, abs=TOL)
    assert lm.q(2, h, v.index["b"]) == pytest.approx(2 / 6, abs=TOL)
    assert lm.q(2, h, v.end_id) == pytest.approx(1 / 6, abs=TOL)


def test_distributions_sum_to_one_seen_and_unseen_histories():
    train = [["the", "cat", "sat"], ["the", "dog", "sat"], ["a", "cat", "ran"], ["the", "cat"]]
    v, _ = whitespace_vocab(train, min_count=1)
    c = NgramCounts(v).fit(train)
    lm = AddOneLM(c)
    s, ids = v.start_id, v.index
    histories = {1: [()],
                 2: [(ids["the"],), (s,), (v.unk_id,)],               # seen, start, unseen
                 3: [(ids["the"], ids["cat"]), (s, s), (ids["sat"], ids["the"])]}
    for k, hs in histories.items():
        for h in hs:
            dist = lm.distribution(k, h)
            assert dist.sum() == pytest.approx(1.0, abs=1e-12)
            assert (dist > 0).all()
    # unseen history -> uniform
    assert np.allclose(lm.distribution(2, (v.unk_id,)), 1 / v.size)


def test_c_h_is_sum_over_vocab():
    train = [["x", "y"], ["y", "x", "x"]]
    v, _ = whitespace_vocab(train, min_count=1)
    c = NgramCounts(v).fit(train)
    h = (v.index["x"],)
    assert c.Ch(h) == sum(c.C(h, w) for w in range(v.size))
    assert c.Ch(()) == sum(c.C((), w) for w in range(v.size)) == c.total_targets() == 7


def test_interpolation_weights_and_normalization():
    grid = lambda_grid(0.25)
    assert len(grid) == 15
    for l in grid:
        assert min(l) >= 0 and sum(l) == pytest.approx(1.0, abs=TOL)
    _, _, lm = _hand_case()
    v = lm.c.vocab
    h2, h3 = (v.start_id,), (v.start_id, v.start_id)
    for l in grid:
        mix = sum(l[0] * lm.q(1, (), w) + l[1] * lm.q(2, h2, w) + l[2] * lm.q(3, h3, w)
                  for w in range(v.size))
        assert mix == pytest.approx(1.0, abs=1e-12)


def test_four_equally_likely_targets_give_two_bits_pp_four():
    v = Vocab("uniform4", ["a", "b", "c", "</s>"], unk=None)
    c = NgramCounts(v).fit([])                 # no training data: every history unseen
    lm = AddOneLM(c)
    q = lm.q_arrays(v.map_tokens(["a", "b", "c"]))    # targets a, b, c, </s>
    bits, nats, T = record_losses(q, "q2")
    res = corpus_loss(bits, nats, T)
    assert T == 4
    assert res["H_bits"] == pytest.approx(2.0, abs=TOL)
    assert res["PP"] == pytest.approx(4.0, abs=1e-9)
    assert res["H_nats"] == pytest.approx(res["H_bits"] * math.log(2), abs=TOL)


def test_corpus_aggregation_matches_hand_calculation():
    v, c, lm = _hand_case()
    recs = [["a", "b"], ["b"]]
    sums = [record_losses(lm.q_arrays(v.map_tokens(r)), "q2") for r in recs]
    # hand: q2(a|<s>)=3/6, q2(b|a)=1/5, q2(</s>|b)=2/4 ; q2(b|<s>)=2/6, q2(</s>|b)=2/4
    hand = -(math.log2(3 / 6) + math.log2(1 / 5) + math.log2(2 / 4)
             + math.log2(2 / 6) + math.log2(2 / 4))
    total_bits = sum(s[0] for s in sums)
    total_T = sum(s[2] for s in sums)
    assert total_T == 5
    assert total_bits == pytest.approx(hand, abs=1e-12)
    # corpus loss is NOT the mean of sentence perplexities
    res = corpus_loss(total_bits, sum(s[1] for s in sums), total_T)
    mean_sentence_pp = np.mean([2 ** (s[0] / s[2]) for s in sums])
    assert res["PP"] == pytest.approx(2 ** (hand / 5), abs=1e-9)
    assert not math.isclose(res["PP"], mean_sentence_pp)


def test_unseen_whitespace_tokens_map_to_unk_and_reserved_symbols():
    train = [["Paris", "is", "big"], ["Paris", "is", "old"]]
    v, stats = whitespace_vocab(train, min_count=2)
    assert set(v.tokens) == {"Paris", "is", "<UNK>", "</s>"}
    assert stats["types_mapped_to_unk"] == 2
    assert v.map_token("big") == v.unk_id and v.map_token("zzz") == v.unk_id
    assert v.map_token("</s>") == v.unk_id and v.map_token("<s>") == v.unk_id
    assert "<s>" not in v.tokens


def test_record_boundaries_reset_context():
    v = Vocab("ctx", ["a", "b", "</s>"], unk=None)
    c = NgramCounts(v).fit([["a", "b"], ["b", "a"]])
    # 'b' ends record 1 and 'b' starts record 2: no cross-record bigram (b -> b) may exist
    assert c.C((v.index["b"],), v.index["b"]) == 0
    assert c.C((v.start_id,), v.index["b"]) == 1
    assert c.C((v.start_id, v.start_id), v.index["a"]) == 1


def test_evaluation_never_changes_counts_or_vocab():
    v, c, lm = _hand_case()
    before, vocab_before = c.snapshot(), list(v.tokens)
    lm.q_arrays(v.map_tokens(["a", "b", "a"]))
    assert c.snapshot() == before and v.tokens == vocab_before
    with pytest.raises(RuntimeError):
        c.fit([["a"]])


def test_unsmoothed_mle_gives_infinite_loss_on_zero_probability():
    v, c, _ = _hand_case()
    h = (v.start_id,)
    assert p_mle(c, h, v.index["a"]) == pytest.approx(2 / 3)
    zero = p_mle(c, h, v.end_id)
    assert zero == 0.0 and nll_bits(zero) == math.inf
    assert p_mle(c, (v.index["b"], v.index["b"]), v.index["a"]) is None   # unseen history


def test_wordpiece_vocab_exclusions():
    hf = {"[PAD]": 0, "[unused0]": 1, "[unused12]": 2, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102,
          "[MASK]": 103, "the": 1996, "##s": 2015}
    v, stats = wordpiece_vocab(hf, ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[unused*]"])
    assert set(v.tokens) == {"[UNK]", "the", "##s", "</s>"}
    assert stats["excluded"] == 6 and v.map_token("[CLS]") == v.unk_id


def test_lambda_tie_break_prefers_simpler_model():
    q = np.full((10, 3), 0.25)                 # every λ gives the same loss
    chosen, table = select_lambdas(q)
    assert chosen == (1.0, 0.0, 0.0)
    assert sum(r["chosen"] for r in table) == 1
    q2 = np.column_stack([np.full(10, 0.1), np.full(10, 0.5), np.full(10, 0.2)])
    assert select_lambdas(q2)[0] == (0.0, 1.0, 0.0)
    assert mix_loss_bits(q2, (0.0, 1.0, 0.0)) == pytest.approx(1.0)


def test_model_probs_p_mix_is_weighted_sum():
    q = np.array([[0.2, 0.4, 0.6]])
    assert model_probs(q, "p_mix", (0.25, 0.25, 0.5))[0] == pytest.approx(0.05 + 0.1 + 0.3)
