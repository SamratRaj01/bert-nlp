"""Engine construction, per-record scoring and corpus loss aggregation.

Owner: Prazol. Phase 2 step B7.

Per record we keep  nll_bits_sum, nll_nats_sum, T  (T counts every scored target including
the one </s>, excluding <s>). Corpus loss aggregates BEFORE dividing:

    H_bits = sum(nll_bits) / sum(T),   H_nats = sum(nll_nats) / sum(T) = H_bits * ln 2,
    PP     = 2 ** H_bits

Sentence perplexities are never averaged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..phase2_config import Phase2Config
from .counts import NgramCounts
from .models import AddOneLM, model_probs
from .vocab import Vocab, whitespace_vocab, wordpiece_vocab


@dataclass
class Engine:
    train_set: str
    tokenizer: str
    pipeline: object
    vocab: Vocab
    vocab_stats: dict
    counts: NgramCounts
    lm: AddOneLM


def build_engine(train_set: str, train_records: list[dict], pipeline,
                 cfg: Phase2Config) -> Engine:
    lm_cfg = cfg.lm
    train_tokens = pipeline.tokenize_batch([r["text"] for r in train_records])
    if pipeline.name == "whitespace":
        vocab, vstats = whitespace_vocab(train_tokens, int(lm_cfg["whitespace_min_count"]),
                                         unk=lm_cfg["whitespace_unk"], end=lm_cfg["end_symbol"],
                                         start=lm_cfg["start_symbol"])
    else:
        vocab, vstats = wordpiece_vocab(pipeline.hf.get_vocab(), lm_cfg["wordpiece_excluded"],
                                        end=lm_cfg["end_symbol"], start=lm_cfg["start_symbol"])
        unk_train = sum(1 for seq in train_tokens for t in seq if vocab.is_unknown(t))
        vstats = {**vstats, "training_tokens": sum(map(len, train_tokens)),
                  "training_tokens_mapped_to_unk": unk_train}
    counts = NgramCounts(vocab).fit(train_tokens)
    return Engine(train_set, pipeline.name, pipeline, vocab, vstats, counts, AddOneLM(counts))


def q_for_records(engine: Engine, token_seqs: list[list[str]]) -> list[np.ndarray]:
    """q1/q2/q3 for every target of every record; read-only on the counts."""
    return [engine.lm.q_arrays(engine.vocab.map_tokens(seq)) for seq in token_seqs]


def record_losses(q: np.ndarray, model: str, lambdas=None) -> tuple[float, float, int]:
    p = model_probs(q, model, lambdas)
    with np.errstate(divide="ignore"):
        bits = float(-np.log2(p).sum())
        nats = float(-np.log(p).sum())
    return bits, nats, len(p)


def corpus_loss(nll_bits_sum: float, nll_nats_sum: float, T: int) -> dict:
    h_bits = nll_bits_sum / T
    return {"T": T, "nll_bits": nll_bits_sum, "H_bits": h_bits,
            "H_nats": nll_nats_sum / T, "PP": math.pow(2.0, h_bits)}
