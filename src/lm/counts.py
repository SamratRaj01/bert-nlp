"""Sparse n-gram counts from clean training data.

Owner: Prazol. Phase 2 steps B3-B4.

Sequence framing (B3): every retained nonempty clean record is one sequence. Context resets
per record. Each record is framed as  <s> <s> w_1 ... w_n </s>  -- the bigram history of
the first target is one <s>, the trigram history two -- and exactly one </s> is a target.
<s> is context only.

Counts (B4), for order k = 1, 2, 3 (history = last k-1 tokens):
    joint[k][(h, w)]  = C(h, w)  number of training events where w follows h
    hist[k][h]        = C(h)     = sum over v in V of C(h, v)  (k=1: total target count)
Keys are packed integers (base |V|+1, <s> = |V|) so the tables stay sparse and compact.
Counts are filled once by `fit` and then frozen; scoring only reads them.
"""
from __future__ import annotations

from collections import Counter

from .vocab import Vocab

MAX_ORDER = 3


class NgramCounts:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.B = vocab.size + 1
        self.joint: dict[int, Counter] = {k: Counter() for k in (1, 2, 3)}
        self.hist: dict[int, Counter] = {k: Counter() for k in (1, 2, 3)}
        self.n_records = 0
        self.frozen = False

    # framing ----------------------------------------------------------
    def frame(self, ids: list[int]) -> list[int]:
        s = self.vocab.start_id
        return [s, s] + list(ids) + [self.vocab.end_id]

    def hkey(self, h: tuple[int, ...]) -> int:
        key = 0
        for x in h:
            key = key * self.B + x
        return key

    def jkey(self, h: tuple[int, ...], w: int) -> int:
        return self.hkey(h) * self.B + w

    # training ---------------------------------------------------------
    def fit(self, token_seqs: list[list[str]]) -> "NgramCounts":
        if self.frozen:
            raise RuntimeError("counts are frozen; evaluation must never refit")
        B = self.B
        for seq in token_seqs:
            f = self.frame(self.vocab.map_tokens(seq))
            a, b, w = f[:-2], f[1:-1], f[2:]
            self.joint[1].update(w)
            self.hist[1][0] += len(w)
            bw = [x * B + y for x, y in zip(b, w)]
            self.joint[2].update(bw)
            self.hist[2].update(b)
            ab = [x * B + y for x, y in zip(a, b)]
            self.joint[3].update(ab_ * B + y for ab_, y in zip(ab, w))
            self.hist[3].update(ab)
            self.n_records += 1
        self.frozen = True
        return self

    # read-only access -------------------------------------------------
    def C(self, h: tuple[int, ...], w: int) -> int:
        return self.joint[len(h) + 1].get(self.jkey(h, w), 0)

    def Ch(self, h: tuple[int, ...]) -> int:
        return self.hist[len(h) + 1].get(self.hkey(h), 0)

    def total_targets(self) -> int:
        return self.hist[1][0]

    def stats(self) -> dict:
        return {"records": self.n_records, "targets": self.total_targets(),
                "distinct_unigrams": len(self.joint[1]), "distinct_bigrams": len(self.joint[2]),
                "distinct_trigrams": len(self.joint[3]),
                "distinct_bigram_histories": len(self.hist[2]),
                "distinct_trigram_histories": len(self.hist[3])}

    def snapshot(self) -> tuple:
        """Cheap fingerprint used to prove evaluation never mutates the counts."""
        return tuple((k, len(self.joint[k]), sum(self.joint[k].values()),
                      len(self.hist[k]), sum(self.hist[k].values())) for k in (1, 2, 3))
