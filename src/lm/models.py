"""Add-one n-gram models, interpolation, and the unsmoothed MLE demonstration.

Owner: Prazol. Phase 2 step B5.

    q_k(w | h)    = (C(h, w) + 1) / (C(h) + |V|)            k = 1, 2, 3; h = last k-1 tokens
    P_mix(w | h)  = λ1 q1(w) + λ2 q2(w | last token) + λ3 q3(w | last two tokens)
    P_MLE(w | h)  = C(h, w) / C(h)          only when C(h) > 0; unseen targets get 0

Unseen histories (C(h) = 0) make q_k uniform, 1/|V|. All orders share one V per tokenizer.

The assignment's three configurations are q1, q2 and P_mix (its third configuration is P_mix,
not a standalone q3). The add-one trigram q3 is additionally scored as a SUPPLEMENTARY model
(team decision) so its standalone behaviour is visible in every output; it never replaces P_mix.
"""
from __future__ import annotations

import math

import numpy as np

from .counts import NgramCounts

REQUIRED_MODELS = ("q1", "q2", "p_mix")     # the assignment's three configurations
SUPPLEMENTARY_MODELS = ("q3",)              # standalone add-one trigram, reported alongside
MODELS = ("q1", "q2", "q3", "p_mix")


class AddOneLM:
    def __init__(self, counts: NgramCounts):
        self.c = counts
        self.V = counts.vocab.size

    def q(self, k: int, h: tuple[int, ...], w: int) -> float:
        assert len(h) == k - 1
        return (self.c.C(h, w) + 1) / (self.c.Ch(h) + self.V)

    def distribution(self, k: int, h: tuple[int, ...]) -> np.ndarray:
        """q_k(. | h) over all of V (used by the normalization tests)."""
        return np.array([self.q(k, h, w) for w in range(self.V)])

    def q_arrays(self, ids: list[int]) -> np.ndarray:
        """(T, 3) array of q1, q2, q3 for every scored target of one framed record."""
        c, B, V = self.c, self.c.B, self.V
        J1, J2, J3 = c.joint[1], c.joint[2], c.joint[3]
        H2, H3 = c.hist[2], c.hist[3]
        n1 = c.hist[1].get(0, 0) + V
        f = c.frame(ids)
        out = np.empty((len(f) - 2, 3), dtype=np.float64)
        for t in range(2, len(f)):
            a, b, w = f[t - 2], f[t - 1], f[t]
            ab = a * B + b
            out[t - 2, 0] = (J1.get(w, 0) + 1) / n1
            out[t - 2, 1] = (J2.get(b * B + w, 0) + 1) / (H2.get(b, 0) + V)
            out[t - 2, 2] = (J3.get(ab * B + w, 0) + 1) / (H3.get(ab, 0) + V)
        return out


def model_probs(q: np.ndarray, model: str, lambdas: tuple[float, float, float] | None = None) -> np.ndarray:
    """Probability of each observed target under one of the scored models."""
    if model == "q1":
        return q[:, 0]
    if model == "q2":
        return q[:, 1]
    if model == "q3":
        return q[:, 2]
    if model == "p_mix":
        l1, l2, l3 = lambdas
        return l1 * q[:, 0] + l2 * q[:, 1] + l3 * q[:, 2]
    raise ValueError(model)


def p_mle(counts: NgramCounts, h: tuple[int, ...], w: int) -> float | None:
    """Unsmoothed MLE. Returns None when the history was never seen (C(h) = 0: undefined)."""
    ch = counts.Ch(h)
    if ch == 0:
        return None
    return counts.C(h, w) / ch


def nll_bits(prob: float) -> float:
    """-log2 P; a zero-probability event gives +inf (kept, not hidden)."""
    return math.inf if prob == 0 else -math.log2(prob)
