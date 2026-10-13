"""Interpolation-weight selection on CLEAN validation data.

Owner: Prazol. Phase 2 step B6.

Grid: every (λ1, λ2, λ3) with λk >= 0, step 0.25, summing to 1 -> 15 combinations.
Objective: corpus loss H_bits of P_mix on the clean validation sample, per tokenizer (and per
training set). Tie-break (frozen in configs/phase2.yaml): among losses within `tol` of the
minimum, prefer the larger λ1, then the larger λ2 -- the simpler model wins ties.
The chosen λ are frozen before any noisy dataset is scored.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np


def lambda_grid(step: float = 0.25) -> list[tuple[float, float, float]]:
    n = Fraction(step).limit_denominator(1000)
    steps = int(1 / n)
    assert Fraction(1, steps) == n, "step must divide 1 exactly"
    grid = []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            grid.append((i / steps, j / steps, k / steps))
    return grid


def mix_loss_bits(q: np.ndarray, lambdas: tuple[float, float, float]) -> float:
    """Corpus H_bits of P_mix over all targets in q (shape (T, 3))."""
    p = q @ np.asarray(lambdas, dtype=np.float64)
    return float(-np.log2(p).sum() / len(p))


def select_lambdas(q: np.ndarray, step: float = 0.25, tol: float = 1e-12):
    """Return (chosen λ tuple, table of every candidate with its clean-validation loss)."""
    table = [{"lambda1": l[0], "lambda2": l[1], "lambda3": l[2],
              "clean_val_H_bits": mix_loss_bits(q, l)} for l in lambda_grid(step)]
    best = min(r["clean_val_H_bits"] for r in table)
    tied = [r for r in table if r["clean_val_H_bits"] <= best + tol]
    chosen = max(tied, key=lambda r: (r["lambda1"], r["lambda2"]))
    for r in table:
        r["chosen"] = r is chosen
        r["tied_with_best"] = r in tied
    return (chosen["lambda1"], chosen["lambda2"], chosen["lambda3"]), table
