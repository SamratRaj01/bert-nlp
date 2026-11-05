"""Cosine similarity and Euclidean distance on raw pooled vectors (float64).

A pair is undefined, never imputed, when either side has no tokens ("empty") or when either
pooled vector has zero norm ("zero_norm").
"""
from __future__ import annotations

import numpy as np

VALID, EMPTY, ZERO_NORM = "valid", "empty", "zero_norm"


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def euclid(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(u - v))


def pair_metrics(u: np.ndarray | None, v: np.ndarray | None
                 ) -> tuple[float | None, float | None, str]:
    """(cosine, distance, status). Metrics are None unless status == "valid"."""
    if u is None or v is None:
        return None, None, EMPTY
    if np.linalg.norm(u) == 0.0 or np.linalg.norm(v) == 0.0:
        return None, None, ZERO_NORM
    return cosine(u, v), euclid(u, v), VALID
