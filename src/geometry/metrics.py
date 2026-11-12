"""G5: cosine similarity and Euclidean distance between a clean and a noisy pooled vector.

    cos(u, v) = u.v / (|u| |v|)        D(u, v) = |u - v|

Both use the RAW pooled vectors (no normalisation). A pair is UNDEFINED, never scored as 0, when
either side has no vector (empty span) or a zero-length vector (cosine would divide by zero).
"""
from __future__ import annotations

import numpy as np

ZERO_NORM = 1e-12


def pair_metrics(u: np.ndarray | None, v: np.ndarray | None) -> tuple[float, float, bool]:
    """Return (cosine, distance, valid). Invalid pairs return (nan, nan, False)."""
    if u is None or v is None:
        return float("nan"), float("nan"), False
    u, v = u.astype(np.float64), v.astype(np.float64)       # float64 so p=0 gives 1 and 0 to ~1e-16
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu <= ZERO_NORM or nv <= ZERO_NORM:
        return float("nan"), float("nan"), False
    return float(u @ v / (nu * nv)), float(np.linalg.norm(u - v)), True
