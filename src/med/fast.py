"""Fast exact unit-cost Levenshtein for the Phase 2 grid.

Owner: Samrat. Phase 2 step A5.

Same definition as `levenshtein.med` (insert = delete = substitute = 1), which remains the
canonical oracle: `med_fast` is cross-checked against it on the known-answer cases and on
random clean/noisy pairs (tests/test_eval.py, phase2.ipynb).

Two exact speed-ups:
  1. a common prefix and suffix never change the distance, so they are stripped first;
  2. each DP row is computed with NumPy. Substitution and deletion are elementwise; the
     insertion chain curr[j] = min(tmp[j], curr[j-1] + 1) unrolls to
     curr[j] = min_{k<=j}(tmp[k] + j - k), i.e. a running minimum of (tmp - j) plus j.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _codes(seq: Sequence, table: dict) -> np.ndarray:
    return np.array([table.setdefault(x, len(table)) for x in seq], dtype=np.int64)


def med_fast(s: Sequence, t: Sequence) -> int:
    # strip common prefix / suffix (exact)
    lo = 0
    hi_s, hi_t = len(s), len(t)
    while lo < hi_s and lo < hi_t and s[lo] == t[lo]:
        lo += 1
    while hi_s > lo and hi_t > lo and s[hi_s - 1] == t[hi_t - 1]:
        hi_s -= 1
        hi_t -= 1
    s, t = s[lo:hi_s], t[lo:hi_t]
    if len(s) < len(t):
        s, t = t, s          # distance is symmetric; keep the row (t) short
    m, n = len(s), len(t)
    if n == 0:
        return m

    table: dict = {}
    t_codes = _codes(t, table)
    s_codes = _codes(s, table)
    idx = np.arange(n + 1, dtype=np.int64)
    prev = idx.copy()
    tmp = np.empty(n + 1, dtype=np.int64)
    for i in range(1, m + 1):
        tmp[0] = i
        np.minimum(prev[:-1] + (t_codes != s_codes[i - 1]),  # substitution / match
                   prev[1:] + 1,                               # deletion
                   out=tmp[1:])
        prev = np.minimum.accumulate(tmp - idx) + idx          # insertion chain
    return int(prev[n])


def normalized_med_fast(clean: Sequence, noisy: Sequence) -> float:
    return med_fast(clean, noisy) / max(1, len(clean))
