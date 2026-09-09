"""Unit-cost Levenshtein distance (Minimum Edit Distance).

Shared component, reviewed by both students.

Standard: insertion = deletion = substitution each cost 1.
    D[i,j] = min(D[i-1,j] + 1, D[i,j-1] + 1, D[i-1,j-1] + [s_i != t_j])
    D[i,0] = i,  D[0,j] = j

This module is deliberately small and fully implemented; the assignment requires it to pass
fixed known-answer tests (see tests/test_med.py).
"""
from __future__ import annotations

from typing import Sequence


def med(s: Sequence, t: Sequence) -> int:
    """Unit-cost Levenshtein distance between two sequences (usually strings)."""
    m, n = len(s), len(t)
    # previous row of the DP matrix; row 0 is [0, 1, 2, ..., n]
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            sub_cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + sub_cost,  # substitution / match
            )
        prev = curr
    return prev[n]


def normalized_med(clean: Sequence, noisy: Sequence) -> float:
    """MED divided by max(1, len(clean))."""
    return med(clean, noisy) / max(1, len(clean))
