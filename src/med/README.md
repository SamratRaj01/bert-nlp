# src/med/  (shared)

Unit-cost Levenshtein (Minimum Edit Distance). Reviewed by both students.

| File | Purpose |
|---|---|
| `levenshtein.py` | `med(s, t) -> int` using the assignment recurrence, and `normalized_med(clean, noisy) -> float = med / max(1, len(clean))`. |

## Standard
Insertion = deletion = substitution each cost **1**.

    D[i,j] = min(D[i-1,j] + 1,
                 D[i,j-1] + 1,
                 D[i-1,j-1] + (0 if s[i-1] == t[j-1] else 1))
    D[i,0] = i,  D[0,j] = j

## Known-answer tests (tests/test_med.py, output saved to evidence/test_outputs/)
| source | target | expected |
|---|---|---|
| `same` | `same` | 0 |
| `` | `nlp` | 3 |
| `kitten` | `sitting` | 3 |
| `flaw` | `lawn` | 2 |
| `intention` | `execution` | 5 |
