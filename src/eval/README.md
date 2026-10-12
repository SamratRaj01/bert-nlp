# src/eval/  (both)

Phase 2 evaluation: score the frozen grid, aggregate the required measures, draw the figures.

| File | Owner | Purpose |
|---|---|---|
| `coverage.py` | Samrat (A6) | Per-record diagnostics: unknown-token rate, pieces per original whitespace span, continuation-piece rate, token-count change; zero-token outputs flagged `rate_undefined`; boundary symbols excluded. |
| `run_grid.py` | both | `select_and_freeze` (B6 λ → `configs/phase2.yaml`), `frozen_lambdas`, `score_grid` (one row per record × condition × model; asserts counts are never mutated). |
| `summarize.py` | both | `scores` (222 required + 74 supplementary q3 per training set), `summary` (6 + 72 required rows + 2 + 24 q3 rows, mean/min/max over seeds), `within_pipeline`, `context_benefit`, `coverage`, `severity`, `write_tables`. |
| `decompose.py` | both | `loss_decomposition`: splits H exactly into share × mean bits per target type, showing why whitespace loss falls (`<UNK>` collapse) and WordPiece loss rises (`##` fragmentation). |
| `figures.py` | both | PNG figures generated only from the saved CSVs; separate tokenizer panels, one y-scale per panel, fixed operation colours + markers. |

Tests: `tests/test_eval.py`.
