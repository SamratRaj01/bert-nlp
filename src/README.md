# src/

All Phase 1 code. Organised by pipeline stage and ownership.

| Package | Owner | Responsibility |
|---|---|---|
| `corpus/` | Samrat | acquire WikiText-2, cut a deterministic slice, audit character categories |
| `noise/` | Samrat | deterministic insertion / deletion / substitution + corruption manifest |
| `tokenizer/` | Prazol | WordPiece tokenizer wrapper, config data card, tokenization diagnostics |
| `med/` | shared | unit-cost Levenshtein distance + normalized MED |
| `integration/` | both | pair-record schema and the end-to-end 20-pair generator |

Each package has its own README describing every file in it.

Run modules with `python -m src.<package>.<module>` from the repo root.
