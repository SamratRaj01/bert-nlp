# src/

All project code (Phase 1 + Phase 2). Organised by pipeline stage and ownership.

| Package | Owner | Responsibility |
|---|---|---|
| `corpus/` | Samrat | acquire WikiText-2, cut a deterministic slice, audit character categories |
| `noise/` | Samrat | deterministic insertion / deletion / substitution + corruption manifest |
| `tokenizer/` | Prazol | WordPiece tokenizer wrapper, config data card, tokenization diagnostics |
| `med/` | shared | unit-cost Levenshtein distance + normalized MED |
| `integration/` | both | pair-record schema and the end-to-end 20-pair generator |
| `phase2_config.py` | both | Phase 2 config loader (`configs/phase2.yaml`), protocol hash, frozen-λ writer, Phase 2 paths |
| `lm/` | Prazol | Phase 2 n-gram engine: tokenizer pipelines, frozen V, sparse counts, add-one models, λ selection, loss |
| `eval/` | both | Phase 2 grid scoring, coverage diagnostics, summary tables, figures |

Each package has its own README describing every file in it.

Run modules with `python -m src.<package>.<module>` from the repo root.
