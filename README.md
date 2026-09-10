# CS620 Research Project — Phase 1 (Foundations)

Reproducible clean/noisy foundation for the semester study **Robust NLP Under Character Noise**.

Phase 1 proves the team can (a) generate **deterministic** character-noise variants,
(b) measure their **edit distance** correctly with unit-cost Levenshtein, and (c) show how the
**fixed BERT WordPiece tokenizer** reacts to matched clean/noisy text. See
`docs/assignment/Phase-1/PHASE1_PLAN.md` for the full breakdown, roles, and step lists, and
`docs/assignment/` for the original assignment PDFs.

**Core rule:** character noise is applied to raw UTF-8 text *before* WordPiece tokenization.
Anything applied after tokenization is token-level noise and is not part of this study.

## Repository layout

| Path | Purpose | Owner |
|---|---|---|
| `raw/` | Untouched WikiText-2 files. Never edited. | Samrat |
| `derived/` | Generated / transformed files (slice, manifests, tokenized output). | both |
| `configs/` | Single source of truth for checkpoint, seed, noise settings, slice rule. | both |
| `src/corpus/` | Corpus acquisition, deterministic slice, character-category audit. | Samrat |
| `src/noise/` | Deterministic insertion / deletion / substitution + corruption manifest. | Samrat |
| `src/tokenizer/` | WordPiece tokenizer wrapper, config data card, tokenization diagnostics. | Prazol |
| `src/med/` | Unit-cost Levenshtein distance + normalized MED. | shared |
| `src/integration/` | Pair-record schema and per-pair build logic. | both |
| `phase1.ipynb` | Runnable entry point: orchestrates the full pipeline and shows the evidence. | both |
| `tests/` | Known-answer and determinism unit tests; outputs saved to `evidence/test_outputs/`. | both |
| `evidence/` | The 20 paired records (readable + CSV/JSON) and saved test outputs. | both |
| `docs/` | Corpus data card, tokenizer data card, contribution record, report outline. | both |

## Environment

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run order

The runnable entry point is **`phase1.ipynb`** — it imports the `src/` helpers and executes
the whole pipeline (corpus → slice → audit → MED tests → noise → tokenizer → 20 pairs →
acceptance checks), writing every artifact under `derived/`, `evidence/`, and `docs/`.

```
jupyter lab phase1.ipynb        # then: Kernel -> Restart & Run All
# or headless:
jupyter nbconvert --to notebook --execute --inplace phase1.ipynb
```

The `src/` modules are also runnable individually for spot-checks:

```
python -m src.corpus.acquire          # download WikiText-2 into raw/
python -m src.corpus.slice            # deterministic slice + checksum -> derived/
python -m src.corpus.audit            # character-category audit -> derived/
python -m src.tokenizer.datacard      # tokenizer config -> docs/tokenizer_card.md
python -m pytest -q                   # known-answer + determinism tests
```

A classmate should be able to recreate the 20-pair table from a fresh checkout by running
`phase1.ipynb` top to bottom.

## Ethics

Character noise is dual-use. This is a BERT encoder study, not a prompt-injection exercise.
Use only instructor-approved data and models; do not probe third-party systems or attempt
filter evasion; keep examples benign; frame findings around detection, normalization,
robustness, and mitigation.
