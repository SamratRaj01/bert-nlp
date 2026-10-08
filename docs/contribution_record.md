# Contribution Record

## Samrat (Student A)
- Corpus acquisition and provenance (`src/corpus/acquire.py`, `raw/SOURCE.md`)
- Deterministic slice + checksum (`src/corpus/slice.py`)
- Character-category audit (`src/corpus/audit.py`, `docs/data_card.md`)
- Noise operations, RNG, corruption manifest (`src/noise/`)
- `tests/test_noise.py`
- Report sections: Data, part of Methods (noise design)

## Prazol (Student B)
- Tokenizer wrapper and data card (`src/tokenizer/wrapper.py`, `src/tokenizer/datacard.py`, `docs/tokenizer_card.md`)
- Tokenization diagnostics + clean baseline (`src/tokenizer/diagnostics.py`)
- BPE-vs-WordPiece write-up
- `tests/test_tokenizer.py`
- Report sections: part of Methods (tokenizer), Verification tables

## Shared
- `src/med/levenshtein.py` and `tests/test_med.py` (pair-programmed by Samrat and Prazol)
- Integration contract: ID scheme, `src/integration/schema.py`, `configs/config.yaml`
- 20-pair generation and inspection (`src/integration/generate_pairs.py`)
- Acceptance checklist sign-off
- Report: Objective, Verification & findings, Limitations & ethics

## Integration decisions log
- 2026-09-07: chose checkpoint `bert-base-uncased` (pinned revision); reason: standard cased-agnostic
  BERT baseline, WordPiece tokenizer used as-is with no vocabulary retraining so clean and noisy
  branches share identical settings.
- 2026-09-07: `intensity` defined as an independent per-character Bernoulli probability of being
  edited (`intensity = 0` → output identical to input, empty log, MED 0).
- 2026-09-07: `example_id` format = `wt2-train-NNNNNN` (zero-padded index into the deterministic
  500-line train slice).
- 2026-09-07: 20 pairs = first 20 slice examples; pair 0 is the intensity-0 control and the
  remaining 19 cycle through insertion / deletion / substitution at intensities 0.05 / 0.10 / 0.15.

---

# Phase 2 (Assignment 2) — Contribution Record

## Samrat (Student A — extend pipeline)
- Frozen validation sample + article IDs (`src/corpus/val_sample.py`, `src/corpus/articles.py`) — A1
- Training sets `train_a1_slice` / `train_full` with checksums — A2
- Noise rule v2 `a2_default` + replay (`src/noise/rule_v2.py`) — A3
- 37-dataset noise grid + manifests (`src/noise/grid.py`) — A4
- Fast MED cross-checked against the reference (`src/med/fast.py`) — A5
- Coverage / fragmentation diagnostics (`src/eval/coverage.py`) — A6
- `tests/test_noise_v2.py` — A8

## Prazol (Student B — build engine)
- Whitespace and WordPiece pipelines (`src/lm/tokenize.py`) — B1
- Frozen vocabularies (`src/lm/vocab.py`) — B2
- Sequence framing + sparse counts (`src/lm/counts.py`) — B3–B4
- Add-one models, P_mix, unsmoothed MLE demo (`src/lm/models.py`) — B5
- λ selection on clean validation (`src/lm/interp.py`) — B6
- Scoring and loss aggregation (`src/lm/score.py`) — B7
- `tests/test_lm.py` — B8

## Shared
- Step 0 freeze: `configs/phase2.yaml`, `src/phase2_config.py`, ID schemes
- Verification: hand check, normalization on real engines, clean invariance, 20-pair regression
- Grid runner, summaries, figures (`src/eval/run_grid.py`, `summarize.py`, `figures.py`), `tests/test_eval.py`
- `phase2.ipynb` (built by `tools/build_phase2_notebook.py`), error analysis, corpus-size extension
- `docs/phase2_summary.md`, acceptance checklist sign-off

## Integration decisions log (Phase 2)
- 2026-10-07: primary training set = the unchanged A1 500-line slice; full WikiText-2 train is a
  separate corpus-size extension (λ re-selected per training set).
- 2026-10-07: adopted the Assignment 2 default noise rule as `a2_default`; the A1 rule is kept only
  to regenerate the 20 diagnostic pairs.
- 2026-10-07: substitution excludes `source.lower()` so `A -> a` (erased by the uncased tokenizer)
  is never chosen.
- 2026-10-07: validation sample = first 500 validation lines passing the A1 line filter (40–600
  chars), file order; article ID = line index of the governing ` = Title = ` heading.
- 2026-10-07: λ tie-break = larger λ1, then larger λ2 (tolerance 1e-12 bits); λ written below the
  frozen marker in `configs/phase2.yaml` and read back for all noisy scoring.
- 2026-10-07: fast MED implemented with NumPy (no new dependency) and cross-checked against the
  Phase 1 reference; `matplotlib` added for figures.
- 2026-10-07: the ACL paper v0.2 is deferred; findings recorded in `docs/phase2_summary.md`.
