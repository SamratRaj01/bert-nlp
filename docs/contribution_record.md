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
