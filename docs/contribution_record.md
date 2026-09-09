# Contribution Record

## Student A — _name_
- Corpus acquisition and provenance (`src/corpus/acquire.py`, `raw/SOURCE.md`)
- Deterministic slice + checksum (`src/corpus/slice.py`)
- Character-category audit (`src/corpus/audit.py`, `docs/data_card.md`)
- Noise operations, RNG, corruption manifest (`src/noise/`)
- `tests/test_noise.py`
- Report sections: Data, part of Methods (noise design)

## Student B — _name_
- Tokenizer wrapper and data card (`src/tokenizer/wrapper.py`, `src/tokenizer/datacard.py`, `docs/tokenizer_card.md`)
- Tokenization diagnostics + clean baseline (`src/tokenizer/diagnostics.py`)
- BPE-vs-WordPiece write-up
- `tests/test_tokenizer.py`
- Report sections: part of Methods (tokenizer), Verification tables

## Shared
- `src/med/levenshtein.py` and `tests/test_med.py` (implemented by _name_, reviewed by _name_)
- Integration contract: ID scheme, `src/integration/schema.py`, `configs/config.yaml`
- 20-pair generation and inspection (`src/integration/generate_pairs.py`)
- Acceptance checklist sign-off
- Report: Objective, Verification & findings, Limitations & ethics

## Integration decisions log
- YYYY-MM-DD: chose checkpoint `bert-base-uncased`; reason: ...
- YYYY-MM-DD: `intensity` defined as per-character Bernoulli probability
- YYYY-MM-DD: example_id format = `wt2-train-NNNNNN`
- YYYY-MM-DD: 20 pairs = first 20 slice examples; operation/intensity schedule = ...
