# Technical Report Outline (3-5 pages + references)

Concise foundation memo, not the final ACL manuscript. Every claim traces to code or a saved
artifact.

## 1. Objective
- The assignment question (deterministic noise, correct MED, WordPiece change on paired text).
- Where Phase 1 sits in the semester project (foundation for baseline -> robustness -> scale).

## 2. Data
- Corpus: WikiText-2, version, license, `wikitext-2-raw-v1` config and why.
- Slice: selection rule, count, checksum. Split discipline (train dev; test untouched).
- UTF-8 handling and WikiText-2 quirks.
- Character-category statistics table (from `docs/data_card.md`).

## 3. Methods
- Noise design: insertion / deletion / substitution; intensity definition; per-example seeded
  RNG; corruption manifest and operation log.
- Fixed tokenizer + checkpoint: `bert-base-uncased`, revision, normalization, special tokens,
  vocab checksum, max length. Same settings both branches.
- MED definition: unit-cost Levenshtein recurrence; normalized MED.
- BPE vs WordPiece (one paragraph).
- Configuration and reproducibility (seed, config file, run order).

## 4. Verification & findings
- Known-answer MED tests: table of 5 cases, all pass (cite `evidence/test_outputs/`).
- Determinism tests: same seed -> same noise+log; intensity 0 -> identity, MED 0; replay == output.
- 20-pair summary: distribution of MED / normalized MED by operation; fertility, [UNK],
  continuation-piece, token-count-change deltas (clean vs noisy).
- 2-3 representative examples, including at least one meaningful WordPiece change.
- Observation only, no causal claims.

## 5. Limitations & ethics
- Coverage limits: one corpus, one checkpoint, small slice, synthetic noise model.
- Dual-use: character noise can obfuscate harmful content / evade filters; this is a BERT
  encoder study, not prompt injection.
- Authorized-testing boundary: instructor-approved data and models only; no third-party probing.
- Next steps: Phase 2 baseline.

## References
Devlin et al. (2019) BERT; Jurafsky & Martin, SLP; Kaplan et al. (2020) (background only).
