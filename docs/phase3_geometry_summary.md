# Phase 3 — Workstream A (geometry): notes

**Status:** complete. All results are **preliminary validation results**; the test split is not opened.
**Run it:** `python noise_vector_drift.py` (about 3 s on CPU) writes `results/phase3/`. The walk-through is `phase3.ipynb`.

## Method
- Frozen table E = `bert.embeddings.word_embeddings.weight` of `bert-base-uncased` @ `86b5e093…` (30,522 × 768, SHA-256 `ca6b5153…cd8105`, identical before and after the run). Only token vectors are used. BERT was pretrained outside this project (BooksCorpus + English Wikipedia), so its vectors may have seen WikiText-like text.
- Spans: first 1,000 nonempty whitespace-delimited spans containing an eligible letter, saved validation order (22 records, 2 articles).
- Noisy twins: the saved Assignment 2 edit logs replayed per span; fully deleted spans stay as empty spans. Rebuilding whole records from spans matches the Assignment 2 noisy text exactly in all 37 conditions.
- Vector per span: each WordPiece piece (no special tokens) is looked up separately in E, rows averaged (raw, unnormalised).
- Metrics: cosine and Euclidean distance on raw pooled vectors. Empty or zero-norm pairs are undefined (NaN, counted, excluded from means).

## Results (37,000 pairs; 36,938 valid; 62 undefined, all from deletion)
- p = 0: cosine 1, distance 0 (max error 2e-16).
- Mean cosine falls from ~0.97–0.98 (p = 0.01) to ~0.64–0.69 (p = 0.20); mean distance rises from ~0.05 to ~0.64–0.71. Seed spread is small.
- Among spans the noise changed, mean cosine stays roughly flat in p (insertion ~0.49–0.53, deletion ~0.39–0.44, substitution ~0.42–0.45): the decline in the overall mean comes mostly from more spans being hit.
- Cosine reaches slightly negative values for single pairs at p = 0.20 (min −0.047).

## Caveats
- The min–max over the 3 corruption seeds is variation across random corruptions, not a confidence interval.
- Narrow sample (22 records); 87% of clean spans are one word piece, so many comparisons are single token vectors.
- Punctuation attached to a word (for example `H.`) is averaged in, following the span definition.
- Geometry is not meaning: a smaller or larger vector change says nothing direct about preserved meaning.
- The explanation that insertion drifts least because it keeps the original letters is untested.

## Files
`configs/phase3.yaml`, `src/geometry/{embeddings,spans,pooling,metrics,drift}.py`, `noise_vector_drift.py`, `tests/test_geometry.py`, `phase3.ipynb`, `results/phase3/{static_span_audit.csv,noise_summary_geometry.csv,span_pairs.parquet,figures/geometry_drift.png,geometry_run_info.json}`, `derived/phase3/{E.sha256,geometry_spans.jsonl}`.
