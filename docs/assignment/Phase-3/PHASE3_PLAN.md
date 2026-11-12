# CS620 Research Project — Phase 3 (Assignment 3) Plan

**Course:** CS620 Research Project, Fall 2026 (instructor: Chhabra)
**Phase:** 3 — *Neural Prediction and Vector Drift*
**Team:** Samrat (Student A: geometry) · Prazol (Student B: prediction)
**Current scope:** Student A geometry workstream is **complete** (G0–G8). Student B neural workstream is **complete** (N0–N7, below). The joint counterexample analysis and paper v0.3 are not started.
**Status:** all results are **preliminary validation results**. The test split stays sealed.

Source: `docs/assignment/CS620 Research Project - Assignment 3 Revised.pdf`.

---

## 1. What the geometry workstream must show

> Does character noise move a **static** vector representation of text, and does the amount of
> movement depend on noise type and rate?

Static means a token ID always gets the same vector, regardless of its neighbours. We pool a span's
token vectors by averaging and compare the clean-span vector `u` with the noisy-span vector `v`:

- `cos(u, v) = u·v / (‖u‖‖v‖)`: direction agreement, −1 to 1.
- `D(u, v) = ‖u − v‖₂`: Euclidean distance, which also reflects magnitude change.

**Not a claim about meaning.** A small displacement does not prove meaning is preserved. Do not promise
monotonically falling cosine or a failure threshold at p > 0.20. Report null, mixed and contrary results.

Deliverable script: `noise_vector_drift.py`. We build it as `src/geometry/` modules plus `phase3.ipynb`, and
export the script once the logic is settled. The notebook is the runnable entry point, as in Phases 1–2.

---

## 2. What carries forward (nothing is regenerated)

| Component | Source | Use |
|---|---|---|
| Validation records, saved order | `derived/phase2/val_sample.jsonl` (500 records, SHA-256 `4894ff5d…`) | span source |
| 37 noisy datasets + edit logs | `derived/phase2/noisy/*.jsonl` | noisy counterparts (**replayed**, never re-sampled) |
| Noise rule `a2_default` | `src/noise/rule_v2.py` (`render_v2`, `eligible_positions`) | edit replay and eligibility |
| Grid | `configs/phase2.yaml`: ops {ins, del, sub}, p ∈ {0.01, 0.05, 0.10, 0.20}, seeds {11, 23, 47}, one shared clean | 37 conditions |
| Tokenizer | `bert-base-uncased` @ `86b5e093…` (`src/tokenizer/wrapper.py`) | WordPiece, no special tokens |

New in Phase 3: `configs/phase3.yaml`, `src/geometry/`, `phase3.ipynb`, `tools/build_phase3_notebook.py`,
`tests/test_geometry.py`, `derived/phase3/`, `results/phase3/`.

---

## 3. Geometry steps (do in order, one part at a time)

Each part ends with a check we can see, so the logic is understood before the next part builds on it.

| Part | What | Check that it worked | Status |
|---|---|---|---|
| **G0** | Freeze Phase 3 config (`configs/phase3.yaml`) | file inherits Phase 2 protocol; records new settings | done |
| **G1** | Extract frozen `E` (30522 × 768) from the pinned checkpoint, record SHA-256 | shape matches, hash saved, array read-only; **re-hash after use must match** | done |
| **G2** | Select the first ≤1,000 nonempty whitespace spans with ≥1 eligible letter, saved record order; keep record ID, span index, offsets | count = 1000, IDs/offsets stable, offsets slice back to the clean text | done |
| **G3** | Recover each span's noisy counterpart from the edit logs; fully deleted → explicit `""` | for all 37 conditions, spans reassembled into records **equal the manifest's `noisy_text` exactly** | done |
| **G4** | Tokenize each span alone (no special tokens), look up rows of `E`, mean-pool into one 768-d vector | hand-check one span; empty span gives no vector | done |
| **G5** | Compute cosine and Euclidean distance; mark empty/zero-norm pairs **undefined**, not 0 | undefined count equals empty-span count; p = 0 gives cos = 1, D = 0 within tolerance | done |
| **G6** | Aggregate per condition over valid pairs: mean cosine, mean D, selected/valid/undefined counts → **37-row static span audit** + span-level evidence file | 37 rows; coverage reported next to every mean | done |
| **G7** | Summaries by operation × rate (mean/min/max across seeds, clean separate); figures | min–max labelled "variation across corruption draws", **not a CI** | done |
| **G8** | Tests, `phase3_geometry` notes in `docs/`, export `noise_vector_drift.py` | tests pass; `noise_vector_drift.py` reproduces the notebook tables exactly | done |

### Design decisions and why

1. **Pair at span level, not by subword position.** Noise changes how many WordPiece pieces a word becomes, so
   "n-th clean piece ↔ n-th noisy piece" is wrong. The rule never edits whitespace, so span boundaries survive.
2. **Replay the saved logs.** Re-sampling noise would change which letters were hit and break the pairing with
   Assignment 2. G3's exact-match check proves the replay is faithful.
3. **Raw (unnormalised) vectors for D.** Normalising makes `D² = 2(1 − cos)`, which is redundant with cosine. Any
   normalised variant must be labelled separately.
4. **Undefined ≠ 0.** A fully deleted span has no vector. Scoring it 0 would make cosine and D look better or worse
   by accident. Count it, report coverage, aggregate over valid pairs only.
5. **Clean is one condition**, not three seed runs.
6. **E is frozen and hashed.** We use only `embeddings.word_embeddings.weight`: no position, segment or LayerNorm,
   no rescaling.
7. **Known limitation.** The first 1,000 spans come from only 22 of the 500 validation records (2 articles), so
   the span sample is narrow. Report the record/article count it covers.
8. **External pretraining disclosure.** BERT's checkpoint was pretrained on BooksCorpus and English Wikipedia, so
   its vectors may have seen the WikiText text. State this in the write-up.

### Outputs

- `derived/phase3/E.sha256`: fingerprint of the embedding table (the table itself is not committed; it is 90 MB).
- `derived/phase3/geometry_spans.jsonl`: the 1,000 selected spans (IDs, offsets, text).
- `results/phase3/static_span_audit.csv`: 37 rows.
- `results/phase3/span_pairs.parquet`: paired span-level evidence (clean text, noisy text, cos, D, validity).
- `results/phase3/noise_summary_geometry.csv`: operation × rate summaries.

---

## 4. Workstream B: neural next-token prediction (`neural_next_token.py`)

Spec: Assignment 3 §9–§14. Settings fixed by the PDF: hidden 64, Adam lr 0.001, batch 128, ≤ 5 epochs, seed 101, no dropout or
weight decay, CrossEntropyLoss on raw logits, clean training records only, epoch kept by lowest clean-validation loss (earlier wins ties).

| Step | What | Check that it worked | Status |
|---|---|---|---|
| **N0** | CPU torch installed, `neural:` block in `configs/phase3.yaml`, versions and device recorded | config loads; run info saved | done |
| **N1** | `BOS BOS w1 … wn EOS` contexts and targets; BOS is a zero vector; K = 29,525 outputs from the Assignment 2 vocabulary | train T = 35,642, clean validation T = 35,971, and all 37 per-condition `T` equal Assignment 2 | done |
| **N2** | `NextTokenNet` (mean / concat) with E as a frozen, non-persistent buffer | trainable 1,968,341 and 2,017,493; frozen 23,440,896; mean ignores token order, concat does not | done |
| **N3** | Seeded training, per-epoch validation, best-epoch selection, E hash before = after | validation loss lowest at epoch 2 for both; E unchanged | done |
| **N4** | Teacher-forced scoring: summed loss / T, perplexity, top-1 accuracy, tokenization diagnostics | hand-computed example matches; untrained loss = ln K (10.29) | done |
| **N5** | 74 condition rows and 24 noise-summary rows | counts exact; seed range labelled as not a confidence interval | done |
| **N6** | 37-row Assignment 2 n-gram control, kept separate; change from each model's own clean row | control `T` equals neural `T` in every condition | done |
| **N7** | `neural_next_token.py`, `tests/test_neural.py`, notebook sections, `docs/phase3_neural_summary.md` | tests pass; saved weights reproduce the table | done |

Still to do under the PDF's "Both: research": interpret counterexamples where vector drift and prediction change disagree, cross-verify each
other's code, and extend the paper to v0.3.
