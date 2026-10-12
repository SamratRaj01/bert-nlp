# CS620 Research Project — Robust NLP Under Character Noise

| Phase | Entry point | Plan | Write-up |
|---|---|---|---|
| 1 — Foundations (corpus, character noise, WordPiece) | `phase1.ipynb` | `docs/assignment/Phase-1/PHASE1_PLAN.md` | `docs/phase1_summary.md` |
| 2 — Baselines (n-gram LMs under noise) | `phase2.ipynb` | `docs/assignment/Phase-2/PHASE2_PLAN.md` | `docs/phase2_summary.md` |

## Phase 1 (Foundations)

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
| `phase2.ipynb` | Phase 2 entry point: noise grid, n-gram models, scores, figures, acceptance checks. | both |
| `configs/phase2.yaml` | Frozen Phase 2 protocol + λ written by the notebook. | both |
| `src/lm/` | Phase 2 n-gram engine: tokenizer pipelines, vocabularies, sparse counts, models, λ, loss. | Prazol |
| `src/eval/` | Phase 2 grid runner, coverage diagnostics, summary tables, figures. | both |
| `results/phase2/` | Phase 2 scores, summaries, per-record metrics, figures. | both |
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

## Phase 2 (Baselines) — run order

```
.venv/bin/python tools/build_phase2_notebook.py      # only if the notebook source changed
.venv/bin/jupyter nbconvert --to notebook --execute --inplace phase2.ipynb
# or: jupyter lab phase2.ipynb, then Kernel -> Restart & Run All   (~1-2 minutes, CPU only)
```

`phase2.ipynb` builds the frozen validation sample and training sets, generates the 37 noise
datasets (`derived/phase2/noisy/`), trains add-one unigram / bigram / interpolated models (plus a
supplementary standalone trigram) for whitespace and WordPiece tokenization, freezes λ into
`configs/phase2.yaml`, scores the 222 required conditions (+74 trigram) per training set, writes tables and figures, and runs the acceptance checklist
(`evidence/phase2/acceptance.txt`).

**Reproduce the headline table:** after the run, `results/phase2/train_a1_slice/summary.csv` holds
the 6 clean + 72 noisy required rows, plus 2 + 24 supplementary trigram rows (mean / min / max
over seeds 11, 23, 47), and
`results/phase2/figures/loss_vs_rate_train_a1_slice.png` plots them. Re-running gives identical
files. The corpus-size comparison is `results/phase2/corpus_size_comparison.csv`.

Phase 2 also needs `matplotlib` (in `requirements.txt`).

## Ethics

Character noise is dual-use. This is a BERT encoder study, not a prompt-injection exercise.
Use only instructor-approved data and models; do not probe third-party systems or attempt
filter evasion; keep examples benign; frame findings around detection, normalization,
robustness, and mitigation.
