# CS620 Research Project — Phase 2 (Assignment 2) Plan

**Course:** CS620 Research Project, Fall 2026 (instructor: Chhabra)
**Project:** Robust NLP Under Character Noise
**Phase:** 2 — Baselines: *Probabilistic Language Modeling and Input Noise*
**Team:** Samrat (Student A) · Prazol (Student B)
**Current scope:** experiment + repo. The ACL paper v0.2 is **deferred** (see §11).

---

## 1. What Phase 2 must prove

> **Research question:** How do character-noise type and intensity change token coverage,
> fragmentation, and predictive behavior under fixed whitespace and WordPiece tokenization?

We train count-based n-gram models on **clean training text** and score one **frozen validation
sample** in clean form and in 36 corrupted forms. The models are **scored, not sampled**. We
measure how surprised they are by the observed next tokens, and we do not generate text.

- Models: add-one unigram `q1`, add-one bigram `q2`, interpolated `P_mix = λ1·q1 + λ2·q2 + λ3·q3`.
- Tokenizers: Python `text.split()` (whitespace) and the **same pinned BERT WordPiece** as Phase 1.
- Everything here is **preliminary validation (development) evidence**. **No test split** is
  scored, inspected, or used for any choice.
- **Out of scope:** BERT forward passes, representations, and neural evaluation. Those come in
  later phases.

**Core rule (unchanged):** noise is applied to raw text **before** tokenization. Each noisy text
is generated **once** and the identical string is passed to both tokenizers.

---

## 2. What carries forward and what changes

### Reused from Phase 1 (do not rewrite)

| Component | Where | Phase 2 use |
|---|---|---|
| Corpus + provenance | `raw/wikitext-2/*.raw`, `raw/SOURCE.md` (`Salesforce/wikitext`, `wikitext-2-raw-v1`, rev `b08601e…`) | same data and hashes |
| Line filter + IDs | `src/corpus/slice.py` → `is_kept_line`, `make_example_id` | validation sample, record IDs |
| A1 train slice | `derived/slice.jsonl` (500 records, ~31k words, checksum in `slice.sha256`) | **primary training set** |
| Per-example RNG | `src/noise/rng.py` → `example_seed`, `example_rng` | per-record, per-condition seeds |
| Manifest + replay | `src/noise/manifest.py` → `build_row`, `replay` | extended for the v2 rule |
| Levenshtein | `src/med/levenshtein.py` → `med`, `normalized_med` | reference MED and test oracle |
| WordPiece wrapper | `src/tokenizer/wrapper.py` → `TokenizerWrapper` (rev `86b5e09…`) | WordPiece pipeline |
| Diagnostics | `src/tokenizer/diagnostics.py` | basis for coverage/fragmentation |
| 20 diagnostic pairs | `src/integration/generate_pairs.py`, `evidence/pairs/` | regression check only |

### New in Phase 2 (gaps in the current code)

| Gap | Why it matters | Owner |
|---|---|---|
| **Noise rule v2.** The A1 rule edits *any* char and inserts *before* (`operations.render`) | A2 default: letters only, insert *after*, no re-corruption | Samrat |
| **Validation sample + article IDs.** None exist yet | needed for the grid, λ selection, and later document-level bootstrap | Samrat |
| **MED at scale.** `med` is pure-Python O(n·m) | 500 records × 36 conditions ≈ 18k pairs of ≤ 600 chars | Samrat |
| **Whitespace pipeline + vocabularies** | second tokenizer, frozen `V` per tokenizer | Prazol |
| **N-gram engine.** Counts, `q1`/`q2`/`q3`, `P_mix`, λ search, loss | the whole predictive side | Prazol |
| **Grid runner, summaries, figures** | 222 scores per training set, 72 summary rows | both |
| **Phase 2 write-up** | `docs/phase2_summary.md`: methods notes, results, caveats (feeds the paper later) | both |

---

## 3. Roles

### Student A — Samrat: extend the pipeline
Owns the **data side** of the experiment. This covers the frozen validation sample with article
IDs, the full-train record set, the v2 noise rule, the 36-condition noisy grid with complete
manifests, MED at scale, and the per-record coverage/fragmentation diagnostics for both
tokenizers.

**Required evidence (§10):** deterministic paired manifests and tokenization diagnostics,
produced by reusing the corpus and noise code.

### Student B — Prazol: build the engine
Owns the **predictive side**. This covers the two tokenizer pipelines, frozen vocabularies,
sparse n-gram counts, the add-one models, interpolation, λ selection on clean validation, and
loss/perplexity aggregation in consistent units.

**Required evidence (§10):** reuse of the tokenizer settings, plus sparse counts,
`q1`/`q2`/`P_mix`, validation selection, and loss aggregation.

### Shared
Step 0 freeze, cross-review of tests, hand calculations against program output, rerunning the
20 A1 pairs, running the grid, error analysis, figures, the Phase 2 write-up, and the contribution
record.

---

## 4. Step 0 — Shared freeze (do together, first)

Create **`configs/phase2.yaml`**. Leave `configs/config.yaml` frozen as the Phase 1 record. Both
students review and approve it before any scoring.

1. **Checkpoint:** keep `bert-base-uncased` @ `86b5e0934494bd15c9632b12f734a8a67f723594`.
   The course default `google-bert/bert-base-uncased` is the same Hub repo, so record that and
   keep the pinned revision.
2. **Noise grid:** operations `[insertion, deletion, substitution]`; rates
   `p ∈ {0, 0.01, 0.05, 0.10, 0.20}`; seeds `{11, 23, 47}` for every nonzero rate. Clean (p = 0)
   is **one** dataset shared across operations and seeds.
3. **Noise rule:** `rule_version: a2_default` (see A3). The A1 rule is kept **only** for
   regenerating the 20 diagnostic pairs.
4. **Validation sample rule:** see A1. It uses `is_kept_line` (non-empty, non-heading,
   40–600 chars), file order, cap 500.
5. **Training sets:**
   - `train_a1_slice`: **primary baseline**. The existing 500-record A1 slice (~31k words).
   - `train_full`: **corpus-size extension**. Every non-empty, non-heading line of `train.raw`
     (~17.5k records, ~2.0M words). Its rule and checksum are recorded the same way.
6. **ID schemes:**
   - Record ID: `wt2-valid-NNNNNN` / `wt2-train-NNNNNN` (existing `make_example_id`).
   - Article ID: `wt2-<split>-doc-NNNNNN` = line index of the governing **top-level heading**
     (` = Title = `, a single `=` on each side; ` = = Sub = = ` does **not** start an article).
   - Condition ID: `clean`, or `<op>-p<rate>-s<seed>`, e.g. `del-p0.05-s23`.
   - Per-record RNG: `example_rng(seed, f"{record_id}|{op}|{p}")`. This depends only on source
     IDs and the condition, never on outcomes.
7. **Model settings:** λ grid step 0.25, nonnegative, summing to 1 (15 combinations). The
   tie-break rule is fixed now (see B6).
8. **Hashes:** record the SHA-256 of the val sample, each training set, and `phase2.yaml`. The
   config hash is copied into every manifest and metric row.
9. **Planned repo additions:**

   | Path | Purpose | Owner |
   |---|---|---|
   | `configs/phase2.yaml` | single source of truth for Phase 2 | both |
   | `src/corpus/val_sample.py`, `src/corpus/articles.py` | val sample, article IDs, full-train set | Samrat |
   | `src/noise/rule_v2.py` (+ `replay_v2`) | A2-default noise rule | Samrat |
   | `src/noise/grid.py` | 37-dataset generation + manifests | Samrat |
   | `src/med/fast.py` | fast MED, cross-checked against `med` | Samrat |
   | `src/eval/coverage.py` | coverage/fragmentation diagnostics | Samrat |
   | `src/lm/{tokenize,vocab,counts,models,interp,score}.py` | n-gram engine | Prazol |
   | `src/eval/{run_grid,summarize,figures}.py` | grid runner, summaries, plots | both |
   | `tests/test_noise_v2.py`, `tests/test_lm.py`, `tests/test_eval.py` | Phase 2 tests | A / B / both |
   | `derived/phase2/` | val sample, training sets, noisy datasets, manifests | Samrat |
   | `results/phase2/` | per-record metrics, scores, summaries, figures | both |
   | `phase2.ipynb` | runnable entry point (same pattern as `phase1.ipynb`) | both |
   | `docs/phase2_summary.md` | Phase 2 write-up (same role as `phase1_summary.md`) | both |

---

## 5. Student A steps (Samrat): extend the pipeline

1. **A1 — Frozen validation sample.**
   - Scan `raw/wikitext-2/validation.raw` in file order and keep lines passing
     `is_kept_line(line, 40, 600)`. Stop at **500 records**. The selection uses source IDs only and
     never looks at model outcomes.
   - For each record, save `record_id`, `article_id`, `source_split`, `source_index`, `text`,
     `char_count`.
   - Pre-check: 846 val lines are eligible across 57 articles, and the first 500 span **36
     articles** (up to line 2176). Report the exclusions: headings, empty lines, and lines outside
     the 40–600 band.
   - Write `derived/phase2/val_sample.jsonl` + `.sha256`. The 40–600 band is the
     **clean-record length cap** for MED runtime. It is applied **before** corruption, and noisy
     variants are never truncated independently.
   - Justify the sample size in `docs/phase2_summary.md` as bounded by compute, and say that it is not the
     20 diagnostic pairs.
2. **A2 — Article IDs for training + full-train set.**
   - Back-fill `article_id` for the A1 train slice. Do this as a new derived file;
     `slice.jsonl` stays byte-identical.
   - Build `derived/phase2/train_full.jsonl` (all non-empty, non-heading train lines, with IDs,
     checksum, and counts).
   - Record clean char and token counts for both training sets.
3. **A3 — Noise rule v2 (`a2_default`).**
   - **Eligible** = ASCII letters `a–z`, `A–Z` in the **original** string. Spaces, punctuation,
     digits, record boundaries, and WikiText sentinels (`@-@`, `@,@`, `@.@`) are never edited.
     Sentinels contain no letters, so they are protected automatically; still keep an explicit
     protected-span list in config. The raw variant contains no `<unk>` strings.
   - For each eligible original position, run one independent Bernoulli(p) draw:
     - **Deletion:** delete the char.
     - **Substitution:** replace it with a uniformly sampled lowercase `a–z` char **different
       from** the source. *Decision to freeze:* exclude `source.lower()`, so that `A → a` is never
       chosen; the uncased tokenizer would erase that change.
     - **Insertion:** insert one uniform lowercase `a–z` char **immediately after** the eligible
       char.
   - Sample once over original positions only. **Inserted chars are never corrupted.**
   - The log stays `(op, index, source_char, replacement_char)` against the clean string, plus
     `rule_version`, `eligible_count`, and `realized_edits`.
   - Add `replay_v2`. Replaying the log must reproduce the noisy text exactly.
   - p = 0 consumes no RNG and returns the input unchanged.
4. **A4 — Generate the grid (37 datasets).**
   - For each `(op, p, seed)` and each val record, generate the noisy text **once** and save it.
     Both tokenizers read these saved files and never regenerate.
   - Each manifest row holds: `article_id`, `record_id`, `condition_id`, `split`, `seed`, `op`,
     requested `p`, `rule_version`, `eligible_count`, the edit log, `realized_edits`, `MED`,
     `normalized_MED`, `source_text_sha256`, `config_sha256`.
   - Keep records that become **empty** after deletion, with an empty `noisy_text`. Never drop
     them.
5. **A5 — MED at scale.**
   - Add a fast exact Levenshtein: a NumPy row-wise DP, or `rapidfuzz.distance.Levenshtein` if we
     add the dependency.
   - Test it against `src/med/levenshtein.med` on all known-answer cases plus ≥ 1,000 random
     clean/noisy pairs. `med` stays the canonical oracle.
   - `normalized_MED = MED / max(1, clean_char_count)`, unchanged from Phase 1.
   - Note in `docs/phase2_summary.md`: `p` ≠ MED ≠ logged edit count, because MED can be smaller than the
     number of edits.
6. **A6 — Coverage and fragmentation diagnostics** (per record × condition × tokenizer).
   - Whitespace **`<UNK>` rate**: the share of tokens outside frozen `V` (`V` comes from Prazol's
     B2).
   - WordPiece **`[UNK]` rate**.
   - **Pieces per original word span** = tokens ÷ number of original nonempty whitespace spans
     of the **clean** record. Deleted spans contribute 0 pieces.
   - **Continuation-piece rate** (`##` pieces ÷ pieces) and **token-count change**
     (noisy − clean).
   - **Exclude boundary symbols** (`<s>`, `</s>`) from all diagnostics.
   - For a zero-token output, save zero counts and set `rate_undefined = true`. Never divide by
     zero, and never drop the record.
   - Define every rate in a short table in `docs/phase2_summary.md`.
7. **A7 — Rerun the 20 A1 diagnostic pairs** with the **A1 rule** and confirm the output is
   byte-identical to `evidence/pairs/`. They are a regression check, not the aggregate sample.
8. **A8 — Tests** (`tests/test_noise_v2.py`):
   - p = 0 gives identical output and an empty log.
   - The same seed gives the same text and log.
   - Replay reproduces the noisy text.
   - Only letters change; digits, punctuation, spaces, and sentinels are untouched.
   - An insertion lands after its char.
   - Inserted chars are never re-edited.
   - A substitution never maps to the same letter, in either case.
   - An empty output is retained.
   - The fast MED agrees with the reference MED.

---

## 6. Student B steps (Prazol): build the engine

1. **B1 — Two tokenizer pipelines, documented.**
   - **WordPiece:** `TokenizerWrapper` at the pinned revision, using pieces **without**
     `[CLS]`/`[SEP]`, padding, or 512 truncation (n-grams need no BERT wrappers). Record casing
     (lowercased), accent stripping, punctuation splitting, and library versions.
   - **Whitespace:** `text.split()`, which keeps case and punctuation exactly as written.
   - State in `docs/phase2_summary.md` that this compares **complete pipelines** (normalization + segmentation
     + vocabulary), not segmentation alone.
2. **B2 — Frozen vocabularies `V`** (one per tokenizer × training set):
   - **Whitespace:** training tokens with count ≥ 2, plus `<UNK>` and the end symbol `</s>`.
     Map every other token to `<UNK>`, both at training and at evaluation.
   - **WordPiece:** all vocab entries **except** `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`, and
     `[unused…]`, plus `</s>`. Keep `[UNK]` and every other entry, including entries unseen in
     training.
   - `<s>` is **context only**, so it is never in `V` and never scored.
   - Freeze `V` and report `|V|`. Expect a small whitespace `|V|` and a high `<UNK>` rate on the
     31k-word slice. That is a finding to report, not a bug.
3. **B3 — Sequence framing.** Each retained nonempty clean record is one sequence. Reset the
   context per record. Pad with one `<s>` for bigram context and two for trigram context, and
   score exactly one `</s>` per record. A record emptied by noise is still scored on its `</s>`.
4. **B4 — Sparse counts.**
   - `C(h, w)` for orders 1–3 as `Counter`/dict keyed by tuples.
   - `C(h) = Σ_{v∈V} C(h, v)`. For the empty history it is the total training target count.
   - Compute probabilities on demand and never materialize dense tables.
   - Counts come from **clean training data only**, are never refit per seed, and are never
     touched by evaluation.
5. **B5 — Models.**
   - `q_k(w|h) = (C(h,w) + 1) / (C(h) + |V|)`. An unseen history becomes uniform `1/|V|`. All
     orders share one `V` per tokenizer.
   - `P_mix(w|h) = λ1·q1(w) + λ2·q2(w|last token) + λ3·q3(w|last two tokens)`. The third
     configuration is `P_mix`, **not** a standalone `q3`.
   - **Unsmoothed MLE demo:** on a small separate test case, show
     `P_MLE = C(h,w)/C(h)` assigning 0 to an unseen event, and keep the resulting **infinite
     loss** in the output instead of hiding it.
6. **B6 — λ selection.**
   - Search all 15 `(λ1, λ2, λ3)` on the 0.25 grid that sum to 1.
   - Minimize **clean-validation** loss **separately for each tokenizer**. For the extension,
     repeat the search for each training set.
   - **Tie-break (frozen in Step 0):** prefer the larger `λ1`, then the larger `λ2`, so the
     simpler model wins ties. Compare losses with a fixed tolerance, e.g. `1e-12`.
   - Write the chosen λ into `phase2.yaml` **before** any noisy scoring, and keep them frozen for
     all 36 noisy datasets.
7. **B7 — Scoring and aggregation.**
   - For each record, save `nll_bits_sum`, `nll_nats_sum`, and `T_record` (targets including
     `</s>`, excluding `<s>`).
   - Corpus loss `H_bits = Σ nll_bits / Σ T`, with `H_bits = H_nats / ln 2`, and `PP = 2^H_bits`.
   - **Never average sentence perplexities.**
   - Per-record rows carry `article_id`, `record_id`, and `condition_id` so that later document
     bootstrap intervals are possible.
8. **B8 — Tests** (`tests/test_lm.py`, §11):
   - For observed **and** unseen histories, each smoothed distribution sums to 1 (within
     tolerance), and every allowed target has `p > 0`.
   - Hand vocab `{a, b, end}` with `C(h,a)=2`, `C(h,b)=1`, `C(h,end)=0` gives `3/6`, `2/6`,
     `1/6`.
   - λ are nonnegative and sum to 1, and `P_mix` is normalized.
   - Four equally likely targets give **2 bits** and **PP 4**. Corpus aggregation matches a hand
     calculation on 2–3 tiny records.
   - Unseen whitespace tokens map to `<UNK>`, record boundaries reset context, and evaluation
     never changes counts or `V`.
   - The MLE demo produces `inf` loss on a zero-probability event.

---

## 7. Shared — verify (before the full grid)

1. **Cross-review:** Samrat reads `test_lm.py`; Prazol reads `test_noise_v2.py` and the
   coverage code.
2. **Hand calculation vs. program:** take a 3-record toy corpus and work `q1`, `q2`, `P_mix`,
   `H_bits`, and `PP` by hand. Check the program output against those values and save both under
   `evidence/phase2/hand_check.md`.
3. **Phase 1 still passes:** `pytest` on all A1 MED and reproducibility tests, plus p = 0
   leaving the input unchanged.
4. **Rerun the 20 A1 diagnostic pairs** (A7) and diff them against `evidence/pairs/`.
5. **Clean invariance:** scoring the p = 0 dataset under each seed must give exactly the same
   clean score.

---

## 8. Shared — evaluate

1. **Freeze** `phase2.yaml` (including λ), record its hash, and tag the commit.
2. **Run the grid on `train_a1_slice` (primary):** 37 datasets × 2 tokenizers × 3 models =
   **222 perplexity scores**. That is 6 clean + 216 noisy.

   | Quantity | Calculation | Total |
   |---|---|---|
   | Model × tokenizer configs | 3 × 2 | 6 |
   | Noisy validation datasets | 3 ops × 4 rates × 3 seeds | 36 |
   | Clean scores | 6 × 1 | 6 |
   | Noisy scores | 6 × 36 | 216 |
   | All individual scores | 6 + 216 | 222 |
   | Noisy summary rows | 6 × 3 ops × 4 rates | 72 |

3. **Machine-readable outputs** (`results/phase2/<train_set>/`):
   - `per_record.parquet|csv`: one row per (record, condition, tokenizer, model) with NLL sums,
     `T`, and the diagnostics.
   - `scores.csv`: 222 rows with `H_bits`, `H_nats`, `PP`, `T`, and `N_records`.
   - `summary.csv`: 6 clean rows and 72 noisy rows, each noisy row giving the **mean, min, and max
     over the 3 seeds**. These summarize existing runs and are not new runs.
   - `within_pipeline.csv`: noisy loss − clean loss, reported with either sign, together with the
     token-count and target changes.
   - `context_benefit.csv`: `q1` loss − `q2` loss and `q1` loss − `P_mix` loss on the same
     tokenized condition (positive = better than `q1`).
   - `coverage.csv`: `<UNK>`/`[UNK]` rates, pieces per original span, continuation rate, and
     token-count change.
   - `severity.csv`: requested `p` against the distribution of `normalized_MED`, paired by
     record ID.
4. **Figures** (`src/eval/figures.py`, read **only** from the saved CSVs):
   - Loss/PP vs. `p` by operation, with **separate tokenizer panels** and min–max bands.
   - Context benefit vs. `p`.
   - Coverage/fragmentation vs. `p`.
   - `normalized_MED` distribution by `p`.

   Every plotted point must trace back to a CSV row, a config, and record IDs.
5. **Uncertainty and error analysis** (§12):
   - Report the 3-seed mean and the min–max range, **not** confidence intervals.
   - Discuss **three traceable examples**, giving record ID, condition, and the before/after
     tokens and NLL for each. **At least one must challenge a simple robustness story**, e.g.
     perplexity *falling* under deletion because targets disappear, or whitespace `<UNK>` collapse
     making noisy text "easier".
6. **Corpus-size extension (`train_full`):**
   - Repeat B2–B7 on `train_full` with the **same** val sample, noise grid, rule, and tie-break.
     λ are re-selected on clean val for this training set and frozen before noisy scoring.
     That gives another 222 scores and 72 summary rows.
   - Compare **within each tokenizer only**: slice against full, clean against noisy deltas,
     context benefit, and `<UNK>` rate.
   - **Caveat:** whitespace `|V|` differs between the training sets, so raw perplexity is not
     strictly comparable even within the whitespace pipeline. WordPiece `V` is fixed and gives the
     cleaner comparison.
   - Frame any scaling remark with Kaplan et al. (2020), and do not claim a power law from two
     points.

**Comparison limit (state in `docs/phase2_summary.md`):** do **not** rank whitespace against WordPiece by raw
perplexity. Their units, vocabularies, normalization, and context spans differ. A falling score
can reflect unknown-token collapse or changed targets rather than robustness. These scores
measure the *surprise of corrupted sequences*, not semantic recovery.

---

## 9. Acceptance checklist (sign together)

- [ ] All Assignment 1 MED and reproducibility tests still pass; p = 0 leaves the input unchanged
- [ ] Noise v2: letters only, insert-after, no re-corruption, sentinels preserved, replay exact
- [ ] Every smoothed distribution sums to 1 for observed **and** unseen histories; all allowed targets > 0
- [ ] Hand vocab `{a, b, end}` gives `3/6`, `2/6`, `1/6`
- [ ] λ nonnegative, sum to 1, selected on clean val, frozen before the noisy grid; `P_mix` normalized
- [ ] Four equally likely targets give 2 bits / PP 4; corpus aggregation matches the hand calculation
- [ ] Unseen whitespace tokens map to `<UNK>`; boundaries reset context; evaluation never mutates counts or `V`
- [ ] Unsmoothed MLE demo shows infinite loss on a zero-probability event
- [ ] p = 0 gives the identical clean score for every seed
- [ ] 222 scores, 6 clean + 72 noisy summary rows saved per training set
- [ ] Every noisy text generated once and shared by both tokenizers; manifests carry source and config hashes
- [ ] Empty-after-noise records kept, `</s>` scored, undefined rates flagged
- [ ] 20 A1 diagnostic pairs reproduce byte-identically
- [ ] Test split never scored, inspected, or used for any choice
- [ ] **§17 progress check:** partner can reproduce one headline table/figure from the README; every
  plotted point traces to saved data + config + record IDs; the write-up states what the metrics
  cannot establish; the next phase can reuse the corpus, manifests, and diagnostics

---

## 10. Deliverables (current scope)

| Artifact | Contents |
|---|---|
| **Repository** | Extended scripts, tests, environment (`requirements.txt` + lockfile), `configs/phase2.yaml`, `phase2.ipynb`, README with Phase 2 run order, updated contribution record. |
| **Results and manifests** | Corpus/split hashes, paired IDs, corruption logs, per-record metrics, aggregate CSV/JSON, figure-generation code. |
| **Phase 2 write-up** | `docs/phase2_summary.md`: data and noise protocol, models and vocabularies, frozen config and λ, clean table, noise curves, diagnostics, 3 traceable examples, corpus-size comparison, limitations and ethics. |

---

## 11. Deferred: ACL paper v0.2

**Not being worked on for now.** The assignment (§13, §16) asks for a 4-page ACL-style PDF plus
LaTeX/Overleaf source that condenses Assignment 1 into the same paper. When we pick it up:

- Build it from `docs/phase2_summary.md` and `docs/phase1_summary.md`. Every number should already
  exist in `results/phase2/`.
- Sections: Abstract (≤ 200 words), Introduction, Related Work, Data and Methods, Experimental
  Setup, Results and Discussion, Conclusion, Limitations, Ethics.
- Sources (§19): Jurafsky & Martin SLP3 ch. 3; Devlin et al. 2019; Kaplan et al. 2020 (cite when
  discussing scaling); Hugging Face WordPiece and perplexity docs; Salesforce WikiText dataset
  card; official ACL style files.
- The paper is 20% of the A2 grade (see §12). Skipping it costs that share unless the instructor
  agrees otherwise.

---

## 12. Assessment weights (§15)

| Category | Weight | Evidence |
|---|---|---|
| Technical implementation | 40% | correct probabilities, boundaries, tokenizer reuse, passing tests |
| Experimental rigor | 20% | frozen splits/config, paired conditions, three seeds, complete logs |
| Analysis and visualization | 20% | correct units, coverage diagnostics, qualified claims, error analysis |
| Cumulative paper *(deferred)* | 20% | ACL format, integration of A1, citations, limitations, clear writing |

---

## 13. Suggested order of work

1. **Together:** Step 0, i.e. write and approve `configs/phase2.yaml` and the ID schemes.
2. **Parallel:**
   - Samrat: A1 val sample → A2 article IDs + full-train set → A3 noise v2 + A8 tests.
   - Prazol: B1 pipelines → B2 vocabularies → B3–B5 counts and models → B8 tests.
3. **Parallel:**
   - Samrat: A4 grid generation → A5 fast MED → A6 diagnostics (needs Prazol's frozen `V`) →
     A7 20-pair regression.
   - Prazol: B6 λ selection on clean val → B7 aggregation.
4. **Together:** verify (§7) with the toy hand calculation, cross-review, clean invariance.
5. **Together:** freeze, then run the `train_a1_slice` grid, then summaries, figures, and error
   analysis.
6. **Together:** run the `train_full` extension and the corpus-size comparison.
7. **Together:** `docs/phase2_summary.md`, README run order, contribution record, and a fresh-checkout
   reproduction of one headline table.

---

## 14. Claims, ethics, and responsible research (§14)

- Carry forward the Phase 1 ethics statement. Character noise is dual-use; this is a robustness
  study of encoders and n-grams, not filter evasion or prompt injection. Use only
  instructor-approved data and models, and never probe third-party systems.
- Add **licensing** (WikiText CC BY-SA 3.0), **privacy** (public encyclopedic text, no personal
  data collected), **bias** (Wikipedia/English coverage), and **compute use** (CPU-only counting,
  bounded sample).
- Report **null or mixed results** as carefully as positive ones. Make no causal claims, and do
  not describe validation results as held-out findings.
- **Test split:** never inspected. Define the final test sampling rule before the semester
  evaluation freeze. If test outcomes are ever seen, notify the instructor and document it.

---

## 15. Open items

- [ ] Obtain the **Core Research Axioms** sheet referenced in §14 (not yet in `docs/assignment/`).
- [ ] Confirm with the instructor that the `train_full` corpus-size run is acceptable as
      **supplementary** evidence alongside the A1-slice baseline.
- [x] Decide `rapidfuzz` vs. a NumPy DP for fast MED. **Decided: NumPy DP** (`src/med/fast.py`, no new
      dependency; matches the reference on 1,000 real pairs, ~13× faster).
- [ ] When the paper resumes: set up the Overleaf project from the official ACL template (see §11).
