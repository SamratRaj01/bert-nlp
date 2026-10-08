# Phase 2 Summary — Probabilistic Language Modeling and Input Noise

**Course:** CS620 Research Project (instructor: Chhabra) · **Project:** Robust NLP Under Character Noise
**Team:** Samrat (Student A — extend pipeline), Prazol (Student B — build engine)

This is the Phase 2 write-up. It records methods, results and caveats, and will feed paper v0.2 later; the ACL
paper itself is deferred (see `docs/assignment/Phase-2/PHASE2_PLAN.md` §11). Every number here is
produced by `phase2.ipynb` and saved under `results/phase2/` or `evidence/phase2/`. **All results are
preliminary validation (development) evidence:** the validation sample also selected λ, and the
test split was never scored or inspected.

> **Research question:** How do character-noise type and intensity change token coverage,
> fragmentation, and predictive behavior under fixed whitespace and WordPiece tokenization?

## 1. Data and noise protocol

- **Corpus:** `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision `b08601e…685c3` (unchanged from
  Phase 1; CC BY-SA 3.0).
- **Validation sample (frozen):** validation lines in file order passing the Phase 1 line filter
  (non-empty, non-heading, 40–600 characters), first 500. 846 lines are eligible across 57 articles; the
  500 kept come from **36 articles**, 166,255 characters, 31,573 whitespace tokens.
  `derived/phase2/val_sample.jsonl` (SHA-256 `4894ff5d…8a7b`). Exclusions while scanning: 764 empty, 364
  heading, 38 too short, 511 too long. The 40–600 band is the MED length cap; it is applied before
  corruption and noisy text is never truncated.
- **Article IDs:** every record carries `wt2-<split>-doc-NNNNNN`, the line index of its top-level
  ` = Title = ` heading, so later bootstrap intervals can resample by document.
- **Training sets:** `train_a1_slice` = the Phase 1 slice, unchanged (500 records, 42 articles,
  31,339 words; checksum re-verified) — **primary**. `train_full` = every non-empty, non-heading train
  line (17,556 records, 629 articles, 2,007,146 words) — **corpus-size extension**.
- **Noise rule `a2_default`** (`src/noise/rule_v2.py`): eligible = ASCII letters of the original
  string; one Bernoulli(p) draw per eligible position; deletion; substitution by a different lowercase
  a–z letter (`source.lower()` excluded); insertion of one lowercase letter *after* the eligible letter.
  Digits, punctuation, spaces and sentinels (`@-@`, `@,@`, `@.@`) are never edited and inserted letters
  are never corrupted. Rates p ∈ {0.01, 0.05, 0.10, 0.20}, seeds {11, 23, 47}; per-record RNG
  `example_seed(seed, f"{record_id}|{op}|{p}")`. The Phase 1 rule is kept only for the 20-pair regression.
- **Grid:** 1 clean + 36 noisy datasets, each generated once (`derived/phase2/noisy/`), with a full
  manifest per record (IDs, seed, rule version, eligible count, edit log, realized edits, MED, normalized
  MED, source-text and protocol hashes). Both tokenizers read the same files.

**Realized severity** (`results/phase2/severity.csv`, figure `severity.png`). The realized edit rate
matches p (e.g. 0.2009 / 0.2007 / 0.2002 for deletion / insertion / substitution at p = 0.20). Mean
normalized MED is ≈ 0.76 p because letters are 76% of all characters and normalized MED divides by all
characters (at p = 0.20: 0.151–0.152, range 0.02–0.29 across records). For insertion and deletion, MED
equals the edit count exactly; only substitution produced MED < edits (6 of 18,000 noisy records),
where a shifted run aligns more cheaply as one insertion + one deletion. **p, MED and the logged edit
count are three different quantities.** No record became empty.

## 2. Classical methods

- **Tokenizers:** *whitespace* = `text.split()` (case and punctuation kept); *WordPiece* = the pinned
  `bert-base-uncased` tokenizer (revision `86b5e09…`), pieces only — no `[CLS]`/`[SEP]`, padding or
  truncation. The comparison is between complete pipelines (normalization + segmentation + vocabulary).
- **Vocabularies (frozen):** whitespace V = training tokens with count ≥ 2 + `<UNK>` + `</s>`
  (|V| = 2,918 on the slice, 44,205 on full train); WordPiece V = BERT vocabulary minus `[PAD] [CLS]
  [SEP] [MASK]` and 994 `[unused…]` entries, plus `</s>` (|V| = 29,525, identical for both training sets).
  `<s>` is context only.
- **Framing:** one sequence per record, `<s> <s> w1 … wn </s>`, context reset per record, one `</s>`
  scored. T counts content targets plus one `</s>` per record.
- **Models:** add-one `q_k(w|h) = (C(h,w)+1)/(C(h)+|V|)` for k = 1, 2, 3 on sparse integer-keyed
  counts; scored configurations q1, q2 and `P_mix = λ1 q1 + λ2 q2 + λ3 q3`. Unsmoothed MLE is shown
  separately (`evidence/phase2/mle_demo.txt`): an unseen target gets probability 0 and loss ∞.
- **λ selection:** 15 grid points (step 0.25), minimum clean-validation loss per tokenizer and per
  training set, ties to larger λ1 then λ2; written to `configs/phase2.yaml` before any noisy scoring.

| training set | whitespace λ | WordPiece λ |
|---|---|---|
| `train_a1_slice` | (1.00, 0.00, 0.00) | (1.00, 0.00, 0.00) |
| `train_full` | (0.75, 0.25, 0.00) | (0.50, 0.50, 0.00) |

- **Loss:** `H_bits = Σ −log2 P / Σ T`, `PP = 2^H_bits`, aggregated over the corpus before dividing;
  sentence perplexities are never averaged. Per-record NLL sums (bits and nats) and T are saved with
  article, record and condition IDs (`per_record.parquet`).

## 3. Verification

- 56/56 unit tests pass (26 Phase 1 + 30 Phase 2; `evidence/phase2/pytest.txt`).
- Hand calculations match the program to 1e-12 (`evidence/phase2/hand_check.md`): the {a, b, end} case
  gives 3/6, 2/6, 1/6; four equally likely targets give 2 bits / PP 4; a 2-record corpus aggregates to
  the hand value; P_mix sums to 1.
- On the real engines, every smoothed distribution sums to 1 for observed, start-symbol and unseen
  histories, with all targets > 0.
- Fast MED (`src/med/fast.py`) agrees with the Phase 1 reference on the known answers and on 1,000 real
  clean/noisy pairs (13× faster; `evidence/phase2/fast_med_check.txt`).
- p = 0 regenerated under every seed and operation gives exactly the six clean scores.
- The 20 Phase 1 diagnostic pairs regenerate byte-identically (`evidence/phase2/a1_pairs_regression.txt`).
- All 16 acceptance checks pass (`evidence/phase2/acceptance.txt`).

## 4. Results (`train_a1_slice`, primary)

**Clean validation scores** (compare within a tokenizer only):

| tokenizer | model | H (bits/target) | PP | T |
|---|---|---|---|---|
| whitespace | q1 | 6.955 | 124.1 | 32,073 |
| whitespace | q2 | 8.258 | 306.2 | 32,073 |
| whitespace | P_mix | 6.955 | 124.1 | 32,073 |
| WordPiece | q1 | 10.890 | 1,897.9 | 35,971 |
| WordPiece | q2 | 13.654 | 12,894.3 | 35,971 |
| WordPiece | P_mix | 10.890 | 1,897.9 | 35,971 |

**Noisy loss at p = 0.20** (3-seed mean; the min–max range across seeds is at most 0.05 bits in any of the 72 cells). All 72 rows are in
`results/phase2/train_a1_slice/summary.csv`.

| tokenizer | model | insertion ΔH | deletion ΔH | substitution ΔH |
|---|---|---|---|---|
| whitespace | q1 / P_mix | −1.918 | −1.656 | −1.912 |
| whitespace | q2 | −2.013 | −1.635 | −2.003 |
| WordPiece | q1 / P_mix | +1.998 | +1.199 | +1.896 |
| WordPiece | q2 | +0.795 | +0.634 | +0.775 |

Observed patterns:

1. **The two pipelines move in opposite directions.** Whitespace loss *falls* steadily with p and
   WordPiece loss *rises*, for all three operations and models (`loss_vs_rate_train_a1_slice.png`).
2. **Whitespace "improvement" is unknown-token collapse.** The whitespace `<UNK>` rate rises from 0.282
   (clean) to 0.58 at p = 0.20. With a 2,918-type vocabulary, `<UNK>` is the single most frequent
   training target (q1(`<UNK>`) = 0.112, 3.16 bits), so every corrupted word becomes a cheap, highly
   predictable symbol. The target count barely moves (T unchanged for insertion and substitution; −356 for
   deletion, where single-letter words vanish).
3. **WordPiece never loses coverage; it fragments.** The WordPiece `[UNK]` count is exactly 43 in every
   condition (non-ASCII characters in the clean text), so its *rate* falls only because the denominator
   grows. Pieces per original word rise from 1.12 to 2.21 (insertion), 1.95 (substitution) and 1.49
   (deletion) at p = 0.20, and the continuation-piece rate rises from 0.08 to 0.53 / 0.47 / 0.31. T grows
   from 35,971 to 70,141 under 20% insertion. Loss rises because targets become rare `##` fragments.
4. **Operation order is consistent.** Insertion ≥ substitution > deletion in WordPiece loss increase and
   fragmentation at every rate. Deletion shortens words, and short fragments are more often whole vocabulary
   entries.
5. **Context does not help on the slice.** With add-one smoothing and only ~32k training targets, q2 is
   worse than q1 (context benefit H(q1) − H(q2) = −1.30 bits whitespace, −2.76 bits WordPiece, clean), and
   λ selection puts all weight on q1, so P_mix = q1 and its context benefit is 0 in every condition.
6. **Seeds barely matter at this sample size.** The largest min–max range across the three seeds is
   0.052 bits (0.060 for `train_full`), small next to the noise effects.

**Three traceable examples** (`results/phase2/train_a1_slice/examples.md`):

- *Challenges a simple robustness story:* `wt2-valid-000924` (article `wt2-valid-doc-000873`,
  `sub-p0.20-s11`, whitespace). Per-target loss falls 8.98 → 4.11 bits because 5 of 6 words become
  `<UNK>`. The model is not more robust; the targets became easier.
- `wt2-valid-000905` (`sub-p0.20-s11`, WordPiece): "posted the following statistics" fragments into
  `pr ##ss ##x ##d th ##t f ##gl ##low ##ick …`; T 8 → 20 and loss +3.18 bits per target.
- `wt2-valid-001855` (`del-p0.20-s11`, WordPiece): 97 deleted letters make the text shorter, yet T grows
  103 → 160. Total NLL and T both change, so the noisy and clean per-target losses score different
  prediction problems.

## 5. Corpus-size extension (`train_full`)

Same validation sample, grid, rule and tie-break; λ re-selected on clean validation
(`results/phase2/corpus_size_comparison.csv`, `corpus_size.png`).

| tokenizer | clean H P_mix (slice → full) | ΔH ins p=0.20 (slice → full) | ΔH del | ΔH sub | clean unknown rate (slice → full) |
|---|---|---|---|---|---|
| whitespace | 6.955 → 9.863 | −1.918 → −2.507 | −1.656 → −1.532 | −1.912 → −2.381 | 0.282 → 0.052 |
| WordPiece | 10.890 → 10.068 | +1.998 → +2.357 | +1.199 → +1.544 | +1.896 → +2.237 | 0.0012 → 0.0012 |

- **WordPiece** (fixed V, the cleaner comparison): more training text lowers clean loss by 0.82 bits and
  gives context real weight (λ = 0.5 / 0.5; clean context benefit +0.34 bits). But the noise penalty
  **grows** (e.g. +2.36 vs +2.00 bits at 20% insertion). A better clean model has more to lose when targets
  turn into fragments. The context benefit shrinks with p and turns slightly negative under 20% deletion
  (−0.03 bits): λ were frozen on clean text.
- **Whitespace:** clean loss *rises* 6.96 → 9.86 bits with 64× more text. This is not a worse model. The
  vocabulary grows 15× and the `<UNK>` rate drops 0.28 → 0.05, so far fewer targets are the cheap `<UNK>`.
  Whitespace perplexities across training sets measure different prediction problems.
- Two training sizes are two points, not a scaling curve; no power-law or scaling claim is made
  (cf. Kaplan et al., 2020).

## 6. What these metrics cannot establish (limitations)

- **Raw perplexity cannot rank whitespace against WordPiece.** Units (words vs. pieces), vocabularies,
  normalization (case and accents) and context spans differ, and ratios or loss differences do not
  remove that.
- **A falling score is not robustness.** Lower whitespace loss under noise reflects `<UNK>` collapse;
  WordPiece changes the targets themselves (T nearly doubles under 20% insertion). Per-target losses
  across conditions therefore score different prediction problems.
- **Scores measure the surprise of corrupted sequences, not semantic recovery.** Nothing here says
  whether meaning survives the noise. That needs the neural and downstream evaluations planned for
  later phases.
- **Add-one smoothing dominates.** With large |V| it over-smooths so heavily that q2 loses to q1 and
  the trigram never receives weight on the 0.25 grid. Better smoothers (Kneser–Ney, finer λ grids)
  could change the context-benefit conclusions.
- **Development evidence only.** λ were tuned on the same validation sample that is reported; 500
  records from 36 articles is a bounded sample. Uncertainty is shown only as the min–max over three
  corruption seeds, not as confidence intervals. Per-record NLL sums and article IDs are saved for
  later paired document-bootstrap intervals.
- **Single corpus and language** (English Wikipedia), one noise family (uniform random letters, no
  keyboard or phonetic noise), and one tokenizer checkpoint.

## 7. Ethics and responsible use

Character noise is dual-use: it is studied here to measure robustness, but the same corruptions
could obfuscate text to evade filters. This is a study of n-gram and (later) BERT encoder behaviour,
not filter evasion or prompt injection. Only instructor-approved public data (WikiText-2,
CC BY-SA 3.0, encyclopedic text with no personal data collected) and the public
`bert-base-uncased` tokenizer are used, and no third-party systems are probed. Wikipedia's
English-centric, topic-skewed coverage limits generality. Compute use is small: CPU-only counting,
about one minute per full run. Null and counter-intuitive results (whitespace loss falling under
noise) are reported as carefully as expected ones.

## 8. Reproduce

```
.venv/bin/jupyter nbconvert --to notebook --execute --inplace phase2.ipynb
```

The headline tables are `results/phase2/train_a1_slice/summary.csv` and
`results/phase2/corpus_size_comparison.csv`. Re-running reproduces the identical files: the protocol
is fixed in `configs/phase2.yaml`, and every RNG is keyed by seed and record ID.
