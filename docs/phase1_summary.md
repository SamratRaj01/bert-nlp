# Phase 1 Summary — Foundations

**Course:** CS620 Research Project (instructor: Chhabra) · **Project:** Robust NLP Under Character Noise
**Team:** Samrat (Student A — corpus + noise), Prazol (Student B — tokenizer + baseline)

This is a narrative summary of everything built and verified in Phase 1. It complements, and
does not replace, `README.md` (run instructions), `docs/data_card.md`, `docs/tokenizer_card.md`,
`docs/contribution_record.md`, `docs/report_outline.md`, and `docs/assignment/Phase-1/PHASE1_PLAN.md`
(the original assignment plan). Every claim below is backed by a file in this repo.

## 1. What Phase 1 set out to prove

One narrow claim: the team can (a) generate **deterministic** character-noise variants of text,
(b) measure edit distance between clean/noisy text **correctly** (unit-cost Levenshtein), and
(c) show how the **fixed BERT WordPiece tokenizer** reacts to matched clean/noisy pairs. No model
training and no performance claims — Phase 1 is a foundation for later phases (baseline →
robustness → scale).

**Core rule enforced throughout:** noise is applied to raw UTF-8 text *before* WordPiece
tokenization. Anything applied after tokenization would be "token-level noise" and is out of
scope here.

## 2. Pipeline built (`phase1.ipynb`, backed by `src/`)

The notebook is the single runnable entry point (`Kernel → Restart & Run All`) and orchestrates
ten stages, each backed by a tested `src/` module:

1. **Corpus acquisition & provenance** (`src/corpus/acquire.py`) — downloads WikiText-2, saves
   raw files verbatim to `raw/wikitext-2/`, and records source/version/license into
   `raw/SOURCE.md` plus a SHA-256 per file. `raw/` is never edited after this.
2. **Deterministic slice** (`src/corpus/slice.py`) — applies the slice rule to `train`, assigns
   each kept line an `example_id`, writes `derived/slice.jsonl` + `derived/slice.sha256`. Pure
   function of (raw file, config) — no randomness.
3. **Character-category audit** (`src/corpus/audit.py`) — buckets every character in the slice by
   Unicode category and writes `derived/category_audit.json`, feeding the stats table in
   `docs/data_card.md`.
4. **Minimum Edit Distance** (`src/med/levenshtein.py`) — unit-cost Levenshtein
   (insert = delete = substitute = 1) plus `normalized_med = med / max(1, len(clean))`. Verified
   against 5 known-answer cases.
5. **Character-noise operations** (`src/noise/`) — deterministic insertion / deletion /
   substitution driven by a per-example seeded RNG, with intensity as an independent
   per-character Bernoulli probability. Every edit is logged as
   `(op, index, source_char, replacement_char)` so the noisy text can be replayed exactly from
   the clean text + log.
6. **WordPiece tokenizer baseline** (`src/tokenizer/`) — loads `bert-base-uncased` (pinned
   revision) as-is, no vocabulary training, and computes a clean-branch baseline
   (`derived/clean_baseline.json`).
7. **20 paired clean/noisy records** (`src/integration/generate_pairs.py`) — for each pair:
   apply noise → write manifest row → MED + normalized MED → tokenize clean & noisy → diagnostics
   → one-line observation.
8. **Paired findings & diagnostics** — aggregated view over the 20 pairs (fertility,
   continuation-piece rate, `[UNK]` rate, sequence-change rate).
9. **Acceptance checklist — programmatic verification** — every checklist item re-checked in
   code and saved to `evidence/test_outputs/acceptance.txt`, plus the full `pytest` run.
10. **Artifacts produced** — a manifest of every file the pipeline writes.

## 3. Data

- **Source:** `Salesforce/wikitext`, config `wikitext-2-raw-v1` (raw variant preserves original
  characters), revision `b08601e...685c3`. License: CC BY-SA 3.0. Retrieved 2026-09-09.
- **Row counts:** train 36,718 / validation 3,760 / test 4,358. Only `train` was used for
  development; `validation` reserved for spot-checks; `test` untouched.
- **Slice rule:** first 500 `train` lines that, after `str.strip()`, are non-empty, not
  `= heading =` lines, and 40–600 characters long. 500 examples, checksummed
  (`derived/slice.sha256` = `ea400026...7e937`).
- **Encoding:** UTF-8 end-to-end. Only transform before noise/tokenization is `str.strip()`;
  WikiText's literal `<unk>` strings and spaced punctuation (`don 't`, ` , `) are left as-is.
- **Character-category audit (500 examples, 167,391 characters, 216 non-ASCII code points):**
  lowercase 72.9%, whitespace 18.4%, uppercase 3.6%, punctuation 3.2%, digit 1.9%, symbol/other
  negligible.

## 4. Methods

### Noise model
- Three operations: **insertion**, **deletion**, **substitution** on Unicode code points (not
  bytes).
- **Intensity** = independent per-character Bernoulli probability of being edited.
  `intensity = 0` → output identical to input, empty log, MED 0.
- Per-example RNG seeded deterministically from `(global_seed, example_id)` — same inputs always
  produce the same noisy text and the same operation log.
- Substitution never replaces a character with itself.
- `replay(clean_text, operation_log)` reproduces the noisy text exactly from the log — verified
  for all 20 pairs.

### Tokenizer
- Fixed checkpoint `bert-base-uncased`, pinned revision `86b5e09...723594`, `transformers`
  5.16.1, `BertTokenizer` (WordPiece), `do_lower_case=True`. Used exactly as pretrained — no
  vocabulary retraining, so clean and noisy branches are guaranteed to share settings.
- Special tokens: `[PAD]`=0, `[UNK]`=100, `[CLS]`=101, `[SEP]`=102, `[MASK]`=103. Vocab size
  30,522, `max_length` 512.
- Diagnostics computed per pair: token count / fertility, continuation-piece rate (leading `##`),
  `[UNK]` rate, truncation flag, and `sequence_change_rate` (normalized edit distance over token
  sequences, clean vs. noisy).
- Clean-branch baseline over the full 500-example slice (`derived/clean_baseline.json`): mean
  70.3 tokens/example, mean fertility 1.134, mean continuation-piece rate 7.7%, **0** `[UNK]`
  tokens, **0** truncated examples.

### MED (shared)
Unit-cost Levenshtein with the assignment recurrence
`D[i,j] = min(D[i-1,j]+1, D[i,j-1]+1, D[i-1,j-1] + [s_i != t_j])`, `D[i,0]=i`, `D[0,j]=j`.
Normalized MED = `MED / max(1, len(clean_text))`.

### BPE vs. WordPiece
BPE grows its vocabulary by repeatedly merging the most frequent adjacent symbol pair (pure
frequency). WordPiece instead adds the merge that most increases the training corpus's
likelihood under a unigram language model (a likelihood/association score, roughly
`count(ab) / (count(a) * count(b))`), not raw frequency alone. BERT's WordPiece marks every
non-word-initial subword with a leading `##`.

## 5. Verification results

**Known-answer MED tests** — all 5 pass (`evidence/test_outputs/med_known_answers.txt`):

| source | target | expected | got |
|---|---|---|---|
| `same` | `same` | 0 | 0 |
| `` | `nlp` | 3 | 3 |
| `kitten` | `sitting` | 3 | 3 |
| `flaw` | `lawn` | 2 | 2 |
| `intention` | `execution` | 5 | 5 |

**Unit test suite:** 26/26 tests pass (`python -m pytest -q`, saved to
`evidence/test_outputs/pytest.txt`) — covering MED boundaries, noise determinism/intensity-0/replay
invariants, and tokenizer special-token/`##`/`[UNK]`/truncation behavior.

**Acceptance checklist — all 12 programmatic checks pass**
(`evidence/test_outputs/acceptance.txt`): same-seed reproducibility, intensity-0 identity, replay
consistency (all 20 pairs), shared tokenizer config across branches, at least one pair with a
meaningful WordPiece change, raw-file checksums unchanged, slice checksum present, 20 records
written in all three formats, tokenizer/corpus data cards written, clean baseline persisted, and
the full unit-test suite passing.

## 6. The 20 paired clean/noisy records

Generated by `pair_schedule`: pair 0 is the intensity-0 control; the remaining 19 cycle through
the three operations (insertion / deletion / substitution) at three severities
(0.05 / 0.10 / 0.15 — `base`, `2×base`, `3×base`), so the set is not monotone. Full text, MED,
and diagnostics for every pair are in `evidence/pairs/pairs.md` (also `.jsonl` / `.csv`).

- **MED ranged 0–70**, normalized MED 0.000–0.200.
- Token count changes ranged **+0 to +99** pieces; every non-control pair fragmented into more
  WordPiece tokens than its clean counterpart (continuation-piece counts rose sharply — e.g. pair
  `wt2-train-000031`, substitution @ intensity 0.15, MED 70, tokens 98→186, +99 continuation
  pieces).
- Two intensity-0/near-zero controls confirm the identity invariant: `wt2-train-000004`
  (intensity 0.0, MED 0, tokens unchanged) and `wt2-train-000076` (intensity 0.05 but too short
  for any position to be sampled — MED 0, tokens unchanged).
- Sequence-change rate (normalized token-sequence edit distance) climbed with intensity, e.g. up
  to 1.75 for `wt2-train-000068` (16-token sentence, substitution @ 0.15).
- **Qualitative pattern:** insertion and substitution reliably increase `[UNK]`-free but
  heavily-fragmented tokenizations (many extra `##` continuation pieces) rather than producing
  `[UNK]`s — consistent with WordPiece falling back to character-level fragments instead of
  losing coverage. This is reported as an observation only; Phase 1 makes no causal claims.

## 7. Repository artifacts produced

| Path | Contents |
|---|---|
| `raw/wikitext-2/*.raw`, `raw/SOURCE.md` | untouched corpus + provenance, checksummed |
| `derived/slice.jsonl`, `derived/slice.sha256` | deterministic 500-example slice + checksum |
| `derived/category_audit.json` | character-category counts |
| `derived/clean_baseline.json` | clean-branch tokenization baseline (500 examples) |
| `derived/corruption_manifest.jsonl` | seed + operation log per noisy example |
| `docs/data_card.md` | generated corpus data card |
| `docs/tokenizer_card.md` | generated tokenizer data card |
| `evidence/pairs/pairs.{jsonl,csv,md}` | the 20 paired clean/noisy records, 3 formats |
| `evidence/test_outputs/med_known_answers.txt`, `pytest.txt`, `acceptance.txt` | saved proof of all checks passing |

## 8. Contribution record (see `docs/contribution_record.md` for the authoritative version)

- **Samrat (Student A):** corpus acquisition/provenance, deterministic slice + checksum,
  character-category audit, noise operations/RNG/manifest, `tests/test_noise.py`.
- **Prazol (Student B):** tokenizer wrapper + data card, tokenization diagnostics + clean
  baseline, BPE-vs-WordPiece write-up, `tests/test_tokenizer.py`.
- **Shared:** MED implementation + tests, integration schema (`src/integration/`), 20-pair
  generation, acceptance checklist, this documentation.

## 9. Limitations & ethics

- Coverage is narrow by design: one corpus (WikiText-2), one checkpoint (`bert-base-uncased`),
  a 500-line slice, a synthetic per-character noise model — not claimed to generalize beyond
  Phase 1's scope.
- **Dual-use note:** character noise can in principle obfuscate harmful content or evade filters.
  This work is a BERT-encoder tokenization study only, not a prompt-injection or filter-evasion
  exercise. Only instructor-approved data/models were used; no third-party systems were probed;
  all examples are benign encyclopedia text from WikiText-2.
- **Next step:** Phase 2 — baseline model evaluation on top of this foundation.
