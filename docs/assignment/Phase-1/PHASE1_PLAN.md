# CS620 Research Project — Phase 1 (Assignment 1) Plan

**Course:** CS620 Research Project, Fall 2026 (instructor: Chhabra)
**Project:** Robust NLP Under Character Noise
**Phase:** 1 — Foundations
**Team:** 2 students

---

## 1. What Phase 1 must prove

One narrow claim, backed by inspectable artifacts:

> The team can (a) generate **deterministic** character-noise variants, (b) measure their
> **edit distance** correctly, and (c) show how the **fixed BERT WordPiece tokenizer** reacts
> to matched clean/noisy text.

No model training, no performance results, no power-law claims. The output is a *foundation*:
a corpus slice, noise functions, a verified Levenshtein implementation, a tokenizer baseline,
and **20 paired clean/noisy records** that a classmate can regenerate from the README.

**Core rule:** noise is applied to raw UTF-8 text, **before** WordPiece. Anything applied after
tokenization is "token-level noise" and must not be called character noise.

---

## 2. Roles

### Student A — Corpus + Noise
Owns the **input side** of the pipeline: obtaining a clean, documented, bounded slice of
WikiText-2, auditing which characters it contains, and building the noise functions that
corrupt it in a controlled, logged, reproducible way.

**Deliverables:** corpus data card, noise module, corruption-manifest format, related unit tests.

### Student B — Tokenizer + Baseline
Owns the **model-facing side**: loading the WordPiece tokenizer paired with the chosen BERT
checkpoint, fully documenting its configuration, and building the diagnostics that reveal how
tokenization changes between clean and noisy text. Also writes the conceptual
BPE-vs-WordPiece distinction.

**Deliverables:** tokenizer data card, tokenization wrapper + diagnostics, clean baseline stats,
BPE/WordPiece paragraph, related unit tests.
**Constraint:** use the pretrained tokenizer as-is — do **not** train a replacement vocabulary.

### Shared
Minimum Edit Distance implementation + known-answer tests, integration contract, the 20 paired
examples, acceptance checklist, report, data/contribution records.

---

## 3. Step 0 — Shared setup (do together, first)

1. **Repo skeleton** — `raw/` (never edited), `derived/` (generated), `src/`, `tests/`,
   `configs/`, `README.md`, environment lockfile.
2. **ID scheme** — how a corpus example is named, e.g. `wt2-train-000042` (split + line index)
   or a content hash. This ID threads through the noise log, the MED record, and the tokenizer
   diagnostics.
3. **Pair record schema** — one row (JSON + CSV) per pair with every field the rubric lists
   (see Step 7).
4. **Global seed** — one integer in a config file. Per-example seeds are derived from it
   deterministically (e.g. `hash(global_seed, example_id)`).
5. **Config file** — checkpoint name, seed, noise operation, intensity, slice rule, max length.
   Everything reproducible lives here; nothing hard-coded.
6. **BERT checkpoint** — pick one now (e.g. `bert-base-uncased`) and never change it.

---

## 4. Student A steps — Corpus + Noise

1. **Acquire the corpus.** Download WikiText-2. Record dataset version, source (URL / HF dataset
   id + revision), license/terms. Save raw files untouched in `raw/`.
2. **Choose a deterministic slice.** Define an explicit rule — e.g. "first 500 non-empty,
   non-heading lines of the `train` split." Record the rule, the example count, and a
   **SHA-256 checksum** of the slice file. Develop on `train`; use `validation` only for
   spot-checks; never inspect `test`.
3. **Document encoding & normalization.** Confirm UTF-8. Write down handling of WikiText-2
   quirks: `= = Section = =` heading lines, literal `<unk>` tokens, pre-tokenized spacing
   around punctuation (`don 't`, `, `). Any stripping happens **in code**, output to
   `derived/`, never editing `raw/`.
4. **Character-category audit.** Using `unicodedata`, count characters by category (upper/lower
   letters, digits, punctuation, whitespace, symbols, non-ASCII). Drives the noise alphabet and
   the report's "category statistics."
5. **Implement the three noise operations** on raw strings:
   - **Insertion** — add a character at position *i*; record inserted char + index.
   - **Deletion** — remove the character at position *i*; record deleted char + index.
   - **Substitution** — replace char at *i* with a different one; record source char,
     replacement, index.

   Each configurable by an **intensity** parameter (define precisely — e.g. per-character
   probability, or a fixed edit count) and driven by a **seeded RNG** passed in per example.
6. **Emit a corruption manifest** per generated example: `example_id`, `seed`, `operation`,
   `intensity`, and an ordered **operation log** of `(op, index, source_char, replacement_char)`.
7. **Build in the controls:**
   - `intensity = 0` -> output byte-identical to input, empty log, MED 0.
   - Same seed + same input -> identical output **and** identical log.
   - Replaying the log against the clean string reproduces the noisy string exactly.
   - Whitespace / Unicode handling documented (operating pre-tokenization, so no special
     tokens exist yet — note this).
8. **Write unit tests** for determinism, the intensity-0 identity, and log <-> output
   consistency.

---

## 5. Student B steps — Tokenizer + Baseline

1. **Load the paired tokenizer** from the checkpoint named in the shared config
   (e.g. `AutoTokenizer.from_pretrained("bert-base-uncased")`).
2. **Record the tokenizer data card:**
   - Library + version (`transformers` / `tokenizers`).
   - Checkpoint name **and** revision/commit hash.
   - Casing & normalization (uncased -> lowercasing + accent stripping; NFC/NFD; control-char
     stripping).
   - Special tokens and IDs: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`.
   - Vocabulary size, vocab file source, **checksum** of the vocab file.
   - Model max length (512 for BERT-base) and truncation behavior.
3. **Write the tokenization wrapper:** text -> `{pieces, input_ids, with/without special tokens,
   truncated?}`, plus helpers to detect **continuation pieces** (leading `##`) and **`[UNK]`**.
4. **Define the diagnostics** (per text and aggregate):
   - **Token count / fertility** — tokens per word; does noise fragment words?
   - **Token-sequence change rate** — how much the subword sequence shifts clean -> noisy.
   - **`[UNK]` rate** — does the tokenizer lose coverage?
   - **Continuation-piece rate** — do internal fragments increase?
   - **Truncation** — did the sequence exceed max length?
5. **Compute the clean baseline.** Tokenize Student A's clean slice; record aggregate fertility,
   `[UNK]` rate, continuation rate, length distribution. This is the clean-branch "baseline
   metrics."
6. **Write the BPE vs WordPiece paragraph.** One or two precise sentences, e.g.: *BPE grows its
   vocabulary by repeatedly merging the most frequent adjacent symbol pair (pure frequency);
   WordPiece instead adds the merge that most increases the training corpus's likelihood under a
   unigram language model (a likelihood/association score, roughly
   `count(ab) / (count(a).count(b))`), not raw frequency alone. BERT's WordPiece marks every
   non-word-initial subword with a leading `##`.* Do **not** call `##` a "BPE merge."
7. **Unit tests:** special-token IDs stable, a known string tokenizes to a known piece list,
   `[UNK]` and `##` detection work, truncation flag fires past 512.

---

## 6. Shared component — Minimum Edit Distance

Either student may own the code (pairs naturally with Student A's noise measurement); **both
review it**.

1. Implement **unit-cost Levenshtein** with the given recurrence:
   `D[i,j] = min(D[i-1,j]+1, D[i,j-1]+1, D[i-1,j-1] + [s_i != t_j])`,
   `D[i,0] = i`, `D[0,j] = j`. Insert = delete = substitute = **cost 1**.
2. **Known-answer tests** (all must pass; output saved):

   | source        | target     | expected MED | purpose                       |
   |---------------|------------|--------------|-------------------------------|
   | `same`        | `same`     | 0            | identity                      |
   | `` (empty)    | `nlp`      | 3            | empty-string boundary         |
   | `kitten`      | `sitting`  | 3            | mixed substitutions/insertion |
   | `flaw`        | `lawn`     | 2            | deletion/insertion            |
   | `intention`   | `execution`| 5            | lecture reference case        |

3. Add **normalized MED** = `MED / max(1, clean_char_count)`.

---

## 7. Step 6 — Integration (both students)

1. **Freeze the shared schema and IDs** — reconcile Student A's manifest fields with Student B's
   diagnostic fields into one pair record.
2. **Select the same 20 examples** deterministically from the slice (e.g. first 20, or
   `random.sample` with the fixed seed). Document the rule.
3. **Spread noise conditions across the 20** — vary operation (insert/delete/substitute) and
   intensity so the set is not monotone. Need **at least one pair with a meaningful WordPiece
   change** (a word fragmenting into extra pieces, or an `[UNK]` appearing).
4. **Generate each pair end-to-end:** clean text -> noise (+ manifest) -> MED + normalized MED
   -> tokenize clean & noisy -> diagnostics -> **one-sentence observation**
   ("substitution at index 4 split `robustness` into `robust ##ness ##x`, +2 tokens").
5. **Cross-review unit tests** — each student reads the other's.
6. **Run the acceptance checklist together and sign it** (Step 8).

---

## 8. Acceptance checklist (sign together)

- [ ] Same seed reproduces the same noise + log
- [ ] Intensity 0 returns the source, MED 0
- [ ] Replaying the recorded edits reproduces the noisy output
- [ ] Clean and noisy use identical tokenizer settings
- [ ] At least one pair shows a real WordPiece change, and BPE is not mislabeled
- [ ] Raw files unchanged; derived files + encoding documented
- [ ] All known-answer + determinism tests pass and their outputs are saved
- [ ] Package has 20 traceable clean/noisy pairs + tokenizer diagnostics
- [ ] README, environment, seed, configuration, and contribution record complete
- [ ] Ethics section states authorized-use limits and dual-use implications

---

## 9. Step 7 — Deliverables package

| Artifact | Contents |
|---|---|
| **Technical report (PDF, 3-5 pp + refs)** | 1. Objective (question + place in the semester project) · 2. Data (corpus source, slice, splits, UTF-8 handling, category stats) · 3. Methods (noise design, fixed tokenizer/checkpoint, MED definition, config) · 4. Verification + findings (test results, 20-pair summary, diagnostics, 2-3 representative examples, no causal claims) · 5. Limitations + ethics (coverage limits, dual-use, authorized-testing boundary, next steps) |
| **Repo / Colab** | runnable source, tests, configs, environment notes, README with run order |
| **Corpus audit / data card** | source + version, license/terms, encoding, split usage, slice rule, count, checksum |
| **Evidence + tests** | 20 paired records in human-readable **and** CSV/JSON form; saved MED test output; saved determinism-test output |
| **Contribution record** | short statement of who did what + shared integration decisions |

### Per-pair record fields (the 20)

`example_id`, source split/index or hash, `clean_text`, `noisy_text`, `seed`, `operation`,
`intensity`, `edit_location`, `operation_log`, `MED`, `normalized_MED`, clean pieces + count,
noisy pieces + count, continuation-piece counts, `[UNK]` flags, truncation flags,
token-count change, `observation`.

---

## 10. Report structure (rubric-aligned)

| Section | Minimum content |
|---|---|
| 1. Objective | assignment question and its role in the semester project |
| 2. Data | corpus source, slice, splits, UTF-8 handling, category statistics |
| 3. Methods | noise design, fixed tokenizer/checkpoint, MED definition, configuration |
| 4. Verification + findings | tests, 20-pair summary, diagnostics, representative examples; no causal overclaim |
| 5. Limitations + ethics | coverage limits, possible misuse, authorized-testing boundary, next steps |

---

## 11. Assessment weights

| Criterion | Weight |
|---|---|
| Data integrity + audit | 20% |
| Noise implementation | 20% |
| Tokenizer baseline | 15% |
| MED + verification | 15% |
| Integration + evidence | 15% |
| Report + responsibility | 15% |

---

## 12. Suggested order of work

1. **Together:** Step 0 (repo, IDs, schema, config, checkpoint choice).
2. **Parallel:** Student A -> corpus slice + audit (A1-A4); Student B -> load + document
   tokenizer (B1-B3) + BPE/WordPiece paragraph.
3. **Parallel:** Student A -> noise + manifest + tests (A5-A8); Student B -> diagnostics +
   baseline + tests (B4-B7). One student writes MED + known-answer tests.
4. **Together:** integration — generate the 20 pairs, review tests, sign the checklist.
5. **Together:** write the report, assemble the data card and contribution record, verify a
   clean checkout reproduces the 20-pair table.

---

## 13. Ethics note (required in report)

Character noise is dual-use: defensively it studies robustness, but it can also obfuscate
harmful instructions or evade filters. This is a BERT encoder study, **not** a prompt-injection
exercise. Use only instructor-approved data and models; do not probe third-party systems or
attempt filter evasion; use benign examples; frame observations around detection,
normalization, robustness, and mitigation.
