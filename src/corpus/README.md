# src/corpus/  (Samrat)

Turns WikiText-2 into a bounded, documented, deterministic slice and audits what is in it.

| File | Purpose |
|---|---|
| `acquire.py` | Download WikiText-2 via `datasets`; copy the raw files into `raw/wikitext-2/`; write `raw/SOURCE.md` with dataset id, resolved version, date, license. Records a SHA-256 per file. |
| `slice.py` | Apply `config.corpus.slice_rule` to the `train` split, assign each kept line an `example_id`, write `derived/slice.jsonl`, and write `derived/slice.sha256` (checksum + rule text). Deterministic: same config -> same slice. |
| `articles.py` | **Phase 2 (A1–A2).** Article IDs from top-level ` = Title = ` headings (`wt2-<split>-doc-NNNNNN`), record builder with exclusion report, back-fill of article IDs for the A1 slice, JSONL + `.sha256` writer. |
| `val_sample.py` | **Phase 2 (A1–A2).** Frozen 500-record validation sample and the two training sets (`train_a1_slice`, `train_full`) under `derived/phase2/`; re-verifies the Phase 1 slice checksum. |
| `audit.py` | Walk the slice character by character, bucket each into a Unicode category (uppercase, lowercase, digit, punctuation, whitespace, symbol, other/non-ASCII), write `derived/category_audit.json`, and drop a summary table into `docs/data_card.md`. |

## Notes
- Dataset id is `Salesforce/wikitext` (the legacy `wikitext` id fails on `datasets` >= 5).
- Use the `wikitext-2-raw-v1` config so original characters survive; document that choice.
- Handle WikiText-2 quirks explicitly: `= = Heading = =` lines, literal `<unk>`, spaced
  punctuation (`don 't`). Whatever you drop, log why.
- Never write into `raw/`. Outputs go to `derived/` and `docs/`.
