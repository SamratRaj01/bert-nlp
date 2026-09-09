# derived/

Everything generated from `raw/` by `phase1.ipynb`. Safe to delete and regenerate.

These files **are** committed to git: generation is deterministic (fixed seed + pinned
config), so re-running produces byte-identical files and a grader can inspect the slice,
checksum, corruption manifest, and baseline without executing anything.

## Expected contents
| File | Produced by | Contents |
|---|---|---|
| `slice.jsonl` | `src.corpus.slice` | the deterministic corpus slice, one example per line (`example_id`, `text`, `source_split`, `source_index`) |
| `slice.sha256` | `src.corpus.slice` | checksum of `slice.jsonl` and the selection rule used |
| `category_audit.json` | `src.corpus.audit` | character-category counts for the slice |
| `corruption_manifest.jsonl` | `src.noise` / `src.integration` | one manifest row per generated noisy example |
| `tokenized_clean.jsonl` | `src.tokenizer` | tokenizer output + diagnostics for the clean slice (baseline) |

## Rules
- Nothing here is hand-edited. If a file looks wrong, fix the code and rerun.
- These files are reproducible from `raw/` + `configs/config.yaml` + the fixed seed.
