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

## Phase 2 (`derived/phase2/`, written by `phase2.ipynb`)
| File | Produced by | Contents |
|---|---|---|
| `val_sample.jsonl` (+ `.sha256`) | `src.corpus.val_sample` | 500 frozen validation records with article IDs |
| `train_a1_slice.jsonl`, `train_full.jsonl` (+ `.sha256`) | `src.corpus.val_sample` | the two clean training sets |
| `noisy/<condition_id>.jsonl` | `src.noise.grid` | 37 datasets (clean + 36 noisy); each row = noisy text + full corruption manifest |
