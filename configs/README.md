# configs/

Single source of truth for every reproducible setting. Nothing in `src/` hard-codes a value
that belongs here.

## Files
- `config.yaml` — the Phase 1 configuration (frozen record; Phase 2 never edits it).
- `phase2.yaml` — the Phase 2 protocol (checkpoint, validation-sample rule, training sets, noise grid,
  vocabulary and λ settings). Everything above the `FROZEN BY phase2.ipynb` marker is the protocol and
  is hashed (`protocol_sha256`) into every manifest row; the `frozen_lambdas` block below it is written
  by the notebook (step B6) before any noisy scoring and must not be edited by hand.

## Fields in `config.yaml`
| Field | Meaning |
|---|---|
| `global_seed` | one integer; per-example seeds are derived from it deterministically |
| `checkpoint` | BERT checkpoint name (fixed across clean/noisy branches) |
| `checkpoint_revision` | pinned commit/revision hash of the checkpoint |
| `max_length` | tokenizer max sequence length (must match the checkpoint) |
| `corpus.dataset` / `corpus.version` | WikiText-2 dataset id and version |
| `corpus.split` | split developed on (`train`) |
| `corpus.slice_rule` | human-readable selection rule |
| `corpus.slice_size` | number of examples in the slice |
| `noise.operations` | which of insertion/deletion/substitution are enabled |
| `noise.intensity` | intensity parameter (definition documented in src/noise/README.md) |
| `noise.alphabet` | character set used for insertion/substitution |
| `pairs.count` | number of paired examples to produce (20) |
| `pairs.selection_rule` | how the 20 are chosen from the slice |

Changing any value here changes the run. Record checksums of this file with the submission.
