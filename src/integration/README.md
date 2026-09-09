# src/integration/  (both students)

Where Student A's noise and Student B's tokenizer meet. Defines the shared record format and
runs the end-to-end generator for the 20 paired examples.

| File | Purpose |
|---|---|
| `schema.py` | The `PairRecord` dataclass + `write_jsonl` / `write_csv` / `write_markdown` / `write_all`. One place both students agree on field names and types. |
| `generate_pairs.py` | `pair_schedule(n, operations, base_intensity)` — the deterministic (operation, intensity) plan for the 20 pairs — and `build_pair_record(example, operation, intensity, wrapper, cfg)` which runs one pair end to end (apply noise → assert replay → MED → tokenize clean & noisy → diagnostics → observation) and returns `(PairRecord, manifest_row)`. |

The **loop** over the 20 examples, the tables, and the acceptance checks live in
`phase1.ipynb`; this module holds the per-pair logic so the notebook stays readable and the
work stays unit-testable.

## Pair selection
`pair_schedule` makes pair 0 the intensity-0 control, then cycles the enabled operations and
three severities (`base`, `2*base`, `3*base`) so the set is not monotone and at least one pair
produces a meaningful WordPiece change (extra fragments or an `[UNK]`).

## PairRecord fields
`example_id`, `source_split`, `source_index`, `clean_text`, `noisy_text`,
`seed`, `operation`, `intensity`, `edit_location`, `operation_log`,
`med`, `normalized_med`,
`clean_pieces`, `clean_token_count`, `noisy_pieces`, `noisy_token_count`,
`clean_continuation_count`, `noisy_continuation_count`,
`clean_unk_count`, `noisy_unk_count`,
`clean_truncated`, `noisy_truncated`,
`token_count_change`, `sequence_change_rate`, `observation`

Run:  python -m src.integration.generate_pairs
