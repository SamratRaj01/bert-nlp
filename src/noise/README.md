# src/noise/  (Samrat)

Deterministic character-level corruption applied to **raw text, before tokenization**.

| File | Purpose |
|---|---|
| `rng.py` | Per-example deterministic RNG. `example_rng(global_seed, example_id)` returns a seeded `random.Random` so the same (seed, id) always yields the same edits. |
| `operations.py` | The three edit operations on a `str`: `insert`, `delete`, `substitute`. Each returns `(new_text, edit_record)` where `edit_record = {op, index, source_char, replacement_char}`. Plus `apply(text, operation, intensity, rng)` which performs 0+ edits and returns `(noisy_text, operation_log)`. |
| `manifest.py` | Serialise a corruption manifest row: `{example_id, seed, operation, intensity, operation_log}` and a `replay(clean_text, operation_log) -> noisy_text` verifier. |

## Intensity definition
`intensity` is the independent probability that each character position is edited
(Bernoulli per character). `intensity = 0` performs no edits and returns the input unchanged
with an empty log. Document any change to this definition here.

## Invariants (enforced by tests/test_noise.py)
- Same `(global_seed, example_id, operation, intensity)` -> identical `noisy_text` and log.
- `intensity == 0` -> `noisy_text == clean_text`, `operation_log == []`.
- `replay(clean_text, operation_log) == noisy_text`.
- Substitution never replaces a character with itself.
- Edits operate on Unicode code points, not bytes.
