# tests/

Known-answer and determinism tests. Every run's output is saved to
`evidence/test_outputs/` so the graded package contains proof they pass.

| File | Owner | Covers |
|---|---|---|
| `test_med.py` | shared | the 5 required known-answer MED cases + normalized MED + empty-string boundary |
| `test_noise.py` | Samrat | determinism (same seed -> same output+log), intensity-0 identity, replay == output, substitution never a no-op, Unicode-safe indexing |
| `test_noise_v2.py` | Samrat | Phase 2 rule: p = 0 identity, determinism, replay, letters only, sentinels/digits preserved, insert-after, no re-corruption, substitution never same letter, empty output retained |
| `test_lm.py` | Prazol | Assignment 2 §11: hand vocab 3/6·2/6·1/6, distributions sum to 1 (seen + unseen histories), λ grid/normalization/tie-break, 4 equiprobable → 2 bits / PP 4, aggregation hand calc, `<UNK>` mapping, context reset, counts never mutated, MLE ∞ loss |
| `test_eval.py` | both | fast MED vs reference, 37 unique condition IDs, zero-token diagnostics flagged, corpus aggregation before dividing, summary seed counts |
| `test_tokenizer.py` | Prazol | special-token IDs stable, a fixed string -> fixed piece list, `##` and `[UNK]` detection, truncation flag fires past `max_length` |

## Run and save output
```
python -m pytest -q | tee evidence/test_outputs/pytest.txt
```

Phase 2 saves its run to `evidence/phase2/pytest.txt` (from `phase2.ipynb`).
