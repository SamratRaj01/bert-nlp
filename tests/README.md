# tests/

Known-answer and determinism tests. Every run's output is saved to
`evidence/test_outputs/` so the graded package contains proof they pass.

| File | Owner | Covers |
|---|---|---|
| `test_med.py` | shared | the 5 required known-answer MED cases + normalized MED + empty-string boundary |
| `test_noise.py` | Student A | determinism (same seed -> same output+log), intensity-0 identity, replay == output, substitution never a no-op, Unicode-safe indexing |
| `test_tokenizer.py` | Student B | special-token IDs stable, a fixed string -> fixed piece list, `##` and `[UNK]` detection, truncation flag fires past `max_length` |

## Run and save output
```
python -m pytest -q | tee evidence/test_outputs/pytest.txt
```
