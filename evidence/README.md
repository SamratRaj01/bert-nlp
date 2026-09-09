# evidence/

The graded artifacts. Everything here is generated, but it is committed so the submission is
self-contained and inspectable without rerunning anything.

| Path | Contents |
|---|---|
| `pairs/pairs.jsonl` | the 20 pair records, machine-readable (one JSON object per line) |
| `pairs/pairs.csv` | the 20 pair records as a flat table |
| `pairs/pairs.md` | the 20 pair records as a readable table for the report appendix |
| `test_outputs/pytest.txt` | saved output of `python -m pytest -q` |
| `test_outputs/med_known_answers.txt` | the 5 MED known-answer results, printed explicitly |

Regenerate with:
```
python -m pytest -q | tee evidence/test_outputs/pytest.txt
python -m src.integration.generate_pairs
```
