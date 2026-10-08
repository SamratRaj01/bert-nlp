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

## Phase 2 (`evidence/phase2/`, written by `phase2.ipynb`)
| Path | Contents |
|---|---|
| `acceptance.txt` | the plan §9 acceptance checklist, checked in code |
| `pytest.txt` | full unit-test run (Phase 1 + Phase 2) |
| `hand_check.md` | hand calculations vs program output |
| `mle_demo.txt` | unsmoothed MLE: zero probability → infinite loss |
| `fast_med_check.txt` | fast MED vs reference on known answers + 1,000 real pairs |
| `a1_pairs_rerun.jsonl`, `a1_pairs_regression.txt` | the 20 A1 pairs regenerated, byte-identical check |
