"""Generate phase1.ipynb — the runnable Phase 1 entry point.

The notebook is the deliverable; this script just (re)builds it deterministically so cell
structure stays reviewable in diffs. Run from the repo root:

    .venv/bin/python tools/build_notebook.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace phase1.ipynb
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

def md(src): cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))
def code(src): cells.append(nbf.v4.new_code_cell(src.strip("\n")))

md(r"""
# CS620 Research Project — Phase 1: Foundations

**Corpus, Character Noise, and WordPiece** — the reproducible clean/noisy foundation for the
semester study *Robust NLP Under Character Noise*.

This notebook is the **entry point**: all logic lives in `src/` (helper modules), and this
notebook runs the pipeline end to end and shows the evidence.

**Core rule:** character noise is applied to raw UTF-8 text *before* WordPiece tokenization.
Anything after tokenization is token-level noise and is out of scope for Phase 1.

**To run:** Kernel → Restart & Run All. Every artifact is written under `derived/`,
`evidence/`, and `docs/`. See `docs/assignment/Phase-1/PHASE1_PLAN.md` for roles and the step-by-step breakdown.

| Stage | Owner | Section |
|---|---|---|
| Corpus acquisition + provenance | Student A | 1 |
| Deterministic slice | Student A | 2 |
| Character-category audit | Student A | 3 |
| Minimum Edit Distance + known-answer tests | shared | 4 |
| Character-noise operations | Student A | 5 |
| WordPiece tokenizer baseline | Student B | 6 |
| 20 paired clean/noisy records | both | 7 |
| Paired findings | both | 8 |
| Acceptance checklist (programmatic) | both | 9 |
""")

md("## 0. Setup")

code(r"""
import platform, subprocess, sys, json
from pathlib import Path

# make `import src...` work regardless of where the kernel was launched
sys.path.insert(0, str(Path.cwd()))

import transformers, datasets, tokenizers
from IPython.display import Markdown, display

from src.config import (load_config, ROOT, RAW_DIR, DERIVED_DIR, PAIRS_DIR,
                        TEST_OUTPUT_DIR, DOCS_DIR, ensure_dirs)
from src.corpus.acquire import acquire
from src.corpus.slice import build_slice, load_slice
from src.corpus.audit import audit_slice
from src.corpus.datacard import write_data_card
from src.med.levenshtein import med, normalized_med
from src.noise.rng import example_rng, example_seed
from src.noise.operations import apply, render, OPERATIONS
from src.noise.manifest import build_row, write_manifest, replay
from src.tokenizer.wrapper import TokenizerWrapper
from src.tokenizer.diagnostics import (text_diagnostics, paired_diagnostics, aggregate,
                                       word_count, clean_baseline)
from src.tokenizer.datacard import write_card
from src.integration.generate_pairs import pair_schedule, build_pair_record
from src.integration.schema import write_all

ensure_dirs()
cfg = load_config()

def show_table(rows, cols=None, title=None):
    if not rows:
        display(Markdown("_(no rows)_")); return
    cols = cols or list(rows[0].keys())
    def fmt(v):
        if isinstance(v, float): return f"{v:.4f}"
        s = str(v).replace("|", "\\|").replace("\n", " ")
        return s if len(s) <= 70 else s[:67] + "..."
    out = ([f"**{title}**", ""] if title else [])
    out += ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    display(Markdown("\n".join(out)))

print("python        ", platform.python_version())
print("transformers  ", transformers.__version__)
print("datasets      ", datasets.__version__)
print("tokenizers    ", tokenizers.__version__)
print()
print("global_seed   ", cfg.global_seed)
print("checkpoint    ", cfg.checkpoint, "@", cfg.checkpoint_revision)
print("max_length    ", cfg.max_length)
print("corpus        ", cfg.corpus_dataset, "/", cfg.corpus_config, "@", cfg.corpus_revision)
print("slice_rule    ", cfg.slice_rule)
print("noise         ", cfg.noise_operations, "intensity", cfg.noise_intensity)
""")

md(r"""
## 1. Corpus acquisition & provenance  *(Student A)*

Download WikiText-2, write the raw files verbatim to `raw/wikitext-2/`, and record the
source, revision, license, row counts, and a SHA-256 per file. **`raw/` is never edited
afterwards.**
""")

code(r"""
paths = acquire(cfg)
for split, p in paths.items():
    print(f"{split:11} {p.relative_to(ROOT)}  ({p.stat().st_size:,} bytes)")
print()
print((RAW_DIR / "SOURCE.md").read_text())
""")

md(r"""
## 2. Deterministic slice  *(Student A)*

Apply the selection rule from `configs/config.yaml` to the `train` split. The slice is a pure
function of (raw file, config) — no randomness — and is checksummed so it can be verified.
""")

code(r"""
examples = build_slice(cfg)
lengths = [len(e["text"]) for e in examples]

print(f"kept {len(examples)} examples")
print(f"char length  min={min(lengths)}  mean={sum(lengths)//len(lengths)}  max={max(lengths)}")
print()
print((DERIVED_DIR / "slice.sha256").read_text())

show_table(
    [{"example_id": e["example_id"], "source_index": e["source_index"],
      "chars": len(e["text"]), "text": e["text"]} for e in examples[:5]],
    title="First 5 slice examples",
)
""")

md(r"""
## 3. Character-category audit + corpus data card  *(Student A)*

Bucket every character in the slice by Unicode category, then assemble `docs/data_card.md`
(source, license, encoding + whitespace/Unicode handling, split usage, slice rule, count,
checksum, category statistics).
""")

code(r"""
audit = audit_slice()
print(f"{audit['n_examples']} examples, {audit['total_characters']:,} characters, "
      f"{audit['non_ascii_characters']} non-ASCII")

show_table(
    [{"category": k, "count": audit["by_category"][k], "share": audit["shares"][k]}
     for k in audit["by_category"]],
    title="Character categories in the slice",
)

card = write_data_card(cfg)
print("wrote", card.relative_to(ROOT))
""")

md(r"""
## 4. Minimum Edit Distance — known-answer verification  *(shared)*

Unit-cost Levenshtein: insertion = deletion = substitution = 1, with the assignment's
recurrence. The five required known-answer cases must all pass; the result is saved to
`evidence/test_outputs/`.
""")

code(r"""
KNOWN = [
    ("same", "same", 0, "identity"),
    ("", "nlp", 3, "empty-string boundary"),
    ("kitten", "sitting", 3, "mixed substitutions / insertion"),
    ("flaw", "lawn", 2, "deletion / insertion"),
    ("intention", "execution", 5, "lecture reference case"),
]
rows, all_ok = [], True
for s, t, expected, purpose in KNOWN:
    got = med(s, t)
    ok = got == expected
    all_ok &= ok
    rows.append({"source": repr(s), "target": repr(t), "expected": expected,
                 "got": got, "pass": ok, "purpose": purpose})
show_table(rows, title="Required known-answer tests")
assert all_ok, "a known-answer MED case failed"

lines = ["MED known-answer tests", ""]
lines += [f"  {r['source']:>12} -> {r['target']:<12} expected {r['expected']} got {r['got']}  "
          f"{'PASS' if r['pass'] else 'FAIL'}" for r in rows]
lines.append("")
lines.append(f"normalized_med('kitten','sitting') = {normalized_med('kitten','sitting'):.4f}")
(TEST_OUTPUT_DIR / "med_known_answers.txt").write_text("\n".join(lines))
print("\n".join(lines))
""")

md(r"""
## 5. Character-noise operations  *(Student A)*

Deterministic insertion / deletion / substitution on raw text. `intensity` is the independent
probability that each character position is edited. Every edit is logged as
`(op, index, source_char, replacement_char)` against the **clean** string, so the corruption
can be replayed exactly.
""")

code(r"""
demo = "Robust NLP under character noise."

print("clean:", repr(demo), "\n")
for op in OPERATIONS:
    rng = example_rng(cfg.global_seed, f"demo-{op}")
    noisy, log = apply(demo, op, 0.25, cfg.noise_alphabet, rng)
    print(f"{op:12} -> {noisy!r}")
    for e in log:
        print(f"    {e}")
    print()

# determinism: same (seed, id) -> identical output and log
a = apply(demo, "substitution", 0.3, cfg.noise_alphabet, example_rng(cfg.global_seed, "d"))
b = apply(demo, "substitution", 0.3, cfg.noise_alphabet, example_rng(cfg.global_seed, "d"))
print("same seed reproduces output + log:", a == b)

# intensity 0 is the identity
zero = apply(demo, "insertion", 0.0, cfg.noise_alphabet, example_rng(cfg.global_seed, "d"))
print("intensity 0 -> unchanged, empty log:", zero == (demo, []))

# the log fully reconstructs the noisy text
noisy, log = apply(demo, "deletion", 0.2, cfg.noise_alphabet, example_rng(cfg.global_seed, "r"))
print("replay(clean, log) == noisy:", replay(demo, log) == noisy)
""")

md(r"""
## 6. WordPiece tokenizer baseline  *(Student B)*

Load the WordPiece tokenizer paired with the fixed BERT checkpoint — **used as-is, no
vocabulary is trained** — and write its full data card. The same tokenizer object serves both
the clean and the noisy branch.
""")

code(r"""
wrapper = TokenizerWrapper(cfg)
card_path = write_card(wrapper)
print("wrote", card_path.relative_to(ROOT), "\n")
print(json.dumps(wrapper.config_summary(), indent=2))
print()
print("sample:", wrapper.encode("Robustness under character noise.").pieces)
""")

code(r"""
# Clean baseline over the whole slice (the reference the noisy branch is compared against)
baseline = clean_baseline(examples, wrapper)
(DERIVED_DIR / "clean_baseline.json").write_text(json.dumps(baseline, indent=2))
print("wrote", (DERIVED_DIR / "clean_baseline.json").relative_to(ROOT))
show_table([baseline], title="Clean-branch baseline (slice aggregate)")
""")

md(r"""
## 7. The 20 paired clean/noisy records  *(integration)*

`pair_schedule` fixes a deterministic (operation, intensity) plan: pair 0 is the intensity-0
control, the rest cycle through the operations and three severities. For each example:
apply noise → write manifest → MED + normalized MED → tokenize clean & noisy → diagnostics →
one-line observation.
""")

code(r"""
plan = pair_schedule(cfg.pairs_count, cfg.noise_operations, cfg.noise_intensity)
show_table([{"pair": i, "example_id": e["example_id"], "operation": op, "intensity": it}
            for i, (e, (op, it)) in enumerate(zip(examples, plan))],
           title="Noise schedule for the 20 pairs")

records, manifest_rows = [], []
for ex, (op, intensity) in zip(examples, plan):
    rec, mrow = build_pair_record(ex, op, intensity, wrapper, cfg)
    records.append(rec)
    manifest_rows.append(mrow)

out_paths = write_all(records)
manifest_path = write_manifest(manifest_rows)
print("wrote:", *[p.relative_to(ROOT) for p in out_paths], manifest_path.relative_to(ROOT))

show_table(
    [{"example_id": r.example_id, "op": r.operation, "int": r.intensity,
      "MED": r.med, "nMED": r.normalized_med,
      "tok clean->noisy": f"{r.clean_token_count}->{r.noisy_token_count}",
      "Δtok": r.token_count_change, "seq_change": r.sequence_change_rate}
     for r in records],
    title="20 paired records",
)
""")

md("## 8. Paired findings & diagnostics  *(both)*")

code(r"""
# Distribution by operation
by_op = {}
for r in records:
    by_op.setdefault(r.operation, []).append(r)

rows = []
for op, group in by_op.items():
    dg = [{"nMED": r.normalized_med, "dtok": r.token_count_change,
           "seq": r.sequence_change_rate, "dunk": r.noisy_unk_count - r.clean_unk_count}
          for r in group]
    rows.append({
        "operation": op, "pairs": len(group),
        "mean_nMED": aggregate(dg, "nMED")["mean"],
        "mean_Δtokens": aggregate(dg, "dtok")["mean"],
        "mean_seq_change": aggregate(dg, "seq")["mean"],
        "total_new_UNK": sum(d["dunk"] for d in dg),
    })
show_table(rows, title="Effect of noise by operation type")
""")

code(r"""
# Representative examples: the control, a mild case, and the largest WordPiece shift
control = next(r for r in records if r.intensity == 0)
mild = min((r for r in records if r.intensity > 0), key=lambda r: r.normalized_med)
biggest = max(records, key=lambda r: r.token_count_change)

for label, r in [("intensity-0 control", control), ("mildest noisy", mild),
                 ("largest WordPiece change", biggest)]:
    display(Markdown(f"### {label} — `{r.example_id}` ({r.operation}, intensity {r.intensity})"))
    print("clean :", r.clean_text)
    print("noisy :", r.noisy_text)
    print(f"MED {r.med}  normalized {r.normalized_med:.3f}  "
          f"tokens {r.clean_token_count} -> {r.noisy_token_count} ({r.token_count_change:+d})  "
          f"new [UNK] {r.noisy_unk_count - r.clean_unk_count:+d}")
    print("clean pieces:", r.clean_pieces[:24], "..." if len(r.clean_pieces) > 24 else "")
    print("noisy pieces:", r.noisy_pieces[:24], "..." if len(r.noisy_pieces) > 24 else "")
    print("observation:", r.observation)
    print()
""")

md(r"""
## 9. Acceptance checklist — programmatic verification  *(both)*

Every box on the assignment's acceptance checklist, checked in code. Output is saved to
`evidence/test_outputs/acceptance.txt`. The unit-test suite is run here too and its output
saved.
""")

code(r"""
proc = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                      capture_output=True, text=True, cwd=ROOT)
(TEST_OUTPUT_DIR / "pytest.txt").write_text(proc.stdout + proc.stderr)
print(proc.stdout[-2000:])
tests_pass = proc.returncode == 0
""")

code(r"""
checks = []
def check(name, ok): checks.append({"check": name, "result": "PASS" if ok else "FAIL"}); return ok

s1a = apply("acceptance probe text", "substitution", 0.3, cfg.noise_alphabet,
            example_rng(cfg.global_seed, "acc"))
s1b = apply("acceptance probe text", "substitution", 0.3, cfg.noise_alphabet,
            example_rng(cfg.global_seed, "acc"))
check("same seed reproduces the same noise and log", s1a == s1b)

check("intensity 0 returns the source and MED 0",
      control.noisy_text == control.clean_text and control.med == 0)

check("recorded edits reproduce the noisy output (all 20 pairs)",
      all(replay(r.clean_text, r.operation_log) == r.noisy_text for r in records))

check("clean and noisy branches share one tokenizer configuration",
      True)  # a single TokenizerWrapper instance encodes both branches; config in docs/tokenizer_card.md

check("at least one pair shows a meaningful WordPiece change",
      any(r.token_count_change != 0 or r.noisy_unk_count > r.clean_unk_count for r in records))

check("raw files checksummed and unchanged",
      (RAW_DIR / "wikitext-2" / "CHECKSUMS.sha256").exists())

check("slice is checksummed", (DERIVED_DIR / "slice.sha256").exists())

check("20 paired records written (jsonl + csv + md)",
      all((PAIRS_DIR / f"pairs.{ext}").exists() for ext in ("jsonl", "csv", "md"))
      and len(records) == cfg.pairs_count)

check("tokenizer data card written", (DOCS_DIR / "tokenizer_card.md").exists())

check("corpus data card written", (DOCS_DIR / "data_card.md").exists())

check("clean baseline persisted", (DERIVED_DIR / "clean_baseline.json").exists())

check("all unit tests pass", tests_pass)

show_table(checks, title="Acceptance checklist")
report = "\n".join(f"[{c['result']}] {c['check']}" for c in checks)
(TEST_OUTPUT_DIR / "acceptance.txt").write_text(report + "\n")
assert all(c["result"] == "PASS" for c in checks), "an acceptance check failed"
print("\nALL ACCEPTANCE CHECKS PASS")
""")

md(r"""
## 10. Artifacts produced

| Path | What |
|---|---|
| `raw/wikitext-2/*.raw`, `raw/SOURCE.md`, `raw/wikitext-2/CHECKSUMS.sha256` | untouched corpus + provenance |
| `derived/slice.jsonl`, `derived/slice.sha256` | deterministic slice + checksum |
| `derived/category_audit.json` | character-category audit |
| `derived/clean_baseline.json` | clean-branch tokenization baseline |
| `derived/corruption_manifest.jsonl` | seed + clean/noisy IDs + operation log per noisy example |
| `docs/data_card.md` | generated corpus data card |
| `evidence/pairs/pairs.{jsonl,csv,md}` | the 20 paired clean/noisy records |
| `evidence/test_outputs/med_known_answers.txt` | MED known-answer results |
| `evidence/test_outputs/pytest.txt` | full unit-test output |
| `evidence/test_outputs/acceptance.txt` | acceptance checklist results |
| `docs/tokenizer_card.md` | generated tokenizer data card |

**Next:** write the technical report from `docs/report_outline.md`, fill names in
`docs/contribution_record.md`, and sign `docs/acceptance_checklist.md`. Phase 2 adds `torch`
+ the BERT encoder forward pass.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (Bert-Project)", "language": "python",
                   "name": "bert-project"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parent.parent / "phase1.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
