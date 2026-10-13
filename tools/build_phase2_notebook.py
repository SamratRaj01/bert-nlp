"""Generate phase2.ipynb — the runnable Phase 2 entry point.

The notebook is the deliverable; this script just (re)builds it deterministically so cell
structure stays reviewable in diffs. Run from the repo root:

    .venv/bin/python tools/build_phase2_notebook.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace phase2.ipynb
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

def md(src): cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))
def code(src): cells.append(nbf.v4.new_code_cell(src.strip("\n")))

md(r"""
# CS620 Research Project — Phase 2: Probabilistic Language Modeling and Input Noise

**Classical baseline** for the semester study *Robust NLP Under Character Noise*.

> **Research question:** How do character-noise type and intensity change token coverage,
> fragmentation, and predictive behavior under fixed whitespace and WordPiece tokenization?

Count-based n-gram models are trained on **clean training text** and *score* (never generate)
one frozen validation sample in clean form and in 36 corrupted forms
(3 operations × 4 rates × 3 seeds). All results are **preliminary validation (development)
evidence**; the test split is never scored, inspected, or used for any choice.

**Core rule:** noise is applied to raw text *before* tokenization; each noisy text is generated
once and the identical string goes to both tokenizers.

**To run:** Kernel → Restart & Run All (≈ 2 minutes on a laptop CPU). Logic lives in `src/`; this
notebook runs it end to end and shows the evidence. Plan and checklist:
`docs/assignment/Phase-2/PHASE2_PLAN.md`.

| Stage | Owner | Section |
|---|---|---|
| Step 0 — shared freeze (`configs/phase2.yaml`) | both | 1 |
| A1 frozen validation sample + article IDs | Samrat | 2 |
| A2 training sets (A1 slice, full train) | Samrat | 3 |
| A3 noise rule v2 (`a2_default`) | Samrat | 4 |
| A5 fast MED, cross-checked | Samrat | 5 |
| A4 noise grid + manifests, corruption severity | Samrat | 6 |
| B1 tokenizer pipelines | Prazol | 7 |
| B2–B4 vocabularies + sparse counts | Prazol | 8 |
| B5 models + unsmoothed MLE demo | Prazol | 9 |
| §7 verify: hand calculation + normalization | both | 10 |
| B6 λ selection on clean validation → freeze | Prazol | 11 |
| B7 + A6 run the grid (A1 slice, primary) | both | 12 |
| Results tables | both | 13 |
| Figures | both | 14 |
| Why the loss moves in opposite directions (decomposition) | both | 14b |
| Error analysis — 3 traceable examples | both | 15 |
| Corpus-size extension (full train) | both | 16 |
| A7 20 A1 diagnostic pairs — regression | both | 17 |
| Tests + acceptance checklist | both | 18 |
""")

md("## 0. Setup")

code(r"""
import json, math, platform, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import transformers, tokenizers
from IPython.display import Image, Markdown, display

from src.config import ROOT, load_config, PAIRS_DIR
from src.phase2_config import (load_phase2_config, ensure_p2_dirs, sha256_file, DERIVED_P2_DIR,
                               NOISY_DIR, RESULTS_P2_DIR, EVIDENCE_P2_DIR, PHASE2_CONFIG_PATH)
from src.corpus.val_sample import build_val_sample, build_training_sets, a1_slice_unchanged, VAL_PATH, TRAIN_PATHS
from src.corpus.slice import load_slice
from src.noise.rng import example_rng
from src.noise.rule_v2 import apply_v2, replay_v2, eligible_positions, RULE_VERSION
from src.noise.grid import generate_grid, conditions, load_condition, corrupt_record
from src.med.levenshtein import med
from src.med.fast import med_fast
from src.lm.tokenize import make_pipelines
from src.lm.vocab import Vocab
from src.lm.counts import NgramCounts
from src.lm.models import AddOneLM, p_mle, nll_bits, MODELS, REQUIRED_MODELS, SUPPLEMENTARY_MODELS
from src.lm.score import build_engine, q_for_records, record_losses, corpus_loss
from src.lm.interp import lambda_grid
from src.eval.run_grid import score_grid, select_and_freeze, frozen_lambdas
from src.eval.summarize import write_tables, severity
from src.eval.decompose import loss_decomposition
from src.eval import figures

ensure_p2_dirs()
FIG_DIR = RESULTS_P2_DIR / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
cfg = load_phase2_config()
cfg1 = load_config()            # Phase 1 record (read-only)
T0 = time.time()

def show_table(rows, cols=None, title=None, floatfmt="{:.4f}"):
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    if not rows:
        display(Markdown("_(no rows)_")); return
    cols = cols or list(rows[0].keys())
    def fmt(v):
        if isinstance(v, float): return "—" if v != v else floatfmt.format(v)
        s = str(v).replace("|", "\\|").replace("\n", " ")
        return s if len(s) <= 90 else s[:87] + "..."
    out = ([f"**{title}**", ""] if title else [])
    out += ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    display(Markdown("\n".join(out)))

print("python        ", platform.python_version())
print("transformers  ", transformers.__version__, "| tokenizers", tokenizers.__version__)
print("numpy         ", np.__version__, "| pandas", pd.__version__)
print("checkpoint    ", cfg.checkpoint, "@", cfg.checkpoint_revision)
print("protocol hash ", cfg.protocol_sha256)
""")

md(r"""
## 1. Step 0 — shared freeze  *(both)*

`configs/phase2.yaml` is the single source of truth for Phase 2; `configs/config.yaml` stays
frozen as the Phase 1 record. Everything above the *FROZEN BY phase2.ipynb* marker is the
experimental protocol, hashed into every manifest row (`protocol_sha256`). Only λ (step B6) are
written below the marker, and only before any noisy scoring.
""")

code(r"""
assert cfg.checkpoint_revision == cfg1.checkpoint_revision, "Phase 2 must reuse the Phase 1 tokenizer revision"
assert cfg.raw["corpus"]["revision"] == cfg1.corpus_revision

n_cfg = len(cfg.lm["models"]) * len(cfg.lm["tokenizers"])
n_noisy = len(cfg.operations) * len(cfg.rates) * len(cfg.seeds)
show_table([
    {"quantity": "model × tokenizer configurations", "calculation": "3 models × 2 tokenizers", "total": n_cfg},
    {"quantity": "noisy validation datasets", "calculation": "3 operations × 4 rates × 3 seeds", "total": n_noisy},
    {"quantity": "clean perplexity scores", "calculation": "6 × 1 clean sample", "total": n_cfg},
    {"quantity": "noisy perplexity scores", "calculation": "6 × 36", "total": n_cfg * n_noisy},
    {"quantity": "all individual scores (per training set)", "calculation": "6 + 216", "total": n_cfg * (1 + n_noisy)},
    {"quantity": "noisy summary rows (per training set)", "calculation": "6 × 3 operations × 4 rates", "total": n_cfg * 3 * len(cfg.rates)},
    {"quantity": "supplementary: standalone trigram q3", "calculation": "1 model × 2 tokenizers × 37 datasets",
     "total": len(cfg.lm["supplementary_models"]) * 2 * (1 + n_noisy)},
], title="Experiment size (assignment p. 2) + supplementary trigram")
assert (n_cfg, n_noisy, n_cfg * (1 + n_noisy)) == (6, 36, 222)
assert tuple(cfg.lm["models"]) == REQUIRED_MODELS and tuple(cfg.lm["supplementary_models"]) == SUPPLEMENTARY_MODELS

show_table([
    {"setting": "noise rule", "value": cfg.noise["rule_version"]},
    {"setting": "operations", "value": ", ".join(cfg.operations)},
    {"setting": "rates (nonzero)", "value": ", ".join(map(str, cfg.rates))},
    {"setting": "seeds", "value": ", ".join(map(str, cfg.seeds))},
    {"setting": "validation sample", "value": cfg.val["rule"]},
    {"setting": "training sets", "value": "; ".join(f"{k} ({v['role']})" for k, v in cfg.training_sets.items())},
    {"setting": "λ grid / tie-break", "value": f"step {cfg.lm['lambda_step']}; {cfg.lm['lambda_tie_break']}"},
    {"setting": "per-record RNG", "value": cfg.noise["rng_key"]},
], title="Frozen protocol (configs/phase2.yaml)")
""")

md(r"""
## 2. A1 — frozen validation sample + article IDs  *(Samrat)*

Validation lines in file order, kept by the Phase 1 line filter (`is_kept_line`, 40–600 chars),
capped at 500. Selection uses source IDs only, never model outcomes. Each record carries the
**article ID** of its governing top-level heading (` = Title = `), for grouping and later
document-level bootstrap. The 40–600 band is the clean-record length cap for MED runtime; it is
applied before corruption, and noisy variants are never truncated.
""")

code(r"""
val, val_report = build_val_sample(cfg)
show_table([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in val_report.items()}],
           title="Validation sample report")
print((VAL_PATH.with_suffix(".sha256")).read_text())
show_table([{"record_id": r["record_id"], "article_id": r["article_id"], "chars": r["char_count"],
             "text": r["text"]} for r in val[:4]], title="First validation records")
assert len(val) == cfg.val["cap"] and all(r["record_id"].startswith("wt2-valid-") for r in val)
""")

md(r"""
## 3. A2 — training sets  *(Samrat)*

* `train_a1_slice` — **primary baseline**: the Assignment 1 slice, unchanged (its Phase 1
  checksum is re-verified); article IDs are back-filled into a new derived file.
* `train_full` — **corpus-size extension**: every non-empty, non-heading train line.

n-gram counts are learned from **clean training text only** and never refit per seed.
""")

code(r"""
assert a1_slice_unchanged(), "derived/slice.jsonl changed since Phase 1"
train_sets = build_training_sets(cfg)
rows = []
for name, recs in train_sets.items():
    rows.append({"training set": name, "role": cfg.training_sets[name]["role"], "records": len(recs),
                 "articles": len({r["article_id"] for r in recs}),
                 "characters": sum(r["char_count"] for r in recs),
                 "whitespace tokens": sum(len(r["text"].split()) for r in recs),
                 "sha256": sha256_file(TRAIN_PATHS[name])[:16] + "…"})
show_table(rows, title="Training sets (clean)")
assert not any(r["record_id"].startswith("wt2-test") for recs in train_sets.values() for r in recs)
""")

md(r"""
## 4. A3 — noise rule v2 (`a2_default`)  *(Samrat)*

| Operation | Rule on original eligible characters |
|---|---|
| deletion | independently delete an eligible character with probability p |
| substitution | with probability p, replace it with a *different* lowercase a–z character (`'A' → 'a'` excluded) |
| insertion | with probability p, insert one uniform lowercase a–z character immediately **after** it |

Eligible = ASCII letters a–z / A–Z. Spaces, punctuation, digits, record boundaries and WikiText
sentinels (`@-@`, `@,@`, `@.@`) are never edited. One draw per *original* position, so inserted
characters are never corrupted. The edit log `(op, index, source_char, replacement_char)` is
indexed against the clean string and replays exactly. The Phase 1 rule (`src/noise/operations.py`)
is untouched and used only for the 20-pair regression (section 17).
""")

code(r"""
demo = next(r for r in val if "@-@" in r["text"] and len(r["text"]) < 260)
print(demo["record_id"], "| eligible letters:", len(eligible_positions(demo["text"], cfg.protected_spans)))
print("clean        :", demo["text"][:200], "\n")
for op in cfg.operations:
    key = f"{demo['record_id']}|{op}|0.2"
    noisy, log, elig = apply_v2(demo["text"], op, 0.2, cfg.alphabet, example_rng(11, key), cfg.protected_spans)
    assert replay_v2(demo["text"], log) == noisy
    print(f"{op:13}:", noisy[:200])
    print(" " * 15, f"{len(log)} edits, first: {log[:2]}\n")

a = apply_v2(demo["text"], "substitution", 0.2, cfg.alphabet, example_rng(23, "k"), cfg.protected_spans)
b = apply_v2(demo["text"], "substitution", 0.2, cfg.alphabet, example_rng(23, "k"), cfg.protected_spans)
z = apply_v2(demo["text"], "insertion", 0.0, cfg.alphabet, example_rng(23, "k"), cfg.protected_spans)
print("same seed -> same text and log:", a == b)
print("p = 0 -> unchanged, empty log :", z[0] == demo["text"] and z[1] == [])
""")

md(r"""
## 5. A5 — fast MED, cross-checked against the reference  *(Samrat)*

The grid needs ~18,000 clean/noisy MEDs of up to 600 characters. `med_fast` (NumPy row DP +
exact common prefix/suffix stripping) is used for the grid; the Phase 1 `med` stays the
canonical oracle. Both must agree on the five known-answer cases and on 1,000 real clean/noisy
pairs generated with the v2 rule.
""")

code(r"""
KNOWN = [("same", "same", 0), ("", "nlp", 3), ("kitten", "sitting", 3), ("flaw", "lawn", 2),
         ("intention", "execution", 5)]
lines = ["Fast MED cross-check (src/med/fast.py vs src/med/levenshtein.py)", ""]
for s, t, e in KNOWN:
    ok = med_fast(s, t) == med(s, t) == e
    lines.append(f"  {s!r:>12} -> {t!r:<12} expected {e}  fast {med_fast(s, t)}  ref {med(s, t)}  {'PASS' if ok else 'FAIL'}")
    assert ok

pairs, mismatches = 0, 0
t_fast = t_ref = 0.0
for i, r in enumerate(val[:334]):
    for op in cfg.operations:
        if pairs == 1000: break
        noisy, _, _ = apply_v2(r["text"], op, cfg.rates[i % 4], cfg.alphabet,
                               example_rng(620, f"medcheck|{r['record_id']}|{op}"), cfg.protected_spans)
        t = time.perf_counter(); f = med_fast(r["text"], noisy); t_fast += time.perf_counter() - t
        t = time.perf_counter(); g = med(r["text"], noisy); t_ref += time.perf_counter() - t
        mismatches += f != g; pairs += 1
lines += ["", f"real clean/noisy pairs checked: {pairs}", f"mismatches: {mismatches}",
          f"time fast {t_fast:.2f}s vs reference {t_ref:.2f}s"]
(EVIDENCE_P2_DIR / "fast_med_check.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
assert pairs == 1000 and mismatches == 0
""")

md(r"""
## 6. A4 — the noise grid (37 datasets) + corruption severity  *(Samrat)*

For every condition and validation record the noisy text is generated **once** and saved to
`derived/phase2/noisy/<condition_id>.jsonl` together with its manifest: article/record/condition
IDs, split, seed, per-record seed, operation, requested p, rule version, eligible count, edit log,
realized edits, MED, normalized MED, and source-text / protocol hashes. Records emptied by
deletion are kept. Both tokenizers read these files; nothing is regenerated downstream.
""")

code(r"""
grid_index = pd.DataFrame(generate_grid(val, cfg))
grid_index.to_csv(RESULTS_P2_DIR / "grid_index.csv", index=False)
condition_rows = {c["condition_id"]: load_condition(c["condition_id"]) for c in conditions(cfg)}
assert len(condition_rows) == 37 and all(len(v) == len(val) for v in condition_rows.values())

show_table(grid_index[["condition_id", "operation", "p", "seed", "records", "eligible_chars",
                       "realized_edits", "empty_after_noise"]].head(13),
           title="Grid index (first 13 of 37 conditions; full table in results/phase2/grid_index.csv)")
""")

code(r"""
manifest = pd.DataFrame([{k: v for k, v in r.items() if k not in ("noisy_text", "operation_log")}
                         for rows in condition_rows.values() for r in rows])
sev = severity(manifest)
sev.to_csv(RESULTS_P2_DIR / "severity.csv", index=False)
sev_records = manifest[manifest.condition_id != "clean"][["article_id", "record_id", "condition_id", "operation",
                                                         "p", "seed", "eligible_count", "realized_edits",
                                                         "med", "normalized_med"]]
sev_records.to_csv(RESULTS_P2_DIR / "severity_records.csv", index=False)

pooled_sev = (sev.groupby(["operation", "p"], sort=False)
                 .agg(realized_edit_rate=("realized_edit_rate", "mean"), nMED_mean=("nMED_mean", "mean"),
                      nMED_min=("nMED_min", "min"), nMED_max=("nMED_max", "max"),
                      records_med_below_edits=("records_med_below_edits", "sum"),
                      empty_after_noise=("empty_after_noise", "sum")).reset_index())
show_table(pooled_sev, title="Requested p vs realized severity (3 seeds pooled; per-seed rows in severity.csv)")
print("p is a per-eligible-letter probability; normalized MED divides by ALL clean characters (letters are ~75%),",
      "so it sits below p. For insertion and deletion MED equals the edit count exactly (the length changes by",
      "that many characters); only substitution can give MED < edits, when shifted text aligns more cheaply",
      "with one insertion + one deletion.")
""")

md(r"""
## 7. B1 — the two tokenizer pipelines  *(Prazol)*

* **whitespace** — Python `text.split()`, keeping case and punctuation exactly as written.
* **wordpiece** — the pinned BERT WordPiece tokenizer from Phase 1, pieces only: no `[CLS]`/`[SEP]`,
  no padding, no 512-token truncation.

This compares **complete pipelines** (normalization + segmentation + vocabulary), not segmentation
alone. The batch path used for scoring is checked against `tokenizer.tokenize` on every clean
validation record.
""")

code(r"""
pipelines = make_pipelines(cfg)
cards = {name: p.card() for name, p in pipelines.items()}
(RESULTS_P2_DIR / "tokenizer_pipelines.json").write_text(json.dumps(cards, indent=2))
print(json.dumps(cards, indent=2))

wp = pipelines["wordpiece"]
clean_texts = [r["noisy_text"] for r in condition_rows["clean"]]
assert wp.tokenize_batch(clean_texts) == [wp.tokenize(t) for t in clean_texts]
print("\nbatch WordPiece == tokenizer.tokenize on all", len(clean_texts), "clean validation records")
print("sample:", wp.tokenize("Robustness under charcter noise @-@ 1994 ."))
""")

md(r"""
## 8. B2–B4 — frozen vocabularies + sparse counts (A1 slice)  *(Prazol)*

* **whitespace V** = training tokens with count ≥ 2, plus `<UNK>` and `</s>`; all other tokens map
  to `<UNK>` at training and evaluation.
* **WordPiece V** = BERT vocabulary minus `[PAD] [CLS] [SEP] [MASK] [unused…]`, plus `</s>`; `[UNK]`
  and unseen entries are retained.
* `<s>` is context only (never in V, never scored). Each record is one sequence framed
  `<s> <s> w1 … wn </s>`; context resets per record; one `</s>` is scored.
* Counts are sparse integer-keyed tables, filled once from clean training data and then frozen.
""")

code(r"""
def build_engines(train_set):
    return {tok: build_engine(train_set, train_sets[train_set], pipe, cfg) for tok, pipe in pipelines.items()}

engines = build_engines("train_a1_slice")
rows = []
for tok, eng in engines.items():
    rows.append({"tokenizer": tok, "|V|": eng.vocab.size, **eng.vocab_stats, **eng.counts.stats()})
vocab_slice = pd.DataFrame(rows)
(RESULTS_P2_DIR / "train_a1_slice").mkdir(exist_ok=True)
vocab_slice.to_csv(RESULTS_P2_DIR / "train_a1_slice" / "vocab.csv", index=False)
show_table(rows, cols=["tokenizer", "|V|", "training_tokens", "training_tokens_mapped_to_unk",
                       "targets", "distinct_unigrams", "distinct_bigrams", "distinct_trigrams"],
           title="Frozen V and count statistics — train_a1_slice", floatfmt="{:.0f}")
""")

md(r"""
## 9. B5 — models, and the unsmoothed MLE demonstration  *(Prazol)*

$q_k(w\mid h)=\dfrac{C(h,w)+1}{C(h)+|V|}$ for $k=1,2,3$; unseen histories become uniform $1/|V|$.
$P_{mix}(w\mid h)=\lambda_1 q_1(w)+\lambda_2 q_2(w\mid\text{last token})+\lambda_3 q_3(w\mid\text{last two tokens})$.
The assignment's three configurations are $q_1$, $q_2$ and $P_{mix}$ (its third configuration is $P_{mix}$,
not a standalone $q_3$). The add-one trigram $q_3$ is **also scored on its own as a supplementary model**
(team decision), so every table and figure shows how the trigram behaves, while the required counts
(222 scores, 6 + 72 summary rows) stay separately verifiable.

The **unsmoothed MLE** $P_{MLE}=C(h,w)/C(h)$ is demonstrated separately on a small case: an unseen
target gets probability 0 and the loss is kept as **∞**, not hidden.
""")

code(r"""
mv = Vocab("mle-demo", ["a", "b", "</s>"], unk=None)
mc = NgramCounts(mv).fit([["a"], ["a"], ["b"]])        # training records: "a", "a", "b"
h = (mv.start_id,)
lines = ["Unsmoothed MLE demonstration (bigram, history <s>)",
         "training records: 'a', 'a', 'b'  ->  C(<s>,a)=2, C(<s>,b)=1, C(<s>,</s>)=0, C(<s>)=3", ""]
test_targets = ["a", "</s>"]                           # test record: the empty string -> only '</s>' after <s>
for w in ["a", "b", "</s>"]:
    p = p_mle(mc, h, mv.index[w])
    lines.append(f"  P_MLE({w:4}| <s>) = {p:.4f}   -log2 P = {nll_bits(p)}")
lines += ["", "Scoring the test record '' (target sequence: </s>) under P_MLE:",
          f"  loss = {nll_bits(p_mle(mc, h, mv.end_id))} bits  (kept as infinite: an unseen event has zero probability)",
          f"  same target under add-one q2: {AddOneLM(mc).q(2, h, mv.end_id):.4f} -> {-math.log2(AddOneLM(mc).q(2, h, mv.end_id)):.4f} bits",
          f"  unseen history P_MLE(a | b b) = {p_mle(mc, (mv.index['b'], mv.index['b']), mv.index['a'])} (undefined: C(h) = 0)"]
(EVIDENCE_P2_DIR / "mle_demo.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
mle_inf = nll_bits(p_mle(mc, h, mv.end_id)) == math.inf
assert mle_inf
""")

md(r"""
## 10. §7 Verify — hand calculations and normalization  *(both)*

Program output is compared with values worked by hand on toy corpora (saved to
`evidence/phase2/hand_check.md`), and normalization is checked on the **real** frozen engines for
observed and unseen histories.
""")

code(r"""
hv = Vocab("hand", ["a", "b", "</s>"], unk=None)
hc = NgramCounts(hv).fit([["a"], ["a"], ["b"]]); hlm = AddOneLM(hc)
S = hv.start_id
hand = []
def hrow(what, by_hand, expected, program):
    hand.append({"quantity": what, "by hand": by_hand, "expected": expected, "program": program,
                 "match": abs(expected - program) < 1e-12})

# §11 hand vocabulary: {a, b, end}, C(h,a)=2, C(h,b)=1, C(h,end)=0 -> 3/6, 2/6, 1/6
hrow("q2(a | <s>)",   "(2+1)/(3+3) = 3/6", 3/6, hlm.q(2, (S,), hv.index["a"]))
hrow("q2(b | <s>)",   "(1+1)/(3+3) = 2/6", 2/6, hlm.q(2, (S,), hv.index["b"]))
hrow("q2(</s> | <s>)", "(0+1)/(3+3) = 1/6", 1/6, hlm.q(2, (S,), hv.end_id))
# unigram: targets a,a,b,</s>x3 -> N = 6
hrow("q1(</s>)", "(3+1)/(6+3) = 4/9", 4/9, hlm.q(1, (), hv.end_id))
hrow("q2(b | a)", "(0+1)/(2+3) = 1/5", 1/5, hlm.q(2, (hv.index["a"],), hv.index["b"]))
# corpus aggregation: records "a b" and "b" under q2
recs = [["a", "b"], ["b"]]
sums = [record_losses(hlm.q_arrays(hv.map_tokens(r)), "q2") for r in recs]
H_hand = -(math.log2(3/6) + math.log2(1/5) + math.log2(2/4) + math.log2(2/6) + math.log2(2/4)) / 5
res = corpus_loss(sum(s[0] for s in sums), sum(s[1] for s in sums), sum(s[2] for s in sums))
hrow("corpus H_bits, records 'a b' + 'b' (T = 5)",
     "-(log2 3/6 + log2 1/5 + log2 2/4 + log2 2/6 + log2 2/4) / 5", H_hand, res["H_bits"])
hrow("H_nats / ln 2", "equals H_bits", H_hand, res["H_nats"] / math.log(2))
# four equally likely targets -> 2 bits, PP 4
uv = Vocab("uniform4", ["a", "b", "c", "</s>"], unk=None)
ulm = AddOneLM(NgramCounts(uv).fit([]))
u = corpus_loss(*record_losses(ulm.q_arrays(uv.map_tokens(["a", "b", "c"])), "q1"))
hrow("H_bits, four equally likely targets", "-log2(1/4) = 2", 2.0, u["H_bits"])
hrow("PP, four equally likely targets", "2**2 = 4", 4.0, u["PP"])
lam = (0.25, 0.25, 0.5)
hrow("sum_w P_mix(w | <s> <s>), λ = (.25,.25,.5)", "normalized", 1.0,
     sum(lam[0]*hlm.q(1, (), w) + lam[1]*hlm.q(2, (S,), w) + lam[2]*hlm.q(3, (S, S), w) for w in range(hv.size)))

show_table(hand, title="Hand calculation vs program", floatfmt="{:.12f}")
hand_ok = all(r["match"] for r in hand)
md_lines = ["# Phase 2 hand check", "", "Toy training records: `a`, `a`, `b`; V = {a, b, </s>}; framing `<s> <s> … </s>`.", "",
            "| quantity | by hand | program | match |", "|---|---|---|---|"]
md_lines[-2:] = ["| quantity | by hand | expected | program | match |", "|---|---|---|---|---|"]
md_lines += [f"| {r['quantity']} | `{r['by hand']}` | {r['expected']:.12f} | {r['program']:.12f} | {r['match']} |" for r in hand]
(EVIDENCE_P2_DIR / "hand_check.md").write_text("\n".join(md_lines) + "\n")
assert hand_ok
""")

code(r"""
# Normalization on the REAL frozen engines: observed, start-symbol and unseen histories
norm_rows = []
for tok, eng in engines.items():
    v, c, lm = eng.vocab, eng.counts, eng.lm
    first_tok = v.map_tokens(pipelines[tok].tokenize(val[0]["text"]))[0]
    unseen = next(i for i in range(v.size) if c.Ch((i,)) == 0)
    for k, h, kind in [(1, (), "empty"), (2, (first_tok,), "observed"), (2, (v.start_id,), "start"),
                       (2, (unseen,), "unseen"), (3, (v.start_id, v.start_id), "start,start"),
                       (3, (unseen, unseen), "unseen")]:
        d = lm.distribution(k, h)
        norm_rows.append({"tokenizer": tok, "order k": k, "history": kind, "C(h)": c.Ch(h),
                          "sum over V": float(d.sum()), "min prob": float(d.min()), "|V|": v.size})
show_table(norm_rows, title="Smoothed distributions sum to 1 (real engines, train_a1_slice)", floatfmt="{:.12g}")
normalization_ok = all(abs(r["sum over V"] - 1) < 1e-9 and r["min prob"] > 0 for r in norm_rows)
assert normalization_ok

ws = engines["whitespace"].vocab
unk_ok = ws.map_token("zzqxj-never-seen") == ws.unk_id and ws.map_token("</s>") == ws.unk_id
print("unseen whitespace token -> <UNK>:", unk_ok)
""")

md(r"""
## 11. B6 — λ selection on clean validation → freeze  *(Prazol)*

All 15 nonnegative (λ1, λ2, λ3) on the 0.25 grid summing to 1 are scored on the **clean**
validation sample, separately per tokenizer. Ties (within `1e-12` bits) go to the larger λ1, then
the larger λ2. The chosen λ are written to `configs/phase2.yaml` *below the frozen marker* and every
noisy score below reads λ back from that file.
""")

code(r"""
cfg, lam_table = select_and_freeze(engines, condition_rows["clean"], cfg, "train_a1_slice")
lam_df = pd.DataFrame(lam_table)
lam_df.to_csv(RESULTS_P2_DIR / "train_a1_slice" / "lambda_selection.csv", index=False)
LAMBDAS = {tok: frozen_lambdas(cfg, "train_a1_slice", tok) for tok in engines}
for tok in engines:
    show_table(lam_df[lam_df.tokenizer == tok].sort_values("clean_val_H_bits").head(5),
               cols=["lambda1", "lambda2", "lambda3", "clean_val_H_bits", "chosen"],
               title=f"{tok}: best 5 of 15 λ on clean validation")
print("frozen λ:", LAMBDAS)
print("config file hash after freeze:", cfg.file_sha256)
""")

md(r"""
## 12. B7 + A6 — run the frozen grid (primary: `train_a1_slice`)  *(both)*

First the **clean-invariance** check: the p = 0 dataset regenerated under every seed and operation
must score identically. Then all 37 datasets × 2 tokenizers × 3 models = **222 required scores**, plus
74 supplementary trigram ($q_3$) scores (296 in total). Every
per-record row keeps article/record/condition IDs, NLL sums in bits and nats, T, MED and the
coverage diagnostics (boundary symbols excluded; zero-token outputs flagged, not dropped).
""")

code(r"""
clean_scores = set()
for seed in cfg.seeds:
    for op in cfg.operations:
        rows0 = [corrupt_record(r, op, 0.0, seed, cfg) for r in val]
        for tok, eng in engines.items():
            toks = eng.pipeline.tokenize_batch([r["noisy_text"] for r in rows0])
            qs = q_for_records(eng, toks)
            for model in MODELS:
                s = [record_losses(q, model, LAMBDAS[tok] if model == "p_mix" else None) for q in qs]
                clean_scores.add((tok, model, sum(x[0] for x in s), sum(x[2] for x in s)))
clean_invariant = len(clean_scores) == 2 * len(MODELS)
print("distinct clean scores across 3 seeds x 3 operations:", len(clean_scores), f"(expected {2 * len(MODELS)})")
assert clean_invariant
""")

code(r"""
def run_training_set(train_set, engines, lambdas):
    t = time.time()
    rows = []
    for tok, eng in engines.items():
        rows += score_grid(eng, lambdas[tok], condition_rows)
    per_record = pd.DataFrame(rows)
    tables = write_tables(per_record, RESULTS_P2_DIR / train_set)
    print(f"{train_set}: {len(per_record):,} per-record rows, {len(tables['scores'])} scores, "
          f"{len(tables['summary'])} summary rows in {time.time() - t:.1f}s")
    return per_record, tables

snap_before = {tok: (eng.counts.snapshot(), eng.vocab.size) for tok, eng in engines.items()}
per_record, T_slice = run_training_set("train_a1_slice", engines, LAMBDAS)
counts_unchanged = all((eng.counts.snapshot(), eng.vocab.size) == snap_before[tok] for tok, eng in engines.items())
def counts_ok(tables):
    sc, sm = tables["scores"], tables["summary"]
    req_sc, req_sm = sc[sc.model.isin(REQUIRED_MODELS)], sm[sm.model.isin(REQUIRED_MODELS)]
    sup_sc, sup_sm = sc[sc.model.isin(SUPPLEMENTARY_MODELS)], sm[sm.model.isin(SUPPLEMENTARY_MODELS)]
    return (len(req_sc) == 222 and (req_sc.condition_id == "clean").sum() == 6
            and (req_sm.operation == "none").sum() == 6 and (req_sm.operation != "none").sum() == 72
            and len(sup_sc) == 74 and (sup_sm.operation == "none").sum() == 2
            and (sup_sm.operation != "none").sum() == 24)

sc, sm = T_slice["scores"], T_slice["summary"]
assert counts_ok(T_slice)
print("required: 222 scores, 6 clean + 72 noisy summary rows; supplementary q3: 74 scores, 2 + 24 rows")
print("evaluation left counts and V unchanged:", counts_unchanged)
""")

md("## 13. Results tables (`train_a1_slice`)  *(both)*")

code(r"""
clean = sm[sm.operation == "none"][["tokenizer", "model", "H_bits_mean", "PP_mean", "T_mean"]]
show_table(clean.rename(columns={"H_bits_mean": "H_bits", "PP_mean": "PP", "T_mean": "T"}),
           title="Clean validation scores (6 required + 2 supplementary q3) — compare WITHIN a tokenizer only")

for tok in ["whitespace", "wordpiece"]:
    s = sm[(sm.tokenizer == tok) & (sm.operation != "none")]
    piv = s.pivot_table(index=["operation", "p"], columns="model", values="H_bits_mean", sort=False).reset_index()
    rng = s.pivot_table(index=["operation", "p"], columns="model", values="delta_H_bits_mean", sort=False).reset_index()
    out = piv.merge(rng, on=["operation", "p"], suffixes=("", " ΔH"))
    show_table(out, title=f"{tok}: noisy H_bits (3-seed mean) and Δ vs clean — q1, q2, q3 (suppl.), P_mix")
""")

code(r"""
cb = T_slice["context_benefit"]
cbs = (cb.groupby(["tokenizer", "operation", "p"], sort=False)[["benefit_q2_bits", "benefit_q3_bits", "benefit_p_mix_bits"]]
         .mean().reset_index())
show_table(cbs[cbs.p.isin([0.0, 0.05, 0.20])], title="Context benefit H(q1) − H(·), bits (positive = better than q1)")

cov = T_slice["coverage"]
covs = (cov.groupby(["tokenizer", "operation", "p"], sort=False)
           [["unk_rate", "pieces_per_span", "continuation_rate", "mean_token_count_change", "zero_token_records"]]
           .mean().reset_index())
show_table(covs[covs.p.isin([0.0, 0.05, 0.20])], title="Coverage and fragmentation (3-seed mean of pooled rates)")
""")

md(r"""
## 14. Figures  *(both)*

Generated only from the saved CSVs in `results/phase2/`. Separate tokenizer panels, one y-scale per
panel, operation = fixed colour + marker, band = min–max over the three seeds, p = 0 is the shared
clean score.
""")

code(r"""
def load_csv(train_set, name):
    return pd.read_csv(RESULTS_P2_DIR / train_set / f"{name}.csv")

figs = [
    figures.loss_vs_rate(load_csv("train_a1_slice", "summary"), "train_a1_slice", FIG_DIR / "loss_vs_rate_train_a1_slice.png"),
    figures.context_benefit_fig(load_csv("train_a1_slice", "context_benefit"), "train_a1_slice", FIG_DIR / "context_benefit_train_a1_slice.png"),
    figures.coverage_fig(load_csv("train_a1_slice", "coverage"), "train_a1_slice", FIG_DIR / "coverage_train_a1_slice.png"),
    figures.severity_fig(pd.read_csv(RESULTS_P2_DIR / "severity_records.csv"), FIG_DIR / "severity.png"),
]
for f in figs:
    display(Image(filename=str(f)))
""")

md(r"""
## 14b. Why the loss moves in opposite directions  *(both)*

Noise creates words the model has never seen, so one might expect the loss to rise everywhere. It
does for WordPiece, but it **falls** for whitespace. The reason is how each pipeline represents an
unseen word:

* **whitespace** maps it to `<UNK>`, the most frequent training target, which costs only ~3 bits;
* **WordPiece** splits it into several rare `##` pieces, ~14 bits each, and the target count grows.

H is an average over targets, so it splits *exactly* into per-type contributions:
$H = \sum_{\text{type}} \text{share(type)} \times \text{mean bits(type)}$. The table and figure below
show that decomposition for $P_{mix}$ (saved to `token_type_breakdown.csv`; the parts add up to the
saved H to within $10^{-9}$). A falling whitespace score is unknown-token collapse, not robustness.
""")

code(r"""
def decompose(train_set, engines, lambdas):
    rows = []
    for tok, eng in engines.items():
        rows += loss_decomposition(eng, lambdas[tok], condition_rows)
    dec = pd.DataFrame(rows)
    dec.to_csv(RESULTS_P2_DIR / train_set / "token_type_breakdown.csv", index=False)
    sc_mix = pd.read_csv(RESULTS_P2_DIR / train_set / "scores.csv")
    sc_mix = sc_mix[sc_mix.model == "p_mix"].set_index(["tokenizer", "condition_id"]).H_bits
    total = dec.groupby(["tokenizer", "condition_id"]).contribution_bits.sum()
    return dec, float((total - sc_mix.loc[total.index]).abs().max())

dec_slice, dec_err_slice = decompose("train_a1_slice", engines, LAMBDAS)
print("max |sum of contributions − saved H| =", dec_err_slice)
view = dec_slice[dec_slice.condition_id.isin(["clean", "ins-p0.20-s11", "del-p0.20-s11", "sub-p0.20-s11"])]
show_table(view[["tokenizer", "condition_id", "token_type", "targets", "share", "mean_bits", "contribution_bits"]],
           title="P_mix loss by target type (seed 11 shown; all seeds in token_type_breakdown.csv)")
display(Image(filename=str(figures.loss_decomposition_fig(
    pd.read_csv(RESULTS_P2_DIR / "train_a1_slice" / "token_type_breakdown.csv"), "train_a1_slice",
    FIG_DIR / "loss_decomposition_train_a1_slice.png"))))
""")

md(r"""
## 15. Error analysis — three traceable examples  *(both)*

Selected deterministically from the saved per-record rows (A1-slice engines, `P_mix`, seed 11,
p = 0.20). Each example names its record ID, article ID and condition, so it traces back to the
manifests. Example 1 is chosen to **challenge a simple robustness story**.
""")

code(r"""
pr = per_record[per_record.model == "p_mix"].copy()
pr["loss_per_target"] = pr.nll_bits_sum / pr["T"]
clean_pr = pr[pr.condition_id == "clean"].set_index(["tokenizer", "record_id"])
def delta(df):
    c = clean_pr.loc[list(zip(df.tokenizer, df.record_id))]
    df = df.copy()
    df["clean_loss_per_target"] = c["loss_per_target"].values
    df["clean_T"] = c["T"].values
    df["clean_n_unk"] = c["n_unk"].values
    df["delta_loss_per_target"] = df.loss_per_target - df.clean_loss_per_target
    return df

rows_by_id = {cid: {r["record_id"]: r for r in rows} for cid, rows in condition_rows.items()}
examples = []

# 1 — challenges robustness: whitespace loss FALLS under heavy substitution
d1 = delta(pr[(pr.tokenizer == "whitespace") & (pr.condition_id == "sub-p0.20-s11")])
e1 = d1.sort_values(["delta_loss_per_target", "record_id"]).iloc[0]
eng_ws = engines["whitespace"]
q_unk = eng_ws.lm.q(1, (), eng_ws.vocab.unk_id)
examples.append(("1. Whitespace loss falls under 20% substitution (challenges 'noise always hurts')", e1,
    f"<UNK> tokens rose from {e1.clean_n_unk} to {e1.n_unk} of {int(e1.n_tokens)}. Under the slice vocabulary "
    f"(|V| = {eng_ws.vocab.size}) <UNK> is the most frequent target: q1(<UNK>) = {q_unk:.3f} "
    f"({-math.log2(q_unk):.2f} bits), far cheaper than a typical in-vocabulary word. Corrupted words collapse into "
    f"that cheap symbol, so the per-target loss drops by {-e1.delta_loss_per_target:.2f} bits. The lower score "
    f"reflects unknown-token collapse, not robustness."))

# 2 — WordPiece: largest per-target loss increase under substitution
d2 = delta(pr[(pr.tokenizer == "wordpiece") & (pr.condition_id == "sub-p0.20-s11")])
e2 = d2.sort_values(["delta_loss_per_target", "record_id"], ascending=[False, True]).iloc[0]
examples.append(("2. WordPiece: largest per-target loss increase under 20% substitution", e2,
    f"The {e2.clean_T}-target clean record became {e2['T']} targets (pieces per original word "
    f"{e2.pieces_per_span:.2f}, continuation rate {e2.continuation_rate:.2f}). Noise changes the prediction "
    f"targets themselves: rare '##' fragments replace frequent whole-word pieces, and the loss rises by "
    f"{e2.delta_loss_per_target:.2f} bits per target."))

# 3 — WordPiece: deletion shortens the text but adds targets
d3 = delta(pr[(pr.tokenizer == "wordpiece") & (pr.condition_id == "del-p0.20-s11")])
d3["target_change"] = d3["T"] - d3.clean_T
e3 = d3.sort_values(["target_change", "record_id"], ascending=[False, True]).iloc[0]
examples.append(("3. WordPiece: deletion removes characters yet adds prediction targets", e3,
    f"Deleting {rows_by_id['del-p0.20-s11'][e3.record_id]['realized_edits']} letters (MED {e3.med}) made the text "
    f"shorter, but the target count grew from {e3.clean_T} to {e3['T']} because damaged words fragment into "
    f"more pieces. Total NLL and T both change, so a per-target loss compares different prediction problems."))

wp_pipe = pipelines["wordpiece"]
md_out = ["# Phase 2 error analysis — train_a1_slice, P_mix", ""]
for title, e, why in examples:
    clean_text = rows_by_id["clean"][e.record_id]["noisy_text"]
    noisy_text = rows_by_id[e.condition_id][e.record_id]["noisy_text"]
    tok = e.tokenizer
    ct = pipelines[tok].tokenize(clean_text)[:18]; nt = pipelines[tok].tokenize(noisy_text)[:18]
    block = [f"## {title}", "",
             f"- record `{e.record_id}` · article `{e.article_id}` · condition `{e.condition_id}` · tokenizer `{tok}`",
             f"- MED {e.med}, normalized MED {e.normalized_med:.3f}",
             f"- loss per target {e.clean_loss_per_target:.3f} → {e.loss_per_target:.3f} bits (Δ {e.delta_loss_per_target:+.3f}); T {e.clean_T} → {e['T']}",
             f"- clean: {clean_text[:170]}…", f"- noisy: {noisy_text[:170]}…",
             f"- clean tokens: `{' '.join(ct)}` …", f"- noisy tokens: `{' '.join(nt)}` …",
             f"- **reading:** {why}", ""]
    md_out += block
    display(Markdown("\n".join(block)))
(RESULTS_P2_DIR / "train_a1_slice" / "examples.md").write_text("\n".join(md_out))
""")

md(r"""
## 16. Corpus-size extension — `train_full`  *(both)*

Same validation sample, same 37 noisy datasets (read from disk, not regenerated), same rule and
tie-break. Only the training text changes; λ are re-selected on clean validation for this training
set and frozen before noisy scoring. Comparisons are made **within a tokenizer only**. Caveat: the
whitespace |V| changes with the training set, so even within the whitespace pipeline raw perplexity
is not strictly comparable; WordPiece V is fixed and gives the cleaner comparison. Two training
sizes are two points, not a scaling law (cf. Kaplan et al., 2020).
""")

code(r"""
engines_full = build_engines("train_full")
vocab_full = pd.DataFrame([{"tokenizer": tok, "|V|": e.vocab.size, **e.vocab_stats, **e.counts.stats()}
                           for tok, e in engines_full.items()])
(RESULTS_P2_DIR / "train_full").mkdir(exist_ok=True)
vocab_full.to_csv(RESULTS_P2_DIR / "train_full" / "vocab.csv", index=False)
show_table(vocab_full, cols=["tokenizer", "|V|", "training_tokens", "training_tokens_mapped_to_unk", "targets",
                             "distinct_bigrams", "distinct_trigrams"], title="Frozen V and counts — train_full",
           floatfmt="{:.0f}")

cfg, lam_table_full = select_and_freeze(engines_full, condition_rows["clean"], cfg, "train_full")
pd.DataFrame(lam_table_full).to_csv(RESULTS_P2_DIR / "train_full" / "lambda_selection.csv", index=False)
LAMBDAS_FULL = {tok: frozen_lambdas(cfg, "train_full", tok) for tok in engines_full}
print("frozen λ (train_full):", LAMBDAS_FULL)

per_record_full, T_full = run_training_set("train_full", engines_full, LAMBDAS_FULL)
assert counts_ok(T_full)
""")

code(r"""
rows = []
for tok in ["whitespace", "wordpiece"]:
    for model in MODELS:
        r = {"tokenizer": tok, "model": model}
        for ts, T in [("slice", T_slice), ("full", T_full)]:
            s = T["summary"]; s = s[(s.tokenizer == tok) & (s.model == model)]
            r[f"clean_H_{ts}"] = float(s[s.operation == "none"].H_bits_mean.iloc[0])
            for op in cfg.operations:
                r[f"ΔH_{op[:3]}_p0.20_{ts}"] = float(s[(s.operation == op) & (s.p == 0.2)].delta_H_bits_mean.iloc[0])
        rows.append(r)
size_cmp = pd.DataFrame(rows)
for ts, e, T in [("slice", engines, T_slice), ("full", engines_full, T_full)]:
    cv = T["coverage"]
    for tok in ["whitespace", "wordpiece"]:
        size_cmp.loc[size_cmp.tokenizer == tok, f"|V|_{ts}"] = e[tok].vocab.size
        size_cmp.loc[size_cmp.tokenizer == tok, f"clean_unk_rate_{ts}"] = float(
            cv[(cv.tokenizer == tok) & (cv.condition_id == "clean")].unk_rate.iloc[0])
size_cmp.to_csv(RESULTS_P2_DIR / "corpus_size_comparison.csv", index=False)
show_table(size_cmp, cols=["tokenizer", "model", "|V|_slice", "|V|_full", "clean_unk_rate_slice", "clean_unk_rate_full",
                           "clean_H_slice", "clean_H_full", "ΔH_sub_p0.20_slice", "ΔH_sub_p0.20_full",
                           "ΔH_del_p0.20_slice", "ΔH_del_p0.20_full"],
           title="Corpus size: A1 slice (31k words) vs full train (2.0M words) — within tokenizer only")

both = pd.concat([load_csv("train_a1_slice", "summary"), load_csv("train_full", "summary")])
figs2 = [figures.loss_vs_rate(load_csv("train_full", "summary"), "train_full", FIG_DIR / "loss_vs_rate_train_full.png"),
         figures.context_benefit_fig(load_csv("train_full", "context_benefit"), "train_full", FIG_DIR / "context_benefit_train_full.png"),
         figures.coverage_fig(load_csv("train_full", "coverage"), "train_full", FIG_DIR / "coverage_train_full.png"),
         figures.corpus_size_fig(both, FIG_DIR / "corpus_size.png")]
for f in figs2[::3]:
    display(Image(filename=str(f)))

dec_full, dec_err_full = decompose("train_full", engines_full, LAMBDAS_FULL)
display(Image(filename=str(figures.loss_decomposition_fig(
    pd.read_csv(RESULTS_P2_DIR / "train_full" / "token_type_breakdown.csv"), "train_full",
    FIG_DIR / "loss_decomposition_train_full.png"))))
""")

md(r"""
## 17. A7 — the 20 Assignment 1 diagnostic pairs (regression)  *(both)*

Regenerated with the **A1 rule** and Phase 1 configuration, then compared byte-for-byte with
`evidence/pairs/pairs.jsonl`. They are a regression check, not the aggregate evaluation sample.
""")

code(r"""
from src.tokenizer.wrapper import TokenizerWrapper
from src.integration.generate_pairs import pair_schedule, build_pair_record
from src.integration.schema import write_jsonl

slice_examples = load_slice()
plan = pair_schedule(cfg1.pairs_count, cfg1.noise_operations, cfg1.noise_intensity)
wrapper = TokenizerWrapper(cfg1)
rerun = [build_pair_record(ex, op, it, wrapper, cfg1)[0] for ex, (op, it) in zip(slice_examples, plan)]
rerun_path = write_jsonl(rerun, EVIDENCE_P2_DIR / "a1_pairs_rerun.jsonl")
pairs_identical = rerun_path.read_bytes() == (PAIRS_DIR / "pairs.jsonl").read_bytes()
msg = f"20 A1 pairs regenerated: {len(rerun)} records; byte-identical to evidence/pairs/pairs.jsonl: {pairs_identical}"
(EVIDENCE_P2_DIR / "a1_pairs_regression.txt").write_text(msg + "\n")
print(msg)
assert pairs_identical
""")

md(r"""
## 18. Tests + acceptance checklist  *(both)*

The full unit-test suite (Phase 1 + Phase 2) is run and saved, then every box of the plan's
acceptance checklist (§9) is checked in code. Output: `evidence/phase2/acceptance.txt`.
""")

code(r"""
proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no"], capture_output=True, text=True, cwd=ROOT)
(EVIDENCE_P2_DIR / "pytest.txt").write_text(proc.stdout + proc.stderr)
print(proc.stdout[-1500:])
tests_pass = proc.returncode == 0
a1_proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no", "tests/test_med.py", "tests/test_noise.py"],
                         capture_output=True, text=True, cwd=ROOT)
a1_tests_pass = a1_proc.returncode == 0
""")

code(r"""
checks = []
def check(name, ok):
    checks.append({"check": name, "result": "PASS" if ok else "FAIL"}); return ok

all_rows = [r for rows in condition_rows.values() for r in rows]
clean_ok = all(r["noisy_text"] == v["text"] for r, v in zip(condition_rows["clean"], val))
check("All Assignment 1 MED and reproducibility tests still pass; p = 0 leaves the input unchanged",
      a1_tests_pass and clean_ok and z[0] == demo["text"])

v2_ok = True
val_text = {v["record_id"]: v["text"] for v in val}
for r in all_rows:
    clean_t = val_text[r["record_id"]]
    log = r["operation_log"]
    v2_ok &= replay_v2(clean_t, log) == r["noisy_text"]
    v2_ok &= all(clean_t[e["index"]].isascii() and clean_t[e["index"]].isalpha() for e in log)
    v2_ok &= all(clean_t.count(sp) == r["noisy_text"].count(sp) for sp in cfg.protected_spans)
    v2_ok &= r["realized_edits"] <= r["eligible_count"]
    if r["operation"] == "insertion":
        v2_ok &= len(r["noisy_text"]) == len(clean_t) + r["realized_edits"]
check("Noise v2: letters only, insert-after, no re-corruption, sentinels preserved, replay exact (all 18,500 rows)", v2_ok)

check("Every smoothed distribution sums to 1 for observed and unseen histories; all allowed targets > 0", normalization_ok)
check("Hand vocab {a, b, end} gives 3/6, 2/6, 1/6", all(r["match"] for r in hand[:3]))

lam_ok = all(min(l) >= 0 and abs(sum(l) - 1) < 1e-12 for L in (LAMBDAS, LAMBDAS_FULL) for l in L.values())
lam_frozen = (load_phase2_config().frozen_lambdas.get("train_a1_slice") == {k: list(v) for k, v in LAMBDAS.items()}
              and load_phase2_config().frozen_lambdas.get("train_full") == {k: list(v) for k, v in LAMBDAS_FULL.items()})
check("λ nonnegative, sum to 1, selected on clean val, frozen in configs/phase2.yaml before the noisy grid; P_mix normalized",
      lam_ok and lam_frozen and hand[-1]["match"])
check("Four equally likely targets give 2 bits / PP 4; corpus aggregation matches the hand calculation",
      all(r["match"] for r in hand[3:9]))
check("Unseen whitespace tokens map to <UNK>; boundaries reset context; evaluation never mutates counts or V",
      unk_ok and counts_unchanged and tests_pass)
check("Unsmoothed MLE demo shows infinite loss on a zero-probability event", mle_inf)
check("p = 0 gives the identical clean score for every seed", clean_invariant)
check("222 scores, 6 clean + 72 noisy summary rows saved per training set (+ supplementary q3: 74 scores, 2 + 24 rows)",
      all(counts_ok({"scores": pd.read_csv(RESULTS_P2_DIR / ts / "scores.csv"),
                     "summary": pd.read_csv(RESULTS_P2_DIR / ts / "summary.csv")})
          for ts in ("train_a1_slice", "train_full")))
same_inputs = all(set(map(tuple, g[["condition_id", "record_id"]].drop_duplicates().values.tolist()))
                  == set((c, r["record_id"]) for c, rows in condition_rows.items() for r in rows)
                  for _, g in per_record.groupby("tokenizer"))
hashes_ok = all(r["protocol_sha256"] == cfg.protocol_sha256 and len(r["source_text_sha256"]) == 64 for r in all_rows)
check("Every noisy text generated once and shared by both tokenizers; manifests carry source and config hashes",
      same_inputs and hashes_ok)
check("Empty-after-noise records kept, </s> scored, undefined rates flagged",
      (per_record["T"] >= 1).all() and (per_record[per_record.n_tokens == 0].rate_undefined).all()
      and len(per_record) == 37 * 500 * 2 * len(MODELS))
check("20 A1 diagnostic pairs reproduce byte-identically", pairs_identical)
no_test = (not per_record.record_id.str.startswith("wt2-test").any()
           and not any(r["record_id"].startswith("wt2-test") for recs in train_sets.values() for r in recs))
check("Test split never scored, inspected, or used for any choice", no_test)
readme = (ROOT / "README.md").read_text()
summary_doc = ROOT / "docs" / "phase2_summary.md"
check("§17: README reproduces a headline table; plotted points trace to saved CSVs; write-up states metric limits; outputs reusable",
      "phase2.ipynb" in readme and "summary.csv" in readme
      and all((FIG_DIR / n).exists() for n in ["loss_vs_rate_train_a1_slice.png", "severity.png", "corpus_size.png"])
      and summary_doc.exists() and "cannot" in summary_doc.read_text())
check("Loss decomposition by target type adds up to the saved H (both training sets)",
      max(dec_err_slice, dec_err_full) < 1e-9)
check("Full unit-test suite passes", tests_pass)

show_table(checks, title="Phase 2 acceptance checklist (plan §9)")
report = "\n".join(f"[{c['result']}] {c['check']}" for c in checks)
(EVIDENCE_P2_DIR / "acceptance.txt").write_text(report + "\n")
assert all(c["result"] == "PASS" for c in checks), "an acceptance check failed"
print("ALL PHASE 2 ACCEPTANCE CHECKS PASS")
""")

code(r"""
run_info = {
    "protocol_sha256": cfg.protocol_sha256,
    "config_file_sha256": sha256_file(PHASE2_CONFIG_PATH),
    "frozen_lambdas": cfg.frozen_lambdas,
    "val_sample_sha256": sha256_file(VAL_PATH),
    "training_set_sha256": {k: sha256_file(p) for k, p in TRAIN_PATHS.items()},
    "grid_index_sha256": sha256_file(RESULTS_P2_DIR / "grid_index.csv"),
    "env": {"python": platform.python_version(), "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    "runtime_seconds": round(time.time() - T0, 1),
}
(RESULTS_P2_DIR / "run_info.json").write_text(json.dumps(run_info, indent=2))
print(json.dumps(run_info, indent=2))
""")

md(r"""
## 19. Artifacts produced

| Path | What |
|---|---|
| `configs/phase2.yaml` | frozen protocol + λ written by step B6 |
| `derived/phase2/val_sample.jsonl` (+ `.sha256`) | 500 validation records with article IDs |
| `derived/phase2/train_a1_slice.jsonl`, `train_full.jsonl` (+ `.sha256`) | the two clean training sets |
| `derived/phase2/noisy/<condition_id>.jsonl` | 37 datasets: noisy text + full corruption manifest per record |
| `results/phase2/grid_index.csv`, `severity.csv`, `severity_records.csv` | grid file hashes; requested p vs realized normalized MED |
| `results/phase2/<train_set>/per_record.parquet` | one row per record × condition × tokenizer × model |
| `results/phase2/<train_set>/scores.csv` | the 222 required scores + 74 supplementary trigram (q3) scores |
| `results/phase2/<train_set>/summary.csv` | 6 clean + 72 noisy required rows + 2 + 24 q3 rows (mean/min/max over seeds) |
| `results/phase2/<train_set>/within_pipeline.csv`, `context_benefit.csv`, `coverage.csv` | required §9 measures |
| `results/phase2/<train_set>/vocab.csv`, `lambda_selection.csv`, `examples.md` | V and counts, λ search, error analysis |
| `results/phase2/<train_set>/token_type_breakdown.csv` | P_mix loss split by target type (word, `##`, unknown, `</s>`) |
| `results/phase2/corpus_size_comparison.csv`, `figures/*.png`, `run_info.json` | extension table, figures, hashes |
| `evidence/phase2/*` | pytest output, acceptance checklist, hand check, MLE demo, fast-MED check, 20-pair regression |
| `docs/phase2_summary.md` | Phase 2 write-up (interpretation, limitations, ethics) |
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (Bert-Project)", "language": "python",
                   "name": "bert-project"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parent.parent / "phase2.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
