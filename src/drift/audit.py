"""Run the vector-drift audit over the 37 Assignment 2 conditions and write the results.

Outputs (results/phase3/vector_drift/):
    selected_spans.jsonl (+ .sha256)  frozen span list, chosen from clean text only
    span_audit.csv                    one row per (condition, span): pieces, cosine, distance, status
    condition_summary.csv             37 rows: coverage, mean cosine/distance over VALID pairs, diagnostics
    noise_summary.csv                 12 rows (op x p): mean/min/max over the three corruption seeds
    run_info.json                     tokenizer, E checksum, input hashes, versions
    examples.md                       outcome-blind candidate spans for the traced examples

Denominators (also repeated in run_info.json):
    coverage              = valid / selected
    mean_cosine, mean_euclidean = mean over valid pairs only
    pieces_per_clean_span = total clean pieces / selected spans
    pieces_per_noisy_span = total noisy pieces / selected spans   (an emptied span counts 0 pieces)
    unk_rate              = noisy [UNK] pieces / noisy pieces      (0 when there are no pieces)
    mean_span_med         = mean unit-cost edit distance between clean and noisy span text, over selected spans
    changed_span_fraction = spans whose noisy text differs from the clean text / selected spans
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from ..med.fast import med_fast
from ..noise.grid import conditions, load_condition
from ..phase2_config import DERIVED_P2_DIR, load_phase2_config, sha256_file
from ..tokenizer.wrapper import TokenizerWrapper
from .embeddings import SpanEmbedder, extract_E, load_E
from .geometry import EMPTY, VALID, ZERO_NORM, pair_metrics
from .paths import DRIFT_DIR, E_PATH, SPAN_LIMIT
from .spans import select_spans, noisy_span

VAL_PATH = DERIVED_P2_DIR / "val_sample.jsonl"
UNK = "[UNK]"
SPAN_FIELDS = ["condition_id", "operation", "p", "seed", "span_id", "record_id", "span_index",
               "start", "end", "clean", "noisy", "clean_pieces", "noisy_pieces", "n_clean_pieces",
               "n_noisy_pieces", "noisy_unk", "span_med", "status", "cosine", "euclidean"]
SUMMARY_FIELDS = ["condition_id", "operation", "p", "seed", "selected", "valid", "undefined",
                  "undefined_empty", "undefined_zero_norm", "coverage", "mean_cosine",
                  "mean_euclidean", "pieces_per_clean_span", "pieces_per_noisy_span", "unk_rate",
                  "mean_span_med", "changed_span_fraction"]
SEED_METRICS = ["mean_cosine", "mean_euclidean", "coverage", "pieces_per_noisy_span", "unk_rate",
                "mean_span_med"]


def lf_sha256(path: Path) -> str:
    """sha256 with CRLF folded to LF: Phase 2 recorded checksums on LF bytes, and a Windows
    checkout (autocrlf) rewrites the line endings on disk."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_val_sample(path: Path = VAL_PATH) -> str:
    recorded = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    digest = lf_sha256(path)
    assert digest == recorded, "val_sample.jsonl no longer matches its recorded checksum"
    return digest


def load_records(path: Path = VAL_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def check_continuity(records: list[dict], rows_by_condition: dict[str, list[dict]]) -> None:
    """Every condition holds exactly the retained records, in order, with matching text hashes."""
    ids = [r["record_id"] for r in records]
    hashes = {r["record_id"]: hashlib.sha256(r["text"].encode("utf-8")).hexdigest()
              for r in records}
    for cid, rows in rows_by_condition.items():
        assert [r["record_id"] for r in rows] == ids, f"{cid}: record IDs/order differ"
        for r in rows:
            assert r["source_text_sha256"] == hashes[r["record_id"]], \
                f"{cid}: text hash mismatch for {r['record_id']}"
            assert r["split"] == "validation", f"{cid}: not a validation row"


def summarize_condition(cond: dict, rows: list[dict]) -> dict:
    """One condition_summary row from that condition's span_audit rows."""
    n = len(rows)
    valid = [r for r in rows if r["status"] == VALID]
    empty = sum(r["status"] == EMPTY for r in rows)
    zero = sum(r["status"] == ZERO_NORM for r in rows)
    noisy_pieces = sum(r["n_noisy_pieces"] for r in rows)
    return {
        "condition_id": cond["condition_id"],
        "operation": cond["operation"] or "none", "p": cond["p"],
        "seed": "" if cond["seed"] is None else cond["seed"],
        "selected": n, "valid": len(valid), "undefined": n - len(valid),
        "undefined_empty": empty, "undefined_zero_norm": zero,
        "coverage": len(valid) / n if n else 0.0,
        "mean_cosine": float(np.mean([r["cosine"] for r in valid])) if valid else "",
        "mean_euclidean": float(np.mean([r["euclidean"] for r in valid])) if valid else "",
        "pieces_per_clean_span": sum(r["n_clean_pieces"] for r in rows) / n if n else 0.0,
        "pieces_per_noisy_span": noisy_pieces / n if n else 0.0,
        "unk_rate": sum(r["noisy_unk"] for r in rows) / noisy_pieces if noisy_pieces else 0.0,
        "mean_span_med": sum(r["span_med"] for r in rows) / n if n else 0.0,
        "changed_span_fraction": sum(r["clean"] != r["noisy"] for r in rows) / n if n else 0.0,
    }


def seed_summary(summaries: list[dict]) -> list[dict]:
    """12 rows (operation x p): mean/min/max of each metric over the corruption seeds."""
    out = []
    noisy = [s for s in summaries if s["operation"] != "none"]
    keys = sorted({(s["operation"], s["p"]) for s in noisy},
                  key=lambda k: (["insertion", "deletion", "substitution"].index(k[0]), k[1]))
    for op, p in keys:
        group = [s for s in noisy if s["operation"] == op and s["p"] == p]
        row = {"operation": op, "p": p, "n_seeds": len(group),
               "seeds": "|".join(str(s["seed"]) for s in group)}
        for m in SEED_METRICS:
            vals = [s[m] for s in group if s[m] != ""]
            row[f"{m}_mean"] = float(np.mean(vals)) if vals else ""
            row[f"{m}_min"] = float(np.min(vals)) if vals else ""
            row[f"{m}_max"] = float(np.max(vals)) if vals else ""
        out.append(row)
    return out


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def _pieces_str(pieces: list[str]) -> str:
    return " ".join(pieces)


def candidate_examples(span_rows: list[dict]) -> str:
    """Outcome-blind candidates: the first spans (saved order) that meet each structural criterion."""
    def pick(cond, pred, k=3):
        return [r for r in span_rows if r["condition_id"] == cond and pred(r)][:k]

    groups = [
        ("Substitution that changes the pieces", "sub-p0.10-s11",
         lambda r: r["noisy"] != r["clean"] and r["status"] == VALID),
        ("Insertion", "ins-p0.10-s11",
         lambda r: r["noisy"] != r["clean"] and r["status"] == VALID),
        ("Fragmentation into more pieces", "ins-p0.20-s11",
         lambda r: r["n_noisy_pieces"] > r["n_clean_pieces"] and r["status"] == VALID),
        ("Fully deleted span (undefined pair)", "del-p0.20-s11",
         lambda r: r["status"] == EMPTY),
    ]
    lines = ["# Candidate spans for the traced examples",
             "",
             "Chosen without looking at any outcome: the first spans, in saved order, that meet each "
             "structural criterion. Join with Workstream B's model outcomes before using.",
             ""]
    for title, cond, pred in groups:
        lines += [f"## {title} (`{cond}`)", "",
                  "| span_id | clean | noisy | clean pieces | noisy pieces | cosine | distance |",
                  "|---|---|---|---|---|---|---|"]
        for r in pick(cond, pred):
            cos = "" if r["cosine"] is None else f"{r['cosine']:.4f}"
            dist = "" if r["euclidean"] is None else f"{r['euclidean']:.4f}"
            lines.append(f"| {r['span_id']} | `{r['clean']}` | `{r['noisy']}` | "
                         f"{r['clean_pieces']} | {r['noisy_pieces']} | {cos} | {dist} |")
        lines.append("")
    return "\n".join(lines)


def run(limit: int = SPAN_LIMIT, out_dir: Path = DRIFT_DIR) -> dict:
    cfg = load_phase2_config()
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = TokenizerWrapper()
    assert wrapper.cfg.checkpoint == cfg.checkpoint
    assert wrapper.cfg.checkpoint_revision == cfg.checkpoint_revision
    tokenizer = wrapper.tokenizer

    if not E_PATH.exists():
        print("E not found; extracting it from the pinned checkpoint ...", file=sys.stderr)
        extract_E(cfg.checkpoint, cfg.checkpoint_revision)
    E, e_sha = load_E()
    assert E.shape[0] >= tokenizer.vocab_size

    val_sha = check_val_sample()
    records = load_records()
    conds = conditions(cfg)
    rows_by_condition = {c["condition_id"]: load_condition(c["condition_id"]) for c in conds}
    check_continuity(records, rows_by_condition)

    spans = select_spans(records, limit, cfg.protected_spans)
    spans_path = out_dir / "selected_spans.jsonl"
    with open(spans_path, "w", encoding="utf-8", newline="") as fh:
        for s in spans:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    spans_sha = sha256_file(spans_path)
    (out_dir / "selected_spans.sha256").write_text(f"{spans_sha}  selected_spans.jsonl\n",
                                                   encoding="utf-8")

    embedder = SpanEmbedder(tokenizer, E)
    span_rows: list[dict] = []
    summaries: list[dict] = []
    for cond in conds:
        by_record = {r["record_id"]: r for r in rows_by_condition[cond["condition_id"]]}
        cond_rows = []
        for s in spans:
            clean_pieces, u = embedder.embed(s["clean"])
            noisy = noisy_span(s, by_record[s["record_id"]]["operation_log"])
            noisy_pieces, v = embedder.embed(noisy)
            cos, dist, status = pair_metrics(u, v)
            cond_rows.append({
                "condition_id": cond["condition_id"], "operation": cond["operation"] or "none",
                "p": cond["p"], "seed": "" if cond["seed"] is None else cond["seed"],
                "span_id": s["span_id"], "record_id": s["record_id"],
                "span_index": s["span_index"], "start": s["start"], "end": s["end"],
                "clean": s["clean"], "noisy": noisy,
                "clean_pieces": _pieces_str(clean_pieces), "noisy_pieces": _pieces_str(noisy_pieces),
                "n_clean_pieces": len(clean_pieces), "n_noisy_pieces": len(noisy_pieces),
                "noisy_unk": sum(p == UNK for p in noisy_pieces),
                "span_med": med_fast(s["clean"], noisy), "status": status,
                "cosine": cos, "euclidean": dist})
        if cond["condition_id"] == "clean":
            for r in cond_rows:
                assert r["status"] == VALID and abs(r["cosine"] - 1.0) < 1e-9 \
                    and r["euclidean"] < 1e-9, f"clean pair not identical: {r['span_id']}"
        span_rows.extend(cond_rows)
        summaries.append(summarize_condition(cond, cond_rows))

    _write_csv(out_dir / "span_audit.csv", SPAN_FIELDS, span_rows)
    _write_csv(out_dir / "condition_summary.csv", SUMMARY_FIELDS, summaries)
    noise = seed_summary(summaries)
    noise_fields = ["operation", "p", "n_seeds", "seeds"] + [
        f"{m}_{a}" for m in SEED_METRICS for a in ("mean", "min", "max")]
    _write_csv(out_dir / "noise_summary.csv", noise_fields, noise)
    (out_dir / "examples.md").write_text(candidate_examples(span_rows), encoding="utf-8")

    import tokenizers, transformers
    info = {
        "workstream": "Assignment 3 / A: vector drift", "label": "preliminary validation results",
        "checkpoint": cfg.checkpoint, "checkpoint_revision": cfg.checkpoint_revision,
        "vocab_size": tokenizer.vocab_size, "embedding_shape": list(E.shape), "E_sha256": e_sha,
        "tokenization": "per span, add_special_tokens=False; raw E rows mean-pooled in float64",
        "val_sample_sha256": val_sha, "protocol_sha256": cfg.protocol_sha256,
        "selected_spans_sha256": spans_sha, "spans_selected": len(spans),
        "records_covered": len({s["record_id"] for s in spans}),
        "conditions": len(conds), "span_audit_rows": len(span_rows),
        "denominators": {"coverage": "valid / selected",
                         "mean_cosine, mean_euclidean": "over valid pairs only",
                         "pieces_per_clean_span": "total clean pieces / selected spans",
                         "pieces_per_noisy_span": "total noisy pieces / selected spans (emptied span = 0)",
                         "unk_rate": "noisy [UNK] pieces / noisy pieces",
                         "mean_span_med": "mean unit-cost edit distance of span text, over selected spans",
                         "changed_span_fraction": "spans with noisy text != clean text / selected spans"},
        "env": {"python": platform.python_version(), "numpy": np.__version__,
                "transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Assignment 3 / Workstream A: measure vector drift")
    ap.add_argument("--limit", type=int, default=SPAN_LIMIT, help="max spans (default 1000)")
    ap.add_argument("--out-dir", type=Path, default=DRIFT_DIR)
    args = ap.parse_args(argv)
    info = run(args.limit, args.out_dir)
    print(f"selected {info['spans_selected']} spans from {info['records_covered']} records; "
          f"{info['conditions']} conditions; wrote {args.out_dir}")
