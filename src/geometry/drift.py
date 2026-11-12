"""G2-G7 as one reproducible pipeline: spans -> noisy twins -> pooled vectors -> cosine/distance
-> 37-row audit -> seed summary. `noise_vector_drift.py` (repo root) is a thin command-line wrapper.

Everything here reuses the Assignment 2 records, noisy files and edit logs. Nothing is regenerated.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT
from ..phase2_config import NOISY_DIR, DERIVED_P2_DIR, load_phase2_config
from .embeddings import (DERIVED_P3_DIR, RESULTS_P3_DIR, load_embedding_table, load_phase3_config,
                         table_sha256)
from .metrics import pair_metrics
from .pooling import load_tokenizer, pool_span
from .spans import noisy_span, reassemble_record, select_spans

OP_SHORT = {"insertion": "ins", "deletion": "del", "substitution": "sub"}
OP_ORDER = ["insertion", "deletion", "substitution"]
METRICS = ["mean_cosine", "mean_distance", "mean_cosine_changed", "mean_distance_changed"]


def condition_ids(p2) -> list[str]:
    return ["clean"] + [f"{OP_SHORT[op]}-p{p:.2f}-s{s}" for op in p2.operations for p in p2.rates for s in p2.seeds]


def parse_condition(cid: str) -> tuple[str, float, int | None]:
    if cid == "clean":
        return "clean", 0.0, None
    op, p, s = cid.split("-")
    return {v: k for k, v in OP_SHORT.items()}[op], float(p[1:]), int(s[1:])


def load_validation_records() -> list[dict]:
    return [json.loads(l) for l in open(DERIVED_P2_DIR / "val_sample.jsonl", encoding="utf-8")]


# ---------------------------------------------------------------------------------------------
def build_pairs(spans, records, p2, tok, E) -> pd.DataFrame:
    """One row per (condition, span): clean/noisy text, piece counts, cosine, distance, validity."""
    by_id = {r["record_id"]: r for r in records}
    used = sorted({s.record_id for s in spans})
    rows, mismatches = [], 0
    for cid in condition_ids(p2):
        manifest = {}
        for line in open(NOISY_DIR / f"{cid}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["record_id"] in used:
                manifest[r["record_id"]] = r
        for rid in used:   # whole-record consistency check against the Assignment 2 manifest
            if reassemble_record(by_id[rid]["text"], manifest[rid]["operation_log"]) != manifest[rid]["noisy_text"]:
                mismatches += 1
        for s in spans:
            log = manifest[s.record_id]["operation_log"]
            rows.append((cid, s.span_id, s.record_id, s.span_index, s.char_start, s.char_end, s.text,
                         noisy_span(by_id[s.record_id]["text"], log, s.char_start, s.char_end)))
    if mismatches:
        raise RuntimeError(f"{mismatches} record reassembly mismatches against the saved noisy text")
    df = pd.DataFrame(rows, columns=["condition_id", "span_id", "record_id", "span_index", "char_start",
                                     "char_end", "clean_span", "noisy_span"])
    pooled = {t: pool_span(t, tok, E) for t in sorted(set(df.clean_span) | set(df.noisy_span))}
    df["n_pieces_clean"] = df.clean_span.map(lambda t: pooled[t].n_pieces)
    df["n_pieces_noisy"] = df.noisy_span.map(lambda t: pooled[t].n_pieces)
    res = [pair_metrics(pooled[a].vector, pooled[b].vector) for a, b in zip(df.clean_span, df.noisy_span)]
    df["cosine"], df["dist"], df["valid"] = [r[0] for r in res], [r[1] for r in res], [r[2] for r in res]
    return df


def audit_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """G6: one row per condition (37). Means are over VALID pairs; coverage is reported beside them."""
    df = pairs.assign(changed=pairs.clean_span != pairs.noisy_span)
    rows = []
    for cid, g in df.groupby("condition_id", sort=False):
        v = g[g.valid]
        ch = v[v.changed]
        op, p, seed = parse_condition(cid)
        rows.append({
            "condition_id": cid, "operation": op, "rate": p, "seed": seed,
            "selected": len(g), "valid": len(v), "undefined": len(g) - len(v),
            "coverage": len(v) / len(g), "changed_spans": int(g.changed.sum()),
            "mean_cosine": v.cosine.mean(), "mean_distance": v.dist.mean(),
            "mean_cosine_changed": ch.cosine.mean() if len(ch) else np.nan,
            "mean_distance_changed": ch.dist.mean() if len(ch) else np.nan,
        })
    return pd.DataFrame(rows)


def seed_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """G7: per operation x rate, mean / min / max of each condition-level metric over the 3 corruption seeds.

    The min-max range shows how results vary across random corruptions of the SAME text. It is not
    a confidence interval (no repeated training, no resampling of records).
    """
    noisy = audit[audit.operation != "clean"]
    out = []
    for (op, p), g in noisy.groupby(["operation", "rate"], sort=False):
        row = {"operation": op, "rate": p, "n_seeds": len(g), "valid_min": int(g.valid.min()),
               "undefined_min": int(g.undefined.min()), "undefined_max": int(g.undefined.max())}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_min"], row[f"{m}_max"] = g[m].mean(), g[m].min(), g[m].max()
        out.append(row)
    s = pd.DataFrame(out)
    s["operation"] = pd.Categorical(s.operation, OP_ORDER, ordered=True)
    return s.sort_values(["operation", "rate"]).reset_index(drop=True).astype({"operation": str})


def plot_summary(summary: pd.DataFrame, path: Path | None = None, show: bool = False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    rates = sorted(summary.rate.unique()); xs = {r: i for i, r in enumerate(rates)}
    for ax, m, lab in zip(axes, ("mean_cosine", "mean_distance"), ("mean cosine (higher = less change)", "mean Euclidean distance")):
        for op in OP_ORDER:
            g = summary[summary.operation == op]
            x = [xs[r] for r in g.rate]
            ax.plot(x, g[f"{m}_mean"], marker="o", label=op)
            ax.fill_between(x, g[f"{m}_min"], g[f"{m}_max"], alpha=0.25)
        ax.set_xticks(range(len(rates)), [f"{r:g}" for r in rates]); ax.set_xlabel("noise rate p"); ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Pooled static-vector drift. Line = mean, band = min-max over 3 corruption seeds (not a CI). "
                 "Preliminary validation results.", fontsize=8)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def pairs_for_parquet(pairs: pd.DataFrame) -> pd.DataFrame:
    meta = pairs.condition_id.map(parse_condition)
    out = pairs.copy()
    out.insert(1, "operation", [m[0] for m in meta]); out.insert(2, "rate", [m[1] for m in meta])
    out.insert(3, "seed", pd.array([m[2] for m in meta], dtype="Int64"))
    return out.rename(columns={"dist": "distance"})


def write_outputs(pairs, audit, summary, out_dir: Path, run_info: dict) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_dir / "static_span_audit.csv", index=False)
    summary.to_csv(out_dir / "noise_summary_geometry.csv", index=False)
    pairs_for_parquet(pairs).to_parquet(out_dir / "span_pairs.parquet", index=False)
    plot_summary(summary, out_dir / "figures" / "geometry_drift.png")
    (out_dir / "geometry_run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")


def run(out_dir: Path | str = RESULTS_P3_DIR, verbose: bool = True) -> dict:
    """The whole geometry workstream, start to finish."""
    import transformers
    p2, p3 = load_phase2_config(), load_phase3_config()
    records = load_validation_records()
    E = load_embedding_table(p3)
    h_before = table_sha256(E)
    tok = load_tokenizer(p3["embedding"]["checkpoint"], p3["embedding"]["revision"])
    spans = select_spans(records, p3["geometry"]["span_cap"], p2.protected_spans)
    pairs = build_pairs(spans, records, p2, tok, E)
    audit, summary = audit_table(pairs), seed_summary(audit_table(pairs))

    cl = pairs[pairs.condition_id == "clean"]
    tol = p3["geometry"]["identity_tolerance"]
    if not (cl.valid.all() and np.abs(cl.cosine - 1).max() < tol and cl.dist.max() < tol):
        raise RuntimeError("p=0 identity check failed")
    if table_sha256(E) != h_before:
        raise RuntimeError("embedding table changed during the run")
    if len(audit) != 37:
        raise RuntimeError(f"expected 37 condition rows, got {len(audit)}")

    info = {
        "label": p3["label"], "checkpoint": p3["embedding"]["checkpoint"], "revision": p3["embedding"]["revision"],
        "E_shape": list(E.shape), "E_sha256": h_before, "E_unchanged_after_run": True,
        "spans_selected": len(spans), "records_covered": len({s.record_id for s in spans}),
        "conditions": len(audit), "pairs": len(pairs), "valid_pairs": int(pairs.valid.sum()),
        "undefined_pairs": int((~pairs.valid).sum()), "identity_check_passed": True,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "transformers": transformers.__version__, "device": "cpu (numpy only, no GPU)",
    }
    write_outputs(pairs, audit, summary, Path(out_dir), info)
    if verbose:
        print(json.dumps(info, indent=2))
    return {"pairs": pairs, "audit": audit, "summary": summary, "info": info}
