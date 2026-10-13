"""Aggregate per-record rows into the required Phase 2 tables (plan §8, assignment §9, §22).

Owner: both.

    scores           one row per individual perplexity score: 222 for the assignment's three
                     configurations + 74 for the supplementary trigram q3 (296 per training set)
    summary          8 clean + 96 noisy rows (required subset: 6 + 72), mean / min / max over seeds
    within_pipeline  noisy loss - clean loss (either sign), with target-count change
    context_benefit  H(q1) - H(q2), H(q1) - H(q3) and H(q1) - H(P_mix) on the same tokenized condition
    coverage         pooled <UNK>/[UNK] rate, pieces per original span, continuation rate
    severity         requested p vs. the distribution of normalized MED (paired by record ID)

Corpus loss is always total NLL / total targets; seed summaries are summaries of existing
scores, not new runs.
"""
from __future__ import annotations

import math

import pandas as pd

SCORE_KEYS = ["train_set", "tokenizer", "model", "condition_id", "operation", "p", "seed"]
CELL_KEYS = ["train_set", "tokenizer", "model", "operation", "p"]


def scores(per_record: pd.DataFrame) -> pd.DataFrame:
    g = (per_record.groupby(SCORE_KEYS, sort=False)
         .agg(N_records=("record_id", "count"), nll_bits=("nll_bits_sum", "sum"),
              nll_nats=("nll_nats_sum", "sum"), T=("T", "sum"))
         .reset_index())
    g["H_bits"] = g["nll_bits"] / g["T"]
    g["H_nats"] = g["nll_nats"] / g["T"]
    g["PP"] = [math.pow(2.0, h) for h in g["H_bits"]]
    return g


def _clean(sc: pd.DataFrame) -> pd.DataFrame:
    c = sc[sc["condition_id"] == "clean"][["train_set", "tokenizer", "model", "H_bits", "PP", "T"]]
    return c.rename(columns={"H_bits": "clean_H_bits", "PP": "clean_PP", "T": "clean_T"})


def within_pipeline(sc: pd.DataFrame) -> pd.DataFrame:
    noisy = sc[sc["condition_id"] != "clean"].merge(_clean(sc), on=["train_set", "tokenizer", "model"])
    noisy["delta_H_bits"] = noisy["H_bits"] - noisy["clean_H_bits"]
    noisy["delta_PP"] = noisy["PP"] - noisy["clean_PP"]
    noisy["delta_T"] = noisy["T"] - noisy["clean_T"]
    return noisy[SCORE_KEYS + ["H_bits", "clean_H_bits", "delta_H_bits", "PP", "clean_PP",
                               "delta_PP", "T", "clean_T", "delta_T"]]


def summary(sc: pd.DataFrame) -> pd.DataFrame:
    clean = sc[sc["condition_id"] == "clean"].copy()
    clean_rows = pd.DataFrame({
        "train_set": clean["train_set"], "tokenizer": clean["tokenizer"], "model": clean["model"],
        "operation": "none", "p": 0.0, "n_seeds": 1,
        "H_bits_mean": clean["H_bits"], "H_bits_min": clean["H_bits"], "H_bits_max": clean["H_bits"],
        "PP_mean": clean["PP"], "PP_min": clean["PP"], "PP_max": clean["PP"],
        "delta_H_bits_mean": 0.0, "delta_H_bits_min": 0.0, "delta_H_bits_max": 0.0,
        "T_mean": clean["T"].astype(float),
    })
    wp = within_pipeline(sc)
    noisy_rows = (wp.groupby(CELL_KEYS, sort=False)
                  .agg(n_seeds=("seed", "nunique"),
                       H_bits_mean=("H_bits", "mean"), H_bits_min=("H_bits", "min"),
                       H_bits_max=("H_bits", "max"),
                       PP_mean=("PP", "mean"), PP_min=("PP", "min"), PP_max=("PP", "max"),
                       delta_H_bits_mean=("delta_H_bits", "mean"),
                       delta_H_bits_min=("delta_H_bits", "min"),
                       delta_H_bits_max=("delta_H_bits", "max"),
                       T_mean=("T", "mean"))
                  .reset_index())
    return pd.concat([clean_rows, noisy_rows], ignore_index=True)


def context_benefit(sc: pd.DataFrame) -> pd.DataFrame:
    keys = ["train_set", "tokenizer", "condition_id", "operation", "p", "seed"]
    piv = sc.pivot_table(index=keys, columns="model", values="H_bits", sort=False).reset_index()
    piv["benefit_q2_bits"] = piv["q1"] - piv["q2"]
    piv["benefit_q3_bits"] = piv["q1"] - piv["q3"]
    piv["benefit_p_mix_bits"] = piv["q1"] - piv["p_mix"]
    piv.columns.name = None
    return piv.rename(columns={"q1": "H_q1", "q2": "H_q2", "q3": "H_q3", "p_mix": "H_p_mix"})


def coverage(per_record: pd.DataFrame) -> pd.DataFrame:
    one = per_record[per_record["model"] == "q1"]     # diagnostics do not depend on the model
    keys = ["train_set", "tokenizer", "condition_id", "operation", "p", "seed"]
    rows = []
    for k, g in one.groupby(keys, sort=False):
        tok = g["n_tokens"].sum()
        cont = g["n_continuation"].dropna()
        rows.append({**dict(zip(keys, k)), "records": len(g), "tokens": int(tok),
                     "unk_rate": g["n_unk"].sum() / tok if tok else None,
                     "pieces_per_span": tok / g["n_spans"].sum(),
                     "continuation_rate": (cont.sum() / tok) if len(cont) and tok else None,
                     "mean_token_count_change": g["token_count_change"].mean(),
                     "zero_token_records": int(g["rate_undefined"].sum())})
    return pd.DataFrame(rows)


def write_tables(per_record: pd.DataFrame, out_dir) -> dict[str, pd.DataFrame]:
    """Write every per-training-set table; return them for display."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_record.to_parquet(out_dir / "per_record.parquet", index=False)
    sc = scores(per_record)
    tables = {"scores": sc, "summary": summary(sc), "within_pipeline": within_pipeline(sc),
              "context_benefit": context_benefit(sc), "coverage": coverage(per_record)}
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return tables


def severity(manifest: pd.DataFrame) -> pd.DataFrame:
    noisy = manifest[manifest["condition_id"] != "clean"]
    rows = []
    for (op, p, seed), g in noisy.groupby(["operation", "p", "seed"], sort=False):
        n = g["normalized_med"]
        rows.append({"operation": op, "p": p, "seed": seed, "records": len(g),
                     "realized_edit_rate": g["realized_edits"].sum() / g["eligible_count"].sum(),
                     "med_total": int(g["med"].sum()), "edits_total": int(g["realized_edits"].sum()),
                     "nMED_mean": n.mean(), "nMED_median": n.median(),
                     "nMED_p10": n.quantile(0.10), "nMED_p90": n.quantile(0.90),
                     "nMED_min": n.min(), "nMED_max": n.max(),
                     "records_med_below_edits": int((g["med"] < g["realized_edits"]).sum()),
                     "empty_after_noise": int(g["empty_after_noise"].sum())})
    return pd.DataFrame(rows)
