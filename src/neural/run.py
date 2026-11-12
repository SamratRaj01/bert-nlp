"""N0-N6 as one reproducible pipeline; `neural_next_token.py` (repo root) is a thin wrapper.

Reuses the Assignment 2 validation/noisy files, output vocabulary and WordPiece pipeline, and the frozen
embedding table from the geometry workstream. Trains on CLEAN training records only; the three corruption
seeds do not retrain anything.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import ROOT
from ..geometry.drift import OP_ORDER, condition_ids, parse_condition
from ..geometry.embeddings import (DERIVED_P3_DIR, RESULTS_P3_DIR, load_embedding_table, load_phase3_config,
                                   table_sha256)
from ..lm.tokenize import make_pipelines
from ..lm.vocab import wordpiece_vocab
from ..noise.grid import load_condition
from ..phase2_config import DERIVED_P2_DIR, load_phase2_config
from .data import encode_texts
from .evaluate import evaluate, to_tensors
from .figures import plot_noise, plot_training
from .train import train_model

MODELS = {"mean": "mean pooling", "concat": "concatenation"}


def set_determinism(cfg: dict) -> None:
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.set_num_threads(int(cfg["torch_threads"]))
    torch.use_deterministic_algorithms(bool(cfg["deterministic_algorithms"]))


def build_output_space(p2):
    pipe = make_pipelines(p2)["wordpiece"]
    hf_ids = pipe.hf.get_vocab()
    vocab, vstats = wordpiece_vocab(hf_ids, p2.lm["wordpiece_excluded"], end=p2.lm["end_symbol"],
                                    start=p2.lm["start_symbol"])
    return pipe, vocab, vstats, hf_ids


def load_records():
    train = [json.loads(l) for l in open(DERIVED_P2_DIR / "train_a1_slice.jsonl", encoding="utf-8")]
    conds = {c: load_condition(c) for c in condition_ids(load_phase2_config())}
    return train, conds


def load_control(cfg: dict) -> pd.DataFrame:
    c = cfg["control"]
    df = pd.read_csv(ROOT / c["source"])
    df = df[(df.train_set == cfg["train_set"]) & (df.tokenizer == c["tokenizer"]) & (df.model == c["model"])]
    out = df.rename(columns={"H_nats": "H", "N_records": "N_records"})[
        ["condition_id", "operation", "p", "seed", "N_records", "T", "H", "PP"]].copy()
    out["operation"] = out.condition_id.map(lambda c: parse_condition(c)[0])
    out["model"] = "ngram (A2 interpolated WordPiece)"
    return out.reset_index(drop=True)


def summarise(rows: pd.DataFrame, group: list[str], metrics: list[str]) -> pd.DataFrame:
    noisy = rows[rows.operation != "clean"]
    g = noisy.groupby(group, sort=False)
    out = g.size().rename("n_seeds").reset_index()
    for m in metrics:
        agg = g[m].agg(["mean", "min", "max"]).reset_index()
        out[f"{m}_mean"], out[f"{m}_min"], out[f"{m}_max"] = agg["mean"], agg["min"], agg["max"]
    out["operation"] = pd.Categorical(out.operation, OP_ORDER, ordered=True)
    return out.sort_values(group).reset_index(drop=True).astype({"operation": str})


def run(out_dir: Path | str = RESULTS_P3_DIR, verbose: bool = True, train_models: bool = True) -> dict:
    import transformers
    p2, p3 = load_phase2_config(), load_phase3_config()
    cfg = p3["neural"]
    set_determinism(cfg)
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    # ---- N1: data ---------------------------------------------------------------------------------
    pipe, vocab, vstats, hf_ids = build_output_space(p2)
    K = vocab.size
    train_rec, cond_rows = load_records()
    E_np = load_embedding_table(p3)
    E_hash_before = table_sha256(E_np)
    E = torch.from_numpy(np.array(E_np))        # writable copy; the original stays read-only for hashing
    train_seq = encode_texts([r["text"] for r in train_rec], pipe, vocab, hf_ids)
    seqs = {c: encode_texts([r["noisy_text"] for r in rows], pipe, vocab, hf_ids) for c, rows in cond_rows.items()}
    control = load_control(cfg)
    T_phase2 = control.set_index("condition_id")["T"]
    parity = {c: (seqs[c].T == int(T_phase2[c])) for c in seqs}
    if not all(parity.values()) or train_seq.T != 35642:
        raise RuntimeError(f"target-count parity with Assignment 2 failed: train T={train_seq.T}, "
                           f"bad conditions={[c for c, ok in parity.items() if not ok]}")
    log(f"K={K} outputs | train targets {train_seq.T} | clean validation targets {seqs['clean'].T} | "
        f"parity with Assignment 2: all 37 conditions OK")

    # ---- N3: train both models on clean text ------------------------------------------------------
    train_t, val_t = to_tensors(train_seq), to_tensors(seqs["clean"])
    trained = {}
    for mode in MODELS:
        trained[mode] = train_model(mode, E, K, train_t, val_t, cfg)
        for h in trained[mode]["history"]:
            log(f"  {mode:6} epoch {h['epoch']}  train {h['train_loss']:.4f}  val {h['val_loss']:.4f}  "
                f"acc {h['val_accuracy']:.4f}{'  <- selected' if h['selected'] else ''}")
    E_hash_after = table_sha256(E_np)
    if E_hash_after != E_hash_before or table_sha256(E.numpy()) != E_hash_before:
        raise RuntimeError("the frozen embedding table changed during training")

    # ---- N4/N5: score every condition with both models ---------------------------------------------
    rows = []
    for cid, seq in seqs.items():
        op, p, seed = parse_condition(cid)
        t = to_tensors(seq)
        for mode, tr in trained.items():
            r = evaluate(tr["model"], *t, chunk=cfg["eval_chunk"])
            rows.append({"model": mode, "condition_id": cid, "operation": op, "p": p,
                         "seed": -1 if seed is None else seed, "H": r["H"], "PP": r["PP"],
                         "accuracy": r["accuracy"], "S": r["S"], **seq.diagnostics()})
    neural = pd.DataFrame(rows)
    assert len(neural) == 74
    metrics = ["H", "PP", "accuracy"]
    clean = neural[neural.condition_id == "clean"].set_index("model")
    neural["dH"] = neural.H - neural.model.map(clean.H)
    summary = summarise(neural, ["model", "operation", "p"], metrics + ["dH"])
    assert len(summary) == 24

    # ---- N6: n-gram control and side-by-side change from each model's own clean row -----------------
    c_clean = control[control.condition_id == "clean"].iloc[0]
    control["dH"] = control.H - c_clean.H
    ctrl_summary = summarise(control, ["model", "operation", "p"], ["H", "PP", "dH"])
    side = pd.concat([summary[["model", "operation", "p", "dH_mean", "dH_min", "dH_max"]],
                      ctrl_summary[["model", "operation", "p", "dH_mean", "dH_min", "dH_max"]]], ignore_index=True)
    assert len(control) == 37

    # ---- bookkeeping --------------------------------------------------------------------------------
    retained = {
        "label": p3["label"], "train_set": cfg["train_set"],
        "train_record_ids": [r["record_id"] for r in train_rec],
        "validation_record_ids": [r["record_id"] for r in cond_rows["clean"]],
        "train_records": len(train_rec), "validation_records": len(cond_rows["clean"]),
        "cap_applied": False, "reduction_reason": None,
        "train_targets": train_seq.T, "clean_validation_targets": seqs["clean"].T,
        "note": "same retained records for both neural models and the n-gram control; noisy files are the saved "
                "Assignment 2 files (not regenerated, not truncated)"}
    model_info = {m: {"best_epoch": t["best_epoch"], "untrained_val_loss": t["untrained_val_loss"],
                      **t["model"].counts()} for m, t in trained.items()}
    info = {
        "label": p3["label"], "K_outputs": K, "vocab_stats": vstats, "settings": cfg, "models": model_info,
        "E_sha256_before": E_hash_before, "E_sha256_after": E_hash_after, "E_unchanged": True,
        "tokenizer": {"checkpoint": p2.checkpoint, "revision": p2.checkpoint_revision, "vocab_entries": len(hf_ids)},
        "parity_with_assignment2": True, "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "transformers": transformers.__version__,
        "device": cfg["device"], "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "note": "a fixed seed does not guarantee identical results on different hardware"}
    history = pd.DataFrame([h for t in trained.values() for h in t["history"]])

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    neural.to_csv(out_dir / "neural_condition_rows.csv", index=False)
    summary.to_csv(out_dir / "neural_noise_summary.csv", index=False)
    control.to_csv(out_dir / "ngram_control.csv", index=False)
    side.to_csv(out_dir / "neural_vs_ngram_change.csv", index=False)
    history.to_csv(out_dir / "neural_training_history.csv", index=False)
    (out_dir / "neural_run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    DERIVED_P3_DIR.mkdir(parents=True, exist_ok=True)
    rp = DERIVED_P3_DIR / "neural_retained_records.json"
    rp.write_text(json.dumps(retained), encoding="utf-8")
    (DERIVED_P3_DIR / "neural_retained_records.sha256").write_text(hashlib.sha256(rp.read_bytes()).hexdigest() + "\n")
    mdir = DERIVED_P3_DIR / "neural"; mdir.mkdir(exist_ok=True)
    for m, t in trained.items():
        torch.save({k: v for k, v in t["model"].state_dict().items()}, mdir / f"{m}_best.pt")
    import matplotlib
    matplotlib.use("Agg")
    plot_training(history, out_dir / "figures" / "neural_training.png")
    plot_noise(summary, ctrl_summary, out_dir / "figures" / "neural_noise.png")
    log(f"wrote {out_dir}")
    return {"neural": neural, "summary": summary, "control": control, "control_summary": ctrl_summary,
            "side": side, "history": history, "info": info, "retained": retained, "models": trained}
