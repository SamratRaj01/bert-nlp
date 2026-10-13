"""Score the frozen validation grid for one engine (training set x tokenizer).

Owner: both. Phase 2 evaluation (plan §8).

Inputs are the saved condition files from `src/noise/grid.py`: the noisy text is read, never
regenerated, and the identical string is passed to both tokenizers. For every record x
condition x model this produces one per-record row carrying its IDs, NLL sums, T, MED and the
coverage diagnostics, so any corpus number can be rebuilt (and later bootstrapped by article).
"""
from __future__ import annotations

import numpy as np

from ..lm.interp import select_lambdas
from ..lm.models import MODELS
from ..lm.score import Engine, q_for_records, record_losses
from ..phase2_config import Phase2Config, load_phase2_config, write_frozen_lambdas
from .coverage import record_diagnostics


def select_and_freeze(engines: dict[str, Engine], clean_rows: list[dict], cfg: Phase2Config,
                      train_set: str) -> tuple[Phase2Config, list[dict]]:
    """B6: choose λ on CLEAN validation per tokenizer, then write them to configs/phase2.yaml.

    Returns the reloaded config (now carrying the frozen λ) and the full selection table.
    Noisy scoring must read λ from that config, never from this function's working values.
    """
    lm = cfg.lm
    chosen, table = {}, []
    for tok, eng in engines.items():
        tokens = eng.pipeline.tokenize_batch([r["noisy_text"] for r in clean_rows])
        q = np.vstack(q_for_records(eng, tokens))
        lam, rows = select_lambdas(q, float(lm["lambda_step"]), float(lm["lambda_tie_tolerance"]))
        chosen[tok] = [float(x) for x in lam]
        table += [{"train_set": train_set, "tokenizer": tok, **r} for r in rows]
    write_frozen_lambdas(train_set, chosen)
    return load_phase2_config(), table


def frozen_lambdas(cfg: Phase2Config, train_set: str, tokenizer: str) -> tuple[float, float, float]:
    lam = cfg.frozen_lambdas[train_set][tokenizer]
    assert len(lam) == 3 and min(lam) >= 0 and abs(sum(lam) - 1) < 1e-12
    return tuple(lam)


def score_grid(engine: Engine, lambdas: tuple[float, float, float],
               condition_rows: dict[str, list[dict]]) -> list[dict]:
    before = engine.counts.snapshot()
    clean = condition_rows["clean"]
    clean_tokens = engine.pipeline.tokenize_batch([r["noisy_text"] for r in clean])
    clean_n = {r["record_id"]: len(t) for r, t in zip(clean, clean_tokens)}
    clean_text = {r["record_id"]: r["noisy_text"] for r in clean}

    out: list[dict] = []
    for cond_id, rows in condition_rows.items():
        tokens = (clean_tokens if cond_id == "clean"
                  else engine.pipeline.tokenize_batch([r["noisy_text"] for r in rows]))
        qs = q_for_records(engine, tokens)
        for r, toks, q in zip(rows, tokens, qs):
            diag = record_diagnostics(clean_text[r["record_id"]], toks, clean_n[r["record_id"]],
                                      engine.vocab, engine.tokenizer)
            base = {
                "train_set": engine.train_set, "tokenizer": engine.tokenizer,
                "condition_id": cond_id, "operation": r["operation"] or "none",
                "p": r["p"], "seed": r["seed"] if r["seed"] is not None else -1,
                "article_id": r["article_id"], "record_id": r["record_id"],
                "med": r["med"], "normalized_med": r["normalized_med"],
                "empty_after_noise": r["empty_after_noise"],
                "protocol_sha256": r["protocol_sha256"], **diag,
            }
            for model in MODELS:
                bits, nats, T = record_losses(q, model, lambdas if model == "p_mix" else None)
                out.append({**base, "model": model, "nll_bits_sum": bits,
                            "nll_nats_sum": nats, "T": T})
    assert engine.counts.snapshot() == before, "evaluation mutated the training counts"
    return out
