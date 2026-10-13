"""Decompose corpus loss H by token type: why noise moves the two pipelines in opposite directions.

Owner: both. Supports the interpretation in docs/phase2_summary.md §4.

H is an average over targets, so it splits exactly into per-type contributions:

    H = sum over types of  share(type) x mean_bits(type),   share = n_type / T

Token types (boundary <s> is context only and never a target):
    whitespace : "word" (in V), "unknown" (<UNK>), "end" (</s>)
    wordpiece  : "word" (word-start piece), "continuation" (## piece), "unknown" ([UNK]), "end" (</s>)

Under noise, whitespace turns corrupted words into the cheap, frequent <UNK> (share up, H down);
WordPiece turns them into rare ## continuation pieces (share and T up, H up).
"""
from __future__ import annotations

import numpy as np

from ..lm.models import model_probs
from ..lm.score import Engine

TOKEN_TYPES = ("word", "continuation", "unknown", "end")


def token_type(tok: str | None, engine: Engine) -> str:
    if tok is None:
        return "end"
    if engine.vocab.is_unknown(tok):
        return "unknown"
    if engine.tokenizer == "wordpiece" and tok.startswith("##"):
        return "continuation"
    return "word"


def loss_decomposition(engine: Engine, lambdas, condition_rows: dict[str, list[dict]],
                       model: str = "p_mix") -> list[dict]:
    """One row per (condition, token type): targets, share, mean bits, contribution to H."""
    out: list[dict] = []
    for cond_id, rows in condition_rows.items():
        token_seqs = engine.pipeline.tokenize_batch([r["noisy_text"] for r in rows])
        n = dict.fromkeys(TOKEN_TYPES, 0)
        bits = dict.fromkeys(TOKEN_TYPES, 0.0)
        for seq in token_seqs:
            q = engine.lm.q_arrays(engine.vocab.map_tokens(seq))
            costs = -np.log2(model_probs(q, model, lambdas if model == "p_mix" else None))
            for tok, cost in zip(list(seq) + [None], costs):
                t = token_type(tok, engine)
                n[t] += 1
                bits[t] += float(cost)
        T = sum(n.values())
        r0 = rows[0]
        for t in TOKEN_TYPES:
            if engine.tokenizer == "whitespace" and t == "continuation":
                continue
            out.append({
                "train_set": engine.train_set, "tokenizer": engine.tokenizer, "model": model,
                "condition_id": cond_id, "operation": r0["operation"] or "none", "p": r0["p"],
                "seed": r0["seed"] if r0["seed"] is not None else -1, "token_type": t,
                "targets": n[t], "T": T, "share": n[t] / T,
                "mean_bits": bits[t] / n[t] if n[t] else None,
                "contribution_bits": bits[t] / T,
            })
    return out
