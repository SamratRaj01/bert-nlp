"""Tokenization diagnostics.

Owner: Student B. Steps B4-B5.

Per-text metrics and paired clean/noisy deltas. These numbers go into every one of the 20
pair records and into the clean baseline table.
"""
from __future__ import annotations

from statistics import mean

from ..med.levenshtein import med
from .wrapper import Encoding

WHITESPACE_SPLIT = None  # str.split() default


def word_count(text: str) -> int:
    return max(1, len(text.split(WHITESPACE_SPLIT)))


def text_diagnostics(enc: Encoding, text: str) -> dict:
    n = enc.token_count
    words = word_count(text)
    return {
        "n_tokens": n,
        "fertility": n / words,
        "n_continuation": enc.continuation_count,
        "continuation_rate": (enc.continuation_count / n) if n else 0.0,
        "n_unk": enc.unk_count(),
        "unk_rate": (enc.unk_count() / n) if n else 0.0,
        "truncated": enc.truncated,
    }


def sequence_change_rate(clean_pieces: list[str], noisy_pieces: list[str]) -> float:
    """Token-level Levenshtein distance / max(1, len(clean_pieces))."""
    return med(clean_pieces, noisy_pieces) / max(1, len(clean_pieces))


def paired_diagnostics(clean_enc: Encoding, noisy_enc: Encoding,
                       clean_text: str, noisy_text: str) -> dict:
    clean = text_diagnostics(clean_enc, clean_text)
    noisy = text_diagnostics(noisy_enc, noisy_text)
    return {
        "clean": clean,
        "noisy": noisy,
        "token_count_change": noisy["n_tokens"] - clean["n_tokens"],
        "fertility_delta": noisy["fertility"] - clean["fertility"],
        "continuation_delta": noisy["n_continuation"] - clean["n_continuation"],
        "unk_delta": noisy["n_unk"] - clean["n_unk"],
        "sequence_change_rate": sequence_change_rate(clean_enc.pieces, noisy_enc.pieces),
    }


def aggregate(rows: list[dict], key: str) -> dict:
    """Mean / min / max of `row[key]` across pair records (for the summary table)."""
    vals = [r[key] for r in rows]
    return {"mean": mean(vals), "min": min(vals), "max": max(vals)}


def clean_baseline(examples: list[dict], wrapper) -> dict:
    """Clean-branch tokenization baseline over the slice.

    This is the reference the noisy branch is compared against (Student B deliverable).
    `wrapper` is a src.tokenizer.wrapper.TokenizerWrapper.
    """
    per = [text_diagnostics(wrapper.encode(e["text"]), e["text"]) for e in examples]
    return {
        "n_examples": len(per),
        "checkpoint": wrapper.cfg.checkpoint,
        "checkpoint_revision": wrapper.cfg.checkpoint_revision,
        "max_length": wrapper.max_length,
        "mean_tokens": aggregate(per, "n_tokens")["mean"],
        "max_tokens": aggregate(per, "n_tokens")["max"],
        "mean_fertility": aggregate(per, "fertility")["mean"],
        "mean_continuation_rate": aggregate(per, "continuation_rate")["mean"],
        "mean_unk_rate": aggregate(per, "unk_rate")["mean"],
        "total_unk": sum(d["n_unk"] for d in per),
        "truncated_examples": sum(d["truncated"] for d in per),
    }
