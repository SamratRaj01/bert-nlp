"""Coverage and fragmentation diagnostics (per record x condition x tokenizer).

Owner: Samrat. Phase 2 step A6.

Boundary symbols (<s>, </s>) are excluded from every diagnostic. Definitions:

    n_tokens            content tokens produced by the tokenizer for this (noisy) text
    n_unk               tokens that map to the unknown symbol of V (whitespace <UNK>;
                        WordPiece [UNK] pieces)
    unk_rate            n_unk / n_tokens
    n_spans             nonempty whitespace spans of the ORIGINAL clean record (fixed per record)
    pieces_per_span     n_tokens / n_spans   (deleted spans contribute zero pieces)
    n_continuation      WordPiece pieces starting with '##' (not applicable to whitespace)
    continuation_rate   n_continuation / n_tokens
    token_count_change  n_tokens(noisy) - n_tokens(clean), same tokenizer

A zero-token output keeps zero counts and sets `rate_undefined = True`; rates are then None
rather than dividing by zero or dropping the record. Corpus-level rates pool counts first
(sum of numerators / sum of denominators).
"""
from __future__ import annotations

from ..lm.vocab import Vocab


def record_diagnostics(clean_text: str, tokens: list[str], clean_n_tokens: int,
                       vocab: Vocab, tokenizer: str) -> dict:
    n = len(tokens)
    n_unk = sum(1 for t in tokens if vocab.is_unknown(t))
    n_spans = len(clean_text.split())
    if tokenizer == "wordpiece":
        n_cont = sum(1 for t in tokens if t.startswith("##"))
    else:
        n_cont = None
    undefined = n == 0
    return {
        "n_tokens": n,
        "n_unk": n_unk,
        "unk_rate": None if undefined else n_unk / n,
        "n_spans": n_spans,
        "pieces_per_span": n / n_spans if n_spans else None,
        "n_continuation": n_cont,
        "continuation_rate": (None if undefined or n_cont is None else n_cont / n),
        "token_count_change": n - clean_n_tokens,
        "rate_undefined": undefined,
    }


def pooled(rows: list[dict]) -> dict:
    """Corpus-level coverage for one (tokenizer, condition)."""
    tok = sum(r["n_tokens"] for r in rows)
    spans = sum(r["n_spans"] for r in rows)
    cont = [r["n_continuation"] for r in rows if r["n_continuation"] is not None]
    return {
        "records": len(rows),
        "tokens": tok,
        "unk_rate": sum(r["n_unk"] for r in rows) / tok if tok else None,
        "pieces_per_span": tok / spans if spans else None,
        "continuation_rate": (sum(cont) / tok) if (cont and tok) else None,
        "mean_token_count_change": sum(r["token_count_change"] for r in rows) / len(rows),
        "zero_token_records": sum(r["rate_undefined"] for r in rows),
    }
