"""G4: tokenize one span alone, look up each piece's row of E, average the rows.

Each WordPiece piece (e.g. '##ace') is looked up on its OWN. There is no row for the whole word.
The pooled vector is the plain mean of those rows (no weighting, no normalisation). A span with no
pieces (the empty string) has no vector: we return None so callers can mark the pair undefined.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from transformers import AutoTokenizer


@dataclass(frozen=True)
class Pooled:
    text: str
    pieces: list[str]
    ids: list[int]
    vector: np.ndarray | None       # shape (d,), or None for an empty span

    @property
    def n_pieces(self) -> int:
        return len(self.ids)


def load_tokenizer(checkpoint: str, revision: str):
    return AutoTokenizer.from_pretrained(checkpoint, revision=revision)


def pool_span(text: str, tokenizer, E: np.ndarray) -> Pooled:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]     # no [CLS]/[SEP]
    pieces = tokenizer.convert_ids_to_tokens(ids)
    vec = E[ids].mean(axis=0) if ids else None                       # mean of raw rows
    return Pooled(text, pieces, list(ids), vec)
