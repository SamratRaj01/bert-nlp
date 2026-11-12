"""N1: turn records into (two-token context -> next token) training/evaluation examples.

For one record with pieces w1..wn the framed sequence is  BOS BOS w1 ... wn  and the targets are
w1 ... wn EOS, so every observed token and one end-of-record token is scored:

    (BOS, BOS) -> w1      (BOS, w1) -> w2      ...      (w_{n-1}, w_n) -> EOS

* An empty record gives the single example (BOS, BOS) -> EOS.  A one-token record gives
  (BOS, w1) -> EOS after (BOS, BOS) -> w1.
* Context resets at every record boundary; nothing is carried across records, no future tokens.
* Context entries are BERT token ids (rows of the frozen table E); BOS is the id -1, which the model
  turns into an all-zero vector. It is never a row of E.
* Targets are indices into the Assignment 2 output set V (K entries, incl. [UNK] and the EOS symbol).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOS = -1


@dataclass
class Sequences:
    ctx: np.ndarray            # (T, 2) int64 BERT ids, BOS = -1
    tgt: np.ndarray            # (T,)   int64 output indices
    n_records: int
    n_tokens: int              # observed pieces (EOS not counted)
    n_unk_targets: int         # observed pieces whose target is the [UNK] output
    n_empty_records: int
    n_continuation: int        # pieces starting with '##'

    @property
    def T(self) -> int:
        return int(len(self.tgt))

    def diagnostics(self) -> dict:
        return {"N_records": self.n_records, "T": self.T, "n_tokens": self.n_tokens,
                "pieces_per_record": self.n_tokens / max(self.n_records, 1),
                "unk_targets": self.n_unk_targets, "unk_share": self.n_unk_targets / max(self.T, 1),
                "continuation_share": self.n_continuation / max(self.n_tokens, 1),
                "empty_records": self.n_empty_records}


def encode_pieces(token_lists: list[list[str]], vocab, bert_ids: dict[str, int]) -> Sequences:
    ctx, tgt = [], []
    n_tok = n_unk = n_empty = n_cont = 0
    for pieces in token_lists:
        ids = [bert_ids[p] for p in pieces]
        out = [vocab.map_token(p) for p in pieces]
        n_tok += len(pieces)
        n_unk += sum(1 for p in pieces if vocab.is_unknown(p))
        n_cont += sum(1 for p in pieces if p.startswith("##"))
        n_empty += not pieces
        f = [BOS, BOS] + ids
        out.append(vocab.end_id)                       # one end-of-record target
        for k, t in enumerate(out):
            ctx.append((f[k], f[k + 1]))
            tgt.append(t)
    return Sequences(np.asarray(ctx, dtype=np.int64).reshape(-1, 2), np.asarray(tgt, dtype=np.int64),
                     len(token_lists), n_tok, n_unk, n_empty, n_cont)


def encode_texts(texts: list[str], pipeline, vocab, bert_ids: dict[str, int]) -> Sequences:
    """Tokenize whole records (no [CLS]/[SEP], no truncation) and build their examples."""
    return encode_pieces(pipeline.tokenize_batch(list(texts)), vocab, bert_ids)
