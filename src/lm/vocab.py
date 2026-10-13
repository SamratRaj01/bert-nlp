"""Frozen output vocabularies V (one per tokenizer x training set).

Owner: Prazol. Phase 2 step B2.

* whitespace: training tokens with count >= 2, plus <UNK> and the end symbol </s>.
              Every other token maps to <UNK>, at training AND at evaluation time.
* wordpiece:  every BERT vocab entry except [PAD], [CLS], [SEP], [MASK] and [unused...],
              plus </s>. [UNK] and entries never seen in training are kept.

The start symbol <s> is context only: it is never in V and never scored. A literal text token
equal to <s> or </s> is treated as unknown so it cannot be confused with a boundary.
Token IDs are 0..|V|-1 in sorted order; <s> gets id |V| (outside V).
"""
from __future__ import annotations

import fnmatch
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Vocab:
    name: str
    tokens: list[str]                 # sorted; defines the ids
    unk: str | None                   # None only for hand-built test vocabularies
    end: str = "</s>"
    start: str = "<s>"
    index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        assert self.start not in self.tokens, "the start symbol must not be in V"
        assert self.end in self.tokens, "the end symbol must be in V"
        assert len(set(self.tokens)) == len(self.tokens)
        self.tokens = sorted(self.tokens)
        self.index = {t: i for i, t in enumerate(self.tokens)}

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def start_id(self) -> int:
        return len(self.tokens)

    @property
    def end_id(self) -> int:
        return self.index[self.end]

    @property
    def unk_id(self) -> int | None:
        return self.index[self.unk] if self.unk is not None else None

    def map_token(self, tok: str) -> int:
        if tok in (self.start, self.end):          # reserved boundary symbols
            tok = self.unk
        i = self.index.get(tok)
        if i is None:
            if self.unk is None:
                raise KeyError(f"{tok!r} not in vocabulary {self.name} and no unknown symbol")
            return self.index[self.unk]
        return i

    def map_tokens(self, toks: list[str]) -> list[int]:
        return [self.map_token(t) for t in toks]

    def is_unknown(self, tok: str) -> bool:
        """True when `tok` maps to the unknown symbol (incl. a literal [UNK] piece)."""
        return self.unk is not None and self.map_token(tok) == self.unk_id

    def card(self) -> dict:
        return {"name": self.name, "size": self.size, "unk": self.unk, "end": self.end,
                "start (context only, not in V)": self.start}


def whitespace_vocab(train_token_seqs: list[list[str]], min_count: int = 2,
                     unk: str = "<UNK>", end: str = "</s>", start: str = "<s>",
                     name: str = "whitespace") -> tuple[Vocab, dict]:
    counts = Counter(t for seq in train_token_seqs for t in seq)
    kept = {t for t, c in counts.items() if c >= min_count and t not in (unk, end, start)}
    vocab = Vocab(name=name, tokens=sorted(kept | {unk, end}), unk=unk, end=end, start=start)
    stats = {"training_types": len(counts), "training_tokens": sum(counts.values()),
             "types_kept": len(kept), "types_mapped_to_unk": len(counts) - len(kept),
             "training_tokens_mapped_to_unk": sum(c for t, c in counts.items() if t not in kept)}
    return vocab, stats


def wordpiece_vocab(hf_vocab: dict[str, int], excluded: list[str], unk: str = "[UNK]",
                    end: str = "</s>", start: str = "<s>",
                    name: str = "wordpiece") -> tuple[Vocab, dict]:
    def is_excluded(tok: str) -> bool:
        return any(fnmatch.fnmatchcase(tok, pat.replace("[", "[[]")) for pat in excluded)

    kept = [t for t in hf_vocab if not is_excluded(t)]
    assert unk in kept, "[UNK] must be retained"
    vocab = Vocab(name=name, tokens=sorted(set(kept) | {end}), unk=unk, end=end, start=start)
    stats = {"bert_vocab_entries": len(hf_vocab), "excluded": len(hf_vocab) - len(kept),
             "excluded_unused": sum(1 for t in hf_vocab if t.startswith("[unused"))}
    return vocab, stats
