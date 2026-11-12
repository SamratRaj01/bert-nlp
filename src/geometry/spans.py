"""Select clean whitespace-delimited spans and recover their noisy counterparts.

Why not pair "the n-th clean subword with the n-th noisy subword"? Noise can split, merge or
delete pieces, so positions drift. Instead we pair at the *span* level: the a2_default rule only
edits ASCII letters, never whitespace, so every whitespace-delimited span in the clean record has
exactly one counterpart in the noisy record, found by replaying the saved edit log on that span's
characters. A span whose letters are all deleted becomes the explicit empty string "".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..noise.rule_v2 import eligible_positions, render_v2

_SPAN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Span:
    span_id: str          # f"{record_id}|s{span_index}"
    record_id: str
    article_id: str
    span_index: int       # index among ALL whitespace-delimited spans of the record
    char_start: int       # offsets into the clean record text, [start, end)
    char_end: int
    text: str             # clean span text


def whitespace_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _SPAN_RE.finditer(text)]


def select_spans(records: list[dict], cap: int, protected_spans: list[str] = ()) -> list[Span]:
    """First `cap` nonempty whitespace spans containing an eligible letter, in saved record order."""
    out: list[Span] = []
    for rec in records:
        text = rec["text"]
        elig = set(eligible_positions(text, protected_spans))
        for k, (s, e) in enumerate(whitespace_spans(text)):
            if any(i in elig for i in range(s, e)):
                out.append(Span(f"{rec['record_id']}|s{k}", rec["record_id"], rec["article_id"],
                                k, s, e, text[s:e]))
                if len(out) == cap:
                    return out
    return out


def noisy_span(clean_record: str, log: list[dict], start: int, end: int) -> str:
    """Replay the edits that fall inside [start, end) on that slice of the clean record."""
    local = [{**ed, "index": ed["index"] - start} for ed in log if start <= ed["index"] < end]
    return render_v2(clean_record[start:end], local)


def reassemble_record(clean_record: str, log: list[dict]) -> str:
    """Rebuild the full noisy record span by span, keeping the clean gaps (whitespace) as-is.

    Used as a consistency check: must equal the manifest's `noisy_text` exactly.
    """
    out, pos = [], 0
    for s, e in whitespace_spans(clean_record):
        out.append(clean_record[pos:s])
        out.append(noisy_span(clean_record, log, s, e))
        pos = e
    out.append(clean_record[pos:])
    return "".join(out)
