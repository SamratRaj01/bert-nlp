"""Select clean spans and recover each span's noisy counterpart from the saved edit log.

Spans are whitespace-delimited chunks of the CLEAN record text. The Assignment 2 rule edits only
ASCII letters and never touches whitespace, so span boundaries cannot move under noise:
the noisy span is the clean span's characters re-rendered with just the edit-log entries whose
(clean-string) index falls inside [start, end). A fully deleted span becomes "" and stays paired.
Nothing here pairs the n-th clean subword with the n-th noisy subword.
"""
from __future__ import annotations

import re
from typing import Iterable

from ..noise.rule_v2 import eligible_positions, render_v2

_CHUNK = re.compile(r"\S+")


def split_chunks(text: str) -> list[dict]:
    """Every whitespace-delimited chunk with its position among all chunks and its offsets."""
    return [{"span_index": i, "start": m.start(), "end": m.end(), "clean": m.group()}
            for i, m in enumerate(_CHUNK.finditer(text))]


def select_spans(records: Iterable[dict], limit: int = 1000,
                 protected_spans: list[str] | tuple[str, ...] = ()) -> list[dict]:
    """First `limit` nonempty chunks containing an eligible letter, in saved record order.

    Selection looks only at the clean text and the eligibility rule, never at any outcome.
    `span_index` counts all whitespace chunks in the record (skipped ones included), so it
    always identifies the same chunk of the clean text.
    """
    out: list[dict] = []
    for rec in records:
        text = rec["text"]
        eligible = set(eligible_positions(text, protected_spans))
        for chunk in split_chunks(text):
            if not any(i in eligible for i in range(chunk["start"], chunk["end"])):
                continue
            out.append({"span_id": f"{rec['record_id']}:{chunk['span_index']}",
                        "record_id": rec["record_id"], "article_id": rec.get("article_id"),
                        **chunk})
            if len(out) >= limit:
                return out
    return out


def noisy_span(span: dict, operation_log: list[dict]) -> str:
    """The span's noisy counterpart ("" if every character was deleted)."""
    start, end = span["start"], span["end"]
    shifted = [{**e, "index": e["index"] - start}
               for e in operation_log if start <= e["index"] < end]
    return render_v2(span["clean"], shifted)


def reassemble(clean_text: str, operation_log: list[dict]) -> str:
    """Rebuild the noisy record by swapping every whitespace chunk for its noisy span.

    Used as an alignment check: the result must equal the saved `noisy_text`.
    """
    out: list[str] = []
    pos = 0
    for chunk in split_chunks(clean_text):
        out.append(clean_text[pos:chunk["start"]])
        out.append(noisy_span(chunk, operation_log))
        pos = chunk["end"]
    out.append(clean_text[pos:])
    return "".join(out)
