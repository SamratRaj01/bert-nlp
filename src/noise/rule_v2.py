"""Assignment 2 default noise rule (`rule_version = "a2_default"`).

Owner: Samrat. Phase 2 step A3.

Differences from the Phase 1 rule in `operations.py` (which stays untouched and is used only
to regenerate the 20 A1 diagnostic pairs):

    * eligible = ASCII letters a-z / A-Z of the ORIGINAL string, outside protected spans;
      spaces, punctuation, digits and WikiText sentinels (@-@, @,@, @.@) are never edited
    * insertion places one lowercase a-z char immediately AFTER the eligible char
    * substitution picks a lowercase a-z char different from source.lower()
      ('A' -> 'a' is excluded: the uncased tokenizer would erase that edit)
    * one Bernoulli(p) draw per eligible original position; inserted chars are never corrupted

The edit log keeps the Phase 1 record shape, indexed against the CLEAN string:
    {"op", "index", "source_char", "replacement_char"}
(for insertion, `source_char` is None and the new char goes after clean[index]).
"""
from __future__ import annotations

import random

from .operations import Operation, _pick

RULE_VERSION = "a2_default"


def _is_ascii_letter(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def protected_mask(text: str, protected_spans: list[str] | tuple[str, ...] = ()) -> list[bool]:
    mask = [False] * len(text)
    for span in protected_spans:
        if not span:
            continue
        start = text.find(span)
        while start != -1:
            for k in range(start, start + len(span)):
                mask[k] = True
            start = text.find(span, start + 1)
    return mask


def eligible_positions(text: str, protected_spans: list[str] | tuple[str, ...] = ()) -> list[int]:
    mask = protected_mask(text, protected_spans)
    return [i for i, ch in enumerate(text) if _is_ascii_letter(ch) and not mask[i]]


def render_v2(clean_text: str, operation_log: list[dict]) -> str:
    """Rebuild the noisy string from the clean string and an a2_default edit log."""
    by_index = {e["index"]: e for e in operation_log}
    out: list[str] = []
    for i, ch in enumerate(clean_text):
        edit = by_index.get(i)
        if edit is None:
            out.append(ch)
        elif edit["op"] == "insertion":
            out.append(ch)
            out.append(edit["replacement_char"])  # immediately AFTER the eligible char
        elif edit["op"] == "deletion":
            pass
        elif edit["op"] == "substitution":
            out.append(edit["replacement_char"])
        else:  # pragma: no cover
            raise ValueError(f"unknown op: {edit['op']}")
    return "".join(out)


def replay_v2(clean_text: str, operation_log: list[dict]) -> str:
    return render_v2(clean_text, operation_log)


def apply_v2(
    text: str,
    operation: Operation,
    p: float,
    alphabet: str,
    rng: random.Random,
    protected_spans: list[str] | tuple[str, ...] = (),
) -> tuple[str, list[dict], int]:
    """Return (noisy_text, operation_log, eligible_count).

    p <= 0 -> (text, [], eligible_count) with no RNG consumed.
    """
    positions = eligible_positions(text, protected_spans)
    if p <= 0 or not positions:
        return text, [], len(positions)

    log: list[dict] = []
    for i in positions:
        if rng.random() >= p:
            continue
        ch = text[i]
        if operation == "insertion":
            log.append({"op": "insertion", "index": i, "source_char": None,
                        "replacement_char": _pick(alphabet, rng)})
        elif operation == "deletion":
            log.append({"op": "deletion", "index": i, "source_char": ch,
                        "replacement_char": None})
        elif operation == "substitution":
            log.append({"op": "substitution", "index": i, "source_char": ch,
                        "replacement_char": _pick(alphabet, rng, exclude=ch.lower())})
        else:  # pragma: no cover
            raise ValueError(f"unknown operation: {operation}")
    return render_v2(text, log), log, len(positions)
