"""Character-level edit operations.

Owner: Student A. Steps A5-A6.

All operations work on Python str (Unicode code points). Each hit produces one structured
edit record that references the position in the CLEAN string:

    edit_record = {
        "op": "insertion" | "deletion" | "substitution",
        "index": int,                    # position in the clean string
        "source_char": str | None,       # char removed/replaced (None for insertion)
        "replacement_char": str | None,  # char added/substituted (None for deletion)
    }

`apply` decides *which* clean positions are edited (Bernoulli(intensity) per position) and
builds the operation log. The noisy string is then produced by `render` -- the same routine
`manifest.replay` uses -- so the log and the output can never disagree.
"""
from __future__ import annotations

import random
from typing import Literal

Operation = Literal["insertion", "deletion", "substitution"]
OPERATIONS: tuple[Operation, ...] = ("insertion", "deletion", "substitution")


def _pick(alphabet: str, rng: random.Random, exclude: str | None = None) -> str:
    choices = [c for c in alphabet if c != exclude] if exclude else list(alphabet)
    if not choices:  # exclude removed the whole alphabet; fall back to full alphabet
        choices = list(alphabet)
    return rng.choice(choices)


def render(clean_text: str, operation_log: list[dict]) -> str:
    """Rebuild the noisy string from the clean string + the edit log.

    At most one edit per clean index (guaranteed by `apply`). Insertion places the new
    character immediately before the clean character at that index.
    """
    by_index = {e["index"]: e for e in operation_log}
    out: list[str] = []
    for i, ch in enumerate(clean_text):
        edit = by_index.get(i)
        if edit is None:
            out.append(ch)
        elif edit["op"] == "insertion":
            out.append(edit["replacement_char"])
            out.append(ch)
        elif edit["op"] == "deletion":
            pass  # drop the character
        elif edit["op"] == "substitution":
            out.append(edit["replacement_char"])
        else:  # pragma: no cover - guarded by Operation type
            raise ValueError(f"unknown op: {edit['op']}")
    # An insertion whose index == len(clean_text) would append at the very end.
    tail = by_index.get(len(clean_text))
    if tail is not None and tail["op"] == "insertion":
        out.append(tail["replacement_char"])
    return "".join(out)


def apply(
    text: str,
    operation: Operation,
    intensity: float,
    alphabet: str,
    rng: random.Random,
) -> tuple[str, list[dict]]:
    """Apply `operation` to each clean position independently with probability `intensity`.

    intensity <= 0 -> returns (text, []) with no RNG consumed.
    Returns (noisy_text, operation_log) with the log ordered by clean-string index.
    """
    if intensity <= 0 or not text:
        return text, []

    log: list[dict] = []
    for i, ch in enumerate(text):
        if rng.random() >= intensity:
            continue
        if operation == "insertion":
            log.append(
                {"op": "insertion", "index": i, "source_char": None,
                 "replacement_char": _pick(alphabet, rng)}
            )
        elif operation == "deletion":
            log.append(
                {"op": "deletion", "index": i, "source_char": ch, "replacement_char": None}
            )
        elif operation == "substitution":
            log.append(
                {"op": "substitution", "index": i, "source_char": ch,
                 "replacement_char": _pick(alphabet, rng, exclude=ch)}
            )
        else:  # pragma: no cover
            raise ValueError(f"unknown operation: {operation}")

    return render(text, log), log


def edit_locations(operation_log: list[dict]) -> list[int]:
    return [e["index"] for e in operation_log]
