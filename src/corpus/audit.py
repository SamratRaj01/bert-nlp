"""Character-category audit of the corpus slice.

Owner: Student A. Step A4.

Reads:  derived/slice.jsonl
Writes: derived/category_audit.json   -- {category: count, ...} plus totals and shares
"""
from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

from ..config import DERIVED_DIR
from .slice import load_slice

BUCKETS = ("uppercase", "lowercase", "digit", "punctuation", "whitespace", "symbol", "other")


def bucket(ch: str) -> str:
    if ch in ("\n", "\t", "\r") or unicodedata.category(ch) in ("Zs", "Zl", "Zp"):
        return "whitespace"
    cat = unicodedata.category(ch)
    if cat == "Lu":
        return "uppercase"
    if cat in ("Ll", "Lt", "Lm", "Lo"):
        return "lowercase"
    if cat == "Nd":
        return "digit"
    if cat.startswith("P"):
        return "punctuation"
    if cat.startswith("S"):
        return "symbol"
    return "other"


def audit_slice(slice_path: Path | str = DERIVED_DIR / "slice.jsonl") -> dict:
    examples = load_slice(slice_path)
    counts: Counter[str] = Counter()
    non_ascii = 0
    total = 0
    for ex in examples:
        for ch in ex["text"]:
            counts[bucket(ch)] += 1
            total += 1
            if ord(ch) > 127:
                non_ascii += 1

    result = {
        "n_examples": len(examples),
        "total_characters": total,
        "non_ascii_characters": non_ascii,
        "by_category": {b: counts.get(b, 0) for b in BUCKETS},
        "shares": {b: (counts.get(b, 0) / total if total else 0.0) for b in BUCKETS},
    }
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    with open(DERIVED_DIR / "category_audit.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result


def main() -> None:
    result = audit_slice()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
