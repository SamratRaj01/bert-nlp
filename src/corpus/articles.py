"""Article (source-document) IDs and the Phase 2 training sets.

Owner: Samrat. Phase 2 steps A1-A2.

WikiText marks each Wikipedia article with a top-level heading line ' = Title = '; section
headings use more '=' (' = = Sub = = '). Every line is assigned to the most recent top-level
heading, giving the source-document ID needed for grouping and later document-level bootstrap:

    article_id = "wt2-<split>-doc-NNNNNN"   (NNNNNN = line index of the governing heading)

Nothing in raw/ is modified; outputs go to derived/phase2/.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import RAW_DIR
from .slice import is_kept_line, make_example_id

_DOC_PREFIX = {"train": "wt2-train-doc", "validation": "wt2-valid-doc", "test": "wt2-test-doc"}


def is_article_heading(raw_line: str) -> bool:
    """True for ' = Title = ' (one '=' each side), False for ' = = Sub = = ' and body text."""
    s = raw_line.strip()
    return (len(s) >= 5 and s.startswith("= ") and s.endswith(" =")
            and not s.startswith("= =") and not s.endswith("= ="))


def make_article_id(split: str, heading_index: int) -> str:
    return f"{_DOC_PREFIX[split]}-{heading_index:06d}"


def read_raw_lines(split: str) -> list[str]:
    with open(RAW_DIR / "wikitext-2" / f"{split}.raw", "r", encoding="utf-8", newline="") as fh:
        return fh.readlines()


def article_index(lines: list[str], split: str) -> list[str | None]:
    """article_id for every line index (None before the first heading)."""
    out: list[str | None] = []
    current: str | None = None
    for i, line in enumerate(lines):
        if is_article_heading(line):
            current = make_article_id(split, i)
        out.append(current)
    return out


def build_records(split: str, min_chars: int = 0, max_chars: int = 10**9,
                  cap: int | None = None) -> tuple[list[dict], dict]:
    """Records kept by `is_kept_line` in file order, with article IDs, plus an exclusion report.

    The exclusion report covers every line scanned (up to the cap), so it says exactly why
    each skipped line was skipped.
    """
    lines = read_raw_lines(split)
    articles = article_index(lines, split)
    records: list[dict] = []
    excl = {"empty": 0, "heading": 0, "too_short": 0, "too_long": 0}
    scanned = 0
    for i, line in enumerate(lines):
        if cap is not None and len(records) >= cap:
            break
        scanned += 1
        if is_kept_line(line, min_chars, max_chars):
            text = line.strip()
            assert articles[i] is not None, f"{split} line {i} precedes any article heading"
            records.append({
                "record_id": make_example_id(split, i),
                "article_id": articles[i],
                "source_split": split,
                "source_index": i,
                "text": text,
                "char_count": len(text),
            })
            continue
        s = line.strip()
        if not s:
            excl["empty"] += 1
        elif s.startswith("=") and s.endswith("="):
            excl["heading"] += 1
        elif len(s) < min_chars:
            excl["too_short"] += 1
        else:
            excl["too_long"] += 1
    total_eligible = sum(1 for line in lines if is_kept_line(line, min_chars, max_chars))
    report = {
        "split": split,
        "lines_in_split": len(lines),
        "lines_scanned": scanned,
        "kept": len(records),
        "excluded_while_scanning": excl,
        "eligible_in_whole_split": total_eligible,
        "articles": len({r["article_id"] for r in records}),
    }
    return records, report


def attach_articles(slice_examples: list[dict], split: str = "train") -> list[dict]:
    """Back-fill article IDs for the A1 slice without touching derived/slice.jsonl."""
    articles = article_index(read_raw_lines(split), split)
    out = []
    for ex in slice_examples:
        out.append({
            "record_id": ex["example_id"],
            "article_id": articles[ex["source_index"]],
            "source_split": ex["source_split"],
            "source_index": ex["source_index"],
            "text": ex["text"],
            "char_count": len(ex["text"]),
        })
    return out


def write_records(records: list[dict], path: Path | str, rule: str, extra: dict | None = None) -> str:
    """Write records as JSONL plus a `.sha256` sidecar (checksum, rule, counts). Returns the hash."""
    import hashlib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = [f"{digest}  {path.name}", f"selection_rule: {rule}", f"count: {len(records)}",
             f"articles: {len({r['article_id'] for r in records})}",
             f"characters: {sum(r['char_count'] for r in records)}",
             f"whitespace_tokens: {sum(len(r['text'].split()) for r in records)}"]
    for k, v in (extra or {}).items():
        lines.append(f"{k}: {v}")
    path.with_suffix(".sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return digest


def load_records(path: Path | str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
