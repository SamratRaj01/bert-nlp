"""Cut the deterministic corpus slice.

Owner: Student A. Step A2.

Reads:  raw/wikitext-2/<split>.raw
Writes: derived/slice.jsonl   -- one example per line:
            {"example_id", "text", "source_split", "source_index"}
        derived/slice.sha256  -- checksum of slice.jsonl + the exact selection rule

The slice is a pure function of (raw file, config). No randomness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import DERIVED_DIR, RAW_DIR, Config, load_config


def is_kept_line(raw_line: str, min_chars: int = 0, max_chars: int = 10**9) -> bool:
    """Keep non-empty, non-heading lines within a character-length band.

    WikiText headings look like ' = Title = ' or ' = = Section = = ' -> after strip()
    they start and end with '='. The length band keeps examples inspectable and mostly
    below the tokenizer's 512-token limit.
    """
    s = raw_line.strip()
    if not s:
        return False
    if s.startswith("=") and s.endswith("="):
        return False
    if not (min_chars <= len(s) <= max_chars):
        return False
    return True


def make_example_id(source_split: str, source_index: int) -> str:
    prefix = {"train": "wt2-train", "validation": "wt2-valid", "test": "wt2-test"}[source_split]
    return f"{prefix}-{source_index:06d}"


def build_slice(cfg: Config | None = None) -> list[dict]:
    """Return the slice as a list of example dicts and write it to derived/."""
    cfg = cfg or load_config()
    raw_path = RAW_DIR / "wikitext-2" / f"{cfg.corpus_split}.raw"
    with open(raw_path, "r", encoding="utf-8", newline="") as fh:
        raw_lines = fh.readlines()

    examples: list[dict] = []
    for idx, raw_line in enumerate(raw_lines):
        if not is_kept_line(raw_line, cfg.slice_min_chars, cfg.slice_max_chars):
            continue
        examples.append(
            {
                "example_id": make_example_id(cfg.corpus_split, idx),
                "text": raw_line.strip(),
                "source_split": cfg.corpus_split,
                "source_index": idx,
            }
        )
        if len(examples) >= cfg.slice_size:
            break

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    slice_path = DERIVED_DIR / "slice.jsonl"
    with open(slice_path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(slice_path.read_bytes()).hexdigest()
    with open(DERIVED_DIR / "slice.sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  slice.jsonl\n")
        fh.write(f"selection_rule: {cfg.slice_rule}\n")
        fh.write(f"count: {len(examples)}\n")
    return examples


def load_slice(path: Path | str = DERIVED_DIR / "slice.jsonl") -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    examples = build_slice()
    print(f"{len(examples)} examples -> {DERIVED_DIR / 'slice.jsonl'}")


if __name__ == "__main__":
    main()
