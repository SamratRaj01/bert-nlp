"""Shared pair-record schema and writers.

Owner: both. Agree here before touching the generator.

One PairRecord per clean/noisy example, written three ways:
    evidence/pairs/pairs.jsonl  -- machine-readable, one JSON object per line
    evidence/pairs/pairs.csv    -- flat table (list fields JSON-encoded per cell)
    evidence/pairs/pairs.md     -- human-readable table for the report appendix
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import PAIRS_DIR


@dataclass
class PairRecord:
    example_id: str
    clean_id: str        # f"{example_id}:clean"
    noisy_id: str        # f"{example_id}:noisy"
    source_split: str
    source_index: int
    clean_text: str
    noisy_text: str

    seed: int
    operation: str
    intensity: float
    edit_location: list[int]
    operation_log: list[dict]

    med: int
    normalized_med: float

    clean_token_count: int
    noisy_token_count: int
    clean_pieces: list[str]
    noisy_pieces: list[str]
    clean_continuation_count: int
    noisy_continuation_count: int
    clean_unk_count: int
    noisy_unk_count: int
    clean_truncated: bool
    noisy_truncated: bool

    token_count_change: int
    sequence_change_rate: float
    observation: str = field(default="")

    def as_dict(self) -> dict:
        return asdict(self)


_LIST_FIELDS = {"edit_location", "operation_log", "clean_pieces", "noisy_pieces"}


def write_jsonl(records: list[PairRecord], path: Path | str = PAIRS_DIR / "pairs.jsonl") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
    return path


def write_csv(records: list[PairRecord], path: Path | str = PAIRS_DIR / "pairs.csv") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(PairRecord.__dataclass_fields__)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = r.as_dict()
            for k in _LIST_FIELDS:
                row[k] = json.dumps(row[k], ensure_ascii=False)
            w.writerow(row)
    return path


def write_markdown(records: list[PairRecord], path: Path | str = PAIRS_DIR / "pairs.md") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["example_id", "operation", "intensity", "med", "normalized_med",
            "clean_token_count", "noisy_token_count", "token_count_change",
            "sequence_change_rate", "observation"]
    lines = ["# 20 paired clean/noisy records", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in records:
        d = r.as_dict()
        cells = []
        for c in cols:
            v = d[c]
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Full text per pair", ""]
    for r in records:
        lines += [
            f"### {r.example_id}  ({r.operation}, intensity {r.intensity})",
            f"- clean: `{r.clean_text}`",
            f"- noisy: `{r.noisy_text}`",
            f"- MED {r.med} (normalized {r.normalized_med:.3f}); "
            f"tokens {r.clean_token_count} -> {r.noisy_token_count} "
            f"(change {r.token_count_change:+d})",
            f"- {r.observation}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_all(records: list[PairRecord]) -> list[Path]:
    return [write_jsonl(records), write_csv(records), write_markdown(records)]
