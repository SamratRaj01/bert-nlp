"""Corruption manifest: serialise and verify what the noise pipeline did.

Owner: Samrat. Step A6-A7.

One manifest row per generated noisy example, written to
derived/corruption_manifest.jsonl:

    {
        "example_id": str,
        "seed": int,                 # example_seed(global_seed, example_id)
        "operation": str,
        "intensity": float,
        "n_edits": int,
        "operation_log": [edit_record, ...],
    }
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import DERIVED_DIR
from .operations import render

MANIFEST_PATH = DERIVED_DIR / "corruption_manifest.jsonl"


def build_row(
    example_id: str,
    seed: int,
    operation: str,
    intensity: float,
    operation_log: list[dict],
) -> dict:
    return {
        "example_id": example_id,
        "clean_id": f"{example_id}:clean",
        "noisy_id": f"{example_id}:noisy",
        "seed": seed,
        "operation": operation,
        "intensity": intensity,
        "n_edits": len(operation_log),
        "operation_log": operation_log,
    }


def write_manifest(rows: list[dict], path: Path | str = MANIFEST_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def replay(clean_text: str, operation_log: list[dict]) -> str:
    """Reconstruct the noisy text by applying the log to the clean text."""
    return render(clean_text, operation_log)
