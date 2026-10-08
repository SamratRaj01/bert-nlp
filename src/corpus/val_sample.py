"""Frozen validation sample and the two Phase 2 training sets.

Owner: Samrat. Phase 2 steps A1-A2.

Writes (all under derived/phase2/, each with a `.sha256` sidecar):
    val_sample.jsonl      -- 500 validation records, file order, A1 line filter (40-600 chars)
    train_a1_slice.jsonl  -- the A1 slice with article IDs back-filled (slice.jsonl untouched)
    train_full.jsonl      -- every non-empty, non-heading train line (corpus-size extension)

Selection uses source IDs and the line filter only, never model outcomes. The 40-600 band is
the clean-record length cap for MED runtime; it is applied before corruption and noisy
variants are never truncated.
"""
from __future__ import annotations

import hashlib

from ..config import DERIVED_DIR
from ..phase2_config import DERIVED_P2_DIR, Phase2Config, load_phase2_config
from .articles import attach_articles, build_records, write_records
from .slice import load_slice

VAL_PATH = DERIVED_P2_DIR / "val_sample.jsonl"
TRAIN_PATHS = {
    "train_a1_slice": DERIVED_P2_DIR / "train_a1_slice.jsonl",
    "train_full": DERIVED_P2_DIR / "train_full.jsonl",
}


def build_val_sample(cfg: Phase2Config | None = None) -> tuple[list[dict], dict]:
    cfg = cfg or load_phase2_config()
    v = cfg.val
    records, report = build_records(v["split"], int(v["min_chars"]), int(v["max_chars"]),
                                    cap=int(v["cap"]))
    report["sha256"] = write_records(records, VAL_PATH, v["rule"],
                                     {"articles_eligible_in_split": _eligible_articles(v)})
    return records, report


def _eligible_articles(v: dict) -> int:
    recs, _ = build_records(v["split"], int(v["min_chars"]), int(v["max_chars"]))
    return len({r["article_id"] for r in recs})


def a1_slice_unchanged() -> bool:
    """derived/slice.jsonl still matches the checksum recorded in Phase 1."""
    recorded = (DERIVED_DIR / "slice.sha256").read_text().split()[0]
    return hashlib.sha256((DERIVED_DIR / "slice.jsonl").read_bytes()).hexdigest() == recorded


def build_training_sets(cfg: Phase2Config | None = None) -> dict[str, list[dict]]:
    cfg = cfg or load_phase2_config()
    assert a1_slice_unchanged(), "derived/slice.jsonl no longer matches slice.sha256"
    ts = cfg.training_sets
    slice_recs = attach_articles(load_slice())
    write_records(slice_recs, TRAIN_PATHS["train_a1_slice"], ts["train_a1_slice"]["rule"],
                  {"source": ts["train_a1_slice"]["source"], "role": ts["train_a1_slice"]["role"]})
    full_recs, _ = build_records("train")
    write_records(full_recs, TRAIN_PATHS["train_full"], ts["train_full"]["rule"],
                  {"role": ts["train_full"]["role"]})
    return {"train_a1_slice": slice_recs, "train_full": full_recs}
