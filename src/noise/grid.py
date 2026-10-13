"""Generate the frozen validation noise grid: 1 clean + 3 ops x 4 rates x 3 seeds = 37 datasets.

Owner: Samrat. Phase 2 step A4.

Each noisy text is generated ONCE and saved; both tokenizers read these files and never
regenerate. One JSONL file per condition under derived/phase2/noisy/<condition_id>.jsonl, one
row per validation record. Each row is the record's corruption manifest plus its noisy text:

    article_id, record_id, condition_id, split, seed, record_seed, operation, p, rule_version,
    eligible_count, operation_log, realized_edits, med, normalized_med, clean_char_count,
    noisy_char_count, empty_after_noise, source_text_sha256, protocol_sha256, noisy_text

Records emptied by deletion are kept (noisy_text == "").
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..med.fast import med_fast
from ..phase2_config import NOISY_DIR, Phase2Config, sha256_file
from .rng import example_rng, example_seed
from .rule_v2 import RULE_VERSION, apply_v2, eligible_positions, replay_v2

OP_CODE = {"insertion": "ins", "deletion": "del", "substitution": "sub"}
CLEAN_ID = "clean"


def condition_id(op: str | None, p: float, seed: int | None) -> str:
    if op is None or p == 0:
        return CLEAN_ID
    return f"{OP_CODE[op]}-p{p:.2f}-s{seed}"


def conditions(cfg: Phase2Config) -> list[dict]:
    """The 37 conditions in a fixed order: clean first, then op -> rate -> seed."""
    out = [{"condition_id": CLEAN_ID, "operation": None, "p": 0.0, "seed": None}]
    for op in cfg.operations:
        for p in cfg.rates:
            for seed in cfg.seeds:
                out.append({"condition_id": condition_id(op, p, seed),
                            "operation": op, "p": p, "seed": seed})
    return out


def rng_key(record_id: str, op: str, p: float) -> str:
    return f"{record_id}|{op}|{p}"


def corrupt_record(rec: dict, op: str | None, p: float, seed: int | None,
                   cfg: Phase2Config) -> dict:
    clean = rec["text"]
    if op is None or p == 0:
        noisy, log = clean, []
        eligible = len(eligible_positions(clean, cfg.protected_spans))
        record_seed = None
    else:
        key = rng_key(rec["record_id"], op, p)
        noisy, log, eligible = apply_v2(clean, op, p, cfg.alphabet, example_rng(seed, key),
                                        cfg.protected_spans)
        record_seed = example_seed(seed, key)
        assert replay_v2(clean, log) == noisy, f"replay mismatch {rec['record_id']} {op} {p} {seed}"
    distance = med_fast(clean, noisy)
    return {
        "article_id": rec["article_id"],
        "record_id": rec["record_id"],
        "condition_id": condition_id(op, p, seed),
        "split": rec["source_split"],
        "seed": seed,
        "record_seed": record_seed,
        "operation": op,
        "p": p,
        "rule_version": RULE_VERSION,
        "eligible_count": eligible,
        "operation_log": log,
        "realized_edits": len(log),
        "med": distance,
        "normalized_med": distance / max(1, len(clean)),
        "clean_char_count": len(clean),
        "noisy_char_count": len(noisy),
        "empty_after_noise": noisy.strip() == "",
        "source_text_sha256": hashlib.sha256(clean.encode("utf-8")).hexdigest(),
        "protocol_sha256": cfg.protocol_sha256,
        "noisy_text": noisy,
    }


def generate_grid(records: list[dict], cfg: Phase2Config,
                  out_dir: Path = NOISY_DIR) -> list[dict]:
    """Write all 37 condition files; return an index row per condition (with file hash)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for cond in conditions(cfg):
        rows = [corrupt_record(r, cond["operation"], cond["p"], cond["seed"], cfg)
                for r in records]
        path = out_dir / f"{cond['condition_id']}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        index.append({**cond, "records": len(rows),
                      "realized_edits": sum(r["realized_edits"] for r in rows),
                      "eligible_chars": sum(r["eligible_count"] for r in rows),
                      "empty_after_noise": sum(r["empty_after_noise"] for r in rows),
                      "file": str(path.name), "sha256": sha256_file(path)})
    return index


def load_condition(condition: str, out_dir: Path = NOISY_DIR) -> list[dict]:
    with open(out_dir / f"{condition}.jsonl", "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
