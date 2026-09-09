"""Build one PairRecord end-to-end, and schedule the noise regime for the 20 pairs.

Owner: both. Integration step.

The *orchestration* (looping over 20 examples, displaying tables, running acceptance checks)
lives in `phase1.ipynb`. This module holds the per-pair logic it calls, so the notebook stays
readable and the heavy lifting stays unit-testable.
"""
from __future__ import annotations

from ..config import Config
from ..noise.manifest import build_row, replay
from ..noise.operations import Operation, apply, edit_locations
from ..noise.rng import example_rng, example_seed
from ..med.levenshtein import med, normalized_med
from ..tokenizer.diagnostics import paired_diagnostics
from ..tokenizer.wrapper import TokenizerWrapper
from .schema import PairRecord


def pair_schedule(n: int, operations: list[str], base_intensity: float) -> list[tuple[str, float]]:
    """Deterministic (operation, intensity) plan across the n pairs.

    - pair 0 is always the intensity-0 control (required by the acceptance checklist)
    - remaining pairs cycle through the enabled operations so each is represented
    - intensity cycles through {base, 2*base, 3*base} so a range of severities is present
    """
    severities = [base_intensity, 2 * base_intensity, 3 * base_intensity]
    plan: list[tuple[str, float]] = [(operations[0], 0.0)]
    for i in range(1, n):
        op = operations[(i - 1) % len(operations)]
        intensity = round(severities[(i - 1) % len(severities)], 4)
        plan.append((op, intensity))
    return plan


def _observe(diag: dict, log: list[dict], intensity: float) -> str:
    tcc = diag["token_count_change"]
    unk = diag["unk_delta"]
    cont = diag["continuation_delta"]
    if not log:
        if intensity == 0:
            return "intensity-0 control: no edits, tokenization identical."
        return (f"intensity {intensity}: no character positions were sampled for this short "
                f"text, tokenization identical.")
    bits = [f"{len(log)} edit(s)"]
    if tcc:
        bits.append(f"token count {tcc:+d}")
    if cont:
        bits.append(f"continuation pieces {cont:+d}")
    if unk:
        bits.append(f"[UNK] {unk:+d}")
    if len(bits) == 1:
        bits.append("subword sequence shifted but token count held")
    return ", ".join(bits) + "."


def build_pair_record(
    example: dict,
    operation: Operation,
    intensity: float,
    wrapper: TokenizerWrapper,
    cfg: Config,
) -> tuple[PairRecord, dict]:
    """Return (pair_record, manifest_row) for a single slice example."""
    clean_text = example["text"]
    example_id = example["example_id"]

    rng = example_rng(cfg.global_seed, example_id)
    seed = example_seed(cfg.global_seed, example_id)
    noisy_text, log = apply(clean_text, operation, intensity, cfg.noise_alphabet, rng)

    assert replay(clean_text, log) == noisy_text, f"replay mismatch for {example_id}"

    distance = med(clean_text, noisy_text)
    clean_enc = wrapper.encode(clean_text)
    noisy_enc = wrapper.encode(noisy_text)
    diag = paired_diagnostics(clean_enc, noisy_enc, clean_text, noisy_text)

    record = PairRecord(
        example_id=example_id,
        clean_id=f"{example_id}:clean",
        noisy_id=f"{example_id}:noisy",
        source_split=example["source_split"],
        source_index=example["source_index"],
        clean_text=clean_text,
        noisy_text=noisy_text,
        seed=seed,
        operation=operation,
        intensity=intensity,
        edit_location=edit_locations(log),
        operation_log=log,
        med=distance,
        normalized_med=normalized_med(clean_text, noisy_text),
        clean_token_count=clean_enc.token_count,
        noisy_token_count=noisy_enc.token_count,
        clean_pieces=clean_enc.pieces,
        noisy_pieces=noisy_enc.pieces,
        clean_continuation_count=clean_enc.continuation_count,
        noisy_continuation_count=noisy_enc.continuation_count,
        clean_unk_count=clean_enc.unk_count(),
        noisy_unk_count=noisy_enc.unk_count(),
        clean_truncated=clean_enc.truncated,
        noisy_truncated=noisy_enc.truncated,
        token_count_change=diag["token_count_change"],
        sequence_change_rate=diag["sequence_change_rate"],
        observation=_observe(diag, log, intensity),
    )
    manifest_row = build_row(example_id, seed, operation, intensity, log)
    return record, manifest_row
