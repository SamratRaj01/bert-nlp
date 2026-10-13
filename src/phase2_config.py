"""Load the Phase 2 configuration (configs/phase2.yaml) and expose Phase 2 paths.

Step 0 (shared freeze). `src/config.py` stays the Phase 1 record; Phase 2 code reads from here.

The YAML has two parts separated by FROZEN_MARKER:
    above  -> the experimental protocol; its hash (`protocol_sha256`) goes into every manifest
    below  -> frozen λ written by the notebook in step B6, before any noisy scoring
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import DERIVED_DIR, EVIDENCE_DIR, ROOT

PHASE2_CONFIG_PATH = ROOT / "configs" / "phase2.yaml"
DERIVED_P2_DIR = DERIVED_DIR / "phase2"
NOISY_DIR = DERIVED_P2_DIR / "noisy"
RESULTS_P2_DIR = ROOT / "results" / "phase2"
EVIDENCE_P2_DIR = EVIDENCE_DIR / "phase2"

FROZEN_MARKER = "# === FROZEN BY phase2.ipynb (step B6) — do not edit by hand ==="


def _split(text: str) -> tuple[str, str]:
    if FROZEN_MARKER not in text:
        raise ValueError(f"{PHASE2_CONFIG_PATH} is missing the frozen-block marker")
    head, tail = text.split(FROZEN_MARKER, 1)
    return head, tail


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Phase2Config:
    raw: dict[str, Any]
    protocol_sha256: str      # hash of everything above the frozen marker
    file_sha256: str          # hash of the whole file (protocol + frozen λ)

    # tokenizer / corpus --------------------------------------------------
    @property
    def checkpoint(self) -> str:
        return self.raw["checkpoint"]

    @property
    def checkpoint_revision(self) -> str:
        return self.raw["checkpoint_revision"]

    # validation sample ---------------------------------------------------
    @property
    def val(self) -> dict:
        return self.raw["val_sample"]

    # noise grid ----------------------------------------------------------
    @property
    def noise(self) -> dict:
        return self.raw["noise"]

    @property
    def operations(self) -> list[str]:
        return list(self.noise["operations"])

    @property
    def rates(self) -> list[float]:
        return [float(p) for p in self.noise["rates"]]

    @property
    def seeds(self) -> list[int]:
        return [int(s) for s in self.noise["seeds"]]

    @property
    def alphabet(self) -> str:
        return self.noise["alphabet"]

    @property
    def protected_spans(self) -> list[str]:
        return list(self.noise.get("protected_spans", []))

    # language models -----------------------------------------------------
    @property
    def lm(self) -> dict:
        return self.raw["lm"]

    @property
    def training_sets(self) -> dict:
        return self.raw["training_sets"]

    @property
    def frozen_lambdas(self) -> dict:
        return (self.raw.get("frozen_lambdas") or {})


def load_phase2_config(path: Path | str = PHASE2_CONFIG_PATH) -> Phase2Config:
    text = Path(path).read_text(encoding="utf-8")
    head, _ = _split(text)
    return Phase2Config(raw=yaml.safe_load(text) or {},
                        protocol_sha256=sha256_text(head),
                        file_sha256=sha256_text(text))


def write_frozen_lambdas(train_set: str, lambdas: dict[str, list[float]],
                         path: Path | str = PHASE2_CONFIG_PATH) -> Phase2Config:
    """Record the selected λ for one training set below the frozen marker.

    If λ for this training set are already frozen, they must match the new selection exactly
    (re-running the deterministic selection must not silently change a frozen setting).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    head, _ = _split(text)
    current = load_phase2_config(path).frozen_lambdas
    previous = current.get(train_set)
    if previous is not None and previous != lambdas:
        raise ValueError(f"frozen λ for {train_set} differ from the new selection: "
                         f"{previous} vs {lambdas}. Clear the block deliberately to re-freeze.")
    current = {**current, train_set: lambdas}
    block = yaml.safe_dump({"frozen_lambdas": current}, sort_keys=True,
                           default_flow_style=None, allow_unicode=True)
    path.write_text(head + FROZEN_MARKER + "\n" + block, encoding="utf-8")
    return load_phase2_config(path)


def ensure_p2_dirs() -> None:
    for d in (DERIVED_P2_DIR, NOISY_DIR, RESULTS_P2_DIR, EVIDENCE_P2_DIR):
        d.mkdir(parents=True, exist_ok=True)
