"""Load the Phase 1 configuration and expose project paths.

Every module reads settings from here so nothing is hard-coded. The notebook and the
helper modules share one config object.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "raw"
DERIVED_DIR = ROOT / "derived"
EVIDENCE_DIR = ROOT / "evidence"
PAIRS_DIR = EVIDENCE_DIR / "pairs"
TEST_OUTPUT_DIR = EVIDENCE_DIR / "test_outputs"
DOCS_DIR = ROOT / "docs"
CONFIG_PATH = ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    @property
    def global_seed(self) -> int:
        return int(self.raw["global_seed"])

    @property
    def checkpoint(self) -> str:
        return self.raw["checkpoint"]

    @property
    def checkpoint_revision(self) -> str | None:
        return self.raw.get("checkpoint_revision")

    @property
    def max_length(self) -> int:
        return int(self.raw["max_length"])

    # corpus -------------------------------------------------------------
    @property
    def corpus_dataset(self) -> str:
        return self.raw["corpus"]["dataset"]

    @property
    def corpus_config(self) -> str:
        return self.raw["corpus"]["config"]

    @property
    def corpus_revision(self) -> str | None:
        return self.raw["corpus"].get("revision")

    @property
    def corpus_split(self) -> str:
        return self.raw["corpus"]["split"]

    @property
    def slice_rule(self) -> str:
        return self.raw["corpus"]["slice_rule"]

    @property
    def slice_size(self) -> int:
        return int(self.raw["corpus"]["slice_size"])

    @property
    def slice_min_chars(self) -> int:
        return int(self.raw["corpus"].get("slice_min_chars", 0))

    @property
    def slice_max_chars(self) -> int:
        return int(self.raw["corpus"].get("slice_max_chars", 10**9))

    # noise ------------------------------------------------------------
    @property
    def noise_operations(self) -> list[str]:
        return list(self.raw["noise"]["operations"])

    @property
    def noise_intensity(self) -> float:
        return float(self.raw["noise"]["intensity"])

    @property
    def noise_alphabet(self) -> str:
        return self.raw["noise"]["alphabet"]

    # pairs ----------------------------------------------------------
    @property
    def pairs_count(self) -> int:
        return int(self.raw["pairs"]["count"])


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        return Config(raw=yaml.safe_load(fh))


def ensure_dirs() -> None:
    for d in (RAW_DIR, DERIVED_DIR, PAIRS_DIR, TEST_OUTPUT_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)
