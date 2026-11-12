"""Load and fingerprint BERT's frozen token-embedding table E (|V| x d).

Only the stored word-embedding matrix is read. No position/segment vectors, no LayerNorm, no
transformer layers, no rescaling. Row j is the static vector for token ID j.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from ..config import ROOT

PHASE3_CONFIG_PATH = ROOT / "configs" / "phase3.yaml"
DERIVED_P3_DIR = ROOT / "derived" / "phase3"
RESULTS_P3_DIR = ROOT / "results" / "phase3"


def load_phase3_config(path: Path | str = PHASE3_CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def weights_path(cfg: dict | None = None) -> Path:
    """Local path of the pinned weights file (downloads it on first use)."""
    from huggingface_hub import hf_hub_download
    e = (cfg or load_phase3_config())["embedding"]
    return Path(hf_hub_download(e["checkpoint"], e["weights_file"], revision=e["revision"]))


def load_embedding_table(cfg: dict | None = None) -> np.ndarray:
    """Return E as a read-only float32 array of shape (|V|, d)."""
    from safetensors import safe_open
    cfg = cfg or load_phase3_config()
    e = cfg["embedding"]
    with safe_open(str(weights_path(cfg)), framework="numpy") as f:
        table = f.get_tensor(e["tensor_name"])
    if list(table.shape) != list(e["expected_shape"]):
        raise ValueError(f"E has shape {table.shape}, expected {e['expected_shape']}")
    table = np.ascontiguousarray(table, dtype=np.float32)
    table.setflags(write=False)          # frozen: any in-place edit raises
    return table


def table_sha256(table: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(table).tobytes()).hexdigest()
