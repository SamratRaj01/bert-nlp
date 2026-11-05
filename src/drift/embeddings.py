"""The frozen static embedding table E and variable-length span pooling.

E is the pretrained input embedding matrix of the pinned checkpoint (rows indexed by WordPiece ID).
It is copied out once to derived/phase3/, checksummed, and loaded read-only. Nothing here writes to it,
rescales it, or touches any other BERT component.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .paths import E_PATH, E_SHA_PATH, E_SHAPE, E_TENSOR_NAME


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_E(checkpoint: str, revision: str, out: Path = E_PATH,
              sha_out: Path = E_SHA_PATH) -> str:
    """Copy the word-embedding tensor from the pinned checkpoint to `out`; return its sha256."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    weights = hf_hub_download(checkpoint, "model.safetensors", revision=revision)
    with safe_open(weights, "numpy") as f:
        table = np.ascontiguousarray(f.get_tensor(E_TENSOR_NAME), dtype=np.float32)
    assert table.shape == E_SHAPE, f"unexpected embedding shape {table.shape}"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, table)
    digest = sha256_file(out)
    sha_out.write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    return digest


def load_E(path: Path = E_PATH, sha_path: Path = E_SHA_PATH) -> tuple[np.ndarray, str]:
    """Load E read-only after checking its recorded checksum; return (E, sha256)."""
    digest = sha256_file(path)
    recorded = sha_path.read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        raise RuntimeError(f"{path.name} does not match {sha_path.name}: E has changed")
    table = np.load(path)
    table.flags.writeable = False
    return table, digest


def pool_pieces(E: np.ndarray, token_ids: list[int]) -> np.ndarray | None:
    """Mean of the raw E rows for `token_ids` (float64); None when there are no tokens."""
    if not token_ids:
        return None
    return E[token_ids].astype(np.float64).mean(axis=0)


class SpanEmbedder:
    """Tokenize a span on its own (no [CLS]/[SEP]) and pool its token vectors."""

    def __init__(self, tokenizer, E: np.ndarray):
        self.tokenizer = tokenizer
        self.E = E
        self._cache: dict[str, tuple[list[str], np.ndarray | None]] = {}

    def embed(self, text: str) -> tuple[list[str], np.ndarray | None]:
        hit = self._cache.get(text)
        if hit is None:
            pieces = self.tokenizer.tokenize(text)  # tokenize() adds no special tokens
            ids = self.tokenizer.convert_tokens_to_ids(pieces)
            hit = (pieces, pool_pieces(self.E, ids))
            self._cache[text] = hit
        return hit
