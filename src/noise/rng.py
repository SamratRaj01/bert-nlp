"""Per-example deterministic RNG.

Owner: Student A. Supports steps A5-A7.

The whole noise pipeline must be reproducible from one integer (`global_seed`) plus the
stable `example_id`. Never call the global `random` module directly in noise code.
"""
from __future__ import annotations

import hashlib
import random


def example_seed(global_seed: int, example_id: str) -> int:
    """Deterministically derive a 64-bit seed from (global_seed, example_id)."""
    h = hashlib.sha256(f"{global_seed}:{example_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def example_rng(global_seed: int, example_id: str) -> random.Random:
    return random.Random(example_seed(global_seed, example_id))
