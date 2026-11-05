"""Paths and pinned constants for Assignment 3 / Workstream A."""
from __future__ import annotations

from ..config import DERIVED_DIR, ROOT

DERIVED_P3_DIR = DERIVED_DIR / "phase3"
RESULTS_P3_DIR = ROOT / "results" / "phase3"
DRIFT_DIR = RESULTS_P3_DIR / "vector_drift"

E_PATH = DERIVED_P3_DIR / "E_bert-base-uncased.npy"
E_SHA_PATH = DERIVED_P3_DIR / "E_bert-base-uncased.sha256"
E_TENSOR_NAME = "bert.embeddings.word_embeddings.weight"
E_SHAPE = (30522, 768)

SPAN_LIMIT = 1000
