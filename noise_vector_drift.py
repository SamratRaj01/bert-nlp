#!/usr/bin/env python
"""Assignment 3, Workstream A: measure how character noise moves static text vectors.

Reuses the Assignment 2 validation records, noisy datasets and edit logs (nothing is regenerated),
pools raw BERT token-embedding rows per whitespace span, and compares each clean span with its noisy
counterpart by cosine similarity and Euclidean distance.

    python noise_vector_drift.py                 # writes results/phase3/
    python noise_vector_drift.py --out some/dir

All results are preliminary validation results; the test split is never opened.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.geometry.drift import run  # noqa: E402
from src.geometry.embeddings import RESULTS_P3_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS_P3_DIR), help="output directory (default: results/phase3)")
    args = ap.parse_args()
    out = run(args.out)
    print("\nstatic span audit (first rows):")
    print(out["audit"].head(6).round(4).to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
