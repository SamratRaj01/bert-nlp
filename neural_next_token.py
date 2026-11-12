#!/usr/bin/env python
"""Assignment 3, Workstream B: next-token prediction with two small neural networks.

Both networks read the two preceding tokens' frozen BERT vectors (mean pooling vs concatenation), use one
64-unit hidden layer, and predict the next token. They are trained once on CLEAN training text, then scored on
the clean validation text and on all 36 saved noisy versions (loss, perplexity, top-1 accuracy). The
Assignment 2 interpolated WordPiece n-gram results are reported alongside as a separate 37-row control.

    python neural_next_token.py                  # writes results/phase3/
    python neural_next_token.py --out some/dir

All results are preliminary validation results; the test split is never opened.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.geometry.embeddings import RESULTS_P3_DIR  # noqa: E402
from src.neural.run import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS_P3_DIR), help="output directory (default: results/phase3)")
    args = ap.parse_args()
    out = run(args.out)
    print("\nclean condition:")
    print(out["neural"][out["neural"].condition_id == "clean"][["model", "H", "PP", "accuracy", "T"]].round(4).to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
