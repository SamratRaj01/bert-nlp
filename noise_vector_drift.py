"""Assignment 3, Workstream A: measure how character noise moves static token-embedding vectors.

    python noise_vector_drift.py            # full 37-condition grid, 1,000 spans

Thin entry point; the logic lives in src/drift/. Results: results/phase3/vector_drift/.
"""
from src.drift.audit import main

if __name__ == "__main__":
    main()
