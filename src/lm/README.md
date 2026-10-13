# src/lm/  (Prazol)

Phase 2 n-gram engine: count-based models trained on clean text that **score** (never generate)
clean and noisy validation text.

| File | Step | Purpose |
|---|---|---|
| `tokenize.py` | B1 | `WhitespacePipeline` (`text.split()`) and `WordPiecePipeline` (pinned BERT WordPiece, pieces only: no `[CLS]`/`[SEP]`, padding or truncation), with documentation cards. |
| `vocab.py` | B2 | Frozen `Vocab`: whitespace = count ≥ 2 + `<UNK>` + `</s>`; WordPiece = BERT vocab minus `[PAD] [CLS] [SEP] [MASK] [unused*]` + `</s>`. `<s>` is context only. |
| `counts.py` | B3–B4 | `NgramCounts`: framing `<s> <s> w… </s>` per record (context resets), sparse integer-keyed `C(h,w)` and `C(h)` for k = 1–3; frozen after `fit`. |
| `models.py` | B5 | `AddOneLM` (`q_k = (C(h,w)+1)/(C(h)+|V|)`), `model_probs` for q1 / q2 / P_mix (the required configurations) and the supplementary standalone trigram q3, unsmoothed `p_mle`, `nll_bits` (∞ on zero probability). |
| `interp.py` | B6 | 15-point λ grid (step 0.25), clean-validation loss, deterministic tie-break (larger λ1, then λ2). |
| `score.py` | B7 | `build_engine`, per-record NLL sums (bits and nats) + T, `corpus_loss` (`H = ΣNLL/ΣT`, `PP = 2^H`). |

Tests: `tests/test_lm.py` (assignment §11).
