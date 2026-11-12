# Phase 3 — Workstream B (neural next-token prediction): notes

**Status:** complete. All results are **preliminary validation results**; the test split is not opened.
**Run it:** `python neural_next_token.py` (about 4 to 7 minutes on one CPU thread) writes `results/phase3/`. Walk-through: `phase3.ipynb`, sections 10–17.

## Method
- **Task.** For every token in a record, predict the next token from the two preceding tokens. Contexts reset at each record, with `BOS BOS w1 … wn` as input and `w1 … wn EOS` as targets. BOS is an all-zero input vector (not a row of BERT's table); EOS is output-only.
- **Output set.** The Assignment 2 WordPiece set: K = 29,525 (BERT vocabulary minus padding, CLS, SEP, MASK and 994 unused entries, plus `[UNK]` and the end-of-record output). Whole records are tokenized without `[CLS]/[SEP]` and without truncation.
- **Models.** Same network, different input: `x_mean = (e1+e2)/2` (768 numbers) or `x_concat = [e1;e2]` (1,536). Then `h = ReLU(W1 x + b1)` with 64 units and `z = W2 h + b2` over K outputs. E (BERT's token table) is frozen and hashed before and after training (unchanged).
  Trainable parameters: 1,968,341 (mean) and 2,017,493 (concat); frozen E: 23,440,896.
- **Training (fixed by the assignment).** Adam, lr 0.001, batch 128, ≤ 5 epochs, seed 101, no dropout or weight decay, CrossEntropyLoss on raw logits, the 500 clean Assignment 2 training records (35,642 targets). After each epoch the clean-validation loss is measured; the epoch with the lowest value is kept (earlier epoch wins an exact tie). Each model is trained once; the corruption seeds only change the evaluation text.
- **Scoring.** Teacher-forced: clean text with clean contexts and targets, noisy text with noisy contexts and targets. Loss is summed over all targets and divided once by the target count T: `H = S/T` (nats), `PP = exp(H)`, accuracy = top-1 correct / T. Per-record perplexities are never averaged.
- **Control.** The 37 Assignment 2 rows for the interpolated WordPiece n-gram on the same training slice and the same validation files, kept separate. Its frozen weights for this slice are `[1, 0, 0]`, so it is the add-one unigram: it ignores the two previous tokens.

## Checks that passed
- Target counts match Assignment 2 exactly: train 35,642, clean validation 35,971, and all 37 per-condition `T`.
- Untrained validation loss 10.29 equals ln K (uniform guessing).
- E hash identical before and after training.
- Two complete runs produce identical result tables (all five CSVs compared).
- Re-scoring from the saved weights reproduces the saved table.
- 83 tests pass, 10 of them for this workstream (framing, BOS zeros, parameter counts, swap behaviour, tie-break, hand-computed loss/perplexity/accuracy, reproducible training).

## Results (74 neural rows, 24 noise-summary rows, 37 control rows)
| clean validation | loss H (nats) | perplexity | top-1 accuracy |
|---|---|---|---|
| mean pooling | 7.503 | 1,814 | 9.8% |
| concatenation | 7.283 | 1,456 | 13.8% |
| n-gram control (A2) | 7.549 | 1,898 | not computed in Assignment 2 |

- Both networks reach their lowest validation loss after **epoch 2**, then validation loss rises while training loss keeps falling (memorising 500 records).
- **Noise raises loss and lowers accuracy** for both models, for all three operations, at every rate; the effect grows with the rate (loss +1.8 nats under 20% insertion; accuracy falls to 3.9% and 5.1%).
- **Neither neural model deteriorates less than the control.** The increase over each model's own clean loss is larger for both neural models than for the control in all 12 operation × rate cells.
- **Concatenation** has lower absolute loss and higher accuracy than mean pooling in every cell, and its loss rises slightly more than the mean model's.
- **Mixed absolute picture.** By raw loss, concatenation is best in 7 of 12 cells (light noise) and the n-gram control in 5 (heavier noise); mean pooling in none.
- **Insertion hurts most, deletion least** at the same rate (for all models).
- All three operations increase the number of word pieces (typos fragment words), so noisy conditions are scored on longer target sequences (T from 35,971 to 70,141 under 20% insertion).

## Caveats
- Small training set (500 records, 35,642 targets) against 29,525 possible outputs; the models overfit after two epochs.
- The control is a unigram, so "neural vs n-gram" is neural vs a model that never used context.
- Concatenation has twice the input weights (98,304 vs 49,152), so differences cannot be credited only to word order.
- Lower loss on noisy text can come from easier targets or more unknown tokens, so diagnostics are saved with every row; in this run no noisy condition has lower loss than clean.
- The min–max over the 3 corruption seeds is variation across random corruptions of one trained model per type, not a confidence interval.
- A fixed seed does not guarantee identical results on different hardware (CPU, 1 thread, deterministic algorithms on, torch 2.15.0).
- No claim is made that vector drift explains prediction change; that comparison (counterexamples) is not done yet.

## Files
`configs/phase3.yaml` (`neural:` block), `src/neural/{data,model,train,evaluate,figures,run}.py`, `neural_next_token.py`, `tests/test_neural.py`, `phase3.ipynb`,
`results/phase3/{neural_condition_rows.csv, neural_noise_summary.csv, ngram_control.csv, neural_vs_ngram_change.csv, neural_training_history.csv, neural_run_info.json, figures/neural_training.png, figures/neural_noise.png}`,
`derived/phase3/{neural_retained_records.json, neural/{mean,concat}_best.pt}`.
