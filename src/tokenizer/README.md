# src/tokenizer/  (Student B)

Loads the WordPiece tokenizer **paired with the fixed BERT checkpoint** and measures how
tokenization responds to noise. The pretrained vocabulary is used as-is — no vocabulary is
trained.

| File | Purpose |
|---|---|
| `wrapper.py` | Thin wrapper around `AutoTokenizer.from_pretrained(cfg.checkpoint)`. `encode(text)` -> `{pieces, input_ids, n_special, truncated}`. Helpers: `is_continuation(piece)` (leading `##`), `unk_count(pieces)`. |
| `diagnostics.py` | Per-text and paired diagnostics: token count / fertility, continuation-piece rate, `[UNK]` rate, truncation flag, and `sequence_change_rate(clean_pieces, noisy_pieces)` (normalised edit distance over token sequences). |
| `datacard.py` | Dumps the full tokenizer configuration to `docs/tokenizer_card.md`: library + versions, checkpoint name + revision, casing/normalization, special tokens + IDs, vocab size, vocab file source + SHA-256, model max length. |

## BPE vs WordPiece (required one-liner, keep in docs/tokenizer_card.md)
> BPE grows its vocabulary by repeatedly merging the most frequent adjacent symbol pair (pure
> frequency). WordPiece instead adds the merge that most increases the training corpus
> likelihood under a unigram LM (a likelihood/association score, roughly
> `count(ab) / (count(a)*count(b))`), not raw frequency alone. BERT's WordPiece marks every
> non-word-initial subword with a leading `##`.

## Rules
- The tokenizer and all its settings are identical for the clean and noisy branches.
- Do not corrupt or inject noise into special tokens ([PAD] [UNK] [CLS] [SEP] [MASK]).
- Save vocab + tokenizer config + checksum with the repo.
