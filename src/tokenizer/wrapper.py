"""WordPiece tokenizer wrapper for the fixed BERT checkpoint.

Owner: Prazol. Steps B1, B3.

Identical configuration is used for clean and noisy text. No vocabulary training -- the
pretrained tokenizer that ships with the checkpoint is used as-is.
"""
from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer

from ..config import Config, load_config

CONTINUATION_PREFIX = "##"
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


@dataclass
class Encoding:
    pieces: list[str]          # WordPiece tokens WITHOUT special tokens
    input_ids: list[int]       # WITH [CLS]/[SEP], truncated to max_length
    n_special: int
    truncated: bool

    @property
    def token_count(self) -> int:
        return len(self.pieces)

    @property
    def continuation_count(self) -> int:
        return sum(1 for p in self.pieces if is_continuation(p))

    def unk_count(self, unk_token: str = "[UNK]") -> int:
        return sum(1 for p in self.pieces if p == unk_token)


def is_continuation(piece: str) -> bool:
    return piece.startswith(CONTINUATION_PREFIX)


class TokenizerWrapper:
    """Thin, fully-documented wrapper so clean and noisy branches share one config."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.checkpoint, revision=self.cfg.checkpoint_revision
        )
        self.max_length = self.cfg.max_length

    # --- introspection for the data card -------------------------------
    def special_token_ids(self) -> dict[str, int]:
        return {t: self.tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}

    def config_summary(self) -> dict:
        t = self.tokenizer
        basic = getattr(t, "basic_tokenizer", None)
        return {
            "checkpoint": self.cfg.checkpoint,
            "revision": self.cfg.checkpoint_revision,
            "tokenizer_class": type(t).__name__,
            "is_fast": bool(getattr(t, "is_fast", False)),
            "vocab_size": t.vocab_size,
            "model_max_length": t.model_max_length,
            "do_lower_case": getattr(t, "do_lower_case", None),
            "do_basic_tokenize": getattr(t, "do_basic_tokenize", None),
            "strip_accents": getattr(basic, "strip_accents", None) if basic else None,
            "tokenize_chinese_chars": (
                getattr(basic, "tokenize_chinese_chars", None) if basic else None
            ),
            "special_token_ids": self.special_token_ids(),
        }

    # --- the operation used everywhere else ---------------------------
    def encode(self, text: str) -> Encoding:
        pieces = self.tokenizer.tokenize(text)
        full_len = len(pieces) + 2  # + [CLS] + [SEP]
        truncated = full_len > self.max_length
        input_ids = self.tokenizer(
            text, truncation=True, max_length=self.max_length
        )["input_ids"]
        return Encoding(
            pieces=pieces,
            input_ids=input_ids,
            n_special=2,
            truncated=truncated,
        )
