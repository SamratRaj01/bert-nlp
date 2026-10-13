"""The two tokenizer pipelines compared in Phase 2.

Owner: Prazol. Phase 2 step B1.

* whitespace -- Python `text.split()`: case and punctuation kept exactly as written.
* wordpiece  -- the pinned BERT WordPiece tokenizer from Phase 1, pieces only: no [CLS]/[SEP],
                no padding, no 512-token truncation (n-grams need none of BERT's wrappers).
                Lowercasing, accent stripping and punctuation splitting happen inside it.

This compares complete pipelines (normalization + segmentation + vocabulary), not
segmentation alone. Both pipelines receive the identical (clean or noisy) string.
"""
from __future__ import annotations

from transformers import AutoTokenizer

from ..phase2_config import Phase2Config


class WhitespacePipeline:
    name = "whitespace"

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def tokenize_batch(self, texts: list[str]) -> list[list[str]]:
        return [t.split() for t in texts]

    def card(self) -> dict:
        return {"tokenizer": "whitespace", "implementation": "str.split() (Python)",
                "casing": "preserved", "accents": "preserved",
                "punctuation": "kept attached to neighbouring characters unless space-separated"}


class WordPiecePipeline:
    name = "wordpiece"

    def __init__(self, cfg: Phase2Config):
        self.cfg = cfg
        self.hf = AutoTokenizer.from_pretrained(cfg.checkpoint, revision=cfg.checkpoint_revision)
        self._backend = self.hf.backend_tokenizer
        self._backend.no_truncation()
        self._backend.no_padding()

    def tokenize(self, text: str) -> list[str]:
        return self.hf.tokenize(text)

    def tokenize_batch(self, texts: list[str]) -> list[list[str]]:
        return [e.tokens for e in self._backend.encode_batch(list(texts), add_special_tokens=False)]

    def card(self) -> dict:
        import tokenizers
        import transformers

        basic = getattr(self.hf, "basic_tokenizer", None)
        return {
            "tokenizer": "wordpiece",
            "checkpoint": self.cfg.checkpoint,
            "revision": self.cfg.checkpoint_revision,
            "class": type(self.hf).__name__,
            "is_fast": bool(getattr(self.hf, "is_fast", False)),
            "casing": "lowercased" if getattr(self.hf, "do_lower_case", True) else "preserved",
            "accents": ("stripped (uncased BERT normalizer)"
                        if getattr(basic, "strip_accents", None) in (None, True) else "preserved"),
            "punctuation": "split into separate pieces by the BERT pre-tokenizer",
            "special_tokens_added": False,
            "truncation": False,
            "vocab_entries": len(self.hf.get_vocab()),
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
        }


def make_pipelines(cfg: Phase2Config) -> dict:
    return {"whitespace": WhitespacePipeline(), "wordpiece": WordPiecePipeline(cfg)}
