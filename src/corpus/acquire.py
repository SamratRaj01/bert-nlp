"""Download WikiText-2 and record its provenance.

Owner: Student A. Step A1.

Writes:
    raw/wikitext-2/{train,validation,test}.raw   -- source text, one dataset row per line
    raw/wikitext-2/CHECKSUMS.sha256              -- SHA-256 of each raw file
    raw/SOURCE.md                                -- dataset id, revision, date, URL, license
"""
from __future__ import annotations

import datetime as _dt
import hashlib
from pathlib import Path

from datasets import load_dataset

from ..config import RAW_DIR, Config, load_config

SPLITS = ("train", "validation", "test")
LICENSE = "Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def acquire(cfg: Config | None = None) -> dict[str, Path]:
    """Materialise the raw corpus files. Returns {split: path}. Idempotent."""
    cfg = cfg or load_config()
    out_dir = RAW_DIR / "wikitext-2"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(cfg.corpus_dataset, cfg.corpus_config, revision=cfg.corpus_revision)

    paths: dict[str, Path] = {}
    for split in SPLITS:
        path = out_dir / f"{split}.raw"
        # Preserve each row verbatim; rows already carry their trailing newline.
        with open(path, "w", encoding="utf-8", newline="") as fh:
            for row in ds[split]["text"]:
                fh.write(row if row.endswith("\n") else row + "\n")
        paths[split] = path

    checksums = {p.name: _sha256(p) for p in paths.values()}
    with open(out_dir / "CHECKSUMS.sha256", "w", encoding="utf-8") as fh:
        for name, digest in checksums.items():
            fh.write(f"{digest}  {name}\n")

    with open(RAW_DIR / "SOURCE.md", "w", encoding="utf-8") as fh:
        fh.write(
            "# Corpus source\n\n"
            f"- Dataset: `{cfg.corpus_dataset}` (config `{cfg.corpus_config}`)\n"
            f"- Revision (sha): `{cfg.corpus_revision}`\n"
            f"- Retrieved: {_dt.date.today().isoformat()}\n"
            f"- URL: https://huggingface.co/datasets/{cfg.corpus_dataset}\n"
            f"- License / terms: {LICENSE}\n\n"
            "## Row counts\n\n"
            + "".join(f"- {s}: {len(ds[s])} rows\n" for s in SPLITS)
            + "\n## SHA-256\n\n"
            + "".join(f"- `{n}`: `{d}`\n" for n, d in checksums.items())
        )
    return paths


def main() -> None:
    paths = acquire()
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
