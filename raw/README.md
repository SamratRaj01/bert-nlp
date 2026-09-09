# raw/

Untouched source corpus. **Never edit anything in this directory.**

## What lives here
- `wikitext-2/` — the WikiText-2 files exactly as downloaded (train / validation / test).
- `SOURCE.md` — where it came from: dataset id, version/revision, download date, URL, license/terms.

## Rules
- All transformation happens in code (`src/corpus/`) and is written to `derived/`.
- If you need to fix encoding or strip markup, do it in code and save the result elsewhere.
- Record a SHA-256 checksum of each file used so the exact bytes can be verified later.

Populated by `python -m src.corpus.acquire`.
