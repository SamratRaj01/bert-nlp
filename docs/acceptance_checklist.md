# Phase 1 Acceptance Checklist

Sign only when every box is checked and the evidence is in the repo.

- [ ] Same seed reproduces the same noise and operation log
- [ ] Intensity 0 returns the source text and MED 0
- [ ] Replaying the recorded edits reproduces the noisy output for all 20 pairs
- [ ] Clean and noisy branches use identical tokenizer settings
- [ ] At least one pair shows a meaningful WordPiece change, and BPE is not mislabeled
- [ ] Raw files unchanged; derived files and UTF-8 handling documented
- [ ] All known-answer and determinism tests pass; outputs saved to `evidence/test_outputs/`
- [ ] Package contains 20 traceable clean/noisy pairs + tokenizer diagnostics
- [ ] README, environment, seed, configuration, and contribution record complete
- [ ] Ethics section states authorized-use limits and dual-use implications

Samrat: ___________________________  Date: __________

Prazol: ___________________________  Date: __________
