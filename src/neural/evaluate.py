"""N4: teacher-forced scoring of a trained model on one evaluation condition.

Every context comes from the OBSERVED tokens of that condition (clean on clean, noisy on noisy); nothing
the model predicts is fed back. Loss is summed over all targets (incl. one EOS per record) in float64
and divided once by the target count: H = S / T (nats), PP = exp(H), accuracy = top-1 correct / T.
Per-record perplexities are never averaged.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import Sequences


def to_tensors(seq: Sequences) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(seq.ctx), torch.from_numpy(seq.tgt)


@torch.no_grad()
def evaluate(model, ctx: torch.Tensor, tgt: torch.Tensor, chunk: int = 1024) -> dict:
    model.eval()
    S, correct = 0.0, 0
    for i in range(0, len(tgt), chunk):
        logits = model(ctx[i:i + chunk])
        S += float(F.cross_entropy(logits, tgt[i:i + chunk], reduction="none").double().sum())
        correct += int((logits.argmax(dim=1) == tgt[i:i + chunk]).sum())
    T = len(tgt)
    H = S / T
    return {"T": T, "S": S, "H": H, "PP": math.exp(H), "accuracy": correct / T}
