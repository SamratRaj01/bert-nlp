"""N2: the two prediction networks. Same layers, different input.

    x_mean   = (e1 + e2) / 2          (m = d)       always divide by two, BOS counts as a zero vector
    x_concat = [e1 ; e2]              (m = 2d)
    h = ReLU(W1 x + b1)   (64 hidden)       z = W2 h + b2   (K logits)       softmax is inside the loss

E is held as a NON-persistent buffer: it is not a Parameter, never reaches the optimizer, and is not
saved with the weights. Only W1, b1, W2, b2 are trainable.
"""
from __future__ import annotations

import torch
from torch import nn


class NextTokenNet(nn.Module):
    def __init__(self, E: torch.Tensor, mode: str, n_outputs: int, hidden: int = 64):
        super().__init__()
        if mode not in ("mean", "concat"):
            raise ValueError(mode)
        self.mode = mode
        d = E.shape[1]
        m = d if mode == "mean" else 2 * d
        self.register_buffer("E", E, persistent=False)         # frozen lookup table
        self.fc1 = nn.Linear(m, hidden)
        self.fc2 = nn.Linear(hidden, n_outputs)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Rows of E for the ids; id -1 (BOS) gives an all-zero vector."""
        return self.E[ids.clamp(min=0)] * (ids >= 0).unsqueeze(-1).to(self.E.dtype)

    def features(self, ctx: torch.Tensor) -> torch.Tensor:
        e1, e2 = self.embed(ctx[:, 0]), self.embed(ctx[:, 1])
        return (e1 + e2) / 2 if self.mode == "mean" else torch.cat([e1, e2], dim=1)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:        # raw logits
        return self.fc2(torch.relu(self.fc1(self.features(ctx))))

    def counts(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": trainable, "frozen": int(self.E.numel())}
