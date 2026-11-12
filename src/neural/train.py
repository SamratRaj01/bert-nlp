"""N3: train one model on clean text and keep the epoch with the lowest clean-validation loss."""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .evaluate import evaluate
from .model import NextTokenNet


def pick_best_epoch(val_losses: list[float]) -> int:
    """1-based epoch with the lowest loss; an exact tie keeps the EARLIER epoch."""
    best, best_i = float("inf"), 0
    for i, v in enumerate(val_losses, start=1):
        if v < best:                       # strict: equal loss never replaces the earlier epoch
            best, best_i = v, i
    return best_i


def train_model(mode: str, E: torch.Tensor, n_outputs: int, train, val, cfg: dict) -> dict:
    """`train`, `val` are (ctx, tgt) tensor pairs. Returns best weights, history and the selected epoch."""
    torch.manual_seed(cfg["seed"])                         # identical initialisation recipe for both models
    model = NextTokenNet(E, mode, n_outputs, cfg["hidden_size"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(cfg["seed"])       # same shuffles for both models
    ctx, tgt = train
    N, bs = len(tgt), cfg["batch_size"]
    untrained = evaluate(model, *val, chunk=cfg["eval_chunk"])
    history, states = [], []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(N, generator=gen)
        total = 0.0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(ctx[idx]), tgt[idx])      # raw logits in, no softmax first
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        v = evaluate(model, *val, chunk=cfg["eval_chunk"])          # no parameter updates
        history.append({"model": mode, "epoch": epoch, "train_loss": total / N, "val_loss": v["H"],
                        "val_accuracy": v["accuracy"]})
        states.append(copy.deepcopy({k: p.detach().clone() for k, p in model.state_dict().items()}))
    best = pick_best_epoch([h["val_loss"] for h in history])
    for h in history:
        h["selected"] = h["epoch"] == best
    model.load_state_dict(states[best - 1])
    model.eval()
    return {"model": model, "history": history, "best_epoch": best,
            "untrained_val_loss": untrained["H"], "untrained_val_accuracy": untrained["accuracy"]}
