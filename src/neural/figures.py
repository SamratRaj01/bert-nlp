"""Figures for workstream B (matplotlib, saved as PNG). Every figure carries the preliminary-results label."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..geometry.drift import OP_ORDER

COL = {"mean": "#2a78d6", "concat": "#eb6834", "ngram": "#1baf7a"}
NAME = {"mean": "mean pooling", "concat": "concatenation", "ngram": "n-gram control (Assignment 2)"}
NOTE = "Preliminary validation results. Band = lowest to highest of 3 corruption seeds (not a confidence interval)."


def plot_training(history: pd.DataFrame, path: Path | None = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for m in ("mean", "concat"):
        h = history[history.model == m]
        ax[0].plot(h.epoch, h.train_loss, ls="--", marker="o", color=COL[m], label=f"{NAME[m]}, train")
        ax[0].plot(h.epoch, h.val_loss, marker="o", color=COL[m], label=f"{NAME[m]}, validation")
        s = h[h.selected].iloc[0]
        ax[0].scatter([s.epoch], [s.val_loss], s=160, facecolors="none", edgecolors="k", zorder=5)
        ax[1].plot(h.epoch, h.val_accuracy, marker="o", color=COL[m], label=NAME[m])
    ax[0].set_ylabel("loss (nats per target)"); ax[1].set_ylabel("clean validation top-1 accuracy")
    for a in ax:
        a.set_xlabel("epoch"); a.set_xticks(sorted(history.epoch.unique())); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=8)
    ax[0].set_title("circle = epoch kept (lowest validation loss)", fontsize=9)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150)
    return fig


def plot_noise(summary: pd.DataFrame, control_summary: pd.DataFrame, path: Path | None = None):
    import matplotlib.pyplot as plt
    rates = sorted(summary.p.unique()); xs = {r: i for i, r in enumerate(rates)}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey="row")
    for j, op in enumerate(OP_ORDER):
        for key, df, mname in (("mean", summary, "mean"), ("concat", summary, "concat"), ("ngram", control_summary, None)):
            g = df[(df.operation == op) & ((df.model == mname) if mname else True)].sort_values("p")
            x = [xs[r] for r in g.p]
            axes[0, j].plot(x, g.dH_mean, marker="o", color=COL[key], label=NAME[key])
            axes[0, j].fill_between(x, g.dH_min, g.dH_max, color=COL[key], alpha=0.2)
            if key != "ngram":
                axes[1, j].plot(x, g.accuracy_mean, marker="o", color=COL[key], label=NAME[key])
                axes[1, j].fill_between(x, g.accuracy_min, g.accuracy_max, color=COL[key], alpha=0.2)
        axes[0, j].set_title(op)
        for i in (0, 1):
            axes[i, j].set_xticks(range(len(rates)), [f"{r:g}" for r in rates]); axes[i, j].grid(alpha=0.3)
        axes[1, j].set_xlabel("noise rate p")
    axes[0, 0].set_ylabel("increase in loss vs own clean (nats)"); axes[1, 0].set_ylabel("top-1 accuracy")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(NOTE, fontsize=8); fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150)
    return fig
