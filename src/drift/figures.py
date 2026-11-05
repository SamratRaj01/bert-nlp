"""Geometry figures for Workstream A, generated ONLY from the saved CSVs in results/phase3/vector_drift/.

Three panels share the x-axis (evenly spaced ordinal rates, p = 0 is the shared clean point):
mean cosine similarity, mean raw Euclidean distance, and valid-pair coverage. One line per operation
(colour plus marker); bands are the min-max over the three corruption seeds, not a confidence interval.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .paths import DRIFT_DIR  # noqa: E402

OPS = ["insertion", "deletion", "substitution"]
OP_COLOR = {"insertion": "#2a78d6", "deletion": "#eb6834", "substitution": "#1baf7a"}
OP_MARKER = {"insertion": "o", "deletion": "s", "substitution": "^"}
RATES = [0.0, 0.01, 0.05, 0.10, 0.20]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 6, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def _read(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x: str) -> float:
    return float(x) if x != "" else float("nan")


def geometry_vs_rate(out_dir: Path = DRIFT_DIR) -> Path:
    cond = _read(out_dir / "condition_summary.csv")
    noise = _read(out_dir / "noise_summary.csv")
    clean = next(r for r in cond if r["operation"] == "none")
    panels = [("mean_cosine", "mean cosine similarity"),
              ("mean_euclidean", "mean Euclidean distance"),
              ("coverage", "valid-pair coverage")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for ax, (metric, label) in zip(axes, panels):
        for op in OPS:
            rows = sorted((r for r in noise if r["operation"] == op), key=lambda r: float(r["p"]))
            xs = [0] + [RATES.index(round(float(r["p"]), 2)) for r in rows]
            mean = [_f(clean[metric])] + [_f(r[f"{metric}_mean"]) for r in rows]
            lo = [_f(clean[metric])] + [_f(r[f"{metric}_min"]) for r in rows]
            hi = [_f(clean[metric])] + [_f(r[f"{metric}_max"]) for r in rows]
            ax.fill_between(xs, lo, hi, color=OP_COLOR[op], alpha=0.18, linewidth=0)
            ax.plot(xs, mean, color=OP_COLOR[op], marker=OP_MARKER[op], label=op)
        ax.set_xticks(range(len(RATES)), ["0 (clean)", "0.01", "0.05", "0.10", "0.20"])
        ax.set_xlabel("corruption rate p (evenly spaced)")
        ax.set_title(label)
    axes[2].set_ylim(0.95, 1.005)
    axes[0].legend(loc="lower left")
    fig.suptitle("Static-vector drift of aligned spans (1,000 spans; preliminary validation results)",
                 fontsize=10)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    path = fig_dir / "geometry_vs_rate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(geometry_vs_rate())
