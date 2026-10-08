"""Figures for Phase 2, generated ONLY from the saved CSVs in results/phase2/.

Owner: both. Plan §8 step 4.

Every figure keeps separate tokenizer panels (raw perplexities of the two pipelines are not
comparable) and one y-scale per panel. Operation identity is a fixed colour (insertion, deletion,
substitution = categorical slots 1-3) plus a marker shape, so colour is never the only cue.
Min-max bands span the three corruption seeds; p = 0 is the shared clean score.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OPS = ["insertion", "deletion", "substitution"]
OP_COLOR = {"insertion": "#2a78d6", "deletion": "#eb6834", "substitution": "#1baf7a"}
OP_MARKER = {"insertion": "o", "deletion": "s", "substitution": "^"}
TOKENIZERS = ["whitespace", "wordpiece"]
MODEL_LABEL = {"q1": "add-one unigram q1", "q2": "add-one bigram q2", "p_mix": "interpolated P_mix"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RATES = [0.0, 0.01, 0.05, 0.10, 0.20]
X_LABEL = "corruption rate p (evenly spaced)"


def _x(ps):
    """Rates are plotted at evenly spaced ordinal positions so 0 and 0.01 stay legible."""
    return [RATES.index(round(float(p), 2)) for p in ps]


def _rate_axis(ax, label=True):
    ax.set_xticks(range(len(RATES)), ["0 (clean)", "0.01", "0.05", "0.10", "0.20"])
    if label:
        ax.set_xlabel(X_LABEL)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.markersize": 6, "font.size": 9, "axes.titlesize": 10,
    "legend.frameon": False,
})


def _with_clean(summary: pd.DataFrame, value: str, train_set: str, tokenizer: str, model: str):
    s = summary[(summary.train_set == train_set) & (summary.tokenizer == tokenizer)
                & (summary.model == model)]
    clean = s[s.operation == "none"].iloc[0]
    for op in OPS:
        o = s[s.operation == op].sort_values("p")
        xs = [0.0] + list(o.p)
        mean = [clean[f"{value}_mean"]] + list(o[f"{value}_mean"])
        lo = [clean[f"{value}_min"]] + list(o[f"{value}_min"])
        hi = [clean[f"{value}_max"]] + list(o[f"{value}_max"])
        yield op, xs, mean, lo, hi


def _legend(fig):
    handles = [plt.Line2D([], [], color=OP_COLOR[op], marker=OP_MARKER[op], label=op) for op in OPS]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))


def loss_vs_rate(summary: pd.DataFrame, train_set: str, out: Path) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 8.5), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))
    for c, tok in enumerate(TOKENIZERS):
        for r, model in enumerate(["q1", "q2", "p_mix"]):
            ax = axes[r, c]
            for op, xs, mean, lo, hi in _with_clean(summary, "H_bits", train_set, tok, model):
                ax.fill_between(_x(xs), lo, hi, color=OP_COLOR[op], alpha=0.18, linewidth=0)
                ax.plot(_x(xs), mean, color=OP_COLOR[op], marker=OP_MARKER[op])
            ax.set_title(f"{tok} · {MODEL_LABEL[model]}")
            _rate_axis(ax, label=(r == 2))
            if c == 0:
                ax.set_ylabel("loss H (bits / target)")
    _legend(fig)
    fig.suptitle(f"Validation loss vs. noise rate — trained on {train_set} (band = min–max over 3 seeds)",
                 y=0.95, fontsize=10)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def context_benefit_fig(cb: pd.DataFrame, train_set: str, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.86))
    for c, tok in enumerate(TOKENIZERS):
        ax = axes[c]
        d = cb[(cb.train_set == train_set) & (cb.tokenizer == tok)]
        clean = d[d.condition_id == "clean"].iloc[0]
        for op in OPS:
            g = d[d.operation == op].groupby("p")["benefit_p_mix_bits"].agg(["mean", "min", "max"])
            xs = [0.0] + list(g.index)
            ax.fill_between(_x(xs), [clean.benefit_p_mix_bits] + list(g["min"]),
                            [clean.benefit_p_mix_bits] + list(g["max"]),
                            color=OP_COLOR[op], alpha=0.18, linewidth=0)
            ax.plot(_x(xs), [clean.benefit_p_mix_bits] + list(g["mean"]),
                    color=OP_COLOR[op], marker=OP_MARKER[op])
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_title(f"{tok}: H(q1) − H(P_mix)")
        _rate_axis(ax)
        if c == 0:
            ax.set_ylabel("context benefit (bits / target)")
    _legend(fig)
    fig.suptitle(f"Context benefit of interpolation over the unigram — {train_set}", y=0.9, fontsize=10)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def coverage_fig(cov: pd.DataFrame, train_set: str, out: Path) -> Path:
    metrics = [("unk_rate", "unknown-token rate"), ("pieces_per_span", "tokens per original word span"),
               ("continuation_rate", "continuation-piece (##) rate")]
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 8), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))
    for c, tok in enumerate(TOKENIZERS):
        d = cov[(cov.train_set == train_set) & (cov.tokenizer == tok)]
        clean = d[d.condition_id == "clean"].iloc[0]
        for r, (m, label) in enumerate(metrics):
            ax = axes[r, c]
            if tok == "whitespace" and m == "continuation_rate":
                ax.text(0.5, 0.5, "not applicable\n(whitespace has no ## pieces)", ha="center",
                        va="center", color=INK2, transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                continue
            for op in OPS:
                g = d[d.operation == op].groupby("p")[m].agg(["mean", "min", "max"])
                xs = [0.0] + list(g.index)
                ax.fill_between(_x(xs), [clean[m]] + list(g["min"]), [clean[m]] + list(g["max"]),
                                color=OP_COLOR[op], alpha=0.18, linewidth=0)
                ax.plot(_x(xs), [clean[m]] + list(g["mean"]), color=OP_COLOR[op], marker=OP_MARKER[op])
            ax.set_title(f"{tok}: {label}")
            _rate_axis(ax, label=(r == 2 or (tok == "whitespace" and r == 1)))
    _legend(fig)
    fig.suptitle(f"Coverage and fragmentation vs. noise rate — V from {train_set}", y=0.95, fontsize=10)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def severity_fig(sev_records: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), sharey=True, constrained_layout=True)
    for ax, op in zip(axes, OPS):
        d = sev_records[sev_records.operation == op]
        ps = sorted(d.p.unique())
        data = [d[d.p == p]["normalized_med"].values for p in ps]
        bp = ax.boxplot(data, tick_labels=[f"{p:.2f}" for p in ps], patch_artist=True,
                        widths=0.55, showfliers=False)
        for box in bp["boxes"]:
            box.set(facecolor=OP_COLOR[op], alpha=0.35, edgecolor=OP_COLOR[op])
        for k in ("medians", "whiskers", "caps"):
            for line in bp[k]:
                line.set(color=INK2)
        ax.set_title(f"{op}")
        ax.set_xlabel("requested p")
    axes[0].set_ylabel("normalized MED (MED / clean chars)")
    fig.suptitle("Realized corruption severity, 500 records × 3 seeds per box", fontsize=10)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def corpus_size_fig(summary: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.84))
    styles = {"train_a1_slice": "-", "train_full": "--"}
    for c, tok in enumerate(TOKENIZERS):
        ax = axes[c]
        for ts, ls in styles.items():
            s = summary[(summary.train_set == ts) & (summary.tokenizer == tok) & (summary.model == "p_mix")]
            for op in OPS:
                o = s[s.operation == op].sort_values("p")
                ax.plot(_x([0.0] + list(o.p)), [0.0] + list(o.delta_H_bits_mean), color=OP_COLOR[op],
                        marker=OP_MARKER[op], linestyle=ls)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_title(f"{tok}: P_mix noisy − clean loss")
        _rate_axis(ax)
        if c == 0:
            ax.set_ylabel("Δ loss (bits / target)")
    handles = [plt.Line2D([], [], color=OP_COLOR[op], marker=OP_MARKER[op], label=op) for op in OPS]
    handles += [plt.Line2D([], [], color=INK2, linestyle=ls, label=ts) for ts, ls in styles.items()]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Within-pipeline loss change: A1 slice (solid) vs. full train (dashed)", y=0.9, fontsize=10)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
