#!/usr/bin/env python3
"""
plot_delay_attack.py — Robustness to timing-delay attacks, XGBoost only.

Left panel : Test-time jitter — pre-trained classifier, test data corrupted.
Right panel: Adversarial retraining — classifier retrained on corrupted data.

Each line = one training dataset, coloured consistently with plot_combined_efficiency.py.
X-axis: max random delay injected (ms). Y-axis: Macro F1.

Usage:
    python plot_delay_attack.py
    python plot_delay_attack.py --traces-dir /path/to/traces
    python plot_delay_attack.py --split ood
    python plot_delay_attack.py --format pdf
    python plot_delay_attack.py --out fig_delay.pdf --format pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CLF    = "XGBoost"
DELAYS = [500, 1000, 2000, 5000]

# NeurIPS-friendly defaults (mirrors efficiency_figs.ipynb)
plt.rcParams.update({
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
    "text.usetex":       False,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10.5,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "lines.linewidth":   2.0,
})

# Wong (2011) colorblind-safe palette — consistent with plot_combined_efficiency.py
DATASETS = [
    # (key, label, color, marker, baseline_tag, jitter_pattern, delayed_pattern)
    ("wiki",     "2WikiMultiHop", "#0072B2", "o",
     "wiki_ood_all",     "wiki_ood_all_jitter_{N}ms",     "wiki_delayed_xgb_{N}ms"),
    ("frames",   "FRAMES",        "#E69F00", "s",
     "frames_ood_all",   "frames_ood_all_jitter_{N}ms",   "frames_delayed_xgb_{N}ms"),
    ("webshop",  "WebShop",       "#009E73", "^",
     "webshop_ood_all",  "webshop_ood_all_jitter_{N}ms",  "webshop_delayed_xgb_{N}ms"),
    ("deepshop", "DeepShop",      "#CC79A7", "D",
     "deepshop_ood_all", "deepshop_ood_all_jitter_{N}ms", "deepshop_delayed_xgb_{N}ms"),
]


def _load_f1(clf_dir: Path, split: str) -> float | None:
    rpath = clf_dir / "results.json"
    if not rpath.exists():
        return None
    res   = json.load(open(rpath))
    model = res.get("models", {}).get(CLF)
    if model is None:
        return None
    if split == "test":
        return (model.get("test_report") or {}).get("macro avg", {}).get("f1-score")
    ood_f1s = [
        r.get("macro avg", {}).get("f1-score")
        for r in (model.get("ood_reports") or {}).values()
        if r.get("macro avg")
    ]
    return float(np.mean(ood_f1s)) if ood_f1s else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"))
    parser.add_argument("--split", choices=["test", "ood"], default="test")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--format", choices=["png", "pdf"], default="png",
                        help="Output format (default: png). Ignored if --out specifies an extension.")
    args = parser.parse_args()

    clf_root = args.traces_dir / "classifiers"
    REF_LS   = (0, (1.5, 2.5))

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(7.2, 3.0), sharey=True, constrained_layout=False
    )

    for _key, label, color, marker, baseline_tag, jitter_pat, delayed_pat in DATASETS:
        baseline_f1 = _load_f1(clf_root / baseline_tag, args.split)

        # ── left: test-time jitter ────────────────────────────────────────────
        jitter_xs, jitter_ys = [], []
        if baseline_f1 is not None:
            jitter_xs.append(0)
            jitter_ys.append(baseline_f1)
        for n in DELAYS:
            f1 = _load_f1(clf_root / jitter_pat.format(N=n), args.split)
            if f1 is not None:
                jitter_xs.append(n)
                jitter_ys.append(f1)
        if jitter_xs:
            ax_l.plot(jitter_xs, jitter_ys, marker=marker, color=color, label=label,
                      markersize=4.8, markeredgewidth=0.7)

        # ── right: adversarial retraining ─────────────────────────────────────
        delayed_xs, delayed_ys = [], []
        if baseline_f1 is not None:
            delayed_xs.append(0)
            delayed_ys.append(baseline_f1)
        for n in DELAYS:
            f1 = _load_f1(clf_root / delayed_pat.format(N=n), args.split)
            if f1 is not None:
                delayed_xs.append(n)
                delayed_ys.append(f1)
        if delayed_xs:
            ax_r.plot(delayed_xs, delayed_ys, marker=marker, color=color, label=label,
                      markersize=4.8, markeredgewidth=0.7)

    # ── shared axis styling ───────────────────────────────────────────────────
    for ax in (ax_l, ax_r):
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xlim(-50, 5250)
        ax.set_xticks([0, 500, 1000, 2000, 5000])
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("Maximum injected random delay (ms)")
        ax.grid(True, axis="both", alpha=0.22)
        ax.axhline(0.8, color="0.45", linewidth=0.9, linestyle=REF_LS, zorder=0)
        ax.text(60, 0.815, "0.80 F1", color="0.35", fontsize=8.3, va="bottom")

    ax_l.set_ylabel("Macro F1")
    ax_l.set_title("A. Timing attack without retraining", loc="left", fontweight="semibold")
    ax_r.set_title("B. Retrained under timing attack",    loc="left", fontweight="semibold")

    ax_l.annotate(
        "Performance drops sharply\nunder timing randomisation",
        xy=(1000, 0.47), xytext=(1900, 0.73),
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.35"),
        fontsize=8.3, color="0.25", ha="left", va="center",
    )
    ax_r.annotate(
        "Retraining substantially\nrestores robustness",
        xy=(1000, 0.72), xytext=(1900, 0.84),
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.35"),
        fontsize=8.3, color="0.25", ha="left", va="center",
    )

    # ── legend ────────────────────────────────────────────────────────────────
    handles, labels = ax_l.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_r.get_legend_handles_labels()
    fig.text(0.5, 0.15,
             "Injected delay is sampled uniformly up to the stated maximum.",
             ha="center", va="center", fontsize=8.2, color="0.35")

    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.03),
               bbox_transform=fig.transFigure,
               ncol=4, frameon=False, handlelength=1.8, columnspacing=1.5)

    fig.subplots_adjust(left=0.09, right=0.995, top=0.88, bottom=0.38, wspace=0.16)

    fmt = args.format
    out = args.out or (clf_root / f"delay_attack.{fmt}")
    out = out.with_suffix(f".{fmt}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=350, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
