#!/usr/bin/env python3
"""
plot_delay_attack.py — Robustness to timing-delay attacks, XGBoost only.

Left panel : Test-time jitter — pre-trained classifier, test data corrupted.
Right panel: Adversarial retraining — classifier retrained on corrupted data.

Each line = one training dataset, coloured consistently with plot_combined.py.
X-axis: max random delay injected (ms). Y-axis: Macro F1.

Usage:
    python plot_delay_attack.py
    python plot_delay_attack.py --traces-dir /path/to/traces
    python plot_delay_attack.py --split ood
    python plot_delay_attack.py --out fig_delay.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLF    = "XGBoost"
DELAYS = [500, 1000, 2000, 5000]   # ms values used in experiments

DATASETS = [
    # (key, label, color, baseline_tag, jitter_pattern, delayed_pattern)
    ("wiki",     "2WikiMultiHop", "#2196F3",
     "wiki_ood_all",     "wiki_ood_all_jitter_{N}ms",     "wiki_delayed_xgb_{N}ms"),
    ("frames",   "FRAMES",        "#FF9800",
     "frames_ood_all",   "frames_ood_all_jitter_{N}ms",   "frames_delayed_xgb_{N}ms"),
    ("webshop",  "WebShop",       "#4CAF50",
     "webshop_ood_all",  "webshop_ood_all_jitter_{N}ms",  "webshop_delayed_xgb_{N}ms"),
    ("deepshop", "DeepShop",      "#9C27B0",
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
    args = parser.parse_args()

    clf_root = args.traces_dir / "classifiers"
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 5))

    for _key, label, color, baseline_tag, jitter_pat, delayed_pat in DATASETS:
        baseline_f1 = _load_f1(clf_root / baseline_tag, args.split)

        # ── left: test-time jitter ─────────────────────────────────────────────
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
            ax_l.plot(jitter_xs, jitter_ys, marker="o", color=color,
                      label=label, linewidth=1.8)

        # ── right: adversarial retraining ──────────────────────────────────────
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
            ax_r.plot(delayed_xs, delayed_ys, marker="o", color=color,
                      label=label, linewidth=1.8)

    x_ticks = [0] + DELAYS
    for ax, title in ((ax_l, "Timing attack (no retraining)"), (ax_r, "Retrained under timing attack")):
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks], rotation=30, ha="right")
        ax.set_xlabel("Max random delay injected (ms)")
        ax.set_ylabel(f"Macro F1 (XGBoost)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.8, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(title)

    handles, labels = ax_l.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_r.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(DATASETS),
               bbox_to_anchor=(0.5, -0.08), frameon=False)

    plt.tight_layout()
    out = args.out or (clf_root / "delay_attack.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
