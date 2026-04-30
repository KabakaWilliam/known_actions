#!/usr/bin/env python3
"""
plot_learning_curve.py — Training sample efficiency: macro F1 vs. traces per agent.

Reads results.json files produced by train_learning_curve.sh and plots how
classifier performance scales with the number of training traces per agent.

Usage:
    python plot_learning_curve.py
    python plot_learning_curve.py --tags wiki_2_frames frames_2_wiki
    python plot_learning_curve.py --traces-dir /path/to/traces
    python plot_learning_curve.py --classifiers RandomForest XGBoost
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLF_COLORS = {
    "RandomForest": "#2196F3",
    "XGBoost":      "#FF9800",
    "LR_L2":        "#4CAF50",
    "LR_Lasso":     "#9C27B0",
    "LSTM":         "#F44336",
}
CLF_ORDER = ["RandomForest", "XGBoost", "LR_L2", "LR_Lasso", "LSTM"]

TAG_LABELS = {
    "wiki_2_frames":       "2WikiMultiHop",
    "frames_2_wiki":       "FRAMES",
    "webshop_2_deepshop":  "WebShop",
    "deepshop_2_webshop":  "DeepShop",
}


def _extract_n(folder_name: str) -> int | None:
    """Parse trailing _n<N> or _nall from folder name. Returns None for 'all'."""
    m = re.search(r"_n(\d+|all)$", folder_name)
    if not m:
        return None
    v = m.group(1)
    return None if v == "all" else int(v)


def load_curve(lc_dir: Path, base_tag: str, n_agents: int) -> dict:
    """Return {clf: [(n_train_per_agent, test_f1, ood_f1), ...]} sorted by n.

    Runs whose test-set size differs from the majority are dropped — this catches
    stale runs produced with a different agent list.
    """
    from collections import Counter
    pattern = f"{base_tag}_n*"
    runs = sorted(lc_dir.glob(pattern))

    raw: list[tuple] = []  # (n_per_agent, test_n, clf, test_f1, ood_f1)
    for run_dir in runs:
        rpath = run_dir / "results.json"
        if not rpath.exists():
            continue
        with open(rpath) as f:
            res = json.load(f)

        n_train_total = res.get("n_episodes", {}).get("train", 0)
        test_n        = res.get("n_episodes", {}).get("test", 0)
        n_per_agent   = n_train_total / n_agents if n_agents else n_train_total

        for clf in CLF_ORDER:
            model = res.get("models", {}).get(clf)
            if model is None:
                continue
            test_f1 = model.get("test_report", {}).get("macro avg", {}).get("f1-score")
            ood_reports = model.get("ood_reports", {})
            ood_f1s = [
                r.get("macro avg", {}).get("f1-score")
                for r in ood_reports.values()
                if r.get("macro avg")
            ]
            ood_f1 = float(np.mean(ood_f1s)) if ood_f1s else None
            if test_f1 is not None:
                raw.append((n_per_agent, test_n, clf, float(test_f1), ood_f1))

    modal_test_n = Counter(r[1] for r in raw).most_common(1)[0][0] if raw else 0
    curves: dict[str, list] = {c: [] for c in CLF_ORDER}
    for n_per_agent, test_n, clf, test_f1, ood_f1 in raw:
        if test_n == modal_test_n:
            curves[clf].append((n_per_agent, test_f1, ood_f1))

    for clf in curves:
        curves[clf].sort(key=lambda x: x[0])

    return curves


def _to_pct(curves: dict, max_n: float) -> dict:
    """Normalise raw trace counts to percentage of the maximum (100 = all traces)."""
    if not max_n:
        return curves
    return {
        clf: [(n / max_n * 100, te, ood) for n, te, ood in pts]
        for clf, pts in curves.items()
    }


def plot_tag(ax_test, ax_ood, curves: dict, title: str) -> None:
    # normalise x axis to % of training data
    all_ns = [p[0] for pts in curves.values() for p in pts]
    max_n = max(all_ns) if all_ns else 1.0
    curves = _to_pct(curves, max_n)

    for clf in CLF_ORDER:
        pts = curves.get(clf, [])
        if not pts:
            continue
        xs     = [p[0] for p in pts]
        ys_te  = [p[1] for p in pts]
        ys_ood = [p[2] for p in pts if p[2] is not None]
        xs_ood = [p[0] for p in pts if p[2] is not None]

        color = CLF_COLORS.get(clf, "gray")
        ax_test.plot(xs, ys_te, marker="o", color=color, label=clf, linewidth=1.8)
        if ys_ood:
            ax_ood.plot(xs_ood, ys_ood, marker="o", color=color, label=clf, linewidth=1.8)

    for ax in (ax_test, ax_ood):
        ax.set_xlabel("% of training traces")
        ax.set_xlim(0, 105)
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.8, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    ax_test.set_title(f"{title}\nTest split")
    ax_ood.set_title(f"{title}\nOOD split")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"))
    parser.add_argument("--tags", nargs="+",
                        default=["wiki_2_frames", "frames_2_wiki",
                                 "webshop_2_deepshop", "deepshop_2_webshop"])
    parser.add_argument("--n-agents", type=int, default=14,
                        help="Number of agent classes (used to compute per-agent trace count).")
    parser.add_argument("--classifiers", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    lc_dir = args.traces_dir / "classifiers" / "learning_curve"
    if not lc_dir.exists():
        print(f"No learning-curve results found at {lc_dir}. Run train_learning_curve.sh first.")
        return

    active_clfs = args.classifiers or CLF_ORDER

    n_tags = len(args.tags)
    fig, axes = plt.subplots(2, n_tags, figsize=(6 * n_tags, 10), squeeze=False)

    for col, tag in enumerate(args.tags):
        curves = load_curve(lc_dir, tag, args.n_agents)
        # filter to requested classifiers
        curves = {c: v for c, v in curves.items() if c in active_clfs}
        label  = TAG_LABELS.get(tag, tag)
        plot_tag(axes[0][col], axes[1][col], curves, label)

    # shared legend
    handles = [
        plt.Line2D([0], [0], color=CLF_COLORS.get(c, "gray"), marker="o", label=c)
        for c in active_clfs if c in CLF_COLORS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle("Learning Curve: Training Traces per Agent vs. Classifier Performance",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    out = args.out or (args.traces_dir / "classifiers" / "learning_curve" / "learning_curve.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
