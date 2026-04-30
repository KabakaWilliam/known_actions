#!/usr/bin/env python3
"""
plot_identification_speed.py — Early identification: macro F1 vs. DOM events observed.

Reads results.json files produced by identification_speed.sh and plots how classifier
F1 grows as more DOM events are revealed at test time.

Usage:
    python plot_identification_speed.py
    python plot_identification_speed.py --tags wiki_2_frames frames_2_wiki
    python plot_identification_speed.py --traces-dir /path/to/traces
    python plot_identification_speed.py --classifiers XGBoost RandomForest
"""

import argparse
import json
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


def load_speed(speed_dir: Path, tag: str) -> tuple[dict, dict]:
    """Return ({clf: [(n_events, test_f1, ood_f1), ...]}, mean_n_events dict)."""
    rpath = speed_dir / tag / "results.json"
    if not rpath.exists():
        print(f"[WARN] No results at {rpath} — skipping.")
        return {}, {}
    with open(rpath) as f:
        res = json.load(f)

    prefix_curve  = res.get("prefix_curve") or {}
    n_events_data = prefix_curve.get("n_events", {})
    mean_n        = res.get("mean_n_events", {})

    # "null" key = classifier saw all events; use mean trace length as x position
    null_x = mean_n.get("test") or 0

    curves: dict[str, list] = {}
    for clf, buckets in n_events_data.items():
        pts = []
        for key, entry in buckets.items():
            n       = null_x if key == "null" else int(key)
            test_f1 = (entry.get("test") or {}).get("macro_f1")
            ood_f1s = [
                v["macro_f1"]
                for v in (entry.get("ood") or {}).values()
                if v.get("macro_f1") is not None
            ]
            ood_f1 = float(np.mean(ood_f1s)) if ood_f1s else None
            if test_f1 is not None:
                pts.append((n, float(test_f1), ood_f1))
        pts.sort(key=lambda x: x[0])
        curves[clf] = pts

    return curves, mean_n


def plot_tag(ax_test, ax_ood, curves: dict, title: str, mean_n: dict) -> None:
    for clf in CLF_ORDER:
        pts = curves.get(clf, [])
        if not pts:
            continue
        xs     = [p[0] for p in pts]
        ys_te  = [p[1] for p in pts]
        xs_ood = [p[0] for p in pts if p[2] is not None]
        ys_ood = [p[2] for p in pts if p[2] is not None]

        color = CLF_COLORS.get(clf, "gray")
        ax_test.plot(xs, ys_te, marker="o", color=color, label=clf, linewidth=1.8)
        if ys_ood:
            ax_ood.plot(xs_ood, ys_ood, marker="o", color=color, label=clf, linewidth=1.8)

    for ax, split_key in ((ax_test, "test"), (ax_ood, "ood")):
        mn = mean_n.get(split_key)
        if mn:
            ax.axvline(mn, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("DOM events observed")
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
    parser.add_argument("--classifiers", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    speed_dir = args.traces_dir / "classifiers" / "identification_speed"
    if not speed_dir.exists():
        print(f"No results at {speed_dir}. Run identification_speed.sh first.")
        return

    active_clfs = args.classifiers or CLF_ORDER

    n_tags = len(args.tags)
    fig, axes = plt.subplots(2, n_tags, figsize=(6 * n_tags, 10), squeeze=False)

    for col, tag in enumerate(args.tags):
        curves, mean_n = load_speed(speed_dir, tag)
        curves = {c: v for c, v in curves.items() if c in active_clfs}
        label  = TAG_LABELS.get(tag, tag)
        plot_tag(axes[0][col], axes[1][col], curves, label, mean_n)

    handles = [
        plt.Line2D([0], [0], color=CLF_COLORS.get(c, "gray"), marker="o", label=c)
        for c in active_clfs if c in CLF_COLORS
    ]
    # dashed gray line represents mean full-trace length marker
    handles.append(plt.Line2D([0], [0], color="gray", linestyle="--",
                               linewidth=1, alpha=0.5, label="mean full trace"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle("Identification Speed: Macro F1 vs. DOM Events Observed",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    out = args.out or (speed_dir / "identification_speed.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
