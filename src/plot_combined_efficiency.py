#!/usr/bin/env python3
"""
plot_combined.py — Learning curve + identification speed side-by-side, XGBoost only.

Left panel : macro F1 vs. % of training traces (learning curve)
Right panel: macro F1 vs. DOM events observed at test time (identification speed)

Each line = one training-dataset direction, coloured consistently across both panels.

Usage:
    python plot_combined.py
    python plot_combined.py --traces-dir /path/to/traces
    python plot_combined.py --split ood
    python plot_combined.py --out fig_combined.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLF = "XGBoost"

# One colour + marker shape per dataset, shared across both panels
DATASET_COLORS = {
    "wiki_2_frames":       "#2196F3",   # blue
    "frames_2_wiki":       "#FF9800",   # orange
    "webshop_2_deepshop":  "#4CAF50",   # green
    "deepshop_2_webshop":  "#9C27B0",   # purple
}

DATASET_MARKERS = {
    "wiki_2_frames":       "o",   # circle
    "frames_2_wiki":       "s",   # square
    "webshop_2_deepshop":  "^",   # triangle
    "deepshop_2_webshop":  "D",   # diamond
}

TAG_LABELS = {
    "wiki_2_frames":       "2WikiMultiHop",
    "frames_2_wiki":       "FRAMES",
    "webshop_2_deepshop":  "WebShop",
    "deepshop_2_webshop":  "DeepShop",
}

DEFAULT_TAGS = ["wiki_2_frames", "frames_2_wiki", "webshop_2_deepshop", "deepshop_2_webshop"]


# ── learning-curve loader (mirrors plot_learning_curve.py) ────────────────────

def _load_lc(lc_dir: Path, tag: str, n_agents: int, split: str) -> list[tuple[float, float]]:
    """Return [(raw_n_per_agent, f1), ...] for XGBoost on the requested split.

    Runs whose test-set size differs from the majority are dropped — this catches
    stale runs produced with a different agent list.
    """
    from collections import Counter
    raw = []
    for run_dir in sorted(lc_dir.glob(f"{tag}_n*")):
        rpath = run_dir / "results.json"
        if not rpath.exists():
            continue
        res = json.load(open(rpath))
        n_total = res.get("n_episodes", {}).get("train", 0)
        test_n  = res.get("n_episodes", {}).get("test", 0)
        n_per   = n_total / n_agents if n_agents else n_total
        model   = res.get("models", {}).get(CLF)
        if model is None:
            continue
        if split == "test":
            f1 = model.get("test_report", {}).get("macro avg", {}).get("f1-score")
        else:
            ood_f1s = [
                r.get("macro avg", {}).get("f1-score")
                for r in model.get("ood_reports", {}).values()
                if r.get("macro avg")
            ]
            f1 = float(np.mean(ood_f1s)) if ood_f1s else None
        if f1 is not None:
            raw.append((n_per, float(f1), test_n))
    if not raw:
        return []
    modal_test_n = Counter(r[2] for r in raw).most_common(1)[0][0]
    pts = [(n, f1) for n, f1, tn in raw if tn == modal_test_n]
    pts.sort()
    return pts


def _to_pct(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not pts:
        return pts
    max_n = max(p[0] for p in pts)
    return [(n / max_n * 100, f1) for n, f1 in pts] if max_n else pts


# ── identification-speed loader (mirrors plot_identification_speed.py) ─────────

def _load_speed(speed_dir: Path, tag: str, split: str) -> tuple[list[tuple[float, float]], float]:
    """Return ([(n_events, f1), ...], mean_full_trace_n) for XGBoost on the requested split."""
    rpath = speed_dir / tag / "results.json"
    if not rpath.exists():
        return [], 0.0
    res           = json.load(open(rpath))
    prefix_curve  = res.get("prefix_curve") or {}
    n_events_data = prefix_curve.get("n_events", {})
    mean_n        = res.get("mean_n_events", {})
    null_x        = mean_n.get(split) or mean_n.get("test") or 0.0

    buckets = n_events_data.get(CLF, {})
    pts = []
    for key, entry in buckets.items():
        n = null_x if key == "null" else int(key)
        # drop fixed-prefix points that meet or exceed mean trace length —
        # they are nearly equivalent to the null point and would appear after
        # the red star on the x-axis, making the line look like it continues
        # past "full trace seen"
        if key != "null" and null_x and n >= null_x:
            continue
        if split == "test":
            f1 = (entry.get("test") or {}).get("macro_f1")
        else:
            ood_f1s = [
                v["macro_f1"]
                for v in (entry.get("ood") or {}).values()
                if v.get("macro_f1") is not None
            ]
            f1 = float(np.mean(ood_f1s)) if ood_f1s else None
        if f1 is not None:
            pts.append((float(n), float(f1)))
    pts.sort()
    return pts, float(null_x)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"))
    parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    parser.add_argument("--n-agents", type=int, default=14)
    parser.add_argument("--split", choices=["test", "ood"], default="test",
                        help="Which evaluation split to plot (default: test).")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    lc_dir    = args.traces_dir / "classifiers" / "learning_curve"
    speed_dir = args.traces_dir / "classifiers" / "identification_speed"

    fig, (ax_lc, ax_sp) = plt.subplots(1, 2, figsize=(10, 5))

    for tag in args.tags:
        color  = DATASET_COLORS.get(tag, "gray")
        marker = DATASET_MARKERS.get(tag, "o")
        label  = TAG_LABELS.get(tag, tag)

        # ── learning curve ────────────────────────────────────────────────────
        lc_pts = _to_pct(_load_lc(lc_dir, tag, args.n_agents, args.split))
        if lc_pts:
            xs, ys = zip(*lc_pts)
            ax_lc.plot(xs, ys, marker=marker, color=color, label=label,
                       linewidth=1.8, markersize=6)

        # ── identification speed ───────────────────────────────────────────────
        sp_pts, null_x = _load_speed(speed_dir, tag, args.split)
        if sp_pts:
            xs, ys = zip(*sp_pts)
            ax_sp.plot(xs, ys, marker=marker, color=color, label=label,
                       linewidth=1.8, markersize=6)
            # red star + drop line at full-trace (mean trace length) point
            if null_x:
                null_y = next((y for x, y in sp_pts if x == null_x), None)
                if null_y is not None:
                    ax_sp.plot([null_x, null_x], [0, null_y],
                               color="red", linewidth=0.8, linestyle="--",
                               alpha=0.4, zorder=3)
                    ax_sp.plot(null_x, null_y, marker="*", color="red",
                               markersize=12, markeredgecolor="white",
                               markeredgewidth=0.5, zorder=5)

    for ax in (ax_lc, ax_sp):
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.8, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_aspect("auto")

    ax_lc.set_xlim(0, 105)
    ax_lc.set_xlabel("% of training traces")
    ax_lc.set_ylabel("Macro F1 (XGBoost)")
    ax_lc.set_title("Sample efficiency")

    ax_sp.set_xlabel("Actions observed at test time")
    ax_sp.set_ylabel("Macro F1 (XGBoost)")
    ax_sp.set_title("Early identification")

    # shared legend: dataset lines + red star explanation
    handles, labels = ax_lc.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], marker="*", color="red", linestyle="none",
                               markersize=10, label="Full trace (mean length)"))
    labels.append("Full trace (mean length)")
    fig.legend(handles, labels, loc="lower center", ncol=len(args.tags) + 1,
               bbox_to_anchor=(0.5, -0.06), frameon=False)

    plt.tight_layout()

    out = args.out or (args.traces_dir / "classifiers" / "combined.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
