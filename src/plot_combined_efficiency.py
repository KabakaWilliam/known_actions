#!/usr/bin/env python3
"""
plot_combined_efficiency.py — Sample efficiency + early identification, XGBoost only.

NeurIPS-style figure matching efficiency_figs.ipynb.

Left panel : macro F1 vs. % of training traces (learning curve)
Right panel: macro F1 vs. DOM events observed at test time (identification speed)

Usage:
    python plot_combined_efficiency.py
    python plot_combined_efficiency.py --split ood
    python plot_combined_efficiency.py --format pdf
    python plot_combined_efficiency.py --out fig.pdf --format pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

CLF = "XGBoost"

# NeurIPS-friendly defaults (mirrors efficiency_figs.ipynb)
plt.rcParams.update({
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

# Wong (2011) colorblind-safe palette
DATASET_COLORS = {
    "wiki_2_frames":       "#0072B2",
    "frames_2_wiki":       "#E69F00",
    "webshop_2_deepshop":  "#009E73",
    "deepshop_2_webshop":  "#CC79A7",
}

DATASET_MARKERS = {
    "wiki_2_frames":       "o",
    "frames_2_wiki":       "s",
    "webshop_2_deepshop":  "^",
    "deepshop_2_webshop":  "D",
}

TAG_LABELS = {
    "wiki_2_frames":       "2WikiMultiHop",
    "frames_2_wiki":       "FRAMES",
    "webshop_2_deepshop":  "WebShop",
    "deepshop_2_webshop":  "DeepShop",
}

DEFAULT_TAGS = ["wiki_2_frames", "frames_2_wiki", "webshop_2_deepshop", "deepshop_2_webshop"]


# ── learning-curve loader ─────────────────────────────────────────────────────

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
        res    = json.load(open(rpath))
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


# ── identification-speed loader ───────────────────────────────────────────────

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
        # drop fixed-prefix points that meet or exceed mean trace length
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"))
    parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    parser.add_argument("--n-agents", type=int, default=14)
    parser.add_argument("--split", choices=["test", "ood"], default="test",
                        help="Which evaluation split to plot (default: test).")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--format", choices=["png", "pdf"], default="png",
                        help="Output format (default: png). Ignored if --out specifies an extension.")
    args = parser.parse_args()

    # Ensure fonts are properly embedded in PDFs
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    lc_dir    = args.traces_dir / "classifiers" / "learning_curve"
    speed_dir = args.traces_dir / "classifiers" / "identification_speed"

    fig, (ax_lc, ax_sp) = plt.subplots(
        1, 2, figsize=(7.2, 2.85), sharey=True, constrained_layout=False
    )

    for tag in args.tags:
        color  = DATASET_COLORS.get(tag, "gray")
        marker = DATASET_MARKERS.get(tag, "o")
        label  = TAG_LABELS.get(tag, tag)

        # ── learning curve ────────────────────────────────────────────────────
        lc_pts = _to_pct(_load_lc(lc_dir, tag, args.n_agents, args.split))
        if lc_pts:
            xs, ys = zip(*lc_pts)
            ax_lc.plot(xs, ys, marker=marker, color=color, label=label,
                       markersize=4.6, markeredgewidth=0.7)

        # ── identification speed ───────────────────────────────────────────────
        sp_pts, null_x = _load_speed(speed_dir, tag, args.split)
        if sp_pts:
            xs, ys = zip(*sp_pts)
            ax_sp.plot(xs, ys, marker=marker, color=color, label=label,
                       markersize=4.6, markeredgewidth=0.7)
            if null_x:
                null_y = next((y for x, y in sp_pts if x == null_x), None)
                if null_y is not None:
                    ax_sp.scatter([null_x], [null_y], marker="*", s=78,
                                  color=color, edgecolor="white", linewidth=0.45, zorder=5)
                    ax_sp.vlines(null_x, 0, null_y,
                                 colors=color, linestyles=(0, (3, 2)),
                                 linewidth=0.85, alpha=0.45, zorder=0)

    # ── shared axis styling ───────────────────────────────────────────────────
    REF_LS = (0, (1.5, 2.5))
    for ax in (ax_lc, ax_sp):
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(True, axis="both", alpha=0.22)
        ax.axhline(0.8, color="0.45", linewidth=0.9, linestyle=REF_LS, zorder=0)
        ax.text(1.5, 0.815, "0.80 F1", color="0.35", fontsize=8.5, va="bottom")

    ax_lc.set_xlim(0, 105)
    ax_lc.set_xticks([0, 20, 40, 60, 80, 100])
    ax_lc.set_xlabel("Training traces used (%)")
    ax_lc.set_ylabel("Macro F1")
    ax_lc.set_title("A. Sample efficiency", loc="left", fontweight="semibold")

    ax_sp.set_xlim(0, 185)
    ax_sp.set_xticks([0, 25, 50, 75, 100, 125, 150, 175])
    ax_sp.set_xlabel("Observed actions at test time")
    ax_sp.set_title("B. Early identification", loc="left", fontweight="semibold")

    ax_lc.annotate(
        "Most gains appear\nby ~1/3 of traces",
        xy=(34, 0.71), xytext=(45, 0.50),
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.35"),
        fontsize=8.5, color="0.25", ha="left", va="center",
    )

    # ── legend ────────────────────────────────────────────────────────────────
    handles, labels = ax_lc.get_legend_handles_labels()
    star_proxy = mlines.Line2D([0], [0], marker="*", color="none",
                             markerfacecolor="0.25", markeredgecolor="white",
                             markersize=9, label="Full trace")
    handles.append(star_proxy)
    labels.append("Full trace")

    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.14),
               ncol=5, frameon=False, handlelength=1.8, columnspacing=1.4)

    fig.text(0.5, 0.02,
             "Dashed vertical lines mark mean full-trace length for each dataset.",
             ha="center", va="center", fontsize=8.2, color="0.35")

    fig.subplots_adjust(left=0.085, right=0.995, top=0.89, bottom=0.32, wspace=0.16)

    fmt = args.format
    out = args.out or (args.traces_dir / "classifiers" / f"combined.{fmt}")
    out = out.with_suffix(f".{fmt}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=350, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
