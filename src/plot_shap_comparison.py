#!/usr/bin/env python3
"""
plot_shap_comparison.py — Compare SHAP feature importances between two XGBoost classifiers.

Modes:
  shift   (default) Wide landscape strip: all features on x-axis, Δ SHAP on y-axis,
           shaded by category (Timing vs Action). Sorted by |delta| within each group.
  grouped Two-panel figure: grouped horizontal bars (top-N) + diverging delta bars.

Usage:
    python plot_shap_comparison.py \
        --dir-a wiki_xgb_ood_frames \
        --dir-b wiki_delayed_xgb_1000ms \
        --label-a "Wiki — no delay" --label-b "Wiki — delayed 1000 ms"

    python plot_shap_comparison.py \
        --dir-a wiki_xgb_ood_frames \
        --dir-b wiki_delayed_xgb_1000ms \
        --mode grouped --top-n 20 --format pdf
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "font.size":    9,
})

BG_COLOUR  = "#FFFFFF"
COLOUR_A   = "#4C72B0"
COLOUR_B   = "#DD8452"
DELTA_POS  = "#2CA02C"
DELTA_NEG  = "#D62728"

# Shift-mode category colours
CLR_TIMING_BG        = "#D6E4F7"
CLR_TIMING_BAR       = "#4C72B0"   # solid blue — timing gained
CLR_TIMING_BAR_LIGHT = "#B8CDE8"   # light blue — timing lost
CLR_ACTION_BG        = "#FDE8D0"
CLR_ACTION_BAR       = "#C8622A"   # solid orange — action gained
CLR_ACTION_BAR_LIGHT = "#F0C8A0"   # light orange — action lost

# Features driven by inter-event timing / dwell / duration
TIMING_FEATURES = {
    "std_iei_ms", "mean_iei_ms", "median_iei_ms",
    "p10_iei_ms", "p90_iei_ms", "iei_trend",
    "mean_click_iei_ms", "std_click_iei_ms",
    "mean_nav_iei_ms",   "std_nav_iei_ms",
    "max_page_dwell_ms", "mean_key_iei_ms",
    "std_key_iei_ms",    "t_first_action_ms",
    "total_duration_s",
}

_PRETTY = {
    "std_iei_ms":           "IEI std",
    "mean_click_iei_ms":    "Click IEI mean",
    "t_first_action_ms":    "Time to 1st action",
    "structural_key_ratio": "Structural key ratio",
    "max_page_dwell_ms":    "Max page dwell",
    "p90_iei_ms":           "IEI p90",
    "p10_iei_ms":           "IEI p10",
    "mean_nav_iei_ms":      "Nav IEI mean",
    "click_x_std":          "Click X std",
    "std_click_iei_ms":     "Click IEI std",
    "mean_iei_ms":          "IEI mean",
    "median_iei_ms":        "IEI median",
    "iei_trend":            "IEI trend",
    "std_nav_iei_ms":       "Nav IEI std",
    "mean_key_iei_ms":      "Key IEI mean",
    "std_key_iei_ms":       "Key IEI std",
    "total_duration_s":     "Total duration",
    "click_y_std":          "Click Y std",
    "click_bbox_area_frac": "Click bbox area",
    "click_top_frac":       "Click top frac",
    "n_clicks":             "# clicks",
    "n_navigations":        "# navigations",
    "n_events_total":       "# events",
    "n_scrolls":            "# scrolls",
    "n_keydowns":           "# keydowns",
    "n_focus":              "# focus",
    "n_link_clicks":        "# link clicks",
    "n_deep_scrolls":       "# deep scrolls",
    "n_unique_domains":     "Unique domains",
    "page_count":           "Page count",
    "link_click_ratio":     "Link click ratio",
    "nav_to_click_ratio":   "Nav:click ratio",
    "scroll_to_click_ratio":"Scroll:click ratio",
    "actions_per_page":     "Actions/page",
    "keydowns_per_page":    "Keydowns/page",
    "focus_per_page":       "Focus/page",
    "popstate_ratio":       "Popstate ratio",
    "scroll_reversals":     "Scroll reversals",
    "max_scroll_pct":       "Max scroll %",
    "mean_scroll_pct":      "Mean scroll %",
    "mean_exit_scroll_pct": "Exit scroll %",
}


def _pretty(name: str) -> str:
    return _PRETTY.get(name, name.replace("_", " "))


def _load_shap(traces_dir: Path, subdir: str, clf: str) -> dict[str, float]:
    path = traces_dir / "classifiers" / subdir / "results.json"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")
    with open(path) as f:
        r = json.load(f)
    shap = ((r.get("models") or {}).get(clf) or {}).get("shap_importances")
    if not shap:
        sys.exit(f"ERROR: no shap_importances for {clf} in {path}.")
    return {k: float(v) for k, v in shap.items()}


def _default_label(d: str) -> str:
    d = d.replace("wiki_", "Wiki — ").replace("xgb_", "").replace("_xgb", "")
    d = d.replace("delayed_", "delayed ").replace("ood_", "OOD: ")
    d = d.replace("_", " ").strip(" —").strip()
    return d


# ── Shift mode ────────────────────────────────────────────────────────────────

def _plot_shift(shap_a, shap_b, label_a, label_b, clf, fmt, out_path,
                top_n_per_group=10, legend_title=None):
    all_features = sorted(set(shap_a) | set(shap_b))

    def _sort_by_abs_delta(feats):
        return sorted(feats,
                      key=lambda f: abs(shap_b.get(f, 0.0) - shap_a.get(f, 0.0)),
                      reverse=True)

    timing  = _sort_by_abs_delta([f for f in all_features if f in TIMING_FEATURES])
    action  = _sort_by_abs_delta([f for f in all_features if f not in TIMING_FEATURES])

    # Keep only top-N per group
    timing  = timing[:top_n_per_group]
    action  = action[:top_n_per_group]
    features = timing + action

    deltas  = np.array([shap_b.get(f, 0.0) - shap_a.get(f, 0.0) for f in features])
    xlabels = [_pretty(f) for f in features]
    x       = np.arange(len(features))

    n_timing = len(timing)
    n_action = len(action)

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(BG_COLOUR)
    ax.set_facecolor(BG_COLOUR)

    # Bars: solid opaque fill for gained; distinct light colour + hatch for lost
    bar_colours       = [CLR_TIMING_BAR       if f in TIMING_FEATURES else CLR_ACTION_BAR
                         for f in features]
    bar_colours_light = [CLR_TIMING_BAR_LIGHT if f in TIMING_FEATURES else CLR_ACTION_BAR_LIGHT
                         for f in features]
    for xi, d, c, cl in zip(x, deltas, bar_colours, bar_colours_light):
        if d >= 0:
            ax.bar(xi, d, width=0.65, color=c,
                   edgecolor="white", linewidth=0.6, zorder=3)
        else:
            ax.bar(xi, d, width=0.65, facecolor=cl,
                   edgecolor=c, linewidth=1.3, hatch="////", zorder=3)

    # Separator between groups
    if n_timing and n_action:
        ax.axvline(n_timing - 0.5, color="#cccccc", linewidth=1.0, zorder=2)

    # Zero baseline
    ax.axhline(0, color="#333333", linewidth=1.0, zorder=4)

    # Category header labels
    y_top = ax.get_ylim()[1]
    if n_timing:
        ax.text((n_timing - 1) / 2, y_top, "Timing features",
                ha="center", va="bottom", fontsize=15, fontweight="bold",
                color=CLR_TIMING_BAR, clip_on=False)
    if n_action:
        ax.text(n_timing + (n_action - 1) / 2, y_top, "Action features",
                ha="center", va="bottom", fontsize=15, fontweight="bold",
                color=CLR_ACTION_BAR, clip_on=False)

    # Value labels
    y_range = deltas.max() - deltas.min() if len(deltas) > 1 else 1.0
    pad = max(abs(y_range) * 0.022, 0.006)
    for xi, d in enumerate(deltas):
        if d >= 0:
            ax.text(xi, d + pad, f"+{d:.2f}", ha="center", va="bottom",
                    fontsize=10, color="#222222")
        else:
            ax.text(xi, d - pad, f"{d:.2f}", ha="center", va="top",
                    fontsize=10, color="#222222")

    # X-axis — remove tick marks (eliminates apostrophe rendering artefact)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=38, ha="right", fontsize=12)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_xlim(-0.7, len(features) - 0.3)

    # Y-axis
    ax.set_ylabel("Δ Mean |SHAP|", fontsize=15, fontweight="bold", labelpad=10)
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5,
                  color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend — framed, lower right; title carries the comparison context
    leg = ax.legend(
        handles=[
            mpatches.Patch(facecolor=CLR_TIMING_BAR, edgecolor="white",
                           label="Timing — gained (after delay)"),
            mpatches.Patch(facecolor=CLR_TIMING_BAR_LIGHT, edgecolor=CLR_TIMING_BAR,
                           linewidth=1.3, hatch="////",
                           label="Timing — lost (after delay)"),
            mpatches.Patch(facecolor=CLR_ACTION_BAR, edgecolor="white",
                           label="Action — gained (after delay)"),
            mpatches.Patch(facecolor=CLR_ACTION_BAR_LIGHT, edgecolor=CLR_ACTION_BAR,
                           linewidth=1.3, hatch="////",
                           label="Action — lost (after delay)"),
        ],
        title=legend_title or f"{label_a}  →  {label_b}",
        loc="lower right", fontsize=11, frameon=True,
        framealpha=1.0, edgecolor="#cccccc", fancybox=False,
        handlelength=1.8, handleheight=1.4,
    )
    leg.get_title().set_fontsize(11)
    leg.get_title().set_fontweight("bold")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Grouped mode (original 2-panel) ──────────────────────────────────────────

def _plot_grouped(shap_a, shap_b, label_a, label_b, clf, top_n, sort_by, fmt, out_path):
    all_features = sorted(set(shap_a) | set(shap_b))

    if sort_by == "a":
        ranked = sorted(all_features, key=lambda f: shap_a.get(f, 0.0), reverse=True)
    elif sort_by == "b":
        ranked = sorted(all_features, key=lambda f: shap_b.get(f, 0.0), reverse=True)
    elif sort_by == "avg":
        ranked = sorted(all_features,
                        key=lambda f: (shap_a.get(f, 0.0) + shap_b.get(f, 0.0)) / 2,
                        reverse=True)
    else:
        ranked = sorted(all_features,
                        key=lambda f: abs(shap_b.get(f, 0.0) - shap_a.get(f, 0.0)),
                        reverse=True)

    features_plot = list(reversed(ranked[:top_n]))
    vals_a  = np.array([shap_a.get(f, 0.0) for f in features_plot])
    vals_b  = np.array([shap_b.get(f, 0.0) for f in features_plot])
    deltas  = vals_b - vals_a
    ylabels = [_pretty(f) for f in features_plot]

    n     = len(features_plot)
    bar_h = 0.38
    y     = np.arange(n)

    fig_h = max(6, n * 0.52 + 2)
    fig, (ax_bars, ax_delta) = plt.subplots(1, 2, figsize=(14, fig_h))
    fig.patch.set_facecolor(BG_COLOUR)
    for ax in (ax_bars, ax_delta):
        ax.set_facecolor(BG_COLOUR)

    ax_bars.barh(y + bar_h / 2, vals_a, height=bar_h,
                 color=COLOUR_A, alpha=0.88, zorder=3)
    ax_bars.barh(y - bar_h / 2, vals_b, height=bar_h,
                 color=COLOUR_B, alpha=0.88, zorder=3)
    ax_bars.set_yticks(y)
    ax_bars.set_yticklabels(ylabels, fontsize=9)
    ax_bars.set_xlabel("Mean |SHAP| value", fontsize=10)
    ax_bars.set_title("Feature importance", fontsize=13, fontweight="bold", pad=8)
    ax_bars.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax_bars.set_axisbelow(True)
    for spine in ax_bars.spines.values():
        spine.set_visible(False)
    ax_bars.set_ylim(-0.5, n - 0.5)
    ax_bars.legend(loc="lower right", fontsize=8, frameon=False,
                   handles=[mpatches.Patch(color=COLOUR_A, label=label_a),
                             mpatches.Patch(color=COLOUR_B, label=label_b)])
    x_max = max(vals_a.max(), vals_b.max())
    pad   = x_max * 0.01
    for xi, (va, vb) in enumerate(zip(vals_a, vals_b)):
        longer_v = max(va, vb)
        ax_bars.text(longer_v + pad, xi, f"{longer_v:.3f}",
                     va="center", ha="left", fontsize=6.5, color="#444")

    colours_d = [DELTA_POS if d >= 0 else DELTA_NEG for d in deltas]
    ax_delta.barh(y, deltas, height=0.55, color=colours_d, alpha=0.85, zorder=3)
    ax_delta.axvline(0, color="#555555", linewidth=0.8, zorder=4)
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels(ylabels, fontsize=9)
    ax_delta.set_xlabel(f"Δ SHAP  ({label_b} − {label_a})", fontsize=10)
    ax_delta.set_title("Importance shift  (B − A)", fontsize=13, fontweight="bold", pad=8)
    ax_delta.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax_delta.set_axisbelow(True)
    for spine in ax_delta.spines.values():
        spine.set_visible(False)
    ax_delta.set_ylim(-0.5, n - 0.5)
    d_pad = (deltas.max() - deltas.min()) * 0.01 + 1e-9
    for xi, d in enumerate(deltas):
        ha  = "left"  if d >= 0 else "right"
        off = d_pad   if d >= 0 else -d_pad
        ax_delta.text(d + off, xi, f"{d:+.3f}",
                      va="center", ha=ha, fontsize=6.5, color="#444")
    ax_delta.legend(loc="lower right", fontsize=8, frameon=False,
                    handles=[mpatches.Patch(color=DELTA_POS, label=f"{label_b} gained"),
                              mpatches.Patch(color=DELTA_NEG, label=f"{label_b} lost")])

    fig.suptitle(
        f"SHAP importance: {label_a}  vs  {label_b}  ({clf}, top {top_n})",
        fontsize=14, fontweight="bold", color="#2a2a2a", y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare SHAP feature importances between two classifiers."
    )
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"))
    parser.add_argument("--dir-a",      required=True)
    parser.add_argument("--dir-b",      required=True)
    parser.add_argument("--label-a",    default=None)
    parser.add_argument("--label-b",    default=None)
    parser.add_argument("--classifier", default="XGBoost")
    parser.add_argument("--mode",       choices=["shift", "grouped"], default="shift",
                        help="Plot mode: shift (default) or grouped (old 2-panel).")
    parser.add_argument("--top-n",            type=int, default=20,
                        help="Top-N features (grouped mode only).")
    parser.add_argument("--top-n-per-group",  type=int, default=10,
                        help="Top-N features per category in shift mode (default: 10).")
    parser.add_argument("--legend-title",     default=None,
                        help="Override legend title in shift mode.")
    parser.add_argument("--sort-by",    choices=["a", "b", "avg", "delta"], default="a",
                        help="Sort key for grouped mode.")
    parser.add_argument("--format",     choices=["png", "pdf"], default="png")
    parser.add_argument("--out",        type=Path, default=None)
    args = parser.parse_args()

    label_a = args.label_a or _default_label(args.dir_a)
    label_b = args.label_b or _default_label(args.dir_b)

    shap_a = _load_shap(args.traces_dir, args.dir_a, args.classifier)
    shap_b = _load_shap(args.traces_dir, args.dir_b, args.classifier)

    fmt = args.format
    if args.out:
        out = args.out.with_suffix(f".{fmt}")
    elif args.mode == "shift":
        out = Path("figures") / f"shap_shift_{args.dir_a}_vs_{args.dir_b}.{fmt}"
    else:
        out = Path("figures") / f"shap_comparison_{args.dir_a}_vs_{args.dir_b}.{fmt}"

    if args.mode == "shift":
        _plot_shift(shap_a, shap_b, label_a, label_b, args.classifier, fmt, out,
                    top_n_per_group=args.top_n_per_group,
                    legend_title=args.legend_title)
    else:
        _plot_grouped(shap_a, shap_b, label_a, label_b, args.classifier,
                      args.top_n, args.sort_by, fmt, out)


if __name__ == "__main__":
    main()
