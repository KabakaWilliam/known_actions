#!/usr/bin/env python3
"""
plot_importance_x_degradation.py — Feature importance × timing-attack degradation figure.

Produces two panels (together or individually):
  LEFT  — Top-N XGBoost SHAP feature importances; timing features in orange, others in grey.
  RIGHT — XGBoost weighted-F1 vs injected random delay (ms), with two curves:
            RED  — Training poisoning: train AND test corrupted (--poison-tags)
            BLUE — Test-time jitter:   train clean, test corrupted (--jitter-tags)

Usage:
    python plot_importance_x_degradation.py \\
        --baseline-tag wiki_xgb_ood_frames \\
        --poison-tags wiki_delayed_xgb_500ms wiki_delayed_xgb_1000ms \\
                      wiki_delayed_xgb_2000ms wiki_delayed_xgb_5000ms \\
        --jitter-tags wiki_jitter_test_500ms wiki_jitter_test_1000ms \\
                      wiki_jitter_test_2000ms wiki_jitter_test_5000ms

    # Single panel, one curve only:
    python plot_importance_x_degradation.py --baseline-tag wiki_xgb_ood_frames \\
        --poison-tags wiki_delayed_xgb_500ms --plot degradation

Delay values are parsed from tag names automatically (e.g. wiki_delayed_xgb_500ms → 500 ms).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Colours ────────────────────────────────────────────────────────────────────
TIMING_COLOUR    = "#FF9800"   # orange — matches XGBoost colour elsewhere
NONTIMING_COLOUR = "#BDBDBD"   # grey
CURVE_COLOUR     = "#E53935"   # red

# ── Timing feature set ─────────────────────────────────────────────────────────
TIMING_FEATURES = {
    "std_iei_ms", "mean_iei_ms", "median_iei_ms",
    "p10_iei_ms", "p90_iei_ms", "iei_trend",
    "mean_click_iei_ms", "std_click_iei_ms",
    "mean_nav_iei_ms", "std_nav_iei_ms", "max_page_dwell_ms",
    "mean_key_iei_ms", "std_key_iei_ms",
    "t_first_action_ms", "total_duration_s",
}

FEATURE_LABELS = {
    "std_iei_ms":            "IEI std dev",
    "mean_click_iei_ms":     "Click IEI mean",
    "t_first_action_ms":     "Time to first action",
    "max_page_dwell_ms":     "Max page dwell",
    "p90_iei_ms":            "IEI p90",
    "p10_iei_ms":            "IEI p10",
    "mean_iei_ms":           "IEI mean",
    "mean_nav_iei_ms":       "Nav IEI mean",
    "structural_key_ratio":  "Structural key ratio",
    "total_duration_s":      "Session duration",
    "click_x_std":           "Click X spread",
    "click_y_std":           "Click Y spread",
    "std_key_iei_ms":        "Keydown IEI std",
    "std_click_iei_ms":      "Click IEI std",
    "mean_key_iei_ms":       "Keydown IEI mean",
    "click_bbox_area_frac":  "Click bbox coverage",
    "std_nav_iei_ms":        "Nav IEI std",
    "nav_to_click_ratio":    "Nav-to-click ratio",
    "iei_trend":             "IEI trend",
    "actions_per_page":      "Actions per page",
    "keydowns_per_page":     "Keydowns per page",
    "link_click_ratio":      "Link click ratio",
    "n_events_total":        "Total events",
    "median_iei_ms":         "IEI median",
    "n_keydowns":            "Keydown count",
    "n_clicks":              "Click count",
    "focus_per_page":        "Focus per page",
    "n_focus":               "Focus count",
    "click_top_frac":        "Clicks in top 25%",
    "n_navigations":         "Navigation count",
    "scroll_to_click_ratio": "Scroll-to-click ratio",
    "n_link_clicks":         "Link click count",
    "page_count":            "Pages visited",
    "max_scroll_pct":        "Max scroll depth",
    "n_scrolls":             "Scroll count",
    "mean_scroll_pct":       "Mean scroll depth",
    "mean_exit_scroll_pct":  "Exit scroll depth",
    "scroll_reversals":      "Scroll reversals",
    "popstate_ratio":        "Back-nav ratio",
    "n_deep_scrolls":        "Deep scroll count",
    "n_unique_domains":      "Unique domains",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_results(traces_dir: Path, tag: str) -> dict:
    path = traces_dir / "classifiers" / tag / "results.json"
    if not path.exists():
        sys.exit(f"ERROR: results.json not found for tag '{tag}' at {path}")
    with open(path) as f:
        return json.load(f)


def _xgb_weighted_f1(results: dict) -> float | None:
    xgb = (results.get("models") or {}).get("XGBoost") or {}
    tr  = xgb.get("test_report") or {}
    return (tr.get("weighted avg") or {}).get("f1-score")


def _shap_importances(results: dict) -> dict | None:
    xgb = (results.get("models") or {}).get("XGBoost") or {}
    imp = xgb.get("shap_importances")
    if imp:
        return {k: abs(v) for k, v in imp.items()}
    return None


def _parse_delay_ms(tag: str) -> int | None:
    m = re.search(r"_(\d+)ms", tag)
    return int(m.group(1)) if m else None


# ── Plotting ───────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def plot_importance(ax, shap_imp: dict, top_n: int):
    ranked = sorted(shap_imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    ranked = list(reversed(ranked))   # bottom-to-top

    features = [f for f, _ in ranked]
    values   = [v for _, v in ranked]
    labels   = [FEATURE_LABELS.get(f, f) for f in features]
    colours  = [TIMING_COLOUR if f in TIMING_FEATURES else NONTIMING_COLOUR
                for f in features]

    y = np.arange(len(features))
    ax.barh(y, values, color=colours, height=0.6, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean |SHAP|", fontsize=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    _style_ax(ax)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=TIMING_COLOUR,    label="Timing feature"),
        Patch(facecolor=NONTIMING_COLOUR, label="Other feature"),
    ], fontsize=8, loc="lower right", framealpha=0.7)


def _plot_curve(ax, delay_f1, colour, label, annotate_baseline=False):
    delay_f1 = sorted(delay_f1, key=lambda x: x[0])
    xs = [d for d, _ in delay_f1]
    ys = [f for _, f in delay_f1]
    ax.plot(xs, ys, color=colour, linewidth=2, label=label,
            marker="o", markersize=5,
            markerfacecolor="white", markeredgewidth=1.5, markeredgecolor=colour)
    if annotate_baseline and xs:
        ax.annotate(f"{ys[0]:.3f}",
                    xy=(xs[0], ys[0]),
                    xytext=(xs[0] + max(xs) * 0.04, ys[0] + 0.002),
                    fontsize=8.5, color=colour, va="bottom")


def plot_degradation(ax,
                     poison_f1: list[tuple[int, float]] | None,
                     jitter_f1: list[tuple[int, float]] | None):
    has_poison = bool(poison_f1)
    has_jitter = bool(jitter_f1)

    if has_poison:
        _plot_curve(ax, poison_f1, CURVE_COLOUR,
                    "Poisoning (train + test corrupted)",
                    annotate_baseline=not has_jitter)
    if has_jitter:
        _plot_curve(ax, jitter_f1, "#1565C0",
                    "Test-time jitter (train clean)",
                    annotate_baseline=True)

    if has_poison or has_jitter:
        ax.legend(fontsize=8.5, framealpha=0.8, loc="upper right")

    ax.set_xlabel("Max injected delay per gap (ms)", fontsize=10)
    ax.set_ylabel("Weighted F1", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.text(0.97, 0.97, "XGBoost", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#333333", fontweight="bold")
    _style_ax(ax)


# ── Entry points ───────────────────────────────────────────────────────────────

def _collect_delay_f1(traces_dir, tags, baseline_f1=None):
    points = []
    if baseline_f1 is not None:
        points.append((0, baseline_f1))
    for tag in tags:
        delay = _parse_delay_ms(tag)
        if delay is None:
            print(f"[WARN] Cannot parse delay from tag '{tag}' — skipping.")
            continue
        res = _load_results(traces_dir, tag)
        f1  = _xgb_weighted_f1(res)
        if f1 is None:
            print(f"[WARN] No XGBoost test F1 in '{tag}' — skipping.")
            continue
        points.append((delay, f1))
    return points or None


def build_data(traces_dir, baseline_tag, poison_tags, jitter_tags):
    baseline    = _load_results(traces_dir, baseline_tag)
    shap_imp    = _shap_importances(baseline)
    baseline_f1 = _xgb_weighted_f1(baseline)

    if shap_imp is None:
        sys.exit(f"ERROR: no shap_importances in '{baseline_tag}'. "
                 "Run add_xgb_explain.py first.")

    poison_f1 = _collect_delay_f1(traces_dir, poison_tags, baseline_f1) if poison_tags else None
    jitter_f1 = _collect_delay_f1(traces_dir, jitter_tags, baseline_f1) if jitter_tags else None

    if not poison_f1 and not jitter_f1:
        sys.exit("ERROR: need at least one of --poison-tags or --jitter-tags with valid data.")

    return shap_imp, poison_f1, jitter_f1


def make_combined(shap_imp, poison_f1, jitter_f1, top_n, out):
    fig, (ax_imp, ax_deg) = plt.subplots(
        1, 2,
        figsize=(12, max(4.5, top_n * 0.42 + 1.5)),
        gridspec_kw={"width_ratios": [1.1, 1]},
    )
    plot_importance(ax_imp, shap_imp, top_n)
    plot_degradation(ax_deg, poison_f1, jitter_f1)
    # fig.suptitle("XGBoost: feature importance and timing-attack degradation",
    #              fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined → {out}")


def make_importance_only(shap_imp, top_n, out):
    fig_h = max(4, top_n * 0.42 + 1.2)
    fig, ax = plt.subplots(figsize=(7, fig_h))
    plot_importance(ax, shap_imp, top_n)
    ax.set_title("XGBoost SHAP feature importances", fontsize=10)
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved importance → {out}")


def make_degradation_only(poison_f1, jitter_f1, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_degradation(ax, poison_f1, jitter_f1)
    ax.set_title("XGBoost F1 under timing attack", fontsize=10)
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved degradation → {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot XGBoost SHAP importances and timing-attack F1 degradation."
    )
    parser.add_argument("--baseline-tag", required=True,
                        help="Experiment tag for the no-delay baseline (must have shap_importances).")
    parser.add_argument("--poison-tags", nargs="+", default=[],
                        help="Tags for train+test corrupted experiments (red line). "
                             "Delay is auto-parsed from tag name (e.g. wiki_delayed_xgb_500ms → 500 ms).")
    parser.add_argument("--jitter-tags", nargs="+", default=[],
                        help="Tags for test-only corrupted experiments (blue line). "
                             "Delay is auto-parsed from tag name (e.g. wiki_jitter_test_500ms → 500 ms).")
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces).")
    parser.add_argument("--top-n", type=int, default=8,
                        help="Number of top features to show (default: 8).")
    parser.add_argument("--plot", choices=["both", "importance", "degradation"],
                        default="both",
                        help="Which panel(s) to produce (default: both).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path. Defaults to traces/classifiers/<baseline-tag>/")
    parser.add_argument("--format", choices=["png", "pdf"], default="png",
                        help="Output format (default: png). Ignored if --out specifies an extension.")
    args = parser.parse_args()

    out_dir = args.traces_dir / "classifiers" / args.baseline_tag
    fmt = args.format
    shap_imp, poison_f1, jitter_f1 = build_data(
        args.traces_dir, args.baseline_tag, args.poison_tags, args.jitter_tags)

    if args.plot == "importance":
        make_importance_only(shap_imp, args.top_n, args.out or out_dir / f"feature_importance.{fmt}")
    elif args.plot == "degradation":
        make_degradation_only(poison_f1, jitter_f1, args.out or out_dir / f"degradation.{fmt}")
    else:  # both
        make_combined(shap_imp, poison_f1, jitter_f1, args.top_n,
                      args.out or out_dir / f"importance_x_degradation.{fmt}")
        make_importance_only(shap_imp, args.top_n, out_dir / f"feature_importance.{fmt}")
        make_degradation_only(poison_f1, jitter_f1, out_dir / f"degradation.{fmt}")


if __name__ == "__main__":
    main()
