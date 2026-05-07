#!/usr/bin/env python3
"""
plot_feature_importance.py — Visualise RF (MDI) and XGBoost (SHAP) feature importances.

Reads feature_importances (RandomForest) and shap_importances (XGBoost) from
results.json and produces a horizontal grouped bar chart.

Usage:
    python plot_feature_importance.py --tag wiki_xgb_ood_frames
    python plot_feature_importance.py --tag wiki_ood_all --top-n 20
    python plot_feature_importance.py --tag wiki_ood_all --out my_fig.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RF_COLOR  = "#2196F3"   # blue  — matches CLF_COLORS in other plot scripts
XGB_COLOR = "#FF9800"   # orange

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


def load_importances(results_path: Path):
    with open(results_path) as f:
        results = json.load(f)

    models = results.get("models", {})
    rf_entry  = models.get("RandomForest") or {}
    xgb_entry = models.get("XGBoost") or {}

    rf_imp   = rf_entry.get("feature_importances")
    shap_imp = xgb_entry.get("shap_importances")

    if shap_imp:
        # Ensure values are positive (old data may have been stored with wrong sign)
        shap_imp = {k: abs(v) for k, v in shap_imp.items()}

    return rf_imp, shap_imp, results


def plot_importance(tag: str, traces_dir: Path, top_n: int, out: Path | None):
    results_path = traces_dir / "classifiers" / tag / "results.json"
    if not results_path.exists():
        sys.exit(f"ERROR: results.json not found at {results_path}")

    rf_imp, shap_imp, results = load_importances(results_path)

    if not rf_imp and not shap_imp:
        sys.exit("ERROR: no feature_importances (RF) or shap_importances (XGBoost) found in results.json.")

    # Gather all features; sort by max importance across available methods
    all_features = set()
    if rf_imp:   all_features |= set(rf_imp)
    if shap_imp: all_features |= set(shap_imp)

    def max_importance(feat):
        vals = []
        if rf_imp and feat in rf_imp:     vals.append(rf_imp[feat])
        if shap_imp and feat in shap_imp: vals.append(shap_imp[feat])
        return max(vals)

    ranked = sorted(all_features, key=max_importance, reverse=True)[:top_n]
    ranked = list(reversed(ranked))   # bottom-to-top for horizontal chart

    labels = [FEATURE_LABELS.get(f, f) for f in ranked]

    # ── Layout ─────────────────────────────────────────────────────────────────
    has_both = rf_imp and shap_imp
    bar_height = 0.35 if has_both else 0.55
    y = np.arange(len(ranked))

    fig_height = max(5, len(ranked) * 0.42 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    if has_both:
        rf_vals   = [rf_imp.get(f, 0)   for f in ranked]
        shap_vals = [shap_imp.get(f, 0) for f in ranked]
        ax.barh(y + bar_height / 2, rf_vals,   bar_height, label="RF (MDI)",       color=RF_COLOR,  alpha=0.85)
        ax.barh(y - bar_height / 2, shap_vals, bar_height, label="XGBoost (SHAP)", color=XGB_COLOR, alpha=0.85)
    elif rf_imp:
        rf_vals = [rf_imp.get(f, 0) for f in ranked]
        ax.barh(y, rf_vals, bar_height, label="RF (MDI)", color=RF_COLOR, alpha=0.85)
    else:
        shap_vals = [shap_imp.get(f, 0) for f in ranked]
        ax.barh(y, shap_vals, bar_height, label="XGBoost (SHAP)", color=XGB_COLOR, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Feature importance", fontsize=10)

    # Title: include test accuracy if available
    models = results.get("models", {})
    acc_parts = []
    for clf, label in [("RandomForest", "RF"), ("XGBoost", "XGB")]:
        entry = models.get(clf) or {}
        tr = entry.get("test_report") or {}
        if tr.get("accuracy") is not None:
            acc_parts.append(f"{label} test F1={tr['accuracy']:.3f}")
    acc_str = "  |  " + ",  ".join(acc_parts) if acc_parts else ""
    # ax.set_title(f"Feature importance — {tag}{acc_str}", fontsize=10, pad=10)
    ax.set_title(f"Feature importance — {tag}{acc_str}", pad=10, fontweight="bold")

    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if out is None:
        out = traces_dir / "classifiers" / tag / "feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot RF MDI and XGBoost SHAP feature importances.")
    parser.add_argument("--tag", required=True, help="Experiment tag")
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top features to show (default: 20)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: traces/classifiers/<tag>/feature_importance.png)")
    args = parser.parse_args()
    plot_importance(args.tag, args.traces_dir, args.top_n, args.out)


if __name__ == "__main__":
    main()
