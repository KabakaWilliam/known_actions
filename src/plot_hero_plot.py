#!/usr/bin/env python3
"""
create_hero_plot.py — per-dataset agent/family identifiability bar chart.

Reads live from results.json; selects classifier automatically (best macro F1
across all panels) or via --classifier; supports identity and family modes.

Usage:
    # Identity (default) — 4 per-dataset panels
    python create_hero_plot.py \\
        --test-set-source wiki_ood_all webshop_ood_all frames_ood_all deepshop_ood_all

    # Family — pass tags that contain _family_
    python create_hero_plot.py --mode family \\
        --test-set-source wiki_family_ood_all webshop_family_ood_all \\
                          frames_family_ood_all deepshop_family_ood_all

    # Specify classifier explicitly
    python create_hero_plot.py --classifier XGBoost \\
        --test-set-source wiki_ood_all frames_ood_all
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ── Colours ────────────────────────────────────────────────────────────────────
PROP_COLOUR = "#e9b59e"
OS_COLOUR   = "#98abd0"
EDGE_COLOUR = "#7a7a7a"

# ── Class membership ───────────────────────────────────────────────────────────
PROPRIETARY_FAMILIES = {"gpt_5", "claude_4", "gemini_3", "gemini_3_flash", "seed_2"}
PROPRIETARY_AGENTS   = {"gpt_5_4", "claude_opus_4_6", "gemini_3_1", "gemini_3_flash", "seed_2_lite"}

# ── Display names ──────────────────────────────────────────────────────────────
AGENT_LABELS = {
    "gpt_5_4":            "GPT-5.4",
    "claude_opus_4_6":    "Claude 4.6",
    "gemini_3_1":         "Gemini-3.1",
    "gemini_3_flash":     "Gemini-3-Flash",
    "gemma-4-31B-it":     "Gemma-4 (31B)",
    "gemma_4_26B_A4B_it": "Gemma-4 (26B)",
    "glm_4.6v":           "GLM-4.6V",
    "glm_4.6v_flash":     "GLM-4.6V-Flash",
    "qwen3vl_8b":         "Qwen3-VL-8B",
    "qwen3vl_30b_a3b":    "Qwen3-VL-30B",
    "qwen3_5_27b":        "Qwen3.5-27B",
    "qwen3_5_9b":         "Qwen3.5-9B",
    "seed_2_lite":        "Seed-2-lite",
    "uitars_7b":          "UI-TARS-1.5",
}

FAMILY_LABELS = {
    "gpt_5":          "GPT-5",
    "claude_4":       "Claude 4",
    "gemini_3":       "Gemini-3",
    "gemini_3_flash": "Gemini-3-Flash",
    "seed_2":         "Seed-2",
    "gemma_4":        "Gemma-4",
    "glm_4.6v":       "GLM-4.6V",
    "qwen3vl":        "Qwen3-VL",
    "qwen35":         "Qwen3.5",
    "uitars_1.5":     "UI-TARS-1.5",
}

DATASET_LABELS = {
    "wiki_ood_all":            "2WikiMultihopQA",
    "wiki_family_ood_all":     "2WikiMultihopQA",
    "webshop_ood_all":         "WebShop",
    "webshop_family_ood_all":  "WebShop",
    "frames_ood_all":          "FRAMES",
    "frames_family_ood_all":   "FRAMES",
    "deepshop_ood_all":        "DeepShop",
    "deepshop_family_ood_all": "DeepShop",
    "webgames_ood_all":        "WebGames",
    "webgames_family_ood_all": "WebGames",
    "universal_wiki_frames": "Wikipedia",
    "universal_ws_deepshop": "Amazon"
}


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_results(traces_dir: Path, tag: str) -> dict:
    path = traces_dir / "classifiers" / tag / "results.json"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found")
    with open(path) as f:
        return json.load(f)


def _macro_f1(results: dict, clf: str) -> float:
    tr = (results["models"].get(clf) or {}).get("test_report") or {}
    return (tr.get("macro avg") or {}).get("f1-score", 0.0)


def pick_best_classifier(all_results: dict) -> str:
    """Best classifier by mean macro F1 across all panels."""
    clf_totals: dict[str, list[float]] = {}
    for res in all_results.values():
        for clf_name, data in res["models"].items():
            tr = (data.get("test_report") or {})
            f1 = (tr.get("macro avg") or {}).get("f1-score", 0.0)
            clf_totals.setdefault(clf_name, []).append(f1)
    return max(clf_totals, key=lambda c: sum(clf_totals[c]) / len(clf_totals[c]))


def extract_per_class_f1(results: dict, clf: str) -> dict[str, float]:
    tr = (results["models"].get(clf) or {}).get("test_report") or {}
    return {
        k: v["f1-score"]
        for k, v in tr.items()
        if isinstance(v, dict) and "f1-score" in v
        and k not in ("macro avg", "weighted avg", "accuracy")
    }


def sort_classes(classes: list[str], mode: str) -> list[str]:
    prop_set = PROPRIETARY_FAMILIES if mode == "family" else PROPRIETARY_AGENTS
    prop = sorted(c for c in classes if c in prop_set)
    oset = sorted(c for c in classes if c not in prop_set)
    return prop + oset


# ── Plotting ───────────────────────────────────────────────────────────────────

def _build_axes(n: int):
    if n == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        return fig, [ax]
    if n == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        return fig, list(axes)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    return fig, axes.flatten().tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-class Macro F1 from classifier test reports."
    )
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces).")
    parser.add_argument("--mode", choices=["identity", "family"], default="identity",
                        help="identity = per-agent bars; family = per-family bars.")
    parser.add_argument("--classifier", default=None,
                        help="Classifier to use. Default: best by mean macro F1.")
    parser.add_argument("--test-set-source", nargs="+", required=True,
                        metavar="TAG",
                        help="Experiment tag(s) to plot (up to 4). "
                             "Family mode requires tags containing '_family_'.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path.")
    args = parser.parse_args()

    tags = args.test_set_source[:4]

    if args.mode == "family":
        bad = [t for t in tags if "_family" not in t]
        if bad:
            sys.exit(
                f"ERROR: --mode family requires tags with '_family' in the name.\n"
                f"  Offending: {bad}\n"
                f"  Expected e.g. wiki_family_ood_all, webshop_family_ood_all"
            )

    all_results = {t: load_results(args.traces_dir, t) for t in tags}

    clf = args.classifier or pick_best_classifier(all_results)
    print(f"Classifier: {clf}")

    # Validate chosen classifier exists in every panel
    for t, res in all_results.items():
        if clf not in res["models"]:
            available = list(res["models"].keys())
            sys.exit(f"ERROR: classifier '{clf}' not in '{t}'. Available: {available}")

    panels = {t: extract_per_class_f1(all_results[t], clf) for t in tags}

    # Union of all classes, sorted proprietary-first
    all_classes = set()
    for p in panels.values():
        all_classes.update(p.keys())
    classes = sort_classes(list(all_classes), args.mode)

    label_map = FAMILY_LABELS if args.mode == "family" else AGENT_LABELS
    prop_set  = PROPRIETARY_FAMILIES if args.mode == "family" else PROPRIETARY_AGENTS

    xlabels = [label_map.get(c, c) for c in classes]
    colours  = [PROP_COLOUR if c in prop_set else OS_COLOUR for c in classes]

    n = len(tags)
    fig, axes = _build_axes(n)

    panel_letters = ["A", "B", "C", "D"]
    x = np.arange(len(classes))
    bar_w = 0.52

    for idx, (tag, ax) in enumerate(zip(tags, axes)):
        values = [panels[tag].get(c, 0.0) * 100 for c in classes]

        bars = ax.bar(x, values, width=bar_w, color=colours,
                      edgecolor=EDGE_COLOUR, linewidth=0.9, zorder=3)

        ax.text(0.01, 1.05, panel_letters[idx], transform=ax.transAxes,
                fontsize=20, fontweight="bold", ha="left", va="bottom")

        ax.set_title(DATASET_LABELS.get(tag, tag), fontsize=24, pad=14)
        ax.set_ylim(0, 101)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.9, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=40, ha="right",
                           fontsize=10 if args.mode == "identity" else 11)
        ax.tick_params(axis="y", labelsize=11)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        for rect, val in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.2,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=8 if args.mode == "identity" else 9)

        # Classifier name inside panel
        ax.text(0.99, 0.03, clf, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color="#555555",
                fontstyle="italic")

    # Hide unused axes (only relevant for n=3 in a 2×2 grid)
    for ax in axes[n:]:
        ax.set_visible(False)

    # Ensure right-column panels show y tick labels
    if n > 2:
        for ax in axes[1::2]:
            ax.tick_params(axis="y", labelleft=True)

    fig.supylabel("Macro F1 (%)", fontsize=20, x=0.03)

    prop_label = "Proprietary families" if args.mode == "family" else "Proprietary agents"
    os_label   = "Open-source families"  if args.mode == "family" else "Open-source agents"
    fig.legend(
        handles=[
            Patch(facecolor=PROP_COLOUR, edgecolor=EDGE_COLOUR, label=prop_label),
            Patch(facecolor=OS_COLOUR,   edgecolor=EDGE_COLOUR, label=os_label),
        ],
        loc="lower center", ncol=2, frameon=False,
        fontsize=15, bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=[0.04, 0.07, 1, 1])

    if args.out:
        out = args.out
    else:
        suffix = "family" if args.mode == "family" else "identity"
        out = Path("figures") / f"{suffix}_identifiability_barplots.png"

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
