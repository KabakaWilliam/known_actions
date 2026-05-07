#!/usr/bin/env python3
"""
plot_main_results.py — Main results figure: closed-set Macro F1 and open-set AUROC.

2 rows × 4 columns of horizontal-bar subplots:
  Row A (top):    closed-set Macro F1 (%), blue bars
  Row B (bottom): open-set AUROC, green bars; dashed chance line at 0.5

Agents sorted by 2WikiMultiHop closed-set F1 descending (shared order across all panels).

Usage:
    python plot_main_results.py
    python plot_main_results.py --classifier RandomForest --format pdf
    python plot_main_results.py --out /tmp/fig.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "font.size":    9,
})

# ── Colours ────────────────────────────────────────────────────────────────────
# Per-dataset palette (column order: 2WikiMultiHop, FRAMES, WebShop, DeepShop)
DATASET_COLOURS = ["#D4735E", "#E8973B", "#6BAA98", "#C49BB5"]
CHANCE_COLOUR   = "#AA3333"
BG_COLOUR       = "#FFFFFF"

# ── Dataset config ─────────────────────────────────────────────────────────────
CLOSED_TAGS = [
    ("wiki_ood_all",     "2WikiMultiHop"),
    ("frames_ood_all",   "FRAMES"),
    ("webshop_ood_all",  "WebShop"),
    ("deepshop_ood_all", "DeepShop"),
]

OPEN_TAGS = [
    ("2wikimultihop_open_set", "2WikiMultiHop"),
    ("frames_open_set",        "FRAMES"),
    ("webshop_open_set",       "WebShop"),
    ("deepshop_open_set",      "DeepShop"),
]

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

# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_closed(traces_dir: Path, tag: str, clf: str) -> dict[str, float]:
    """Returns {agent_id: f1_fraction} from test_report for the given classifier."""
    path = traces_dir / "classifiers" / tag / "results.json"
    if not path.exists():
        print(f"WARNING: {path} not found — skipping", file=sys.stderr)
        return {}
    with open(path) as f:
        results = json.load(f)
    tr = (results.get("models", {}).get(clf) or {}).get("test_report") or {}
    out = {}
    for k, v in tr.items():
        if isinstance(v, dict) and "f1-score" in v and k not in ("macro avg", "weighted avg", "accuracy"):
            out[k] = float(v["f1-score"])
    return out


def _load_open(traces_dir: Path, loo_subdir: str, clf: str) -> dict[str, float]:
    """Returns {agent_id: auroc} from LOO open-set dirs for the given classifier."""
    loo_dir = traces_dir / "classifiers" / loo_subdir
    if not loo_dir.exists():
        print(f"WARNING: {loo_dir} not found — skipping", file=sys.stderr)
        return {}
    out = {}
    for exp_dir in sorted(loo_dir.iterdir()):
        if not exp_dir.name.startswith("open_set_loo_"):
            continue
        agent_id = exp_dir.name[len("open_set_loo_"):]
        rpath = exp_dir / "results.json"
        if not rpath.exists():
            continue
        with open(rpath) as f:
            r = json.load(f)
        auroc = ((r.get("open_set") or {}).get(clf) or {}).get("auroc")
        if auroc is not None:
            out[agent_id] = float(auroc)
    return out


def _draw_panel(ax, agents: list[str], values: dict[str, float],
                colour: str, xlim: tuple, xticks: list,
                xlabel: str, show_names: bool, val_fmt: str,
                chance_line: float | None = None,
                bg: str | None = None):
    if bg:
        ax.set_facecolor(bg)
    y = np.arange(len(agents))
    vals = [values.get(a, 0.0) for a in agents]
    ax.barh(y, vals, color=colour, height=0.72, zorder=3)
    if chance_line is not None:
        ax.axvline(chance_line, color=CHANCE_COLOUR, linestyle="--",
                   linewidth=1.2, zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.5, len(agents) - 0.5)
    ax.set_yticks(y)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.invert_yaxis()
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    pad = (xlim[1] - xlim[0]) * 0.01
    for xi, v in zip(y, vals):
        if v > 0:
            ax.text(v + pad, xi, val_fmt.format(v),
                    va="center", ha="left", fontsize=7, color="#444444")

    if show_names:
        anchor_trans = mtransforms.blended_transform_factory(
            ax.transAxes, ax.transData)
        text_shift = mtransforms.ScaledTranslation(
            -6 / 72.0, 0, ax.get_figure().dpi_scale_trans)
        for yi, agent in enumerate(agents):
            ax.text(0, yi, AGENT_LABELS.get(agent, agent),
                    transform=anchor_trans + text_shift,
                    ha="right", va="center", fontsize=12, clip_on=False)


def main():
    parser = argparse.ArgumentParser(
        description="Plot closed-set Macro F1 and open-set AUROC main results figure."
    )
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"))
    parser.add_argument("--classifier",  default="XGBoost",
                        help="Classifier name (default: XGBoost).")
    parser.add_argument("--panel", choices=["both", "closed", "open"], default="both",
                        help="Which panel to plot (default: both).")
    parser.add_argument("--format", choices=["png", "pdf"], default="png")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    clf   = args.classifier
    panel = args.panel

    closed_data = {tag: _load_closed(args.traces_dir, tag, clf) for tag, _ in CLOSED_TAGS}
    open_data   = {tag: _load_open(args.traces_dir, tag, clf)   for tag, _ in OPEN_TAGS}

    all_agents: set[str] = set()
    for d in closed_data.values():
        all_agents.update(d.keys())
    if not all_agents:
        sys.exit("ERROR: no closed-set data found. Check --traces-dir and --classifier.")

    anchor = closed_data.get(CLOSED_TAGS[0][0], {})
    agents = sorted(all_agents, key=lambda a: anchor.get(a, 0.0), reverse=True)

    n_agents = len(agents)
    n_rows   = 1 if panel != "both" else 2
    fig_h    = max(4.5, n_agents * 0.38 + 1.5) if n_rows == 1 else max(8, n_agents * 0.65 + 3)
    fig_w    = 18 if n_rows == 1 else 16
    fig, axes_grid = plt.subplots(n_rows, 4, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG_COLOUR)

    # Normalise to always be shape (n_rows, 4)
    if n_rows == 1:
        axes_grid = [list(axes_grid)]

    if panel in ("both", "closed"):
        row = axes_grid[0]
        for col, (tag, label) in enumerate(CLOSED_TAGS):
            colour = DATASET_COLOURS[col]
            vals_pct = {a: v * 100 for a, v in closed_data[tag].items()}
            _draw_panel(
                row[col], agents, vals_pct,
                colour=colour,
                xlim=(0, 115), xticks=[0, 25, 50, 75, 100],
                xlabel="Macro F1 (%)", show_names=(col == 0),
                val_fmt="{:.1f}", bg=BG_COLOUR,
            )
            row[col].set_title(label, fontsize=15, pad=8,
                               fontweight="bold", color=colour)
        title_fs = 19 if n_rows == 1 else 17
        fig.text(0.5, 0.97,
                 f"{'A.  ' if panel == 'both' else ''}Closed-set Attribution — Per-model F1 (%) ↑",
                 ha="center", fontsize=title_fs, fontweight="bold", color="#2a2a2a")

    if panel in ("both", "open"):
        row = axes_grid[1] if panel == "both" else axes_grid[0]
        for col, (tag, label) in enumerate(OPEN_TAGS):
            colour = DATASET_COLOURS[col]
            _draw_panel(
                row[col], agents, open_data[tag],
                colour=colour,
                xlim=(0, 1.12), xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
                xlabel="AUROC", show_names=(col == 0),
                val_fmt="{:.2f}", chance_line=0.5, bg=BG_COLOUR,
            )
            row[col].set_title(label, fontsize=15, pad=8,
                               fontweight="bold", color=colour)
        title_y  = 0.465 if panel == "both" else 0.97
        title_fs = 19 if n_rows == 1 else 17
        fig.text(0.5, title_y,
                 f"{'B.  ' if panel == 'both' else ''}Open-set Unknown-Agent Detection — AUROC ↑  [dashed = chance 0.5]",
                 ha="center", fontsize=title_fs, fontweight="bold", color="#2a2a2a")

    top = 0.96 if panel == "both" else 0.94
    fig.tight_layout(rect=(0.0, 0.01, 1.0, top))
    if panel == "both":
        fig.subplots_adjust(left=0.18, hspace=0.25, wspace=0.04)
    else:
        fig.subplots_adjust(left=0.18, wspace=0.04)

    fmt      = args.format
    stem     = {"both": "main_results", "closed": "main_results_closed", "open": "main_results_open"}[panel]
    out = (args.out.with_suffix(f".{fmt}") if args.out
           else Path("figures") / f"{stem}.{fmt}")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
