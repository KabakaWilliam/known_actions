#!/usr/bin/env python3
"""
plot_eval_universal.py — Per-agent XGBoost F1 for the universal_wiki_frames model
evaluated separately on 2WikiMultiHopQA and FRAMES test splits.

Usage:
    python plot_eval_universal.py
    python plot_eval_universal.py --format pdf
    python plot_eval_universal.py --out /tmp/fig.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "font.size":    9,
})

DATASET_COLOURS = ["#D4735E", "#E8973B"]
BG_COLOUR       = "#FFFFFF"

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


def _load_closed(traces_dir: Path, tag: str, clf: str) -> dict[str, float]:
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


def _draw_panel(ax, agents: list[str], values: dict[str, float],
                colour: str, show_names: bool,
                ref_values: dict[str, float] | None = None):
    ax.set_facecolor(BG_COLOUR)
    y    = np.arange(len(agents))
    vals = [values.get(a, 0.0) * 100 for a in agents]

    matplotlib.rcParams["hatch.linewidth"] = 1.4
    light_fill = mc.to_rgba(colour, alpha=0.15)

    if ref_values:
        refs = [ref_values.get(a, 0.0) * 100 for a in agents]
        # solid bar = dedicated single-dataset model (drawn first, behind)
        ax.barh(y, refs, facecolor=colour, height=0.68, zorder=2)

    # hatched bar = universal model (drawn on top)
    bars = ax.barh(y, vals, facecolor=light_fill, height=0.68,
                   hatch="//", linewidth=0.0, zorder=3)
    for bar in bars:
        bar.set_edgecolor(mc.to_rgba(colour, alpha=1.0))

    ax.set_xlim(0, 115)
    ax.set_ylim(-0.5, len(agents) - 0.5)
    ax.set_yticks(y)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.invert_yaxis()
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("F1 (%)", fontsize=9)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
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
        description="Plot per-agent F1 for the universal_wiki_frames model on each dataset."
    )
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"))
    parser.add_argument("--classifier",  default="XGBoost")
    parser.add_argument("--format", choices=["png", "pdf"], default="png")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    clf = args.classifier

    wiki_data     = _load_closed(args.traces_dir, "eval_universal_wiki_frames_on_wiki",   clf)
    frames_data   = _load_closed(args.traces_dir, "eval_universal_wiki_frames_on_frames", clf)
    wiki_ref      = _load_closed(args.traces_dir, "wiki_ood_all",                         clf)
    frames_ref    = _load_closed(args.traces_dir, "frames_ood_all",                       clf)

    if not wiki_data:
        sys.exit("ERROR: no wiki eval data found. Run eval_universal_wiki_frames.sh first.")

    agents = sorted(wiki_data.keys(), key=lambda a: wiki_data.get(a, 0.0), reverse=True)

    macro_wiki   = sum(wiki_data.values())   / len(wiki_data)
    macro_frames = sum(frames_data.values()) / len(frames_data) if frames_data else 0.0

    n_agents = len(agents)
    fig_h    = max(4.5, n_agents * 0.38 + 1.5)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, fig_h))
    fig.patch.set_facecolor(BG_COLOUR)

    _draw_panel(ax0, agents, wiki_data,   colour=DATASET_COLOURS[0], show_names=True,
                ref_values=wiki_ref or None)
    _draw_panel(ax1, agents, frames_data, colour=DATASET_COLOURS[1], show_names=False,
                ref_values=frames_ref or None)

    ax0.set_title(f"2WikiMultiHop  (Macro F1: {macro_wiki*100:.1f}%)",
                  fontsize=15, pad=8, fontweight="bold", color=DATASET_COLOURS[0])
    ax1.set_title(f"FRAMES  (Macro F1: {macro_frames*100:.1f}%)",
                  fontsize=15, pad=8, fontweight="bold", color=DATASET_COLOURS[1])

    fig.text(0.5, 0.97,
             "Universal Classifier (2Wiki + FRAMES) — Per-model F1 (%) ↑",
             ha="center", fontsize=17, fontweight="bold", color="#2a2a2a")

    import matplotlib.patches as mpatches
    solid_patch  = mpatches.Patch(color="#888888",
                                  label="Dedicated single-dataset model")
    hatch_patch  = mpatches.Patch(facecolor=mc.to_rgba("#888888", 0.15),
                                  edgecolor="#888888", hatch="//",
                                  label="Universal model (2Wiki + FRAMES)")
    fig.legend(handles=[solid_patch, hatch_patch], loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=9)

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    fig.subplots_adjust(left=0.18, wspace=0.04)

    fmt = args.format
    out = (args.out.with_suffix(f".{fmt}") if args.out
           else Path("figures") / f"eval_universal_wiki_frames.{fmt}")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
