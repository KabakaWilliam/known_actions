#!/usr/bin/env python3
"""Connected dot plot of per-agent task accuracy across datasets.

Each row is one agent. Three dots show per-dataset accuracy (colour + shape
encoded by dataset). A thin grey line spans the min–max range, and the
overall mean is printed to the right.

Usage:
    python plot_benchmark.py
    python plot_benchmark.py --split test
    python plot_benchmark.py --out figures/benchmark.pdf
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

DATASET_LABELS = {
    "2wikimultihop": "2WikiMultihop",
    "frames":        "FRAMES",
    "webgames":      "WebGames",
}

# Okabe–Ito: blue / orange / bluish-green — safe for all common colour-vision deficiencies
DATASET_COLORS = {
    "2wikimultihop": "#0072B2",
    "frames":        "#E69F00",
    "webgames":      "#009E73",
}

DATASET_MARKERS = {
    "2wikimultihop": "o",
    "frames":        "s",
    "webgames":      "D",
}

# Darkened Okabe–Ito hues — readable as text on white
FAMILY_COLORS = {
    "gpt_5":          "#111111",
    "claude_4":       "#C47C00",
    "gemini_3":       "#005F9E",
    "gemini_3_flash": "#3E8EC4",
    "gemma_4":        "#007A57",
    "glm_4.6v":       "#9A4080",
    "qwen35":         "#A84000",
    "qwen3vl":        "#8B7200",
    "uitars_1.5":     "#484848",
    "seed_2":         "#888888",
}

mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8,
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "text.color":        "#1a1a1a",
    "axes.labelcolor":   "#1a1a1a",
    "xtick.color":       "#555555",
    "ytick.color":       "#1a1a1a",
})


def _load_config(config_path: Path = _CONFIG_PATH) -> tuple[dict, dict]:
    try:
        import yaml
        cfg     = yaml.safe_load(config_path.read_text())
        names   = {a["agent_id"]: a.get("display_name", a["agent_id"])
                   for a in cfg.get("agents", [])}
        fams    = {a["agent_id"]: a.get("family",       a["agent_id"])
                   for a in cfg.get("agents", [])}
        return names, fams
    except Exception:
        return {}, {}


def collect(traces_dir: Path, splits: list[str] | None) -> dict:
    stats: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )
    for path in traces_dir.rglob("*.json"):
        parts = path.relative_to(traces_dir).parts
        if parts[0].startswith("classifiers") or len(parts) < 3:
            continue
        agent_id     = parts[0]
        dataset_name = parts[1]
        base         = dataset_name.rsplit("_", 1)[0]
        suffix       = dataset_name.rsplit("_", 1)[-1] if "_" in dataset_name else dataset_name
        if base not in DATASET_LABELS:
            continue
        if splits is not None and suffix not in splits:
            continue
        try:
            ep = json.loads(path.read_text())
        except Exception:
            continue
        v = ep.get("verification")
        if not v or not v.get("ground_truth"):
            continue
        stats[agent_id][base]["total"]   += 1
        stats[agent_id][base]["correct"] += int(bool(v.get("correct")))
    return stats


def plot(stats: dict, display_names: dict, families: dict, out: Path) -> None:
    datasets = list(DATASET_LABELS.keys())

    def overall_acc(agent: str) -> float:
        c = t = 0
        for ds in datasets:
            d = stats[agent].get(ds, {})
            c += d.get("correct", 0); t += d.get("total", 0)
        return c / t if t else 0.0

    agents = sorted(stats.keys(), key=overall_acc, reverse=True)
    n      = len(agents)

    # Per-agent accuracy dict
    acc: dict[str, dict[str, float | None]] = {}
    for agent in agents:
        acc[agent] = {}
        for ds in datasets:
            d = stats[agent].get(ds, {})
            c, t = d.get("correct", 0), d.get("total", 0)
            acc[agent][ds] = c / t if t else None
        acc[agent]["overall"] = overall_acc(agent) or None

    # ── Figure ───────────────────────────────────────────────────────────
    fig_h = max(4.2, n * 0.44 + 1.6)
    fig, ax = plt.subplots(figsize=(6.8, fig_h))

    # agents are listed top-to-bottom, so highest-accuracy agent is at y = n-1
    y_pos = {agent: (n - 1 - i) for i, agent in enumerate(agents)}

    # Alternating row shading for readability
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f7f7f7", linewidth=0, zorder=0)

    for agent in agents:
        y   = y_pos[agent]
        fam = families.get(agent, "")
        vals = [v for ds in datasets if (v := acc[agent][ds]) is not None]

        # Range spine
        if len(vals) > 1:
            ax.hlines(y, min(vals), max(vals),
                      color="#cccccc", linewidth=1.8, zorder=1)

        # Dataset dots
        for ds in datasets:
            v = acc[agent][ds]
            if v is not None:
                ax.scatter(v, y,
                           color=DATASET_COLORS[ds],
                           marker=DATASET_MARKERS[ds],
                           s=52, zorder=3, linewidths=0)

        # Overall % label on the right
        ov = acc[agent]["overall"]
        if ov is not None:
            ax.text(1.045, y, f"{ov:.0%}", va="center", fontsize=7.5,
                    color=FAMILY_COLORS.get(fam, "#555555"), fontweight="semibold",
                    transform=ax.get_yaxis_transform())

    # ── Y-axis: agent names coloured by family ────────────────────────────
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels([display_names.get(a, a) for a in agents])
    for label, agent in zip(ax.get_yticklabels(), agents):
        fam = families.get(agent, "")
        label.set_color(FAMILY_COLORS.get(fam, "#333333"))
        label.set_fontweight("semibold")

    # ── X-axis ────────────────────────────────────────────────────────────
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Task Accuracy")
    ax.xaxis.set_tick_params(length=3, width=0.5)

    # Reference lines
    for x in [0.25, 0.5, 0.75]:
        ax.axvline(x, color="#e0e0e0", linewidth=0.7, zorder=0)

    # ── Spines ────────────────────────────────────────────────────────────
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(left=False)
    ax.set_ylim(-0.7, n - 0.3)

    # ── Dataset legend (FRAMES → WebGames → 2WikiMultihop) ───────────────
    legend_order = ["frames", "webgames", "2wikimultihop"]
    ds_handles = [
        mlines.Line2D([], [],
                      color=DATASET_COLORS[ds],
                      marker=DATASET_MARKERS[ds],
                      linewidth=0, markersize=7,
                      label=DATASET_LABELS[ds])
        for ds in legend_order
    ]
    ax.legend(handles=ds_handles,
              loc="lower right",
              fontsize=8,
              frameon=True,
              framealpha=0.95,
              edgecolor="#dddddd",
              borderpad=0.7,
              handletextpad=0.5,
              labelcolor="#333333")

    # ── "Overall" column header above the right-side % labels ────────────
    ax.set_title("Agent Task Accuracy across Benchmarks", pad=10)
    ax.text(1.045, 1.01, "Overall",
            ha="center", va="bottom", fontsize=7.5, color="#888888",
            fontweight="semibold", transform=ax.get_yaxis_transform())

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"))
    parser.add_argument("--split", nargs="+", default=None,
                        metavar="SPLIT",
                        help="e.g. --split test  (default: all splits)")
    parser.add_argument("--out", type=Path,
                        default=Path("./figures/benchmark.png"))
    cli = parser.parse_args()

    display_names, families = _load_config()
    stats = collect(cli.traces_dir, cli.split)
    plot(stats, display_names, families, cli.out)
