#!/usr/bin/env python3
"""Heatmap of per-agent task accuracy across datasets.

Usage:
    python plot_benchmark.py
    python plot_benchmark.py --split test
    python plot_benchmark.py --out figures/benchmark.pdf
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

DATASET_LABELS = {
    "2wikimultihop": "2WikiMultihop",
    "frames":        "FRAMES",
    "webgames":      "WebGames",
}

FAMILY_COLORS = {
    "claude_4":       "#D4A853",
    "gemini_3":       "#4285F4",
    "gemini_3_flash": "#7BAAF7",
    "gemma_4":        "#34A853",
    "glm_4.6v":       "#9C5FBA",
    "gpt_5":          "#00A67E",
    "qwen35":         "#E8701A",
    "qwen3vl":        "#F4B942",
    "seed_2":         "#E53935",
    "uitars_1.5":     "#546E7A",
}


def _load_config(config_path: Path = _CONFIG_PATH) -> tuple[dict, dict]:
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        names   = {a["agent_id"]: a.get("display_name", a["agent_id"])
                   for a in cfg.get("agents", [])}
        families = {a["agent_id"]: a.get("family", a["agent_id"])
                    for a in cfg.get("agents", [])}
        return names, families
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

    # Compute overall per agent and sort descending
    def overall(agent):
        c = t = 0
        for ds in datasets:
            d = stats[agent].get(ds, {})
            c += d.get("correct", 0)
            t += d.get("total", 0)
        return c / t if t else 0.0

    agents = sorted(stats.keys(), key=overall, reverse=True)

    # Build matrix  (agents × datasets+overall)
    col_keys = datasets + ["overall"]
    matrix   = np.full((len(agents), len(col_keys)), np.nan)

    for i, agent in enumerate(agents):
        tc = tt = 0
        for j, ds in enumerate(datasets):
            d = stats[agent].get(ds, {})
            c, t = d.get("correct", 0), d.get("total", 0)
            tc += c; tt += t
            if t:
                matrix[i, j] = c / t
        if tt:
            matrix[i, -1] = tc / tt

    # --- Figure layout ---
    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 6),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
    )

    col_labels = [DATASET_LABELS[d] for d in datasets]

    # Left: per-dataset heatmap
    ax = axes[0]
    im = ax.imshow(matrix[:, :-1], aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1, interpolation="none")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(
        [display_names.get(a, a) for a in agents],
        fontsize=9,
    )
    ax.tick_params(left=False, bottom=False)
    ax.set_title("Task Accuracy per Dataset", fontsize=12, pad=10)

    # Annotate cells
    for i in range(len(agents)):
        for j in range(len(datasets)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "black" if 0.25 < v < 0.75 else "white"
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, color=color)
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="#aaaaaa")

    # Family colour strip on the left
    strip_w = 0.018
    for i, agent in enumerate(agents):
        fam   = families.get(agent, "")
        color = FAMILY_COLORS.get(fam, "#cccccc")
        ax.add_patch(mpatches.FancyBboxPatch(
            (-0.48, i - 0.45), strip_w * 10, 0.9,
            boxstyle="round,pad=0", linewidth=0,
            facecolor=color, transform=ax.transData, clip_on=False,
        ))

    # Right: overall bar chart
    ax2 = axes[1]
    bar_colors = [FAMILY_COLORS.get(families.get(a, ""), "#cccccc") for a in agents]
    overall_vals = [matrix[i, -1] if not np.isnan(matrix[i, -1]) else 0
                    for i in range(len(agents))]

    bars = ax2.barh(range(len(agents)), overall_vals, color=bar_colors,
                    height=0.65, edgecolor="white", linewidth=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, len(agents) - 0.5)
    ax2.invert_yaxis()
    ax2.set_yticks([])
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax2.set_title("Overall", fontsize=12, pad=10)
    ax2.axvline(0.5, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax2.tick_params(left=False)
    ax2.spines[["top", "right", "left"]].set_visible(False)

    for i, v in enumerate(overall_vals):
        if v > 0:
            ax2.text(min(v + 0.02, 0.97), i, f"{v:.0%}",
                     va="center", fontsize=8,
                     ha="left" if v < 0.9 else "right")

    # Family legend
    seen = {}
    for agent in agents:
        fam = families.get(agent, "")
        if fam and fam not in seen:
            seen[fam] = FAMILY_COLORS.get(fam, "#cccccc")
    legend_handles = [
        mpatches.Patch(facecolor=c, label=f)
        for f, c in seen.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    plt.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02,
                 label="Accuracy", format=lambda x, _: f"{x:.0%}")

    fig.suptitle("Agent Task Performance", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
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
