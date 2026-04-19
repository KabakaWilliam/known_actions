#!/usr/bin/env python3
"""Scatter plot: per-agent task accuracy (capability) vs classifier F1 (identifiability).

Two panels side by side — 2WikiMultihop accuracy and FRAMES accuracy on the x-axis,
identifiability F1 on the y-axis. Spearman ρ annotated on each panel.

Usage:
    python capability_vs_identifiability.py \\
        --results-json src/traces/classifiers/wiki_ood_all/results.json

    python capability_vs_identifiability.py \\
        --results-json src/traces/classifiers/wiki_ood_all/results.json \\
        --traces-dir src/traces \\
        --out figures/capability_vs_identifiability.pdf
"""
import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from scipy import stats

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

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
        names    = {a["agent_id"]: a.get("display_name", a["agent_id"])
                    for a in cfg.get("agents", [])}
        families = {a["agent_id"]: a.get("family", a["agent_id"])
                    for a in cfg.get("agents", [])}
        return names, families
    except Exception:
        return {}, {}


# ── Identifiability ───────────────────────────────────────────────────────────

def load_identifiability(results_json: Path) -> tuple[str, dict[str, float]]:
    """Return (best_clf_name, {agent_id: f1}) from the classifier results file."""
    if not results_json.exists():
        print(f"ERROR: {results_json} not found.\n"
              f"Run the classifier training first:\n"
              f"  bash src/train_classifiers.sh", file=sys.stderr)
        sys.exit(1)

    data   = json.loads(results_json.read_text())
    models = data.get("models") or {}
    if not models:
        print(f"ERROR: no 'models' key in {results_json}", file=sys.stderr)
        sys.exit(1)

    # Pick best classifier by test macro-F1
    best_clf  = max(
        models,
        key=lambda k: ((models[k].get("test_report") or {})
                       .get("macro avg", {}).get("f1-score", -1)),
    )
    best_data = models[best_clf]
    test_rep  = best_data.get("test_report") or {}
    macro_f1  = test_rep.get("macro avg", {}).get("f1-score", float("nan"))
    print(f"Best classifier: {best_clf}  (test macro-F1 = {macro_f1:.3f})")

    per_agent: dict[str, float] = {}
    for key, val in test_rep.items():
        if key in ("macro avg", "weighted avg", "accuracy"):
            continue
        if isinstance(val, dict) and "f1-score" in val:
            per_agent[key] = val["f1-score"]

    return best_clf, per_agent


# ── Capability ────────────────────────────────────────────────────────────────

def load_capability(
    traces_dir: Path,
    datasets: list[str],
    split: str = "test",
) -> dict[str, dict[str, float]]:
    """Returns {agent_id: {base_dataset: accuracy}}."""
    stats: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"c": 0, "n": 0})
    )
    for path in traces_dir.rglob("*.json"):
        parts = path.relative_to(traces_dir).parts
        if parts[0].startswith("classifiers") or len(parts) < 3:
            continue
        agent_id     = parts[0]
        dataset_name = parts[1]
        base         = dataset_name.rsplit("_", 1)[0]
        suffix       = dataset_name.rsplit("_", 1)[-1] if "_" in dataset_name else ""
        if base not in datasets:
            continue
        if split is not None and suffix not in split:
            continue
        try:
            ep = json.loads(path.read_text())
        except Exception:
            continue
        v = ep.get("verification")
        if not v or not v.get("ground_truth"):
            continue
        stats[agent_id][base]["n"] += 1
        stats[agent_id][base]["c"] += int(bool(v.get("correct")))

    return {
        agent: {ds: d["c"] / d["n"] for ds, d in ds_data.items() if d["n"]}
        for agent, ds_data in stats.items()
    }


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot(
    identifiability: dict[str, float],
    capability: dict[str, dict[str, float]],
    datasets: list[str],
    display_names: dict,
    families: dict,
    best_clf: str,
    out: Path,
) -> None:
    ds_labels = {
        "2wikimultihop": "2WikiMultihop accuracy",
        "frames":        "FRAMES accuracy",
    }

    # Build per-panel data — inner join on agents that have all three scores
    panels: list[tuple[str, list, list, list]] = []  # (ds, agents, x, y)
    for ds in datasets:
        agents, xs, ys = [], [], []
        for agent, f1 in identifiability.items():
            cap = capability.get(agent, {}).get(ds)
            if cap is None:
                warnings.warn(f"No {ds} capability score for {agent} — skipped")
                continue
            agents.append(agent)
            xs.append(cap)
            ys.append(f1)
        panels.append((ds, agents, xs, ys))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (ds, agents, xs, ys) in zip(axes, panels):
        rgba = [mcolors.to_rgba(FAMILY_COLORS.get(families.get(a, ""), "#888888"))
                for a in agents]

        # x = identifiability (F1), y = accuracy (capability)
        ax.scatter(ys, xs, c=rgba, s=90, zorder=3, edgecolors="white", linewidths=0.6)

        for agent, x, y in zip(agents, xs, ys):
            label = display_names.get(agent, agent)
            ax.annotate(
                label, (y, x),
                textcoords="offset points", xytext=(6, 3),
                fontsize=7.5, color="#333333",
            )

        # Spearman correlation
        if len(xs) >= 3:
            rho, pval = stats.spearmanr(xs, ys)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            ax.text(0.04, 0.96, f"Spearman ρ = {rho:.2f}{sig}\np = {pval:.3f}  n = {len(xs)}",
                    transform=ax.transAxes, fontsize=9,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

            m, b = np.polyfit(ys, xs, 1)
            y_range = np.linspace(min(ys), max(ys), 100)
            ax.plot(y_range, m * y_range + b, color="#aaaaaa",
                    linewidth=1.2, linestyle="--", zorder=2)

        ax.set_ylabel(ds_labels.get(ds, ds), fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(ds_labels.get(ds, ds).split(" ")[0], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

        n_agents = len(agents)
        print(f"  {ds}: {n_agents} agents, Spearman ρ = {rho:.3f}, p = {pval:.4f}"
              if len(xs) >= 3 else f"  {ds}: {n_agents} agents (too few for correlation)")

    fig.supxlabel(f"Identifiability (F1, {best_clf})", fontsize=11, y=0.01)

    # Family legend
    seen = {}
    for agent in identifiability:
        fam = families.get(agent, "")
        if fam and fam not in seen:
            seen[fam] = FAMILY_COLORS.get(fam, "#888888")
    handles = [mpatches.Patch(facecolor=c, label=f) for f, c in seen.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Capability vs Identifiability", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-json", type=Path, required=True,
                        help="Path to classifiers/{tag}/results.json")
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces)")
    parser.add_argument("--datasets", nargs="+",
                        default=["2wikimultihop", "frames"],
                        help="Datasets to use for capability (default: 2wikimultihop frames)")
    parser.add_argument("--split", nargs="+", default=None,
                        help="Directory suffixes to include for capability, e.g. --split train val test "
                             "(default: all splits with ground truth)")
    parser.add_argument("--out", type=Path,
                        default=Path("./figures/capability_vs_identifiability.png"))
    cli = parser.parse_args()

    display_names, families = _load_config()

    print("Loading identifiability scores...")
    best_clf, identifiability = load_identifiability(cli.results_json)
    print(f"  {len(identifiability)} agents with F1 scores")

    print("\nLoading capability scores...")
    capability = load_capability(cli.traces_dir, cli.datasets, cli.split or None)
    print(f"  {len(capability)} agents with accuracy scores")

    print("\nPlotting...")
    plot(identifiability, capability, cli.datasets,
         display_names, families, best_clf, cli.out)
