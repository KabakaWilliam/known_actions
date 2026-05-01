#!/usr/bin/env python3
"""
plot_open_set_summary.py — 2-panel open-set classification figure.

Panel A: Per-agent AUROC (leave-one-out on 2WikiMultihopQA).
         RF (blue) and XGBoost (orange) bars per agent, sorted by mean AUROC.
         Background stripe indicates proprietary (salmon) vs open-source (slate).
         Dashed line at AUROC=0.5 (random chance).

Panel B: Scatter — closed-set Macro F1 vs open-set AUROC.
         Points coloured proprietary/open-source; agent labels shown.
         Highlights whether closed-set identifiability predicts open-set detectability.

Usage:
    python plot_open_set_summary.py
    python plot_open_set_summary.py --traces-dir src/traces --out my_fig.png
"""

SCATTER_LEAD = (
    r"\paragraph{Closed-set identifiability does not predict open-set detectability.}"
    " Agents that are easiest to classify when their identity is known are not "
    "easiest to flag as unknown when withheld from training. The negative trend in "
    r"\cref{fig:open_set_scatter} shows these two axes of identifiability are "
    "largely orthogonal, with Seed-2 as the sharpest illustration: it achieves "
    "the highest per-agent F1 under closed-set evaluation yet an open-set AUROC "
    "of only 0.47 — indistinguishable from chance."
)

SCATTER_CAPTION = (
    "Closed-set identifiability versus open-set detectability for 14 agents on "
    "2WikiMultihopQA. Each point shows one agent's XGBoost Macro F1 under "
    "closed-set classification (x-axis) against the best AUROC achieved by any "
    "classifier in a leave-one-out open-set experiment (y-axis). Points are "
    "coloured by model family (proprietary: orange; open-source: blue). The "
    "dashed line marks chance-level AUROC (0.5). The negative trend indicates "
    "that high closed-set identifiability does not predict open-set detectability: "
    "agents whose behaviour is most distinctive within the training distribution "
    "are not necessarily easier to detect as unknown. Seed-2 exemplifies this "
    "dissociation — it achieves the highest per-agent F1 in closed-set evaluation "
    "yet the lowest open-set AUROC (0.47), suggesting its features occupy an "
    "isolated region of the feature space that classifiers cannot generalise from."
)

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Colours ────────────────────────────────────────────────────────────────────
RF_COLOUR       = "#2196F3"   # blue
XGB_COLOUR      = "#FF9800"   # orange
PROP_BAR_BG     = "#FFEBE0"   # light salmon background stripe
OS_BAR_BG       = "#E8EEF8"   # light slate background stripe
PROP_POINT      = "#E07850"   # salmon — proprietary scatter points
OS_POINT        = "#6A8DB8"   # slate  — open-source scatter points
CHANCE_COLOUR   = "#9E9E9E"   # grey dashed line

# ── Proprietary families ───────────────────────────────────────────────────────
PROPRIETARY_FAMILIES = {"gpt_5", "claude_4", "gemini_3", "gemini_3_flash"}

# ── Display name mapping ───────────────────────────────────────────────────────
AGENT_LABELS = {
    "gpt_5_4":           "GPT-5",
    "claude_opus_4_6":   "Claude 4",
    "gemini_3_1":        "Gemini-3",
    "gemini_3_flash":    "Gemini-3-Flash",
    "gemma-4-31B-it":    "Gemma-4 (31B)",
    "gemma_4_26B_A4B_it":"Gemma-4 (26B)",
    "glm_4.6v":          "GLM-4.6V",
    "glm_4.6v_flash":    "GLM-4.6V-Flash",
    "qwen3vl_8b":        "Qwen3-VL-8B",
    "qwen3vl_30b_a3b":   "Qwen3-VL-30B",
    "qwen3_5_27b":       "Qwen3.5-27B",
    "qwen3_5_9b":        "Qwen3.5-9B",
    "seed_2_lite":       "Seed-2",
    "uitars_7b":         "UI-TARS-1.5",
}


# ── Per-agent marker + colour (unique across all 14 agents) ───────────────────

AGENT_STYLES = {
    "gpt_5_4":            {"marker": "o",  "color": "#E53935"},
    "claude_opus_4_6":    {"marker": "s",  "color": "#FF7043"},
    "gemini_3_1":         {"marker": "^",  "color": "#FFA726"},
    "gemini_3_flash":     {"marker": "D",  "color": "#FFCA28"},
    "gemma-4-31B-it":     {"marker": "v",  "color": "#66BB6A"},
    "gemma_4_26B_A4B_it": {"marker": "P",  "color": "#26C6DA"},
    "glm_4.6v":           {"marker": ">",  "color": "#42A5F5"},
    "glm_4.6v_flash":     {"marker": "<",  "color": "#7E57C2"},
    "qwen3vl_8b":         {"marker": "*",  "color": "#AB47BC"},
    "qwen3vl_30b_a3b":    {"marker": "h",  "color": "#EC407A"},
    "qwen3_5_27b":        {"marker": "H",  "color": "#78909C"},
    "qwen3_5_9b":         {"marker": "X",  "color": "#8D6E63"},
    "uitars_7b":          {"marker": "p",  "color": "#5C6BC0"},
    "seed_2_lite":        {"marker": "8",  "color": "#26A69A"},
}


# ── Dataset registry ───────────────────────────────────────────────────────────

DATASET_SUBDIRS = [
    ("2wikimultihop_open_set", "2WikiMultiHop"),
    ("frames_open_set",        "FRAMES"),
    ("webshop_open_set",       "WebShop"),
    ("deepshop_open_set",      "DeepShop"),
    ("wiki_frames_open_set",   "Wiki+FRAMES"),
    ("ws_deepshop_open_set",   "WebShop+DeepShop"),
]


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_family_map(traces_dir: Path) -> dict[str, str]:
    """Load agent_id → family from config.yaml."""
    try:
        import yaml
    except ImportError:
        sys.exit("ERROR: pyyaml not installed. Run: pip install pyyaml")
    config_path = traces_dir.parent / "config.yaml"
    if not config_path.exists():
        # Try sibling of traces_dir
        config_path = traces_dir / ".." / "config.yaml"
    if not config_path.exists():
        print("[WARN] config.yaml not found — proprietary/OS split unavailable.")
        return {}
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {a["agent_id"]: a["family"] for a in cfg.get("agents", []) if "agent_id" in a}


def load_open_set_auroc(traces_dir: Path, loo_subdir: str = "2wikimultihop_open_set") -> dict[str, dict]:
    """
    Returns {agent_id: {"rf": auroc, "xgb": auroc}} for all LOO experiments.
    """
    loo_dir = traces_dir / "classifiers" / loo_subdir
    if not loo_dir.exists():
        # When called from load_all_datasets, missing subdirs are silently skipped.
        # When called directly (--plot bars/scatter), caller decides whether to exit.
        return {}


    results = {}
    for exp_dir in sorted(loo_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("open_set_loo_"):
            continue
        agent_id = exp_dir.name[len("open_set_loo_"):]
        rpath = exp_dir / "results.json"
        if not rpath.exists():
            continue
        with open(rpath) as f:
            r = json.load(f)
        os_data = r.get("open_set") or {}
        rf_auroc  = (os_data.get("RandomForest") or {}).get("auroc")
        xgb_auroc = (os_data.get("XGBoost") or {}).get("auroc")
        if rf_auroc is None and xgb_auroc is None:
            continue
        results[agent_id] = {
            "rf":  rf_auroc  or 0.0,
            "xgb": xgb_auroc or 0.0,
        }
    return results


def load_all_datasets(traces_dir: Path) -> list[dict]:
    """Return [{label, rf_aurocs, xgb_aurocs}, ...] for each available dataset."""
    rows = []
    for subdir, label in DATASET_SUBDIRS:
        data = load_open_set_auroc(traces_dir, subdir)
        if not data:
            continue
        rows.append({
            "label":      label,
            "rf_aurocs":  [v["rf"]  for v in data.values()],
            "xgb_aurocs": [v["xgb"] for v in data.values()],
        })
    return rows


def load_datasets_xgb(traces_dir: Path, subdirs: list[tuple]) -> list[dict]:
    """Return [{label, agent_aurocs: {agent_id: xgb_auroc}}, ...] for each subdir."""
    rows = []
    for subdir, label in subdirs:
        loo_dir = traces_dir / "classifiers" / subdir
        if not loo_dir.exists():
            continue
        agent_aurocs = {}
        for exp_dir in sorted(loo_dir.iterdir()):
            if not exp_dir.is_dir() or not exp_dir.name.startswith("open_set_loo_"):
                continue
            agent_id = exp_dir.name[len("open_set_loo_"):]
            rpath = exp_dir / "results.json"
            if not rpath.exists():
                continue
            with open(rpath) as f:
                r = json.load(f)
            auroc = ((r.get("open_set") or {}).get("XGBoost") or {}).get("auroc")
            if auroc is not None:
                agent_aurocs[agent_id] = float(auroc)
        if agent_aurocs:
            rows.append({"label": label, "agent_aurocs": agent_aurocs})
    return rows


def load_closed_set_f1(traces_dir: Path, closed_set_tag: str = "wiki_ood_all") -> dict[str, float]:
    """Per-agent Macro F1 from XGBoost for the given closed-set experiment tag."""
    rpath = traces_dir / "classifiers" / closed_set_tag / "results.json"
    if not rpath.exists():
        print(f"[WARN] {closed_set_tag}/results.json not found — Panel B unavailable.")
        return {}
    with open(rpath) as f:
        r = json.load(f)
    tr = ((r.get("models") or {}).get("XGBoost") or {}).get("test_report") or {}
    return {
        agent: v["f1-score"]
        for agent, v in tr.items()
        if isinstance(v, dict) and "f1-score" in v
        and agent not in ("macro avg", "weighted avg", "accuracy")
    }


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def _is_proprietary(agent_id: str, family_map: dict) -> bool:
    return family_map.get(agent_id, "") in PROPRIETARY_FAMILIES


def plot_panel_a(ax, auroc_data: dict, family_map: dict, label: str = "A"):
    agents = list(auroc_data.keys())
    agents.sort(key=lambda a: (auroc_data[a]["rf"] + auroc_data[a]["xgb"]) / 2)

    n = len(agents)
    y = np.arange(n)
    h = 0.35

    rf_vals  = [auroc_data[a]["rf"]  for a in agents]
    xgb_vals = [auroc_data[a]["xgb"] for a in agents]
    labels   = [AGENT_LABELS.get(a, a) for a in agents]

    # Background stripe per agent row
    for i, agent in enumerate(agents):
        bg = PROP_BAR_BG if _is_proprietary(agent, family_map) else OS_BAR_BG
        ax.axhspan(i - 0.5, i + 0.5, color=bg, alpha=0.55, zorder=0)

    ax.barh(y + h/2, rf_vals,  h, color=RF_COLOUR,  alpha=0.88, label="Random Forest", zorder=2)
    ax.barh(y - h/2, xgb_vals, h, color=XGB_COLOUR, alpha=0.88, label="XGBoost",       zorder=2)

    # Value annotations
    for i, (rv, xv) in enumerate(zip(rf_vals, xgb_vals)):
        ax.text(rv + 0.008, i + h/2, f"{rv:.2f}", va="center", fontsize=7.5, color=RF_COLOUR)
        ax.text(xv + 0.008, i - h/2, f"{xv:.2f}", va="center", fontsize=7.5, color=XGB_COLOUR)

    # Chance line
    ax.axvline(0.5, color=CHANCE_COLOUR, linestyle="--", linewidth=1.2, zorder=1, label="Chance (0.5)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("AUROC", fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    # Legend: classifiers + proprietary/OS
    clf_handles = [
        mpatches.Patch(color=RF_COLOUR,  label="Random Forest"),
        mpatches.Patch(color=XGB_COLOUR, label="XGBoost"),
        plt.Line2D([0], [0], color=CHANCE_COLOUR, linestyle="--", label="Chance"),
    ]
    prop_handles = [
        mpatches.Patch(facecolor=PROP_BAR_BG, label="Proprietary", edgecolor="#888", linewidth=0.7),
        mpatches.Patch(facecolor=OS_BAR_BG,   label="Open-source", edgecolor="#888", linewidth=0.7),
    ]
    ax.legend(handles=clf_handles + prop_handles, fontsize=8, loc="lower right", framealpha=0.8)
    if label:
        ax.set_title(label, fontsize=11, fontweight="bold", loc="left")

    _style_ax(ax)


def plot_panel_b(ax, auroc_data: dict, closed_f1: dict, family_map: dict, label: str = "B"):
    common = [a for a in auroc_data if a in closed_f1]
    if not common:
        ax.text(0.5, 0.5, "No matching data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="grey")
        return

    xs = [closed_f1[a] * 100 for a in common]
    ys = [max(auroc_data[a]["rf"], auroc_data[a]["xgb"]) for a in common]
    colours = [PROP_POINT if _is_proprietary(a, family_map) else OS_POINT for a in common]
    labels  = [AGENT_LABELS.get(a, a) for a in common]

    ax.scatter(xs, ys, c=colours, s=60, zorder=3, edgecolors="white", linewidth=0.6)

    # Agent labels with simple collision avoidance
    for x, y_val, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y_val),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5, color="black", zorder=4)

    # Reference line
    ax.axhline(0.5, color=CHANCE_COLOUR, linestyle="--", linewidth=1.2, label="Chance AUROC")

    # Trend line
    if len(xs) > 2:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        xr = np.linspace(min(xs), max(xs), 100)
        ax.plot(xr, p(xr), color="#AAAAAA", linewidth=1.2, linestyle=":", zorder=2)

    ax.set_xlabel("Closed-set Macro F1 (%)", fontsize=10)
    ax.set_ylabel("Open-set AUROC (best of RF/XGB)", fontsize=10)
    ax.set_ylim(0.25, 1.0)

    prop_handles = [
        mpatches.Patch(color=PROP_POINT, label="Proprietary"),
        mpatches.Patch(color=OS_POINT,   label="Open-source"),
        plt.Line2D([0], [0], color=CHANCE_COLOUR, linestyle="--", label="Chance AUROC"),
    ]
    ax.legend(handles=prop_handles, fontsize=8, loc="lower right", framealpha=0.8)
    if label:
        ax.set_title(label, fontsize=11, fontweight="bold", loc="left")
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    _style_ax(ax)


# ── Cross-dataset panel ────────────────────────────────────────────────────────

def plot_cross_dataset(ax, rows: list[dict]):
    offsets    = {"rf": -0.15, "xgb": +0.15}
    colours    = {"rf": RF_COLOUR, "xgb": XGB_COLOUR}
    clf_labels = {"rf": "Random Forest", "xgb": "XGBoost"}

    for i, row in enumerate(rows):
        for clf in ("rf", "xgb"):
            vals   = np.array(row[f"{clf}_aurocs"])
            x0     = i + offsets[clf]
            rng    = np.random.default_rng(42)
            jitter = rng.uniform(-0.06, 0.06, len(vals))
            ax.scatter(x0 + jitter, vals,
                       color=colours[clf], alpha=0.55, s=22, zorder=3,
                       label=clf_labels[clf] if i == 0 else "_nolegend_")
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.add_patch(mpatches.FancyBboxPatch(
                (x0 - 0.10, q1), 0.20, q3 - q1,
                boxstyle="square,pad=0",
                linewidth=1.2, edgecolor=colours[clf],
                facecolor=colours[clf], alpha=0.18, zorder=2))
            ax.plot([x0 - 0.10, x0 + 0.10], [med, med],
                    color=colours[clf], linewidth=2.0, zorder=4)

    ax.axhline(0.5, color=CHANCE_COLOUR, linestyle="--",
               linewidth=1.2, label="Chance (0.5)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["label"] for r in rows], fontsize=10)
    ax.set_ylabel("Open-set AUROC", fontsize=10)
    ax.set_ylim(0.3, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.8)
    _style_ax(ax)


# ── XGBoost-only per-agent strip ──────────────────────────────────────────────

def plot_xgb_strip(ax, rows: list[dict]):
    seen_agents: set = set()
    for i, row in enumerate(rows):
        agents = list(row["agent_aurocs"].keys())
        vals   = [row["agent_aurocs"][a] for a in agents]
        rng    = np.random.default_rng(42 + i)
        jitter = rng.uniform(-0.18, 0.18, len(agents))
        for j, (agent, auroc) in enumerate(zip(agents, vals)):
            style = AGENT_STYLES.get(agent, {"marker": "o", "color": "#999999"})
            lbl   = AGENT_LABELS.get(agent, agent) if agent not in seen_agents else "_nolegend_"
            seen_agents.add(agent)
            ax.scatter(i + jitter[j], auroc,
                       marker=style["marker"], color=style["color"],
                       s=55, alpha=0.85, zorder=3,
                       edgecolors="white", linewidths=0.4,
                       label=lbl)
        if vals:
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.add_patch(mpatches.FancyBboxPatch(
                (i - 0.22, q1), 0.44, q3 - q1,
                boxstyle="square,pad=0",
                linewidth=1.0, edgecolor="#888888",
                facecolor="#888888", alpha=0.12, zorder=2))
            ax.plot([i - 0.22, i + 0.22], [med, med],
                    color="#555555", linewidth=1.8, zorder=4)

    ax.axhline(0.5, color=CHANCE_COLOUR, linestyle="--",
               linewidth=1.2, label="Chance (0.5)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["label"] for r in rows], fontsize=10)
    ax.set_ylabel("Open-set AUROC", fontsize=10)
    ax.set_ylim(0.3, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    _style_ax(ax)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot open-set (LOO) AUROC summary for 2WikiMultihopQA."
    )
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default varies by --plot mode).")
    parser.add_argument("--format", choices=["png", "pdf"], default="png",
                        help="Output format (default: png). Ignored if --out has an extension.")
    parser.add_argument("--plot", choices=["both", "bars", "scatter", "cross_dataset", "xgb_strip"],
                        default="both",
                        help="Which panel(s) to produce: both (default), bars (Panel A only), "
                             "scatter (Panel B only), cross_dataset (all datasets summary), "
                             "xgb_strip (XGBoost per-agent strip, use with --loo-subdirs).")
    parser.add_argument("--loo-subdirs", nargs="+",
                        default=["2wikimultihop_open_set", "frames_open_set",
                                 "webshop_open_set", "deepshop_open_set"],
                        help="Classifier subdirs to compare in xgb_strip mode "
                             "(default: the 4 single-dataset open-set dirs).")
    parser.add_argument("--loo-subdir", default="2wikimultihop_open_set",
                        help="Subdirectory under traces/classifiers/ containing open_set_loo_* "
                             "experiment dirs (default: 2wikimultihop_open_set).")
    parser.add_argument("--closed-set-tag", default="wiki_ood_all",
                        help="Experiment tag for the closed-set F1 baseline used in Panel B "
                             "(default: wiki_ood_all).")
    args = parser.parse_args()

    if args.plot == "cross_dataset":
        rows = load_all_datasets(args.traces_dir)
        if not rows:
            sys.exit("ERROR: no open-set data found for any dataset.")
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_cross_dataset(ax, rows)
        ax.set_title("Open-set unknown-agent detection",
                     fontsize=10, loc="left", fontweight="semibold")
        plt.tight_layout()
        out_dir = args.traces_dir / "classifiers"
        out = args.out or out_dir / "open_set_cross_dataset.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")
        return

    if args.plot == "xgb_strip":
        subdir_label_map = dict(DATASET_SUBDIRS)
        subdirs = [(s, subdir_label_map.get(s, s)) for s in args.loo_subdirs]
        rows = load_datasets_xgb(args.traces_dir, subdirs)
        if not rows:
            sys.exit("ERROR: no XGBoost open-set data found for any specified subdir.")
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_xgb_strip(ax, rows)
        ax.set_title("Open-set unknown-agent detection (XGBoost)",
                     fontsize=11, loc="left", fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.10), ncol=5,
                   frameon=False, fontsize=8, handlelength=1.2)
        fig.subplots_adjust(bottom=0.18)
        fmt     = args.format
        fig_dir = args.traces_dir.parent / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        out = args.out or fig_dir / f"open_set_xgb_strip.{fmt}"
        out = out.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")
        return

    auroc_data = load_open_set_auroc(args.traces_dir, args.loo_subdir)
    TITLE_MAP = {
        "2wikimultihop_open_set": '2wikimultihop',
        "ws_deepshop_open_set": "Amazon",
        "wiki_frames_open_set": "Wikipedia",
    }
    if not auroc_data:
        sys.exit("ERROR: no open-set LOO results found.")
    print(f"Loaded AUROC data for {len(auroc_data)} agents.")

    closed_f1  = load_closed_set_f1(args.traces_dir, args.closed_set_tag)
    family_map = _load_family_map(args.traces_dir)
    out_dir    = args.traces_dir / "classifiers"

    if args.plot == "scatter":
        if not closed_f1:
            sys.exit(f"ERROR: {args.closed_set_tag}/results.json not found — scatter unavailable.")
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_panel_b(ax, auroc_data, closed_f1, family_map, label="")
        ax.set_title(f"Open-set agent detection — {TITLE_MAP[args.loo_subdir]} (leave-one-out)",
                     fontsize=10)
        plt.tight_layout()
        out = args.out or out_dir / "open_set_scatter.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")
        return

    if args.plot == "bars":
        n_agents   = len(auroc_data)
        fig_height = max(5, n_agents * 0.48 + 1.5)
        fig, ax = plt.subplots(figsize=(8, fig_height))
        plot_panel_a(ax, auroc_data, family_map, label="")
        ax.set_title("Open-set agent detection — 2WikiMultihopQA (leave-one-out)",
                     fontsize=10)
        plt.tight_layout()
        out = args.out or out_dir / "open_set_bars.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")
        return

    # both
    n_agents  = len(auroc_data)
    fig_height = max(5, n_agents * 0.48 + 1.5)
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(13, fig_height),
        gridspec_kw={"width_ratios": [1.4, 1]},
    )

    plot_panel_a(ax_a, auroc_data, family_map)
    if closed_f1:
        plot_panel_b(ax_b, auroc_data, closed_f1, family_map)
    else:
        ax_b.set_visible(False)

    fig.suptitle("Open-set agent detection — 2WikiMultihopQA (leave-one-out)",
                 fontsize=11, y=1.01)
    plt.tight_layout()

    out = args.out or out_dir / "open_set_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
