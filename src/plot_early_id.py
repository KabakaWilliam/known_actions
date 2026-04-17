#!/usr/bin/env python3
"""
plot_early_id.py — Visualise early-identification prefix curves from run_early_id.sh results.

Reads the 'prefix_curve' key from traces/models/{tag}/results.json and produces:
  1. early_id_accuracy_curve.png  — Accuracy vs prefix size for all classifiers
  2. early_id_per_agent_heatmap.png — Per-agent OOD F1 at key prefix sizes

Usage:
    python plot_early_id.py                           # both wiki_2_frames and frames_2_wiki
    python plot_early_id.py --tags wiki_2_frames      # single experiment
    python plot_early_id.py --mode t_ms               # time-based x-axis instead of event count
    python plot_early_id.py --traces-dir /path/traces # custom traces directory
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

# ── Style constants ────────────────────────────────────────────────────────────
CLF_COLORS = {
    "RandomForest": "#2196F3",   # blue
    "XGBoost":      "#FF9800",   # orange
    "LR_L2":        "#4CAF50",   # green
    "LR_Lasso":     "#9C27B0",   # purple
    "LSTM":         "#F44336",   # red
}
CLF_ORDER = ["RandomForest", "XGBoost", "LR_L2", "LR_Lasso", "LSTM"]

TAG_LABELS = {
    "wiki_2_frames":       "train: 2WikiMultiHop, OOD: FRAMES",
    "frames_2_wiki":       "train: FRAMES, OOD: 2WikiMultiHop",
    "wiki_2_frames_early": "train: 2WikiMultiHop, OOD: FRAMES",
    "frames_2_wiki_early": "train: FRAMES, OOD: 2WikiMultiHop",
    "deepshop_2_webshop": "train: deepshop, OOD: webshop",
    "webshop_2_deepshop": "train: webshop, OOD: deepshop"
}

HEATMAP_COLS_EVENTS = [5, 10, 20, 30, 50, 100, "null"]
HEATMAP_COLS_MS     = [1000, 2000, 5000, 10000, 20000, "null"]
HEATMAP_COL_LABELS_EVENTS = ["5", "10", "20", "30", "50", "100", "all"]
HEATMAP_COL_LABELS_MS     = ["1s", "2s", "5s", "10s", "20s", "all"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean_events_from_traces(
    traces_dir: Path,
    dataset_splits: list[str],
    agents: list[str],
) -> float | None:
    """
    Compute the mean number of DOM events per trace by reading raw trace files.
    traces_dir/  {agent}/  {dataset_split}/  {timestamp}/  {episode}.json
    """
    counts = []
    for agent in agents:
        for ds_split in dataset_splits:
            ds_path = traces_dir / agent / ds_split
            if not ds_path.exists():
                continue
            for ep_file in ds_path.rglob("*.json"):
                try:
                    with open(ep_file) as f:
                        ep = json.load(f)
                    n = len(ep.get("dom_trace", {}).get("events", []))
                    counts.append(n)
                except Exception:
                    pass
    return float(sum(counts) / len(counts)) if counts else None


def load_results(results_path: Path, mode: str) -> tuple[dict | None, dict | None]:
    """Return (prefix_curve[mode], mean_n_events) from a results.json.

    mean_n_events = {"test": float, "ood": float}
    Falls back to computing from raw trace files when the key is absent
    (i.e. results were generated before the field was added).
    """
    try:
        with open(results_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"  [warn] results.json not found: {results_path}", file=sys.stderr)
        return None, None

    pc = data.get("prefix_curve")
    if pc is None:
        print(f"  [warn] 'prefix_curve' key absent in {results_path}  "
              f"(run run_stepwise_classifier.sh first)", file=sys.stderr)
        return None, None

    curve = pc.get(mode)
    if curve is None:
        print(f"  [warn] prefix_curve.{mode} absent in {results_path}", file=sys.stderr)
        return None, None

    mean_n_events = data.get("mean_n_events") or {}
    # If missing (old results), compute from trace files
    if not mean_n_events.get("test") and not mean_n_events.get("ood"):
        traces_dir = results_path.parent.parent.parent  # traces/models/{tag}/ → traces/
        agents     = data.get("class_names", [])
        test_ds    = [d.rsplit("_", 1)[0] + "_" + d.rsplit("_", 1)[1]
                      for d in data.get("test_datasets", [])]
        ood_ds     = [d.rsplit("_", 1)[0] + "_" + d.rsplit("_", 1)[1]
                      for d in data.get("ood_datasets", [])]
        # test_datasets are stored as base names (e.g. "2wikimultihop"); find _test splits
        test_splits = []
        ood_splits  = []
        if traces_dir.exists() and agents:
            sample_agent = agents[0]
            for ds_base in data.get("test_datasets", []):
                for d in (traces_dir / sample_agent).iterdir() if (traces_dir / sample_agent).exists() else []:
                    if d.name.startswith(ds_base) and d.name.endswith("_test"):
                        test_splits.append(d.name)
            for ds_base in data.get("ood_datasets", []):
                for d in (traces_dir / sample_agent).iterdir() if (traces_dir / sample_agent).exists() else []:
                    if d.name.startswith(ds_base):
                        ood_splits.append(d.name)
        mean_test = _mean_events_from_traces(traces_dir, test_splits, agents) if test_splits else None
        mean_ood  = _mean_events_from_traces(traces_dir, ood_splits,  agents) if ood_splits  else None
        if mean_test or mean_ood:
            mean_n_events = {"test": mean_test, "ood": mean_ood}
            print(f"  mean test event len: {mean_test:.0f}" if mean_test else "  mean test event len: n/a")
            print(f"  mean ood  event len: {mean_ood:.0f}"  if mean_ood  else "  mean ood  event len: n/a")

    return curve, mean_n_events


def sorted_sizes(size_keys: list[str]) -> tuple[list, list[str]]:
    """
    Sort prefix size keys numerically, keeping 'null' at the end.
    Returns (numeric_or_none_values, display_labels).
    """
    numeric = sorted([k for k in size_keys if k != "null"], key=int)
    ordered = numeric + (["null"] if "null" in size_keys else [])
    labels  = [k if k != "null" else "all" for k in ordered]
    return ordered, labels


def ood_name(curve_at_size: dict) -> str:
    """Return the first (and usually only) OOD dataset name from a size entry."""
    ood_dict = curve_at_size.get("ood", {})
    if not ood_dict:
        return ""
    return next(iter(ood_dict))


def extract_metric(size_entry: dict, split: str, ood_key: str = "") -> float | None:
    """Pull macro F1 from a size entry for either 'test' or the OOD split.

    Prefers the stored 'macro_f1' key (available after the scaling bugfix re-run).
    Falls back to computing it from per_class_f1, then to 'accuracy' for old results.
    """
    if split == "test":
        d = size_entry.get("test", {})
    else:
        d = size_entry.get("ood", {}).get(ood_key, {})
    if not d:
        return None
    if "macro_f1" in d:
        return d["macro_f1"]
    pcf = d.get("per_class_f1", {})
    if pcf:
        return float(np.mean(list(pcf.values())))
    return d.get("accuracy")  # last-resort fallback for very old results


def elbow_index(values: list[float | None], threshold_frac: float = 0.80) -> int | None:
    """
    Index of the first prefix size where value >= threshold_frac * full-trace value.
    'Full-trace' is the last element (which corresponds to size='null').
    Returns None if no valid values found.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    full = vals[-1]
    target = threshold_frac * full
    for i, v in enumerate(values):
        if v is not None and v >= target:
            return i
    return None



# ── Figure 1: accuracy curves ─────────────────────────────────────────────────

def plot_accuracy_curves(
    tags: list[str],
    curves_by_tag: dict,          # tag → curve dict (prefix_curve[mode])
    mean_n_events_by_tag: dict,   # tag → {"test": float, "ood": float}
    mode: str,
    out_path: Path,
    n_classes: int = 6,
):
    n_tags = len(tags)
    fig, axes = plt.subplots(1, n_tags, figsize=(7 * n_tags, 5), sharey=True)
    if n_tags == 1:
        axes = [axes]

    chance = 1.0 / n_classes

    for ax, tag in zip(axes, tags):
        curve = curves_by_tag.get(tag)
        if curve is None:
            ax.set_title(f"{TAG_LABELS.get(tag, tag)}\n(no data)")
            continue

        # Collect sizes from the first available classifier
        first_clf = next((c for c in CLF_ORDER if c in curve), None)
        if first_clf is None:
            ax.set_title(f"{TAG_LABELS.get(tag, tag)}\n(empty curve)")
            continue

        raw_keys  = list(curve[first_clf].keys())
        sizes_ord, size_labels = sorted_sizes(raw_keys)

        # x positions: use numeric values for proper log scale, map 'null' to max*1.5
        x_numeric = []
        for s in sizes_ord:
            if s == "null":
                x_numeric.append(None)   # placeholder
            else:
                x_numeric.append(int(s))

        # replace None with 1.5 * last numeric
        last_num = max(v for v in x_numeric if v is not None)
        x_vals = [v if v is not None else last_num * 1.5 for v in x_numeric]

        # Determine OOD key once
        first_entry = next(iter(curve[first_clf].values()))
        ood_key = ood_name(first_entry)

        # Track best OOD accs per size (for elbow)
        best_ood_accs = [None] * len(sizes_ord)

        for clf_name in CLF_ORDER:
            if clf_name not in curve:
                continue
            clf_data = curve[clf_name]
            test_accs = [extract_metric(clf_data.get(s, {}), "test") for s in sizes_ord]
            ood_accs  = [extract_metric(clf_data.get(s, {}), "ood", ood_key) for s in sizes_ord]

            color = CLF_COLORS[clf_name]
            ax.plot(x_vals, test_accs, color=color, linewidth=1.8, linestyle="-",
                    label=clf_name)
            ax.plot(x_vals, ood_accs,  color=color, linewidth=1.8, linestyle="--")

            # Update best OOD
            for i, v in enumerate(ood_accs):
                if v is not None and (best_ood_accs[i] is None or v > best_ood_accs[i]):
                    best_ood_accs[i] = v

        # Chance level
        ax.axhline(chance, color="black", linewidth=1.0, linestyle=":",
                   label=f"Chance (1/{n_classes})")

        # Elbow shading
        elbow_i = elbow_index(best_ood_accs)
        if elbow_i is not None:
            lo = x_vals[max(0, elbow_i - 1)]
            hi = x_vals[elbow_i]
            mid = (lo + hi) / 2
            ax.axvspan(lo, hi + (hi - lo) * 0.5, alpha=0.12, color="#FF5722",
                       label="80% elbow")
            ax.axvline(x_vals[elbow_i], color="#FF5722", linewidth=1.0,
                       linestyle="-.", alpha=0.7)

        # Mean trace length lines
        if mode == "n_events":
            mean_stats = mean_n_events_by_tag.get(tag, {})
            mean_test = mean_stats.get("test")
            mean_ood  = mean_stats.get("ood")
            if mean_test:
                ax.axvline(mean_test, color="black", linewidth=2.0, linestyle="-",
                           alpha=0.85, label=f"mean test ({mean_test:.0f} ev)")
            if mean_ood:
                ax.axvline(mean_ood, color="#141414", linewidth=2.0, linestyle="--",
                           alpha=0.9, label=f"mean OOD ({mean_ood:.0f} ev)")

        # x-axis ticks: show only the prefix sizes we used
        ax.set_xscale("log")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(size_labels, rotation=45, ha="right", fontsize=9)

        xlabel = "DOM events observed" if mode == "n_events" else "Milliseconds observed"
        ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
        ax.set_ylabel("Agent identifiability (macro F1)", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_title(TAG_LABELS.get(tag, tag), fontsize=10, pad=8)
        ax.grid(True, which="both", alpha=0.3)

    # Shared legend (classifiers + line style)
    legend_handles = []
    for clf_name in CLF_ORDER:
        if any(clf_name in curves_by_tag.get(t, {}) for t in tags):
            legend_handles.append(
                mpatches.Patch(color=CLF_COLORS[clf_name], label=clf_name)
            )
    legend_handles += [
        plt.Line2D([0], [0], color="gray", linestyle="-",  label="Test split"),
        plt.Line2D([0], [0], color="gray", linestyle="--", label="OOD split"),
        plt.Line2D([0], [0], color="black", linestyle=":", label=f"Chance (1/{n_classes})"),
        mpatches.Patch(color="#FF5722", alpha=0.3, label="80% of peak OOD F1"),
        plt.Line2D([0], [0], color="black",   linewidth=2.0, linestyle="-",  label="Mean test length"),
        plt.Line2D([0], [0], color="#141414", linewidth=2.0, linestyle="--", label="Mean OOD length"), #00BCD4
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(len(legend_handles), 5),
               bbox_to_anchor=(0.5, -0.08), fontsize=9, framealpha=0.9)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ── Figure 2: per-agent heatmap ───────────────────────────────────────────────

def _best_clf_at_full(curve: dict) -> str:
    """Return the classifier with the highest OOD accuracy at the full-trace size ('null')."""
    best_clf, best_acc = "RandomForest", -1.0
    for clf_name in CLF_ORDER:
        if clf_name not in curve:
            continue
        null_entry = curve[clf_name].get("null", {})
        ood_key = ood_name(null_entry)
        acc = extract_metric(null_entry, "ood", ood_key)
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_clf = clf_name
    return best_clf


def plot_per_agent_heatmap(
    tags: list[str],
    curves_by_tag: dict,
    mode: str,
    out_path: Path,
    low_f1_threshold: float = 0.3,
):
    heatmap_cols   = HEATMAP_COLS_EVENTS   if mode == "n_events" else HEATMAP_COLS_MS
    col_labels     = HEATMAP_COL_LABELS_EVENTS if mode == "n_events" else HEATMAP_COL_LABELS_MS

    n_tags = len(tags)
    fig, axes = plt.subplots(1, n_tags, figsize=(4 * len(col_labels) / 7 * n_tags + 2, 5))
    if n_tags == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        curve = curves_by_tag.get(tag)
        if curve is None:
            ax.set_title(f"{TAG_LABELS.get(tag, tag)}\n(no data)")
            continue

        best_clf = _best_clf_at_full(curve)
        clf_data = curve.get(best_clf, {})

        # Get agent list from the full-trace entry
        full_entry = clf_data.get("null", {})
        ood_key    = ood_name(full_entry)
        ood_report = full_entry.get("ood", {}).get(ood_key, {})
        agents     = sorted([k for k in ood_report.get("per_class_f1", {}).keys()])

        if not agents:
            ax.set_title(f"{TAG_LABELS.get(tag, tag)}\n(no per-class F1)")
            continue

        # Build matrix: rows=agents, cols=prefix sizes
        n_agents = len(agents)
        n_cols   = len(heatmap_cols)
        mat = np.full((n_agents, n_cols), np.nan)

        for j, size in enumerate(heatmap_cols):
            size_str = str(size)
            entry = clf_data.get(size_str, {})
            ood_e  = entry.get("ood", {}).get(ood_key, {})
            per_f1 = ood_e.get("per_class_f1", {})
            for i, agent in enumerate(agents):
                val = per_f1.get(agent)
                if val is not None:
                    mat[i, j] = val

        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="YlGn", aspect="auto")

        # Cell annotations + red borders for low F1
        for i in range(n_agents):
            for j in range(n_cols):
                val = mat[i, j]
                if not np.isnan(val):
                    txt_color = "black" if val > 0.5 else "dimgray"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=txt_color)
                    if val < low_f1_threshold:
                        rect = plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, edgecolor="#D32F2F", linewidth=2.0,
                        )
                        ax.add_patch(rect)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(n_agents))
        ax.set_yticklabels(agents, fontsize=9)
        xlabel = "DOM events observed" if mode == "n_events" else "Time observed"
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(
            f"{TAG_LABELS.get(tag, tag)}\nbest clf: {best_clf}",
            fontsize=9, pad=6,
        )
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="OOD F1")

    fig.suptitle(
        "Per-agent OOD F1 at truncated prefixes"
        f"  ({'n events' if mode == 'n_events' else 'ms'})",
        fontsize=12, y=1.02,
    )
    # Legend for red border
    red_patch = mpatches.Patch(fill=False, edgecolor="#D32F2F", linewidth=2, label="F1 < 0.30")
    fig.legend(handles=[red_patch], loc="lower center",
               bbox_to_anchor=(0.5, -0.05), fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot early-identification prefix curves from run_early_id.sh results."
    )
    parser.add_argument(
        "--tags", nargs="+",
        default=["wiki_2_frames_early", "frames_2_wiki_early"],
        help="Experiment tags to include (default: wiki_2_frames_early frames_2_wiki_early)",
    )
    parser.add_argument(
        "--mode", choices=["n_events", "t_ms"], default="n_events",
        help="Prefix axis: 'n_events' (DOM event count) or 't_ms' (milliseconds). Default: n_events",
    )
    parser.add_argument(
        "--traces-dir", default="./traces",
        help="Root traces directory (default: ./traces)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for PNGs (default: traces-dir/models/<first-tag>/)",
    )
    args = parser.parse_args()

    traces_dir = Path(args.traces_dir)
    mode       = args.mode
    tags       = args.tags

    # Load curves
    curves_by_tag: dict = {}
    mean_n_events_by_tag: dict = {}
    for tag in tags:
        results_path = traces_dir / "models" / tag / "results.json"
        curve, mean_n_events = load_results(results_path, mode)
        if curve is not None:
            curves_by_tag[tag] = curve
        if mean_n_events is not None:
            mean_n_events_by_tag[tag] = mean_n_events
        
        # If mean_n_events is empty, try to compute from traces
        if tag not in mean_n_events_by_tag or not mean_n_events_by_tag[tag]:
            computed = compute_mean_n_events_from_traces(tag, traces_dir)
            if computed:
                mean_n_events_by_tag[tag] = computed
    
    print(mean_n_events_by_tag)

    if not curves_by_tag:
        print("No prefix_curve data found for any requested tag. "
              "Run run_stepwise_classifier.sh first.", file=sys.stderr)
        sys.exit(1)

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # Save alongside the first tag that has data
        first_tag = next(iter(curves_by_tag))
        out_dir = traces_dir / "models" / first_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_suffix = "events" if mode == "n_events" else "ms"

    print(f"Generating figures for tags: {', '.join(curves_by_tag.keys())}  (mode={mode})")

    # Figure 1 — accuracy vs prefix size
    plot_accuracy_curves(
        tags=[t for t in tags if t in curves_by_tag],
        curves_by_tag=curves_by_tag,
        mean_n_events_by_tag=mean_n_events_by_tag,
        mode=mode,
        out_path=out_dir / f"early_id_accuracy_curve_{mode_suffix}.png",
    )

    # Figure 2 — per-agent OOD F1 heatmap
    plot_per_agent_heatmap(
        tags=[t for t in tags if t in curves_by_tag],
        curves_by_tag=curves_by_tag,
        mode=mode,
        out_path=out_dir / f"early_id_per_agent_heatmap_{mode_suffix}.png",
    )

    print("Done.")


if __name__ == "__main__":
    main()
