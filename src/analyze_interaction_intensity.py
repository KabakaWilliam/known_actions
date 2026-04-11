"""
analyze_interaction_intensity.py

Quantifies "interaction intensity" per dataset from existing DOM traces,
then cross-tabulates with classifier transfer (OOD) performance to test
the thesis that fingerprinting strength tracks along the spectrum:

  2wikimultihop (reasoning-heavy)
  → webshop (structured interaction)
  → deepshop (repeated refinement)
  → webgames (sustained, diverse interaction)

Usage:
  python analyze_interaction_intensity.py
  python analyze_interaction_intensity.py --traces-dir ./traces --results-dir ./traces/models
"""

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

# ── helpers ───────────────────────────────────────────────────────────────────

def _shannon_entropy(counts: Counter) -> float:
    """Shannon entropy (bits) over a distribution of event types."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def _episode_metrics(ep: dict) -> dict | None:
    """Extract per-episode interaction intensity metrics from episode JSON."""
    dom = ep.get("dom_trace", {})
    events = dom.get("events", [])
    if not events:
        return None

    duration_ms = dom.get("episodeDuration", 0)
    types = Counter(e["type"] for e in events)
    urls  = {e.get("url", "") for e in events if e.get("url")}

    # Sequential diversity metrics
    event_types_seq = [e["type"] for e in events]
    bigrams = Counter(zip(event_types_seq[:-1], event_types_seq[1:]))
    transition_entropy = _shannon_entropy(bigrams)

    n_types = len(types)
    action_entropy = _shannon_entropy(types)
    action_entropy_norm = (action_entropy / math.log2(n_types)) if n_types > 1 else 0.0

    action_target_pairs = Counter(
        (e["type"], e.get("target_tag", "")) for e in events
    )
    n_unique_action_targets = len(action_target_pairs)

    n_navigations = types.get("navigate", 0)
    url_revisit_rate = (n_navigations / len(urls)) if urls else 0.0

    switches = sum(1 for a, b in zip(event_types_seq[:-1], event_types_seq[1:]) if a != b)
    action_switch_rate = switches / (len(events) - 1) if len(events) > 1 else 0.0

    # Inter-event intervals (ms) — only events with t_episode
    ts = sorted(e["t_episode"] for e in events if "t_episode" in e)
    ieis = [ts[i+1] - ts[i] for i in range(len(ts) - 1)] if len(ts) > 1 else []

    # Midscene task-level metrics
    tasks = []
    for ex in ep.get("midscene_log", {}).get("executions", []):
        tasks.extend(ex.get("tasks", []))
    subtypes    = [t.get("subType") for t in tasks if t.get("subType")]
    plan_count  = subtypes.count("Plan")
    action_subtypes = [s for s in subtypes if s not in ("Plan", "Locate", "Query", "Error")]
    token_total = sum(t.get("usage", {}).get("total_tokens", 0) for t in tasks)

    return {
        "n_events":                 len(events),
        "n_unique_urls":            len(urls),
        "duration_ms":              duration_ms,
        "action_entropy":           action_entropy,
        "action_entropy_norm":      action_entropy_norm,
        "transition_entropy":       transition_entropy,
        "n_unique_action_targets":  n_unique_action_targets,
        "url_revisit_rate":         url_revisit_rate,
        "action_switch_rate":       action_switch_rate,
        "n_action_types":           n_types,
        "mean_iei_ms":              float(np.mean(ieis)) if ieis else 0.0,
        "n_navigations":            n_navigations,
        "n_clicks":                 types.get("click", 0),
        "n_keypresses":             types.get("keydown", 0) + types.get("keypress", 0),
        "n_scrolls":                types.get("scroll", 0),
        "n_plans":                  plan_count,
        "n_action_steps":           len(action_subtypes),
        "total_tokens":             token_total,
        "has_error":                ep.get("error") is not None,
    }


def load_dataset_metrics(traces_dir: Path) -> dict[str, list[dict]]:
    """Walk traces/ and collect per-episode metrics grouped by dataset base name."""
    # Directory layout: traces/{agent_id}/{dataset_name}/{run_ts}/*.json
    dataset_episodes: dict[str, list[dict]] = {}

    for agent_dir in sorted(traces_dir.iterdir()):
        if not agent_dir.is_dir() or agent_dir.name == "models":
            continue
        for ds_dir in sorted(agent_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds_name = ds_dir.name
            for run_dir in sorted(ds_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                for ep_file in sorted(run_dir.glob("*.json")):
                    try:
                        ep = json.loads(ep_file.read_text())
                    except Exception:
                        continue
                    m = _episode_metrics(ep)
                    if m is None:
                        continue
                    m["agent_id"] = ep.get("meta", {}).get("agent_id", "unknown")
                    dataset_episodes.setdefault(ds_name, []).append(m)

    return dataset_episodes


def aggregate(episodes: list[dict], keys: list[str]) -> dict[str, float]:
    """Mean ± std for each metric key over a list of episode dicts."""
    out = {}
    for k in keys:
        vals = [e[k] for e in episodes if k in e]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{k}_std"]  = float(np.std(vals))  if vals else float("nan")
    return out


INTENSITY_KEYS = [
    "n_events", "n_unique_urls", "duration_ms", "action_entropy",
    "action_entropy_norm", "transition_entropy", "n_unique_action_targets",
    "url_revisit_rate", "action_switch_rate",
    "n_action_types", "n_navigations", "n_clicks", "n_keypresses",
    "n_scrolls", "n_plans", "n_action_steps", "total_tokens",
]

# Canonical dataset ordering for the spectrum table
SPECTRUM_ORDER = [
    "2wikimultihop_train", "2wikimultihop_val", "2wikimultihop_test",
    "webshop_train",        "webshop_val",        "webshop_test",
    "deepshop_ood",
    "webgames_train",       "webgames_val",        "webgames_test",
]


def canonical_base(ds_name: str) -> str:
    """Strip _train/_val/_test/_ood suffix to get the base dataset name."""
    for suffix in ("_train", "_val", "_test", "_ood"):
        if ds_name.endswith(suffix):
            return ds_name[: -len(suffix)]
    return ds_name


# ── OOD transfer performance ───────────────────────────────────────────────────

def load_ood_performance(results_dir: Path) -> dict[str, dict[str, float]]:
    """
    Load OOD macro F1 per (experiment_tag, ood_dataset) from results.json files.

    Returns: { experiment_tag: { ood_dataset_base: macro_f1 } }
    """
    out: dict[str, dict[str, float]] = {}
    for tag_dir in sorted(results_dir.iterdir()):
        if not tag_dir.is_dir():
            continue
        rfile = tag_dir / "results.json"
        if not rfile.exists():
            continue
        try:
            data = json.loads(rfile.read_text())
        except Exception:
            continue
        tag = tag_dir.name
        out[tag] = {}
        # results.json structure: { models: { classifier: { ood_reports: { ds_base: report } } } }
        clf_f1s: dict[str, list[float]] = {}
        for clf_results in (data.get("models") or {}).values():
            if not isinstance(clf_results, dict):
                continue
            for ood_name, report in (clf_results.get("ood_reports") or {}).items():
                if not isinstance(report, dict):
                    continue
                f1 = report.get("macro avg", {}).get("f1-score", float("nan"))
                clf_f1s.setdefault(ood_name, []).append(f1)
        for ood_name, f1s in clf_f1s.items():
            valid = [f for f in f1s if not math.isnan(f)]
            out[tag][ood_name] = float(np.mean(valid)) if valid else float("nan")
    return out


# ── formatting ────────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 20) -> str:
    filled = int(round(value / max_val * width)) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def print_intensity_table(agg_by_base: dict[str, dict]) -> None:
    COLS = [
        ("n_events",      "Events/ep"),
        ("n_unique_urls", "Unique URLs"),
        ("n_clicks",      "Clicks"),
        ("n_keypresses",  "Keypresses"),
        ("n_scrolls",     "Scrolls"),
        ("n_navigations", "Navigations"),
        ("action_entropy","Entropy(bits)"),
        ("n_plans",       "Plan steps"),
        ("duration_ms",   "Duration(s)"),
        ("total_tokens",  "Tokens"),
    ]

    print("\n" + "═" * 100)
    print("  INTERACTION INTENSITY BY DATASET")
    print("═" * 100)
    header = f"  {'Dataset':<25}" + "".join(f"  {label:>13}" for _, label in COLS)
    print(header)
    print("─" * 100)

    for base in sorted(agg_by_base.keys()):
        a = agg_by_base[base]
        n = a.get("_n", 0)
        row = f"  {base:<25}"
        for key, _ in COLS:
            val = a.get(f"{key}_mean", float("nan"))
            if key == "duration_ms":
                display = f"{val/1000:.1f}s"
            elif key == "action_entropy":
                display = f"{val:.2f}"
            elif math.isnan(val):
                display = "—"
            else:
                display = f"{val:.1f}"
            row += f"  {display:>13}"
        row += f"  (n={n})"
        print(row)

    print("═" * 100)


def print_diversity_table(agg_by_base: dict[str, dict]) -> None:
    """Print the sequential interaction diversity metrics."""
    COLS = [
        ("transition_entropy",      "Trans.Entropy"),
        ("action_entropy_norm",     "Entropy(norm)"),
        ("n_unique_action_targets", "ActionTargets"),
        ("url_revisit_rate",        "URLRevisitRate"),
        ("action_switch_rate",      "SwitchRate"),
    ]

    print("\n" + "═" * 90)
    print("  INTERACTION DIVERSITY BY DATASET")
    print("  (transition_entropy = sequential unpredictability; switch_rate = type-change frequency)")
    print("═" * 90)
    header = f"  {'Dataset':<25}" + "".join(f"  {label:>15}" for _, label in COLS)
    print(header)
    print("─" * 90)

    for base in sorted(agg_by_base.keys()):
        a = agg_by_base[base]
        n = a.get("_n", 0)
        row = f"  {base:<25}"
        for key, _ in COLS:
            val = a.get(f"{key}_mean", float("nan"))
            if math.isnan(val):
                display = "—"
            elif key in ("action_entropy_norm", "action_switch_rate", "url_revisit_rate"):
                display = f"{val:.3f}"
            else:
                display = f"{val:.2f}"
            row += f"  {display:>15}"
        row += f"  (n={n})"
        print(row)

    print("═" * 90)


def print_spectrum_ranking(agg_by_base: dict[str, dict]) -> None:
    """Rank datasets by composite interaction intensity + diversity score."""
    print("\n" + "═" * 70)
    print("  INTERACTION INTENSITY + DIVERSITY SPECTRUM  (composite rank)")
    print("═" * 70)

    scores = {}
    for base, a in agg_by_base.items():
        # Composite: volume + structural diversity
        scores[base] = (
            a.get("n_events_mean", 0) / 200 +
            a.get("n_unique_urls_mean", 0) / 20 +
            a.get("transition_entropy_mean", 0) / 3 +
            a.get("n_unique_action_targets_mean", 0) / 10 +
            a.get("url_revisit_rate_mean", 0) / 5 +
            a.get("n_plans_mean", 0) / 50
        )

    max_score = max(scores.values()) if scores else 1
    for base, score in sorted(scores.items(), key=lambda x: x[1]):
        bar = _bar(score, max_score)
        print(f"  {base:<30}  {bar}  {score:.2f}")

    print("═" * 70)


def print_transfer_table(ood_perf: dict[str, dict[str, float]]) -> None:
    if not ood_perf:
        return
    print("\n" + "═" * 80)
    print("  OOD TRANSFER PERFORMANCE  (macro F1, averaged over classifiers)")
    print("═" * 80)
    for tag, ood_map in sorted(ood_perf.items()):
        print(f"\n  Experiment: {tag}")
        for ds, f1 in sorted(ood_map.items(), key=lambda x: x[1]):
            bar = _bar(f1, 1.0, width=20)
            print(f"    {ds:<30}  {bar}  {f1:.3f}")
    print("═" * 80)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse interaction intensity across datasets and compare with OOD transfer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--traces-dir",  type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces)")
    parser.add_argument("--results-dir", type=Path, default=Path("./traces/models"),
                        help="Results directory containing per-tag subdirs with results.json")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional: write per-episode metrics to a CSV file")
    args = parser.parse_args()

    print(f"Loading traces from {args.traces_dir} ...")
    ds_episodes = load_dataset_metrics(args.traces_dir)
    if not ds_episodes:
        print("No episode data found. Run the orchestrator first.")
        return

    # Aggregate by base dataset name
    agg_by_base: dict[str, dict] = {}
    for ds_name, episodes in sorted(ds_episodes.items()):
        base = canonical_base(ds_name)
        existing = agg_by_base.get(base, {})
        if not existing:
            agg_by_base[base] = aggregate(episodes, INTENSITY_KEYS)
            agg_by_base[base]["_n"] = len(episodes)
        else:
            # merge multiple splits (train/val/test) into one aggregate
            merged = aggregate(
                [e for ds2 in ds_episodes if canonical_base(ds2) == base
                   for e in ds_episodes[ds2]],
                INTENSITY_KEYS,
            )
            merged["_n"] = sum(
                len(eps) for ds2, eps in ds_episodes.items()
                if canonical_base(ds2) == base
            )
            agg_by_base[base] = merged

    print_intensity_table(agg_by_base)
    print_diversity_table(agg_by_base)
    print_spectrum_ranking(agg_by_base)

    # OOD transfer
    if args.results_dir.exists():
        print(f"\nLoading OOD results from {args.results_dir} ...")
        ood_perf = load_ood_performance(args.results_dir)
        if ood_perf:
            print_transfer_table(ood_perf)
        else:
            print("No OOD results found yet.")
    else:
        print(f"\n(results-dir {args.results_dir} not found — skipping OOD table)")

    # Optional CSV dump
    if args.csv:
        import csv
        all_rows = []
        for ds_name, episodes in sorted(ds_episodes.items()):
            for e in episodes:
                row = {"dataset": ds_name, "dataset_base": canonical_base(ds_name)}
                row.update(e)
                all_rows.append(row)
        if all_rows:
            keys = list(all_rows[0].keys())
            with open(args.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(all_rows)
            print(f"\nPer-episode CSV written to {args.csv}")


if __name__ == "__main__":
    main()
