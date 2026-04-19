#!/usr/bin/env python3
"""Aggregate per-agent task-accuracy from verification.correct across datasets.

Only datasets where ground_truth is stored in the trace are included
(2wikimultihop, frames, webgames). Shopping datasets have no verification.

Usage:
    python benchmark.py                        # all splits, all agents
    python benchmark.py --split test           # test split only
    python benchmark.py --datasets 2wikimultihop webgames
    python benchmark.py --out results.csv      # also write CSV
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_display_names(config_path: Path = _CONFIG_PATH) -> dict[str, str]:
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        return {a["agent_id"]: a.get("display_name", a["agent_id"])
                for a in cfg.get("agents", [])}
    except Exception:
        return {}


def _load_family_map(config_path: Path = _CONFIG_PATH) -> dict[str, str]:
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        return {a["agent_id"]: a.get("family", a["agent_id"])
                for a in cfg.get("agents", [])}
    except Exception:
        return {}


def collect(
    traces_dir: Path,
    datasets: list[str] | None,
    splits: list[str] | None,
) -> dict[str, dict[str, dict]]:
    """Returns {agent_id: {base_dataset: {correct: int, total: int}}}"""
    stats: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for path in sorted(traces_dir.rglob("*.json")):
        parts = path.relative_to(traces_dir).parts
        if parts[0].startswith("classifiers") or len(parts) < 3:
            continue

        agent_id     = parts[0]
        dataset_name = parts[1]
        base         = dataset_name.rsplit("_", 1)[0]
        suffix       = dataset_name.rsplit("_", 1)[-1] if "_" in dataset_name else dataset_name

        if datasets is not None and base not in datasets:
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


def print_table(
    stats: dict,
    display_names: dict[str, str],
    family_map: dict[str, str],
    out_csv: Path | None,
) -> None:
    all_agents   = sorted(stats.keys())
    all_datasets = sorted({ds for agent_data in stats.values() for ds in agent_data})

    if not all_agents:
        print("No verification data found.", file=sys.stderr)
        return

    # Header
    col_w   = 22
    ds_w    = 14
    header  = f"{'agent':<{col_w}}"
    header += "".join(f"{ds:>{ds_w}}" for ds in all_datasets)
    header += f"{'overall':>{ds_w}}"
    print(header)
    print("─" * len(header))

    rows = []
    for agent_id in all_agents:
        agent_data = stats[agent_id]
        total_c = total_n = 0
        cells = []
        for ds in all_datasets:
            d = agent_data.get(ds, {})
            c, n = d.get("correct", 0), d.get("total", 0)
            total_c += c
            total_n += n
            cells.append(f"{c/n:.1%} ({n})" if n else "—")

        overall = f"{total_c/total_n:.1%} ({total_n})" if total_n else "—"
        name    = display_names.get(agent_id, agent_id)
        row     = f"{name:<{col_w}}"
        row    += "".join(f"{cell:>{ds_w}}" for cell in cells)
        row    += f"{overall:>{ds_w}}"
        print(row)
        rows.append([agent_id, name, family_map.get(agent_id, ""), *cells, overall])

    # Family averages
    families: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for agent_id, agent_data in stats.items():
        fam = family_map.get(agent_id, agent_id)
        for ds, d in agent_data.items():
            families[fam][ds]["correct"] += d["correct"]
            families[fam][ds]["total"]   += d["total"]

    print()
    print("Family averages:")
    print("─" * len(header))
    for fam in sorted(families):
        fam_data = families[fam]
        total_c = total_n = 0
        cells = []
        for ds in all_datasets:
            d = fam_data.get(ds, {})
            c, n = d.get("correct", 0), d.get("total", 0)
            total_c += c; total_n += n
            cells.append(f"{c/n:.1%} ({n})" if n else "—")
        overall = f"{total_c/total_n:.1%} ({total_n})" if total_n else "—"
        row  = f"{fam:<{col_w}}"
        row += "".join(f"{cell:>{ds_w}}" for cell in cells)
        row += f"{overall:>{ds_w}}"
        print(row)

    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["agent_id", "display_name", "family",
                             *all_datasets, "overall"])
            writer.writerows(rows)
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        metavar="NAME",
                        help="Base dataset names to include (default: all with verification)")
    parser.add_argument("--split", nargs="+", default=None,
                        metavar="SPLIT",
                        help="Directory suffixes to include, e.g. --split test val "
                             "(default: all splits)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write results to a CSV file")
    cli = parser.parse_args()

    display_names = _load_display_names()
    family_map    = _load_family_map()

    stats = collect(cli.traces_dir, cli.datasets, cli.split)
    print_table(stats, display_names, family_map, cli.out)
