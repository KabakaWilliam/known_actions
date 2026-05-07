#!/usr/bin/env python3
"""
export_split_records.py — Dry-run of the resplit logic; persists split membership
to JSON for audit and reproducibility.

Mirrors the exact shuffle/cap/split algorithm in load_dataset() (trace_analyzer.py
lines 478-508): Random(seed=42), per-agent shuffle, optional cap, 50/25/25 split.

Output: {out_dir}/{dataset}_sampled_split_record.json  (for resplit datasets)
         {out_dir}/{dataset}_split_record.json          (for datasets with natural splits)

Each file contains episode-level records with:
  episode_id, agent_id, question, ground_truth, predicted_answer, split

Usage:
    python export_split_records.py --datasets frames \
        --resplit-datasets frames --resplit-n-per-agent 300

    python export_split_records.py \
        --datasets 2wikimultihop webshop deepshop frames \
        --resplit-datasets frames deepshop
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

NATURAL_SPLITS = {"train", "val", "test", "ood"}


def _base_and_split(dataset_name: str):
    """'frames_test' → ('frames', 'test');  '2wikimultihop_train' → ('2wikimultihop', 'train')."""
    parts = dataset_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in NATURAL_SPLITS:
        return parts[0], parts[1]
    return dataset_name, None


def _load_record(path: Path) -> dict | None:
    """Load the 5 metadata fields from a single trace JSON. Returns None if invalid."""
    try:
        with open(path) as f:
            ep = json.load(f)
    except Exception:
        return None
    events = (ep.get("dom_trace") or {}).get("events")
    if not events:
        return None
    meta = ep.get("meta") or {}
    ver  = ep.get("verification") or {}
    episode_id      = meta.get("episode_id")
    agent_id        = meta.get("agent_id")
    question        = meta.get("question")
    ground_truth    = ver.get("ground_truth")
    predicted       = ver.get("predicted")
    if not episode_id or not agent_id:
        return None
    return {
        "episode_id":       episode_id,
        "agent_id":         agent_id,
        "question":         question,
        "ground_truth":     ground_truth,
        "predicted_answer": predicted,
    }


def collect(traces_dir: Path, datasets: set[str], resplit_datasets: set[str]):
    """
    Scan all trace files and bucket them.

    Returns:
      natural[base][split] = [record, ...]
      resplit_pool[base][agent_id] = [record, ...]
    """
    natural: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    pool:    dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for path in sorted(traces_dir.rglob("*.json")):
        # Expected: traces/{agent_id}/{dataset_name}/{ts}/{episode_id}.json
        parts = path.relative_to(traces_dir).parts
        if len(parts) < 4:
            continue
        dataset_name = parts[1]
        base, split  = _base_and_split(dataset_name)
        if base not in datasets:
            continue
        # Skip _ood dirs unless this dataset is being resplit
        if split == "ood" and base not in resplit_datasets:
            continue

        record = _load_record(path)
        if record is None:
            continue

        if base in resplit_datasets:
            pool[base][record["agent_id"]].append(record)
        else:
            if split is None:
                split = "train"
            natural[base][split].append(record)

    return natural, pool


def apply_resplit(
    pool: dict[str, dict[str, list]],
    resplit_n_per_agent: int | None,
    resplit_seed: int,
    resplit_fracs: tuple = (0.5, 0.25, 0.25),
) -> dict[str, dict[str, list]]:
    """
    Mirror of load_dataset() resplit loop (trace_analyzer.py lines 478-508).
    Returns {base: {"train": [...], "val": [...], "test": [...]}}
    """
    tr_f, va_f, _ = resplit_fracs
    rng = random.Random(resplit_seed)
    result: dict[str, dict[str, list]] = {}

    for base, by_agent in pool.items():
        splits: dict[str, list] = {"train": [], "val": [], "test": []}
        for agent_items in by_agent.values():
            rng.shuffle(agent_items)
            if resplit_n_per_agent is not None:
                agent_items = agent_items[:resplit_n_per_agent]
            n       = len(agent_items)
            n_train = int(n * tr_f)
            n_val   = int(n * va_f)
            splits["train"].extend(agent_items[:n_train])
            splits["val"].extend(  agent_items[n_train : n_train + n_val])
            splits["test"].extend( agent_items[n_train + n_val :])
        result[base] = splits

    return result


def write_record(
    base: str,
    splits: dict[str, list],
    out_dir: Path,
    is_resplit: bool,
    resplit_seed: int,
    resplit_n_per_agent: int | None,
):
    suffix  = "sampled_split_record" if is_resplit else "split_record"
    outfile = out_dir / f"{base}_{suffix}.json"

    # Annotate each record with its split assignment
    records = []
    for split_name in ("train", "val", "test", "ood"):
        for rec in splits.get(split_name, []):
            records.append({**rec, "split": split_name})

    records.sort(key=lambda r: (r["split"], r["agent_id"], r["episode_id"]))

    counts = {sp: len(splits.get(sp, [])) for sp in ("train", "val", "test", "ood")}
    counts = {k: v for k, v in counts.items() if v > 0}

    payload: dict = {
        "dataset":              base,
        "resplit_seed":         resplit_seed,
        "resplit_n_per_agent":  resplit_n_per_agent,
        "counts":               counts,
        "records":              records,
    }

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(payload, f, indent=2)

    total = sum(counts.values())
    print(f"Saved → {outfile}  ({total} episodes: {counts})")


def main():
    parser = argparse.ArgumentParser(
        description="Export deterministic split membership records to JSON."
    )
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces).")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset base names to process (e.g. frames webshop deepshop 2wikimultihop).")
    parser.add_argument("--resplit-datasets", nargs="+", default=None,
                        help="Subset of --datasets to resplit (default: same as --datasets).")
    parser.add_argument("--resplit-n-per-agent", type=int, default=None,
                        help="Cap per agent before splitting (e.g. 300 for frames).")
    parser.add_argument("--resplit-seed", type=int, default=42,
                        help="RNG seed for resplit shuffle (default: 42, must match trace_analyzer.py).")
    parser.add_argument("--out-dir", type=Path, default=Path("./split_records"),
                        help="Output directory (default: ./split_records).")
    args = parser.parse_args()

    datasets         = set(args.datasets)
    resplit_datasets = set(args.resplit_datasets if args.resplit_datasets is not None else args.datasets)

    print(f"Scanning {args.traces_dir} ...")
    natural, pool = collect(args.traces_dir, datasets, resplit_datasets)

    # Write natural-split datasets
    for base, splits in natural.items():
        write_record(base, splits, args.out_dir,
                     is_resplit=False,
                     resplit_seed=args.resplit_seed,
                     resplit_n_per_agent=args.resplit_n_per_agent)

    # Apply resplit and write
    if pool:
        resplit_result = apply_resplit(pool, args.resplit_n_per_agent, args.resplit_seed)
        for base, splits in resplit_result.items():
            write_record(base, splits, args.out_dir,
                         is_resplit=True,
                         resplit_seed=args.resplit_seed,
                         resplit_n_per_agent=args.resplit_n_per_agent)


if __name__ == "__main__":
    main()
