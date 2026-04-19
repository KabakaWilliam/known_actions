#!/usr/bin/env python3
"""Upload collected browser-agent traces to a HuggingFace private dataset repo.

The dataset is structured as multiple configurations (one per base dataset, e.g.
2wikimultihop, webshop, webgames, frames, deepshop) with train/val/test/ood splits
matching the on-disk directory layout under traces/{agent_id}/{dataset_name}/{run_ts}/.

Usage:
    python upload_to_hub.py --repo-id your-org/known-actions-traces --traces-dir ./traces
    HF_TOKEN=hf_xxx python upload_to_hub.py --repo-id your-org/known-actions-traces

To load after uploading:
    from datasets import load_dataset
    ds = load_dataset("your-org/known-actions-traces", "2wikimultihop", token="hf_xxx")
    print(ds)  # DatasetDict with train/val/test splits

To train classifiers directly from the Hub:
    python trace_analyzer.py --hf-repo your-org/known-actions-traces \\
        --train-datasets 2wikimultihop --ood-datasets webshop
"""
import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path


def _infer_split(dataset_name: str) -> str | None:
    if dataset_name.endswith("_train"): return "train"
    if dataset_name.endswith("_val"):   return "val"
    if dataset_name.endswith("_test"):  return "test"
    if dataset_name.endswith("_ood"):   return "ood"
    return None


def _build_row(path: Path, agent_id: str, dataset_name: str, run_ts: str) -> dict | None:
    try:
        with open(path) as f:
            episode = json.load(f)
    except Exception as e:
        warnings.warn(f"Skipping {path.name}: {e}")
        return None

    meta         = episode.get("meta") or {}
    verification = episode.get("verification") or {}
    dom_trace    = episode.get("dom_trace") or {}
    events       = dom_trace.get("events") or []

    return {
        "episode_id":       meta.get("episode_id", path.stem),
        "agent_id":         agent_id,
        "model_family":     meta.get("model_family", ""),
        "dataset_name":     dataset_name,
        "run_ts":           run_ts,
        "question":         meta.get("question", ""),
        "start_url":        meta.get("start_url", ""),
        "correct":          bool(verification.get("correct")),
        "ground_truth":     str(verification.get("ground_truth") or ""),
        "predicted_answer": str(verification.get("predicted") or ""),
        "error":            episode.get("error") or "",
        "n_events":         len(events),
        "duration_ms":      float(dom_trace.get("episodeDuration") or 0.0),
        "dom_events_json":  json.dumps(events),
        "meta_json":        json.dumps(meta),
    }


def collect_rows(traces_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """Returns {base_dataset: {split: [row, ...]}}"""
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    skipped = 0

    paths = sorted(traces_dir.rglob("*.json"))
    print(f"Found {len(paths):,} JSON files under {traces_dir}")

    for path in paths:
        rel_parts = path.relative_to(traces_dir).parts
        if rel_parts[0].startswith("classifiers"):
            continue
        if len(rel_parts) < 3:
            continue

        agent_id     = rel_parts[0]
        dataset_name = rel_parts[1]
        run_ts       = rel_parts[2]
        base         = dataset_name.rsplit("_", 1)[0]
        split        = _infer_split(dataset_name)

        if split is None:
            skipped += 1
            continue

        row = _build_row(path, agent_id, dataset_name, run_ts)
        if row is None:
            skipped += 1
            continue

        data[base][split].append(row)

    if skipped:
        print(f"  Skipped {skipped} files (unrecognised suffix or parse error)")
    return data


def push(data: dict, repo_id: str, token: str | None, dry_run: bool, private: bool = True) -> None:
    from datasets import Dataset, DatasetDict

    total = sum(len(rows) for splits in data.values() for rows in splits.values())
    print(f"\nConfigurations: {sorted(data.keys())}")
    print(f"Total episodes:  {total:,}\n")

    for base in sorted(data.keys()):
        splits = data[base]
        split_summary = {s: len(rows) for s, rows in splits.items()}
        print(f"  {base}: {split_summary}")
        if dry_run:
            continue

        ds_dict = DatasetDict({
            split: Dataset.from_list(rows)
            for split, rows in splits.items()
        })
        ds_dict.push_to_hub(
            repo_id,
            config_name=base,
            private=private,
            token=token,
        )
        print(f"    pushed {base}")

    if dry_run:
        print("\n[dry-run] No data was uploaded.")
    else:
        print(f"\nDataset available at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces)")
    parser.add_argument("--repo-id", required=True,
                        help="HuggingFace dataset repo, e.g. your-org/known-actions-traces")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (default: reads HF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary of what would be uploaded without pushing")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true", default=True,
                            help="Make the HuggingFace repo private (default)")
    visibility.add_argument("--public", dest="private", action="store_false",
                            help="Make the HuggingFace repo public")
    cli = parser.parse_args()

    token = cli.token or os.environ.get("HF_TOKEN")
    if not token and not cli.dry_run:
        parser.error("Provide --token or set HF_TOKEN env var")

    print(f"Scanning {cli.traces_dir} ...")
    data = collect_rows(cli.traces_dir)
    push(data, cli.repo_id, token, cli.dry_run, cli.private)
