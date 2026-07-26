#!/usr/bin/env python3
"""Evaluate an old-wave MidScene classifier on a newly collected test wave."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import f1_score

from experiments.cross_harness.pipeline import (
    _manifest_path,
    _sample_std,
    _sha256_file,
    _task_id,
    _trace_valid,
    evaluate_model,
    summarize_results,
    train_model,
)
from experiments.policy_normalization.pipeline import _task_success
from experiments.reporting import markdown_table, number, read_csv, relative_link


WAVES = ("original", "future")


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else config_path.parent / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    cfg = yaml.safe_load(path.read_text())
    experiment = cfg["experiment"]
    for key in ("artifact_root", "original_traces_dir", "future_traces_dir"):
        experiment[key] = _resolve(path, experiment[key])
    cfg["_config_path"] = path
    return cfg


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_frozen(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"refusing to replace frozen manifest: {path}")
    path.write_text(content)


def _timestamp(episode: dict[str, Any], path: Path) -> str:
    recorded = str((episode.get("meta") or {}).get("timestamp") or "")
    if recorded:
        return recorded
    return dt.datetime.fromtimestamp(
        path.stat().st_mtime, tz=dt.timezone.utc
    ).isoformat()


def scan(cfg: dict[str, Any]) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    """Select one valid MidScene trace per wave/model/split/task."""
    selected: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    roots = {
        "original": cfg["experiment"]["original_traces_dir"],
        "future": cfg["experiment"]["future_traces_dir"],
    }
    for wave, root in roots.items():
        splits = ("train", "val", "test") if wave == "original" else ("test",)
        for agent_id in cfg["agents"]:
            aliases = set(cfg["model_aliases"][agent_id])
            for dataset in cfg["datasets"]:
                for split in splits:
                    directory = root / agent_id / f"{dataset}_{split}"
                    if wave == "original":
                        # Historical MidScene traces predate explicit harness
                        # directories and use dataset/run-id/file.json. This
                        # shape deliberately excludes browser_use traces.
                        trace_paths = directory.glob("*/*.json")
                    else:
                        # New orchestrator collections include the harness:
                        # dataset/midscene/run-id/file.json.
                        trace_paths = (directory / "midscene").glob("*/*.json")
                    for path in sorted(trace_paths):
                        try:
                            episode = json.loads(path.read_text())
                        except Exception:
                            continue
                        if not _trace_valid(episode, None):
                            continue
                        meta = episode.get("meta") or {}
                        if str(meta.get("model_name") or "") not in aliases:
                            continue
                        metadata_agent = str(meta.get("agent_id") or "")
                        if metadata_agent and metadata_agent != agent_id:
                            continue
                        question = str(meta.get("question") or "")
                        if not question:
                            continue
                        task_id = _task_id(dataset, question)
                        key = (wave, agent_id, dataset, split, task_id)
                        timestamp = _timestamp(episode, path)
                        candidate = {
                            "episode_id": str(meta.get("episode_id") or path.stem),
                            "task_id": task_id,
                            "question": question,
                            "agent_id": agent_id,
                            "dataset": dataset,
                            "split": split,
                            "harness": "midscene",
                            "wave": wave,
                            "trace_path": str(path.resolve()),
                            "collection_timestamp": timestamp,
                            "task_success": _task_success(episode),
                            "_order": (timestamp, path.stat().st_mtime_ns),
                        }
                        old = selected.get(key)
                        if old is None or candidate["_order"] > old["_order"]:
                            selected[key] = candidate
    return selected


def _task_set(
    records: dict[tuple[str, str, str, str, str], dict[str, Any]],
    wave: str,
    agent: str,
    dataset: str,
    split: str,
) -> set[str]:
    return {
        key[4]
        for key in records
        if key[:4] == (wave, agent, dataset, split)
    }


def inventory(
    cfg: dict[str, Any],
    records: dict[tuple[str, str, str, str, str], dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    output: dict[str, dict[str, list[str]]] = {}
    for dataset in cfg["datasets"]:
        original_train = [
            _task_set(records, "original", agent, dataset, "train")
            for agent in cfg["agents"]
        ]
        original_val = [
            _task_set(records, "original", agent, dataset, "val")
            for agent in cfg["agents"]
        ]
        matched_test = [
            _task_set(records, wave, agent, dataset, "test")
            for wave in WAVES
            for agent in cfg["agents"]
        ]
        output[dataset] = {
            "train": sorted(set.intersection(*original_train)),
            "val": sorted(set.intersection(*original_val)),
            "test": sorted(set.intersection(*matched_test)),
        }
    return output


def audit(cfg: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    records = scan(cfg)
    common = inventory(cfg, records)
    print(f"Valid wave/model/split/task records: {len(records)}")
    for dataset, splits in common.items():
        expected = cfg["datasets"][dataset]["expected_tasks"]
        print(dataset)
        print(f"  original train common={len(splits['train'])}/{expected['train']}")
        print(f"  original val   common={len(splits['val'])}/{expected['val']}")
        print(f"  matched test   common={len(splits['test'])}/{expected['test']}")
        for wave in WAVES:
            print(f"  {wave} test by model:")
            for agent in cfg["agents"]:
                count = len(_task_set(records, wave, agent, dataset, "test"))
                print(f"    {agent}={count}")
    return common


def prepare(cfg: dict[str, Any]) -> Path:
    records = scan(cfg)
    common = inventory(cfg, records)
    hashes: dict[str, str] = {}
    for dataset, task_ids in common.items():
        requirements = cfg["datasets"][dataset]
        for split in ("train", "val"):
            minimum = int(requirements["minimum_original_tasks"][split])
            if len(task_ids[split]) < minimum:
                raise RuntimeError(
                    f"{dataset}/{split}: {len(task_ids[split])} original common "
                    f"tasks; minimum is {minimum}"
                )
        test_minimum = int(requirements["minimum_matched_test_tasks"])
        if len(task_ids["test"]) < test_minimum:
            raise RuntimeError(
                f"{dataset}/test: {len(task_ids['test'])} matched old/future "
                f"tasks; minimum is {test_minimum}"
            )
        train_tasks = set(task_ids["train"])
        val_tasks = set(task_ids["val"])
        test_tasks = set(task_ids["test"])
        overlaps = {
            "train/val": train_tasks & val_tasks,
            "train/test": train_tasks & test_tasks,
            "val/test": val_tasks & test_tasks,
        }
        if any(overlaps.values()):
            details = ", ".join(
                f"{name}={len(values)}" for name, values in overlaps.items()
            )
            raise RuntimeError(f"{dataset}: task leakage detected ({details})")

        for split in ("train", "val"):
            rows = []
            for agent in cfg["agents"]:
                for task_id in task_ids[split]:
                    row = dict(
                        records[("original", agent, dataset, split, task_id)]
                    )
                    row.pop("_order", None)
                    rows.append(row)
            path = _manifest_path(cfg, dataset, split, "original")
            _write_frozen(path, rows)
            hashes[str(path)] = _sha256_file(path)

        for wave in WAVES:
            rows = []
            for agent in cfg["agents"]:
                for task_id in task_ids["test"]:
                    row = dict(records[(wave, agent, dataset, "test", task_id)])
                    row.pop("_order", None)
                    rows.append(row)
            path = _manifest_path(cfg, dataset, "test", wave)
            _write_frozen(path, rows)
            hashes[str(path)] = _sha256_file(path)

    manifest = {
        "schema_version": 1,
        "experiment_id": cfg["experiment"]["id"],
        "config": str(cfg["_config_path"]),
        "protocol": {
            "train_wave": "original",
            "validation_wave": "original",
            "test_waves": list(WAVES),
            "matched_test_tasks": True,
        },
        "manifest_sha256": hashes,
    }
    output = cfg["experiment"]["artifact_root"] / "experiment_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, indent=2) + "\n"
    if output.exists() and output.read_text() != content:
        raise RuntimeError(f"refusing to replace frozen experiment manifest: {output}")
    output.write_text(content)
    print(f"Prepared frozen temporal manifests → {output.parent}")
    return output


def _prediction_rows(result_path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(result_path.with_name("predictions.jsonl"))


def _paired_delta_summary(cfg: dict[str, Any]) -> Path:
    """Task-clustered paired bootstrap for future-minus-original macro-F1."""
    summary_root = cfg["experiment"]["artifact_root"] / "summaries"
    results = read_csv(summary_root / "results.csv")
    grouped: dict[tuple[str, str], dict[int, dict[str, Path]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in results:
        if row["train_policy"] != "original":
            continue
        wave = row["eval_policy"]
        if wave not in WAVES:
            continue
        key = (row["dataset"], row["feature_group"])
        grouped[key][int(row["seed"])][wave] = Path(row["results_path"])

    output_rows = []
    bootstrap_samples = int(cfg["evaluation"]["bootstrap_samples"])
    confidence = float(cfg["evaluation"]["bootstrap_confidence"])
    for (dataset, feature_group), by_seed in sorted(grouped.items()):
        complete = {
            seed: paths
            for seed, paths in by_seed.items()
            if set(paths) == set(WAVES)
        }
        if not complete:
            continue
        seed_deltas = []
        aligned: dict[int, dict[str, dict[tuple[str, str], dict[str, Any]]]] = {}
        reference_keys: list[tuple[str, str]] | None = None
        for seed, paths in sorted(complete.items()):
            aligned[seed] = {}
            scores = {}
            for wave, result_path in paths.items():
                rows = _prediction_rows(result_path)
                indexed = {(row["task_id"], row["agent_id"]): row for row in rows}
                aligned[seed][wave] = indexed
                keys = sorted(indexed)
                if reference_keys is None:
                    reference_keys = keys
                elif keys != reference_keys:
                    raise RuntimeError(
                        f"unaligned paired predictions for {dataset}/{feature_group}"
                    )
                labels = sorted(cfg["agents"])
                scores[wave] = f1_score(
                    [indexed[key]["true_label"] for key in keys],
                    [indexed[key]["predicted_label"] for key in keys],
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            seed_deltas.append(float(scores["future"] - scores["original"]))

        assert reference_keys is not None
        task_ids = sorted({key[0] for key in reference_keys})
        indices_by_task = {
            task_id: [
                index
                for index, key in enumerate(reference_keys)
                if key[0] == task_id
            ]
            for task_id in task_ids
        }
        labels = sorted(cfg["agents"])
        seed_material = f"{dataset}\0{feature_group}\0temporal".encode()
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        )
        bootstrap_deltas = []
        for _ in range(bootstrap_samples):
            sampled_tasks = rng.choice(task_ids, size=len(task_ids), replace=True)
            sampled_indices = [
                index
                for task_id in sampled_tasks
                for index in indices_by_task[str(task_id)]
            ]
            replicate_seed_deltas = []
            for seed in sorted(aligned):
                wave_scores = {}
                for wave in WAVES:
                    rows = [
                        aligned[seed][wave][reference_keys[index]]
                        for index in sampled_indices
                    ]
                    wave_scores[wave] = f1_score(
                        [row["true_label"] for row in rows],
                        [row["predicted_label"] for row in rows],
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                replicate_seed_deltas.append(
                    wave_scores["future"] - wave_scores["original"]
                )
            bootstrap_deltas.append(float(np.mean(replicate_seed_deltas)))
        alpha = (1.0 - confidence) / 2.0
        output_rows.append(
            {
                "dataset": dataset,
                "feature_group": feature_group,
                "seeds": json.dumps(sorted(complete)),
                "n_seeds": len(complete),
                "n_matched_tasks": len(task_ids),
                "future_minus_original_macro_f1_mean": float(
                    np.mean(seed_deltas)
                ),
                "seed_delta_sample_sd": _sample_std(seed_deltas),
                "paired_task_bootstrap_ci_lower": float(
                    np.quantile(bootstrap_deltas, alpha)
                ),
                "paired_task_bootstrap_ci_upper": float(
                    np.quantile(bootstrap_deltas, 1.0 - alpha)
                ),
            }
        )
    output = summary_root / "paired_temporal_deltas.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "feature_group",
        "seeds",
        "n_seeds",
        "n_matched_tasks",
        "future_minus_original_macro_f1_mean",
        "seed_delta_sample_sd",
        "paired_task_bootstrap_ci_lower",
        "paired_task_bootstrap_ci_upper",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    return output


def run_grid(
    cfg: dict[str, Any],
    *,
    feature_groups: list[str],
    seeds: list[int],
    xgb_device: str | None,
    force: bool,
) -> None:
    for seed in seeds:
        for feature_group in feature_groups:
            for dataset in cfg["datasets"]:
                train_model(
                    cfg,
                    dataset,
                    "original",
                    "XGBoost",
                    seed,
                    feature_group=feature_group,
                    quick=False,
                    xgb_device=xgb_device,
                    force=force,
                )
                for wave in WAVES:
                    evaluate_model(
                        cfg,
                        dataset,
                        "original",
                        wave,
                        "XGBoost",
                        seed,
                        feature_group=feature_group,
                        force=force,
                    )
    summarize_results(cfg)
    _paired_delta_summary(cfg)
    write_report(cfg)


def write_report(cfg: dict[str, Any]) -> Path:
    root = cfg["experiment"]["artifact_root"]
    report_path = root / "REPORT.md"
    summary_root = root / "summaries"
    aggregates = read_csv(summary_root / "seed_aggregates.csv")
    deltas = read_csv(summary_root / "paired_temporal_deltas.csv")
    per_model = read_csv(summary_root / "per_model_seed_aggregates.csv")
    aggregate_index = {
        (row["feature_group"], row["eval_policy"]): row
        for row in aggregates
        if row["train_policy"] == "original"
    }
    delta_index = {row["feature_group"]: row for row in deltas}

    headline = []
    for feature_group in ("full", "timing_only", "non_timing"):
        original = aggregate_index.get((feature_group, "original"))
        future = aggregate_index.get((feature_group, "future"))
        delta = delta_index.get(feature_group)
        if not (original and future and delta):
            continue
        original_sd = (
            number(original["macro_f1_std"])
            if int(original["n_seeds"]) >= 2
            else "—"
        )
        future_sd = (
            number(future["macro_f1_std"])
            if int(future["n_seeds"]) >= 2
            else "—"
        )
        headline.append(
            [
                feature_group,
                original["n_seeds"],
                f"{number(original['macro_f1_mean'])} ± {original_sd}",
                f"{number(future['macro_f1_mean'])} ± {future_sd}",
                number(delta["future_minus_original_macro_f1_mean"]),
                (
                    f"[{number(delta['paired_task_bootstrap_ci_lower'])}, "
                    f"{number(delta['paired_task_bootstrap_ci_upper'])}]"
                ),
            ]
        )

    per_model_index = {
        (row["eval_policy"], row["agent_id"]): row
        for row in per_model
        if row["train_policy"] == "original" and row["feature_group"] == "full"
    }
    per_model_rows = []
    for agent in cfg["agents"]:
        original = per_model_index.get(("original", agent))
        future = per_model_index.get(("future", agent))
        if original and future:
            per_model_rows.append(
                [
                    agent,
                    number(original["f1_mean"]),
                    number(future["f1_mean"]),
                    number(float(future["f1_mean"]) - float(original["f1_mean"])),
                ]
            )

    timestamps = []
    for wave in WAVES:
        rows = _read_jsonl(_manifest_path(cfg, "webshop", "test", wave))
        values = sorted(
            row["collection_timestamp"]
            for row in rows
            if row.get("collection_timestamp")
        )
        if values:
            timestamps.append([wave, values[0], values[-1], len(rows)])

    lines = [
        "# Experiment report: future-wave MidScene generalization",
        "",
        f"Experiment ID: `{cfg['experiment']['id']}`",
        "",
        "A four-class XGBoost classifier is trained only on the original "
        "MidScene WebShop training traces and selected only with the original "
        "validation traces. The frozen classifier is then evaluated on the "
        "original test wave and on newly collected traces for exactly the same "
        "test tasks. No future trace enters training or model selection.",
        "",
        "The reported delta is `future − original`; a negative value is "
        "temporal degradation. Its confidence interval is a paired bootstrap "
        "that resamples task IDs and keeps all four model traces for each "
        "sampled task together.",
        "",
        "## Temporal result",
        "",
        markdown_table(
            [
                "Features",
                "Seeds",
                "Original macro-F1",
                "Future macro-F1",
                "Delta",
                "95% paired task-bootstrap CI",
            ],
            headline,
        )
        if headline
        else "_The classifier grid has not been run yet._",
        "",
        "## Full-feature per-model F1",
        "",
        markdown_table(
            ["Model", "Original", "Future", "Delta"], per_model_rows
        )
        if per_model_rows
        else "_Per-model results are not available yet._",
        "",
        "## Collection-wave provenance",
        "",
        markdown_table(
            ["Wave", "Earliest trace", "Latest trace", "Matched traces"],
            timestamps,
        )
        if timestamps
        else "_Run `prepare` after collection to freeze provenance._",
        "",
        "## Leakage controls",
        "",
        "- Original train, original validation, and matched test task IDs are "
        "pairwise disjoint.",
        "- Original and future test manifests contain the same task IDs for "
        "every model.",
        "- Only original-wave train and validation manifests are referenced by "
        "the fitted model bundle.",
        "",
        "## Artifact map",
        "",
        f"- Per-seed metrics: "
        f"{relative_link(summary_root / 'results.csv', report_path)}",
        f"- Seed aggregates: "
        f"{relative_link(summary_root / 'seed_aggregates.csv', report_path)}",
        f"- Per-model metrics: "
        f"{relative_link(summary_root / 'per_model_seed_aggregates.csv', report_path)}",
        f"- Paired temporal deltas: "
        f"{relative_link(summary_root / 'paired_temporal_deltas.csv', report_path)}",
        "- Frozen task manifests: `frozen_temporal_splits/`.",
        "- Frozen old-wave classifiers: `temporal_model_identity/`.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"Wrote human-readable report → {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs")
        / "webshop_future_midscene_analysis.yaml",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("prepare")
    run = sub.add_parser("run-grid")
    run.add_argument(
        "--feature-groups",
        nargs="+",
        default=["full", "timing_only", "non_timing"],
    )
    run.add_argument("--seeds", nargs="+", type=int, default=None)
    run.add_argument("--xgb-device", choices=["cpu", "cuda"], default=None)
    run.add_argument("--force", action="store_true")
    sub.add_parser("summarize")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.command == "audit":
        audit(cfg)
    elif args.command == "prepare":
        prepare(cfg)
    elif args.command == "run-grid":
        run_grid(
            cfg,
            feature_groups=args.feature_groups,
            seeds=args.seeds or list(cfg["evaluation"]["classifier_seeds"]),
            xgb_device=args.xgb_device,
            force=args.force,
        )
    elif args.command == "summarize":
        summarize_results(cfg)
        _paired_delta_summary(cfg)
        write_report(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
