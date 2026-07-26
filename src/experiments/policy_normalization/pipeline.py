#!/usr/bin/env python3
"""Matched WebShop behavioral-policy normalization defense experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from experiments.cross_harness.pipeline import (
    SPLITS,
    _manifest_path,
    _read_jsonl,
    _sha256_file,
    _task_id,
    _trace_valid,
    _utc_now,
    _write_frozen_jsonl,
    evaluate_model,
    summarize_results,
    train_model,
)
from experiments.reporting import markdown_table, number, read_csv, relative_link


CONDITIONS = ("canonical", "normalized_policy")
GRID = {
    "canonical": ("canonical", "normalized_policy"),
    "normalized_policy": ("normalized_policy",),
    "mixed50": ("canonical", "normalized_policy"),
}


def _task_success(episode: dict[str, Any]) -> bool | None:
    """Read task completion from current traces, with legacy compatibility."""
    value = episode.get("task_success")
    if isinstance(value, bool):
        return value
    value = (episode.get("verification") or {}).get("correct")
    return value if isinstance(value, bool) else None


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else config_path.parent / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["experiment"]["artifact_root"] = _resolve(
        path, cfg["experiment"]["artifact_root"]
    )
    # Shared classifier functions require this field but prompt inventory uses
    # the condition-specific roots below.
    cfg["experiment"]["traces_dir"] = _resolve(
        path, cfg["conditions"]["canonical"]["traces_dir"]
    )
    for condition in cfg["conditions"].values():
        condition["traces_dir"] = _resolve(path, condition["traces_dir"])
    cfg["_config_path"] = path
    return cfg


def _latest_valid_records(cfg: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    selected: dict[tuple, dict[str, Any]] = {}
    for condition_name, condition in cfg["conditions"].items():
        root: Path = condition["traces_dir"]
        harness = condition.get("harness", "browser_use")
        for agent_id in cfg["agents"]:
            for dataset in cfg["datasets"]:
                for split in SPLITS:
                    directory = root / agent_id / f"{dataset}_{split}" / harness
                    for path in sorted(directory.glob("*/*.json")):
                        try:
                            episode = json.loads(path.read_text())
                        except Exception:
                            continue
                        if not _trace_valid(episode, None):
                            continue
                        meta = episode.get("meta") or {}
                        question = str(meta.get("question") or "")
                        if not question:
                            continue
                        task_id = _task_id(dataset, question)
                        key = (
                            condition_name,
                            agent_id,
                            dataset,
                            split,
                            task_id,
                        )
                        timestamp = str(meta.get("timestamp") or "")
                        candidate = {
                            "episode_id": str(meta.get("episode_id") or path.stem),
                            "task_id": task_id,
                            "question": question,
                            "agent_id": agent_id,
                            "dataset": dataset,
                            "split": split,
                            "harness": condition_name,
                            "trace_path": str(path.resolve()),
                            "collection_run_id": path.parent.name,
                            "task_success": _task_success(episode),
                            "_order": (timestamp, path.stat().st_mtime_ns),
                        }
                        previous = selected.get(key)
                        if previous is None or candidate["_order"] > previous["_order"]:
                            selected[key] = candidate
    return selected


def _common_tasks(
    cfg: dict[str, Any],
    records: dict[tuple, dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    common: dict[str, dict[str, list[str]]] = {}
    for dataset in cfg["datasets"]:
        common[dataset] = {}
        for split in SPLITS:
            sets = []
            for condition in CONDITIONS:
                for agent_id in cfg["agents"]:
                    sets.append(
                        {
                            key[4]
                            for key in records
                            if key[:4] == (condition, agent_id, dataset, split)
                        }
                    )
            common[dataset][split] = sorted(set.intersection(*sets)) if sets else []
    return common


def audit(cfg: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    records = _latest_valid_records(cfg)
    common = _common_tasks(cfg, records)
    print(f"Valid condition/model/task records: {len(records)}")
    for dataset, splits in common.items():
        print(dataset)
        for split, task_ids in splits.items():
            expected = cfg["datasets"][dataset]["expected_tasks"][split]
            print(f"  {split}: common={len(task_ids)}/{expected}")
    return common


def _mixed_assignment(task_ids: list[str], seed: int) -> dict[str, str]:
    ordered = sorted(task_ids)
    random.Random(seed).shuffle(ordered)
    midpoint = len(ordered) // 2
    return {
        task_id: ("canonical" if index < midpoint else "normalized_policy")
        for index, task_id in enumerate(ordered)
    }


def prepare(cfg: dict[str, Any], force: bool = False) -> Path:
    records = _latest_valid_records(cfg)
    common = _common_tasks(cfg, records)
    sampling_seed = int(cfg["experiment"]["sampling_seed"])
    manifest_hashes = {}
    for dataset, splits in common.items():
        for split, task_ids in splits.items():
            minimum = int(cfg["datasets"][dataset]["minimum_common_tasks"][split])
            if len(task_ids) < minimum:
                raise RuntimeError(
                    f"{dataset}/{split}: {len(task_ids)} common tasks; "
                    f"minimum is {minimum}"
                )
            assignment = _mixed_assignment(task_ids, sampling_seed)
            for policy in (*CONDITIONS, "mixed50"):
                rows = []
                for agent_id in cfg["agents"]:
                    for task_id in task_ids:
                        condition = (
                            policy if policy in CONDITIONS else assignment[task_id]
                        )
                        row = dict(
                            records[
                                (
                                    condition,
                                    agent_id,
                                    dataset,
                                    split,
                                    task_id,
                                )
                            ]
                        )
                        row.pop("_order", None)
                        row["policy"] = policy
                        rows.append(row)
                path = _manifest_path(cfg, dataset, split, policy)
                _write_frozen_jsonl(path, rows, force)
                manifest_hashes[str(path)] = _sha256_file(path)
    experiment_manifest = {
        "schema_version": 1,
        "experiment_id": cfg["experiment"]["id"],
        "created_at": _utc_now(),
        "config": str(cfg["_config_path"]),
        "conditions": {
            name: {
                "traces_dir": str(spec["traces_dir"]),
                "harness": spec.get("harness", "browser_use"),
            }
            for name, spec in cfg["conditions"].items()
        },
        "manifest_sha256": manifest_hashes,
    }
    output = cfg["experiment"]["artifact_root"] / "experiment_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        old = json.loads(output.read_text())
        comparable = {
            key: value
            for key, value in experiment_manifest.items()
            if key != "created_at"
        }
        old_comparable = {
            key: value for key, value in old.items() if key != "created_at"
        }
        if comparable != old_comparable:
            raise RuntimeError(f"refusing to replace frozen manifest: {output}")
    output.write_text(json.dumps(experiment_manifest, indent=2) + "\n")
    summarize_task_success(cfg)
    print(f"Prepared policy-condition manifests under {output.parent / 'splits'}")
    return output


def _wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_task_success(cfg: dict[str, Any]) -> Path:
    """Report defense utility using only traces with known success labels."""
    grouped: dict[tuple, list[bool]] = defaultdict(list)
    for dataset in cfg["datasets"]:
        for split in SPLITS:
            for condition in CONDITIONS:
                rows = _read_jsonl(_manifest_path(cfg, dataset, split, condition))
                for row in rows:
                    value = row.get("task_success")
                    if isinstance(value, bool):
                        grouped[(dataset, split, condition, row["agent_id"])].append(
                            value
                        )
                        grouped[(dataset, split, condition, "ALL")].append(value)
    results = []
    for key, values in sorted(grouped.items()):
        successes = sum(values)
        lower, upper = _wilson_interval(successes, len(values))
        results.append(
            {
                "dataset": key[0],
                "split": key[1],
                "condition": key[2],
                "agent_id": key[3],
                "n_labeled": len(values),
                "n_success": successes,
                "success_rate": successes / len(values),
                "success_ci_lower": lower,
                "success_ci_upper": upper,
                "ci_method": "wilson_95",
            }
        )
    summary_root = cfg["experiment"]["artifact_root"] / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    json_path = summary_root / "task_success.json"
    csv_path = summary_root / "task_success.csv"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    fields = (
        list(results[0])
        if results
        else [
            "dataset",
            "split",
            "condition",
            "agent_id",
            "n_labeled",
            "n_success",
            "success_rate",
            "success_ci_lower",
            "success_ci_upper",
            "ci_method",
        ]
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Summarized task-success utility → {csv_path}")
    return csv_path


def write_human_report(cfg: dict[str, Any]) -> Path:
    """Write a self-contained guide to the policy experiment and its results."""
    artifact_root: Path = cfg["experiment"]["artifact_root"]
    summary_root = artifact_root / "summaries"
    report_path = artifact_root / "REPORT.md"
    aggregate_rows = read_csv(summary_root / "seed_aggregates.csv")
    per_model_rows = read_csv(summary_root / "per_model_seed_aggregates.csv")
    utility_rows = read_csv(summary_root / "task_success.csv")

    comparison_order = [
        ("canonical", "canonical"),
        ("canonical", "normalized_policy"),
        ("normalized_policy", "normalized_policy"),
        ("mixed50", "canonical"),
        ("mixed50", "normalized_policy"),
    ]
    comparison_labels = {
        ("canonical", "canonical"): "Baseline / canonical control",
        ("canonical", "normalized_policy"): "Fixed attacker under defense",
        ("normalized_policy", "normalized_policy"): "Defense-aware attacker",
        ("mixed50", "canonical"): "Mixed attacker → canonical",
        ("mixed50", "normalized_policy"): "Mixed attacker → normalized",
    }
    by_comparison = {
        (row["train_policy"], row["eval_policy"], row["feature_group"]): row
        for row in aggregate_rows
    }

    full_rows = []
    for train_policy, eval_policy in comparison_order:
        row = by_comparison.get((train_policy, eval_policy, "full"))
        if not row:
            continue
        seed_sd = (
            number(row["macro_f1_std"]) if int(row["n_seeds"]) >= 2 else "—"
        )
        full_rows.append(
            [
                comparison_labels[(train_policy, eval_policy)],
                f"`{train_policy} → {eval_policy}`",
                row["n_seeds"],
                number(row["macro_f1_mean"]),
                seed_sd,
                (
                    f"[{number(row['macro_f1_ci_lower'])}, "
                    f"{number(row['macro_f1_ci_upper'])}]"
                ),
            ]
        )

    ablation_rows = []
    for feature_group in ("timing_only", "non_timing"):
        for train_policy, eval_policy in comparison_order:
            row = by_comparison.get((train_policy, eval_policy, feature_group))
            if not row:
                continue
            ablation_rows.append(
                [
                    feature_group,
                    comparison_labels[(train_policy, eval_policy)],
                    row["n_seeds"],
                    number(row["macro_f1_mean"]),
                    (
                        f"[{number(row['macro_f1_ci_lower'])}, "
                        f"{number(row['macro_f1_ci_upper'])}]"
                    ),
                ]
            )

    fixed_per_model = [
        row
        for row in per_model_rows
        if row["train_policy"] == "canonical"
        and row["eval_policy"] == "normalized_policy"
        and row["feature_group"] == "full"
    ]
    per_model_table = [
        [
            row["agent_id"],
            row["n_seeds"],
            f"{number(row['f1_mean'])} ± {number(row['f1_std'])}",
            number(row["recall_mean"]),
        ]
        for row in sorted(fixed_per_model, key=lambda item: item["agent_id"])
    ]

    utility_table = []
    for row in utility_rows:
        if row["split"] != "test" or row["agent_id"] != "ALL":
            continue
        utility_table.append(
            [
                row["condition"],
                f"{row['n_success']}/{row['n_labeled']}",
                number(row["success_rate"]),
                (
                    f"[{number(row['success_ci_lower'])}, "
                    f"{number(row['success_ci_upper'])}]"
                ),
            ]
        )

    observations = []
    baseline = by_comparison.get(("canonical", "canonical", "full"))
    fixed = by_comparison.get(("canonical", "normalized_policy", "full"))
    aware = by_comparison.get(
        ("normalized_policy", "normalized_policy", "full")
    )
    if baseline and fixed:
        drop = float(baseline["macro_f1_mean"]) - float(fixed["macro_f1_mean"])
        observations.append(
            f"- Fixed-attacker macro-F1 drops by **{drop:.3f}** under the "
            "normalized policy."
        )
    if fixed and aware:
        recovery = float(aware["macro_f1_mean"]) - float(fixed["macro_f1_mean"])
        observations.append(
            f"- Defense-aware retraining recovers **{recovery:.3f}** macro-F1."
        )
    observations.append(
        "- Treat task success as an agent-reported completion proxy gated on "
        "browser activity, not authoritative WebShop reward."
    )

    lines = [
        "# Experiment report: WebShop behavioral-policy normalization",
        "",
        f"Experiment ID: `{cfg['experiment']['id']}`",
        "",
        "## What this experiment tests",
        "",
        "This is a four-class **model-identification** experiment over one "
        "browser-use harness and two execution policies. It asks whether a "
        "classifier trained on canonical traces survives a normalized action "
        "policy, and whether an attacker can recover by retraining.",
        "",
        "- `canonical`: original browser-use traces.",
        "- `normalized_policy`: newly collected traces under the fixed "
        "normalized behavioral policy.",
        "- `mixed50`: a balanced training construction using different tasks "
        "from both conditions; it is not a third collection condition.",
        "- `A → B`: train/validate on policy A, then evaluate once on policy B.",
        "",
        "## Headline full-feature results",
        "",
        "Macro-F1 is the unweighted mean of the four per-model F1 scores. "
        "Seed SD is the sample standard deviation (`ddof=1`) across classifier "
        "seeds and is undefined for one seed. The interval is a 95% stratified "
        "bootstrap over test traces after averaging seed scores.",
        "",
        markdown_table(
            [
                "Scientific comparison",
                "Train → test",
                "Seeds",
                "Macro-F1 mean",
                "Seed SD",
                "95% trace-bootstrap CI",
            ],
            full_rows,
        ),
        "",
        "## Direct reading",
        "",
        *observations,
        "",
        "## Feature ablations",
        "",
        "These rows are currently seed 42 only unless `Seeds` says otherwise. "
        "Do not describe their intervals as multi-seed uncertainty.",
        "",
        markdown_table(
            ["Feature view", "Comparison", "Seeds", "Macro-F1", "95% trace CI"],
            ablation_rows,
        ),
        "",
        "## Fixed-attacker result by model",
        "",
        "This isolates `canonical → normalized_policy` with full features.",
        "",
        markdown_table(
            ["Model", "Seeds", "F1 mean ± seed SD", "Recall mean"],
            per_model_table,
        ),
        "",
        "## Task-completion proxy on the test split",
        "",
        markdown_table(
            ["Condition", "Completed", "Rate", "95% Wilson CI"], utility_table
        ),
        "",
        "## Artifact map",
        "",
        f"- One row per classifier seed: "
        f"{relative_link(summary_root / 'results.csv', report_path)}",
        f"- Seed-aggregated headline metrics: "
        f"{relative_link(summary_root / 'seed_aggregates.csv', report_path)}",
        f"- One row per model and seed: "
        f"{relative_link(summary_root / 'per_model_metrics.csv', report_path)}",
        f"- Per-model seed aggregates: "
        f"{relative_link(summary_root / 'per_model_seed_aggregates.csv', report_path)}",
        f"- Utility proxy: "
        f"{relative_link(summary_root / 'task_success.csv', report_path)}",
        "- Trained models and individual evaluations: `models/`.",
        "- Frozen matched-task manifests: `splits/`.",
        "",
        "This file is regenerated by the policy pipeline's `summarize` command.",
        "",
    ]
    report_path.write_text("\n".join(lines))
    print(f"Wrote human-readable report → {report_path}")
    return report_path


def run_condition_grid(
    cfg: dict[str, Any],
    *,
    feature_groups: list[str],
    seeds: list[int],
    quick: bool,
    xgb_device: str | None,
    force: bool,
) -> None:
    for seed in seeds:
        for feature_group in feature_groups:
            for dataset in cfg["datasets"]:
                for train_policy, eval_policies in GRID.items():
                    train_model(
                        cfg,
                        dataset,
                        train_policy,
                        "XGBoost",
                        seed,
                        feature_group=feature_group,
                        quick=quick,
                        xgb_device=xgb_device,
                        force=force,
                    )
                    for eval_policy in eval_policies:
                        evaluate_model(
                            cfg,
                            dataset,
                            train_policy,
                            eval_policy,
                            "XGBoost",
                            seed,
                            feature_group=feature_group,
                            force=force,
                        )
    summarize_results(cfg)
    summarize_task_success(cfg)
    write_human_report(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the matched behavioral-policy normalization defense experiment."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "webshop_full_analysis.yaml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--force", action="store_true")
    run = subparsers.add_parser("run-grid")
    run.add_argument(
        "--feature-groups",
        nargs="+",
        default=["full", "timing_only", "non_timing"],
    )
    run.add_argument("--seeds", nargs="+", type=int, default=None)
    run.add_argument("--quick", action="store_true")
    run.add_argument("--xgb-device", choices=["cpu", "cuda"], default=None)
    run.add_argument("--force", action="store_true")
    subparsers.add_parser("summarize")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.command == "audit":
        audit(cfg)
    elif args.command == "prepare":
        prepare(cfg, force=args.force)
    elif args.command == "run-grid":
        run_condition_grid(
            cfg,
            feature_groups=args.feature_groups,
            seeds=(args.seeds or list(cfg["evaluation"]["classifier_seeds"])),
            quick=args.quick,
            xgb_device=args.xgb_device,
            force=args.force,
        )
    elif args.command == "summarize":
        summarize_results(cfg)
        summarize_task_success(cfg)
        write_human_report(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
