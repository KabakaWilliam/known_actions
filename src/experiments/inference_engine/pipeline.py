#!/usr/bin/env python3
"""Matched WebShop model-identification grid across vLLM and SGLang."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from experiments.cross_harness.pipeline import (
    SPLITS,
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


ENGINES = ("vllm", "sglang")
GRID = {
    "vllm": ("vllm", "sglang"),
    "sglang": ("sglang", "vllm"),
    "mixed50": ("vllm", "sglang"),
}


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else config_path.parent / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["experiment"]["artifact_root"] = _resolve(
        path, cfg["experiment"]["artifact_root"]
    )
    cfg["experiment"]["traces_dir"] = _resolve(
        path, cfg["experiment"]["traces_dir"]
    )
    for condition in cfg["conditions"].values():
        condition["traces_dir"] = _resolve(path, condition["traces_dir"])
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


def scan(cfg: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    selected: dict[tuple, dict[str, Any]] = {}
    for engine, condition in cfg["conditions"].items():
        root: Path = condition["traces_dir"]
        harness = condition.get("harness", "browser_use")
        for agent_id in cfg["agents"]:
            allowed_models = set(cfg["model_aliases"][agent_id])
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
                        if str(meta.get("model_name") or "") not in allowed_models:
                            continue
                        question = str(meta.get("question") or "")
                        if not question:
                            continue
                        task_id = _task_id(dataset, question)
                        key = (engine, agent_id, dataset, split, task_id)
                        candidate = {
                            "episode_id": str(meta.get("episode_id") or path.stem),
                            "task_id": task_id,
                            "question": question,
                            "agent_id": agent_id,
                            "dataset": dataset,
                            "split": split,
                            "harness": engine,
                            "engine": engine,
                            "trace_path": str(path.resolve()),
                            "collection_run_id": path.parent.name,
                            "task_success": _task_success(episode),
                            "_order": (
                                str(meta.get("timestamp") or ""),
                                path.stat().st_mtime_ns,
                            ),
                        }
                        old = selected.get(key)
                        if old is None or candidate["_order"] > old["_order"]:
                            selected[key] = candidate
    return selected


def common_tasks(
    cfg: dict[str, Any], records: dict[tuple, dict[str, Any]]
) -> dict[str, dict[str, list[str]]]:
    output = {}
    for dataset in cfg["datasets"]:
        output[dataset] = {}
        for split in SPLITS:
            sets = [
                {
                    key[4]
                    for key in records
                    if key[:4] == (engine, agent, dataset, split)
                }
                for engine in ENGINES
                for agent in cfg["agents"]
            ]
            output[dataset][split] = sorted(set.intersection(*sets))
    return output


def audit(cfg: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    records = scan(cfg)
    common = common_tasks(cfg, records)
    print(f"Valid engine/model/task records: {len(records)}")
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
        task_id: ("vllm" if index < midpoint else "sglang")
        for index, task_id in enumerate(ordered)
    }


def prepare(cfg: dict[str, Any]) -> Path:
    records = scan(cfg)
    common = common_tasks(cfg, records)
    hashes = {}
    sampling_seed = int(cfg["experiment"]["sampling_seed"])
    for dataset, splits in common.items():
        for split, task_ids in splits.items():
            minimum = int(
                cfg["datasets"][dataset]["minimum_common_tasks"][split]
            )
            if len(task_ids) < minimum:
                raise RuntimeError(
                    f"{dataset}/{split}: {len(task_ids)} common tasks; "
                    f"minimum is {minimum}"
                )
            assignment = _mixed_assignment(task_ids, sampling_seed)
            for policy in (*ENGINES, "mixed50"):
                rows = []
                for agent in cfg["agents"]:
                    for task_id in task_ids:
                        engine = (
                            policy if policy in ENGINES else assignment[task_id]
                        )
                        row = dict(
                            records[(engine, agent, dataset, split, task_id)]
                        )
                        row.pop("_order", None)
                        row["policy"] = policy
                        rows.append(row)
                path = _manifest_path(cfg, dataset, split, policy)
                _write_frozen(path, rows)
                hashes[str(path)] = _sha256_file(path)
    manifest = {
        "schema_version": 1,
        "experiment_id": cfg["experiment"]["id"],
        "config": str(cfg["_config_path"]),
        "conditions": {
            key: str(value["traces_dir"])
            for key, value in cfg["conditions"].items()
        },
        "manifest_sha256": hashes,
    }
    output = cfg["experiment"]["artifact_root"] / "experiment_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, indent=2) + "\n"
    if output.exists() and output.read_text() != content:
        raise RuntimeError(f"refusing to replace frozen manifest: {output}")
    output.write_text(content)
    summarize_utility(cfg)
    print(f"Prepared frozen engine manifests → {output.parent}")
    return output


def summarize_utility(cfg: dict[str, Any]) -> Path:
    rows = []
    for dataset in cfg["datasets"]:
        for split in SPLITS:
            for engine in ENGINES:
                manifest_rows = _read_jsonl(
                    _manifest_path(cfg, dataset, split, engine)
                )
                values = [
                    row["task_success"]
                    for row in manifest_rows
                    if isinstance(row.get("task_success"), bool)
                ]
                if values:
                    rows.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "engine": engine,
                            "n_labeled": len(values),
                            "n_success": sum(values),
                            "success_rate": sum(values) / len(values),
                        }
                    )
    output = cfg["experiment"]["artifact_root"] / "summaries" / "task_success.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "split", "engine", "n_labeled", "n_success", "success_rate"
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
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
                for train_engine, eval_engines in GRID.items():
                    train_model(
                        cfg,
                        dataset,
                        train_engine,
                        "XGBoost",
                        seed,
                        feature_group=feature_group,
                        quick=False,
                        xgb_device=xgb_device,
                        force=force,
                    )
                    for eval_engine in eval_engines:
                        evaluate_model(
                            cfg,
                            dataset,
                            train_engine,
                            eval_engine,
                            "XGBoost",
                            seed,
                            feature_group=feature_group,
                            force=force,
                        )
    summarize_results(cfg)
    summarize_utility(cfg)
    write_report(cfg)


def write_report(cfg: dict[str, Any]) -> Path:
    root = cfg["experiment"]["artifact_root"]
    report_path = root / "REPORT.md"
    summary_root = root / "summaries"
    aggregates = read_csv(summary_root / "seed_aggregates.csv")
    utility = read_csv(summary_root / "task_success.csv")
    index = {
        (r["train_policy"], r["eval_policy"], r["feature_group"]): r
        for r in aggregates
    }
    comparisons = [
        ("vllm", "vllm", "vLLM → vLLM control"),
        ("vllm", "sglang", "vLLM → SGLang engine transfer"),
        ("sglang", "sglang", "SGLang → SGLang control"),
        ("sglang", "vllm", "SGLang → vLLM reverse transfer"),
        ("mixed50", "vllm", "Mixed engines → vLLM"),
        ("mixed50", "sglang", "Mixed engines → SGLang"),
    ]
    headline = []
    for train_engine, eval_engine, label in comparisons:
        row = index.get((train_engine, eval_engine, "full"))
        if row:
            sd = (
                number(row["macro_f1_std"])
                if int(row["n_seeds"]) >= 2
                else "—"
            )
            headline.append(
                [
                    label,
                    row["n_seeds"],
                    f"{number(row['macro_f1_mean'])} ± {sd}",
                    (
                        f"[{number(row['macro_f1_ci_lower'])}, "
                        f"{number(row['macro_f1_ci_upper'])}]"
                    ),
                ]
            )
    ablations = []
    for feature_group in ("timing_only", "non_timing"):
        for train_engine, eval_engine, label in comparisons:
            row = index.get((train_engine, eval_engine, feature_group))
            if row:
                ablations.append(
                    [
                        feature_group,
                        label,
                        row["n_seeds"],
                        number(row["macro_f1_mean"]),
                    ]
                )
    utility_rows = [
        [
            row["split"],
            row["engine"],
            f"{row['n_success']}/{row['n_labeled']}",
            number(row["success_rate"]),
        ]
        for row in utility
    ]
    lines = [
        "# Experiment report: vLLM versus SGLang",
        "",
        f"Experiment ID: `{cfg['experiment']['id']}`",
        "",
        "This three-class WebShop experiment changes only the local inference "
        "engine used to serve Qwen3.5-27B, GLM-4.6V, and Gemma-4-26B-A4B. "
        "The browser-use harness, model checkpoints, and matched tasks remain "
        "fixed.",
        "",
        "`A → B` means the classifier is trained and validated on traces "
        "collected with engine A and tested on matched traces from engine B. "
        "`mixed50` uses a task-balanced mixture from both engines.",
        "",
        "## Full-feature results",
        "",
        markdown_table(
            ["Comparison", "Seeds", "Macro-F1 mean ± sample SD", "95% trace CI"],
            headline,
        ) if headline else "_The classifier grid has not been run yet._",
        "",
        "## Feature ablations",
        "",
        markdown_table(
            ["Features", "Comparison", "Seeds", "Macro-F1"], ablations
        ) if ablations else "_The feature ablations have not been run yet._",
        "",
        "## Task-completion proxy",
        "",
        markdown_table(
            ["Split", "Engine", "Completed", "Rate"], utility_rows
        ),
        "",
        "## Artifact map",
        "",
        f"- Per-seed metrics: "
        f"{relative_link(summary_root / 'results.csv', report_path)}",
        f"- Seed aggregates: "
        f"{relative_link(summary_root / 'seed_aggregates.csv', report_path)}",
        f"- Per-model metrics: "
        f"{relative_link(summary_root / 'per_model_metrics.csv', report_path)}",
        "- Frozen matched tasks: `frozen_engine_splits/`.",
        "- Models/evaluations: `engine_model_identity/`.",
        "",
    ]
    report_path.write_text("\n".join(lines))
    print(f"Wrote human-readable report → {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "webshop_sglang_analysis.yaml",
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
        summarize_utility(cfg)
        write_report(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
