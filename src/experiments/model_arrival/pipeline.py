#!/usr/bin/env python3
"""Simulate cheaply adding a newly released model to a closed-set classifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from experiments.cross_harness.pipeline import (
    SPLITS,
    _classification_metrics,
    _load_examples,
    _sample_std,
    _task_id,
    _trace_valid,
)
from experiments.reporting import markdown_table, number, read_csv, relative_link

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else config_path.parent / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["experiment"]["traces_dir"] = _resolve(
        path, cfg["experiment"]["traces_dir"]
    )
    cfg["experiment"]["artifact_root"] = _resolve(
        path, cfg["experiment"]["artifact_root"]
    )
    cfg["_config_path"] = path
    return cfg


def _manifest_path(cfg: dict[str, Any], dataset: str, split: str) -> Path:
    return (
        cfg["experiment"]["artifact_root"]
        / "frozen_midscene_splits"
        / dataset
        / f"{split}.jsonl"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_jsonl(
    path: Path, rows: list[dict[str, Any]], *, force: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    if path.exists() and path.read_text() != content and not force:
        raise RuntimeError(f"refusing to replace frozen manifest: {path}")
    path.write_text(content)


def scan(cfg: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    selected: dict[tuple, dict[str, Any]] = {}
    root: Path = cfg["experiment"]["traces_dir"]
    for agent_id in cfg["agents"]:
        allowed_models = set(cfg["model_aliases"][agent_id])
        for dataset in cfg["datasets"]:
            for split in SPLITS:
                directory = root / agent_id / f"{dataset}_{split}"
                # Exactly two levels selects historical MidScene run/file paths
                # and excludes browser_use/run/file paths.
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
                    key = (agent_id, dataset, split, task_id)
                    candidate = {
                        "episode_id": str(meta.get("episode_id") or path.stem),
                        "task_id": task_id,
                        "question": question,
                        "agent_id": agent_id,
                        "dataset": dataset,
                        "split": split,
                        "harness": "midscene",
                        "trace_path": str(path.resolve()),
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
    output: dict[str, dict[str, list[str]]] = {}
    for dataset in cfg["datasets"]:
        output[dataset] = {}
        for split in SPLITS:
            sets = [
                {
                    key[3]
                    for key in records
                    if key[:3] == (agent_id, dataset, split)
                }
                for agent_id in cfg["agents"]
            ]
            output[dataset][split] = sorted(set.intersection(*sets))
    return output


def audit(cfg: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    records = scan(cfg)
    common = common_tasks(cfg, records)
    print(f"Valid model/dataset/split/task records: {len(records)}")
    for dataset, splits in common.items():
        print(dataset)
        for split, task_ids in splits.items():
            minimum = cfg["datasets"][dataset]["minimum_common_tasks"][split]
            status = "OK" if len(task_ids) >= minimum else "INSUFFICIENT"
            print(f"  {split}: common={len(task_ids)} minimum={minimum} [{status}]")
    return common


def prepare(cfg: dict[str, Any]) -> Path:
    records = scan(cfg)
    common = common_tasks(cfg, records)
    hashes = {}
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
            rows = []
            for agent_id in cfg["agents"]:
                for task_id in task_ids:
                    row = dict(records[(agent_id, dataset, split, task_id)])
                    row.pop("_order", None)
                    rows.append(row)
            path = _manifest_path(cfg, dataset, split)
            _write_jsonl(path, rows)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "experiment_id": cfg["experiment"]["id"],
        "config": str(cfg["_config_path"]),
        "agents": cfg["agents"],
        "budgets": cfg["update"]["budgets"],
        "classifier_seeds": cfg["update"]["classifier_seeds"],
        "manifest_sha256": hashes,
    }
    output = cfg["experiment"]["artifact_root"] / "experiment_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, indent=2) + "\n"
    if output.exists() and output.read_text() != content:
        raise RuntimeError(f"refusing to replace frozen experiment manifest: {output}")
    output.write_text(content)
    print(f"Prepared frozen model-arrival manifests → {output.parent}")
    return output


def _budget_label(value: Any) -> str:
    return str(value).lower()


def _result_path(
    cfg: dict[str, Any],
    dataset: str,
    held_out_agent: str,
    budget: Any,
    seed: int,
) -> Path:
    return (
        cfg["experiment"]["artifact_root"]
        / "updates"
        / dataset
        / f"arriving={held_out_agent}"
        / f"budget={_budget_label(budget)}"
        / f"seed={seed}"
        / "results.json"
    )


def _nested_sample(
    rows: list[dict[str, Any]], dataset: str, agent_id: str, seed: int
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["task_id"])
    digest = hashlib.sha256(f"{dataset}|{agent_id}|{seed}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(ordered)
    return ordered


def run_one(
    cfg: dict[str, Any],
    dataset: str,
    held_out_agent: str,
    budget: Any,
    seed: int,
    *,
    xgb_device: str | None,
    force: bool,
) -> Path:
    output = _result_path(cfg, dataset, held_out_agent, budget, seed)
    if output.exists() and not force:
        print(f"[SKIP] {output}")
        return output
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")

    train_rows = _read_jsonl(_manifest_path(cfg, dataset, "train"))
    test_rows = _read_jsonl(_manifest_path(cfg, dataset, "test"))
    established = [row for row in train_rows if row["agent_id"] != held_out_agent]
    arriving_all = [
        row for row in train_rows if row["agent_id"] == held_out_agent
    ]
    arriving_ordered = _nested_sample(
        arriving_all, dataset, held_out_agent, seed
    )
    n_new = len(arriving_all) if str(budget) == "all" else int(budget)
    if n_new > len(arriving_all):
        raise ValueError(
            f"budget {budget} exceeds {len(arriving_all)} traces for {dataset}"
        )
    selected_new = arriving_ordered[:n_new]
    fit_rows = established + selected_new

    X_train, _, _, feature_names = _load_examples(
        fit_rows, False, cfg, cfg.get("feature_group", "full")
    )
    X_test, _, _, test_feature_names = _load_examples(
        test_rows, False, cfg, cfg.get("feature_group", "full")
    )
    if feature_names != test_feature_names:
        raise RuntimeError("feature schema mismatch")
    encoder = LabelEncoder().fit(cfg["agents"])
    y_train = encoder.transform([row["agent_id"] for row in fit_rows])
    y_test = encoder.transform([row["agent_id"] for row in test_rows])

    params = dict(cfg["classifier"]["parameters"])
    params.update(
        {
            "device": xgb_device or cfg["classifier"].get("device", "cpu"),
            "random_state": seed,
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }
    )
    model = XGBClassifier(**params)
    sample_weight = (
        compute_sample_weight("balanced", y_train)
        if cfg["update"].get("class_balanced_weights", True)
        else None
    )
    started = time.perf_counter()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    fit_seconds = time.perf_counter() - started
    predictions = np.asarray(model.predict(X_test))
    metrics = _classification_metrics(y_test, predictions, list(encoder.classes_))
    report = metrics["classification_report"]
    new_model_f1 = float(report[held_out_agent]["f1-score"])
    old_model_f1s = [
        float(report[agent]["f1-score"])
        for agent in cfg["agents"]
        if agent != held_out_agent
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
            "feature_names": feature_names,
            "parameters": params,
        },
        output.with_name("model.joblib"),
    )
    prediction_rows = [
        {
            "episode_id": row["episode_id"],
            "task_id": row["task_id"],
            "true_agent": row["agent_id"],
            "predicted_agent": encoder.inverse_transform(
                [int(predictions[index])]
            )[0],
        }
        for index, row in enumerate(test_rows)
    ]
    _write_jsonl(
        output.with_name("predictions.jsonl"), prediction_rows, force=force
    )
    result = {
        "schema_version": 1,
        "task": "closed_set_model_arrival",
        "dataset": dataset,
        "arriving_agent": held_out_agent,
        "new_trace_budget": n_new,
        "budget_label": _budget_label(budget),
        "seed": seed,
        "n_train_established": len(established),
        "n_train_arriving": len(selected_new),
        "n_test": len(test_rows),
        "fit_seconds": fit_seconds,
        "class_balanced_weights": bool(
            cfg["update"].get("class_balanced_weights", True)
        ),
        "metrics": metrics,
        "new_model_f1": new_model_f1,
        "old_model_macro_f1": float(np.mean(old_model_f1s)),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[UPDATED] {dataset} arriving={held_out_agent} budget={budget} "
        f"seed={seed} new_f1={new_model_f1:.3f} "
        f"macro_f1={metrics['macro_f1']:.3f} fit={fit_seconds:.2f}s"
    )
    return output


def summarize(cfg: dict[str, Any]) -> Path:
    root = cfg["experiment"]["artifact_root"]
    rows = []
    for path in sorted((root / "updates").rglob("results.json")):
        result = json.loads(path.read_text())
        rows.append(
            {
                "dataset": result["dataset"],
                "arriving_agent": result["arriving_agent"],
                "budget_label": result["budget_label"],
                "new_trace_budget": result["new_trace_budget"],
                "seed": result["seed"],
                "new_model_f1": result["new_model_f1"],
                "overall_macro_f1": result["metrics"]["macro_f1"],
                "old_model_macro_f1": result["old_model_macro_f1"],
                "fit_seconds": result["fit_seconds"],
                "results_path": str(path),
            }
        )
    summary_root = root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    raw_csv = summary_root / "results.csv"
    fields = list(rows[0]) if rows else [
        "dataset", "arriving_agent", "budget_label", "new_trace_budget", "seed",
        "new_model_f1", "overall_macro_f1", "old_model_macro_f1", "fit_seconds",
        "results_path",
    ]
    with raw_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    per_arrival = []
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["dataset"], row["arriving_agent"], row["budget_label"])
        ].append(row)
    for (dataset, agent, budget), seed_rows in sorted(grouped.items()):
        per_arrival.append(
            {
                "dataset": dataset,
                "arriving_agent": agent,
                "budget_label": budget,
                "new_trace_budget": seed_rows[0]["new_trace_budget"],
                "n_seeds": len(seed_rows),
                "new_model_f1_mean": float(
                    np.mean([r["new_model_f1"] for r in seed_rows])
                ),
                "new_model_f1_seed_sd": _sample_std(
                    r["new_model_f1"] for r in seed_rows
                ),
                "overall_macro_f1_mean": float(
                    np.mean([r["overall_macro_f1"] for r in seed_rows])
                ),
                "old_model_macro_f1_mean": float(
                    np.mean([r["old_model_macro_f1"] for r in seed_rows])
                ),
                "fit_seconds_median": float(
                    np.median([r["fit_seconds"] for r in seed_rows])
                ),
            }
        )
    per_arrival_csv = summary_root / "per_arriving_model.csv"
    arrival_fields = list(per_arrival[0]) if per_arrival else []
    with per_arrival_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=arrival_fields)
        if arrival_fields:
            writer.writeheader()
            writer.writerows(per_arrival)

    upper = {
        (row["dataset"], row["arriving_agent"]): row["new_model_f1_mean"]
        for row in per_arrival
        if row["budget_label"] == "all"
    }
    aggregate = []
    by_budget: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in per_arrival:
        by_budget[(row["dataset"], row["budget_label"])].append(row)
    budget_order = {
        _budget_label(value): index
        for index, value in enumerate(cfg["update"]["budgets"])
    }
    for (dataset, budget), model_rows in sorted(
        by_budget.items(), key=lambda item: (item[0][0], budget_order[item[0][1]])
    ):
        recoveries = [
            row["new_model_f1_mean"] / upper[(dataset, row["arriving_agent"])]
            for row in model_rows
            if upper.get((dataset, row["arriving_agent"]), 0) > 0
        ]
        aggregate.append(
            {
                "dataset": dataset,
                "budget_label": budget,
                "new_trace_budget": model_rows[0]["new_trace_budget"],
                "n_arriving_models": len(model_rows),
                "seeds_per_model_min": min(r["n_seeds"] for r in model_rows),
                "new_model_f1_mean": float(
                    np.mean([r["new_model_f1_mean"] for r in model_rows])
                ),
                "new_model_f1_model_sd": _sample_std(
                    r["new_model_f1_mean"] for r in model_rows
                ),
                "overall_macro_f1_mean": float(
                    np.mean([r["overall_macro_f1_mean"] for r in model_rows])
                ),
                "old_model_macro_f1_mean": float(
                    np.mean([r["old_model_macro_f1_mean"] for r in model_rows])
                ),
                "full_update_f1_recovered_mean": (
                    float(np.mean(recoveries))
                    if len(recoveries) == len(model_rows)
                    else None
                ),
                "fit_seconds_median": float(
                    np.median([r["fit_seconds_median"] for r in model_rows])
                ),
            }
        )
    aggregate_csv = summary_root / "budget_curve.csv"
    aggregate_fields = list(aggregate[0]) if aggregate else []
    with aggregate_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        if aggregate_fields:
            writer.writeheader()
            writer.writerows(aggregate)
    plot_results(cfg)
    write_report(cfg)
    print(f"Summarized {len(rows)} update fits → {aggregate_csv}")
    return aggregate_csv


def plot_results(cfg: dict[str, Any]) -> list[Path]:
    """Create publication-ready model-arrival learning-curve figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = cfg["experiment"]["artifact_root"]
    summary_root = root / "summaries"
    per_model = read_csv(summary_root / "per_arriving_model.csv")
    aggregate = read_csv(summary_root / "budget_curve.csv")
    if not per_model or not aggregate:
        return []

    budget_labels = [_budget_label(value) for value in cfg["update"]["budgets"]]
    budget_index = {label: index for index, label in enumerate(budget_labels)}
    tick_labels = [label.title() if label == "all" else label for label in budget_labels]
    datasets = list(cfg["datasets"])
    display = {
        "2wikimultihop": "2WikiMultiHopQA",
        "webshop": "WebShop",
    }
    colors = {
        "2wikimultihop": "#2878B5",
        "webshop": "#D55E00",
    }
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Absolute arriving-model F1, including every simulated arrival fold.
    figure, axes = plt.subplots(
        1, len(datasets), figsize=(10.2, 3.8), sharex=True, sharey=True
    )
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, datasets):
        dataset_rows = [row for row in per_model if row["dataset"] == dataset]
        agents = sorted({row["arriving_agent"] for row in dataset_rows})
        for agent in agents:
            rows = sorted(
                (row for row in dataset_rows if row["arriving_agent"] == agent),
                key=lambda row: budget_index[row["budget_label"]],
            )
            axis.plot(
                range(len(rows)),
                [float(row["new_model_f1_mean"]) for row in rows],
                color=colors[dataset],
                alpha=0.16,
                linewidth=0.9,
            )
        mean_rows = sorted(
            (row for row in aggregate if row["dataset"] == dataset),
            key=lambda row: budget_index[row["budget_label"]],
        )
        x = np.arange(len(mean_rows))
        means = np.asarray(
            [float(row["new_model_f1_mean"]) for row in mean_rows]
        )
        model_sd = np.asarray(
            [float(row["new_model_f1_model_sd"]) for row in mean_rows]
        )
        axis.fill_between(
            x,
            np.clip(means - model_sd, 0, 1),
            np.clip(means + model_sd, 0, 1),
            color=colors[dataset],
            alpha=0.18,
            linewidth=0,
            label="±1 SD across arriving models",
        )
        axis.plot(
            x,
            means,
            color=colors[dataset],
            marker="o",
            markersize=4.5,
            linewidth=2.2,
            label="Mean across 14 arrivals",
        )
        axis.set_title(display.get(dataset, dataset))
        axis.set_xticks(x, tick_labels)
        axis.set_xlabel("Labeled traces from arriving model")
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    axes[0].set_ylabel("Arriving-model F1")
    axes[-1].legend(frameon=False, loc="lower right")
    figure.tight_layout()
    learning_png = root / "model_arrival_learning_curve.png"
    learning_pdf = root / "model_arrival_learning_curve.pdf"
    figure.savefig(learning_png, dpi=300, bbox_inches="tight")
    figure.savefig(learning_pdf, bbox_inches="tight")
    plt.close(figure)

    # Normalized recovery communicates label efficiency; refit time communicates
    # the computational cost of updating the closed-set model.
    figure, (recovery_axis, time_axis) = plt.subplots(
        1, 2, figsize=(10.2, 3.7), sharex=True
    )
    for dataset in datasets:
        rows = sorted(
            (row for row in aggregate if row["dataset"] == dataset),
            key=lambda row: budget_index[row["budget_label"]],
        )
        x = np.arange(len(rows))
        label = display.get(dataset, dataset)
        recovery_axis.plot(
            x,
            [float(row["full_update_f1_recovered_mean"]) for row in rows],
            color=colors[dataset],
            marker="o",
            markersize=4.5,
            linewidth=2,
            label=label,
        )
        time_axis.plot(
            x,
            [float(row["fit_seconds_median"]) for row in rows],
            color=colors[dataset],
            marker="o",
            markersize=4.5,
            linewidth=2,
            label=label,
        )
    recovery_axis.axhline(
        1.0, color="#555555", linestyle="--", linewidth=1, label="Full update"
    )
    recovery_axis.set_ylabel("Fraction of full-update F1")
    recovery_axis.set_ylim(0, 1.08)
    recovery_axis.set_title("Performance recovered")
    time_axis.set_ylabel("Median refit time (seconds)")
    time_axis.set_title("Computational update cost")
    for axis in (recovery_axis, time_axis):
        axis.set_xticks(range(len(budget_labels)), tick_labels)
        axis.set_xlabel("Labeled traces from arriving model")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    recovery_axis.legend(frameon=False, loc="lower right")
    time_axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    efficiency_png = root / "model_arrival_update_efficiency.png"
    efficiency_pdf = root / "model_arrival_update_efficiency.pdf"
    figure.savefig(efficiency_png, dpi=300, bbox_inches="tight")
    figure.savefig(efficiency_pdf, bbox_inches="tight")
    plt.close(figure)
    return [learning_png, learning_pdf, efficiency_png, efficiency_pdf]


def write_report(cfg: dict[str, Any]) -> Path:
    root = cfg["experiment"]["artifact_root"]
    report_path = root / "REPORT.md"
    summary_root = root / "summaries"
    rows = read_csv(summary_root / "budget_curve.csv")
    table_rows = [
        [
            row["dataset"],
            row["budget_label"],
            row["n_arriving_models"],
            row["seeds_per_model_min"],
            (
                f"{number(row['new_model_f1_mean'])} ± "
                f"{number(row['new_model_f1_model_sd'])}"
            ),
            number(row["overall_macro_f1_mean"]),
            number(row["old_model_macro_f1_mean"]),
            number(row["full_update_f1_recovered_mean"]),
            number(row["fit_seconds_median"], 2),
        ]
        for row in rows
    ]
    lines = [
        "# Experiment report: adding a newly released model",
        "",
        f"Experiment ID: `{cfg['experiment']['id']}`",
        "",
        "## Question",
        "",
        "How many labeled traces from a newly released model are needed to "
        "update the existing closed-set classifier?",
        "",
        "Each of the 14 MidScene models is treated as the arriving model once. "
        "The other 13 models retain their complete training data. We add the "
        "specified number of arriving-model traces, refit the same fixed "
        "class-balanced XGBoost configuration, and evaluate on the unchanged "
        "14-model test set. Five classifier/sampling seeds are used.",
        "",
        markdown_table(
            [
                "Dataset",
                "New labeled traces",
                "Arrival folds",
                "Seeds/fold",
                "Arriving-model F1 mean ± model SD",
                "Overall macro-F1",
                "Established-model macro-F1",
                "Fraction of full-update F1",
                "Median refit seconds",
            ],
            table_rows,
        ) if table_rows else "_The update grid has not been run yet._",
        "",
        "The SD above is across the completed simulated arriving-model folds "
        "after averaging available classifier seeds within each model. The "
        "final grid contains all 14 folds and five seeds per fold. This SD "
        "measures how much update difficulty varies by model.",
        "",
        "## Learning curves",
        "",
        "![Arriving-model F1 as labeled traces are added]"
        "(model_arrival_learning_curve.png)",
        "",
        "Thin lines show each of the 14 simulated arriving models. The thick "
        "line is their mean, and the shaded region is ±1 SD across arriving "
        "models after averaging the five classifier/sampling seeds. The band "
        "therefore represents variation in update difficulty across models, "
        "not a confidence interval.",
        "",
        "![Fraction of full-update performance recovered and refit time]"
        "(model_arrival_update_efficiency.png)",
        "",
        "The left panel normalizes each arriving model by its all-traces F1 "
        "before averaging, while the right panel shows the median XGBoost "
        "refit time. Together they summarize the labeled-data and compute "
        "needed to update the closed set.",
        "",
        "## Artifact map",
        "",
        f"- Every fit: {relative_link(summary_root / 'results.csv', report_path)}",
        f"- Per arriving model: "
        f"{relative_link(summary_root / 'per_arriving_model.csv', report_path)}",
        f"- Headline budget curve: "
        f"{relative_link(summary_root / 'budget_curve.csv', report_path)}",
        "- Learning-curve figure: "
        "[PNG](model_arrival_learning_curve.png), "
        "[PDF](model_arrival_learning_curve.pdf).",
        "- Update-efficiency figure: "
        "[PNG](model_arrival_update_efficiency.png), "
        "[PDF](model_arrival_update_efficiency.pdf).",
        "- Models and predictions: `updates/`.",
        "- Frozen matched MidScene splits: `frozen_midscene_splits/`.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"Wrote human-readable report → {report_path}")
    return report_path


def run_grid(
    cfg: dict[str, Any],
    *,
    xgb_device: str | None,
    force: bool,
    quick: bool,
) -> None:
    budgets = cfg["update"]["budgets"]
    seeds = cfg["update"]["classifier_seeds"]
    agents = cfg["agents"]
    if quick:
        budgets = [1, 10]
        seeds = [42]
        agents = agents[:2]
    for dataset in cfg["datasets"]:
        for agent in agents:
            for budget in budgets:
                for seed in seeds:
                    run_one(
                        cfg,
                        dataset,
                        agent,
                        budget,
                        int(seed),
                        xgb_device=xgb_device,
                        force=force,
                    )
    summarize(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "midscene_14model.yaml",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("prepare")
    run = sub.add_parser("run-grid")
    run.add_argument("--xgb-device", choices=["cpu", "cuda"], default=None)
    run.add_argument("--force", action="store_true")
    run.add_argument("--quick", action="store_true")
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
            xgb_device=args.xgb_device,
            force=args.force,
            quick=args.quick,
        )
    elif args.command == "summarize":
        summarize(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
