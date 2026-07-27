#!/usr/bin/env python3
"""Measure model-identification scaling as the closed set grows to 14 models."""

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

import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder

from experiments.cross_harness.pipeline import (
    _classification_metrics,
    _load_examples,
    _sample_std,
    _sha256_file,
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
    experiment = cfg["experiment"]
    experiment["artifact_root"] = _resolve(path, experiment["artifact_root"])
    experiment["source_manifests"] = _resolve(path, experiment["source_manifests"])
    cfg["_config_path"] = path
    return cfg


def _source_manifest(cfg: dict[str, Any], dataset: str, split: str) -> Path:
    return cfg["experiment"]["source_manifests"] / dataset / f"{split}.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(cfg: dict[str, Any]) -> None:
    expected_agents = set(cfg["agents"])
    for dataset in cfg["datasets"]:
        print(dataset)
        for split in ("train", "val", "test"):
            path = _source_manifest(cfg, dataset, split)
            rows = _read_jsonl(path)
            agents = {row["agent_id"] for row in rows}
            counts = {
                agent: sum(row["agent_id"] == agent for row in rows)
                for agent in sorted(agents)
            }
            if agents != expected_agents:
                raise RuntimeError(
                    f"{dataset}/{split}: model roster differs from configured agents"
                )
            unique_counts = sorted(set(counts.values()))
            if len(unique_counts) != 1:
                raise RuntimeError(
                    f"{dataset}/{split}: unbalanced traces per model: {counts}"
                )
            print(
                f"  {split}: {len(rows)} traces, {len(agents)} models, "
                f"{unique_counts[0]} traces/model"
            )


def prepare(cfg: dict[str, Any]) -> Path:
    audit(cfg)
    source_hashes = {}
    for dataset in cfg["datasets"]:
        for split in ("train", "val", "test"):
            path = _source_manifest(cfg, dataset, split)
            source_hashes[str(path)] = _sha256_file(path)
    manifest = {
        "schema_version": 1,
        "experiment_id": cfg["experiment"]["id"],
        "config": str(cfg["_config_path"]),
        "protocol": "nested_random_class_sets",
        "agents": cfg["agents"],
        "class_counts": cfg["scaling"]["class_counts"],
        "seeds": cfg["scaling"]["seeds"],
        "source_manifest_sha256": source_hashes,
    }
    output = cfg["experiment"]["artifact_root"] / "experiment_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, indent=2) + "\n"
    if output.exists() and output.read_text() != content:
        raise RuntimeError(f"refusing to replace frozen experiment manifest: {output}")
    output.write_text(content)
    print(f"Prepared class-count scaling experiment → {output.parent}")
    return output


def _ordered_agents(cfg: dict[str, Any], dataset: str, seed: int) -> list[str]:
    agents = sorted(cfg["agents"])
    digest = hashlib.sha256(f"{dataset}|{seed}|class-order".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(agents)
    return agents


def _result_path(
    cfg: dict[str, Any], dataset: str, class_count: int, seed: int
) -> Path:
    return (
        cfg["experiment"]["artifact_root"]
        / "fits"
        / dataset
        / f"classes={class_count}"
        / f"seed={seed}"
        / "results.json"
    )


def run_one(
    cfg: dict[str, Any],
    dataset: str,
    class_count: int,
    seed: int,
    *,
    xgb_device: str | None,
    force: bool,
) -> Path:
    output = _result_path(cfg, dataset, class_count, seed)
    requested_device = xgb_device or cfg["classifier"].get("device", "cpu")
    if output.exists() and not force:
        existing = json.loads(output.read_text())
        existing_device = (existing.get("parameters") or {}).get("device")
        if existing_device == requested_device:
            print(f"[SKIP] {output}")
            return output
        print(
            f"[RERUN] {output}: device changed from "
            f"{existing_device!r} to {requested_device!r}"
        )
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")
    order = _ordered_agents(cfg, dataset, seed)
    selected = set(order[:class_count])
    train_rows = [
        row
        for row in _read_jsonl(_source_manifest(cfg, dataset, "train"))
        if row["agent_id"] in selected
    ]
    test_rows = [
        row
        for row in _read_jsonl(_source_manifest(cfg, dataset, "test"))
        if row["agent_id"] in selected
    ]
    X_train, _, train_labels, feature_names = _load_examples(
        train_rows, False, cfg, cfg.get("feature_group", "full")
    )
    X_test, _, test_labels, test_feature_names = _load_examples(
        test_rows, False, cfg, cfg.get("feature_group", "full")
    )
    if feature_names != test_feature_names:
        raise RuntimeError("feature schema mismatch")
    encoder = LabelEncoder().fit(sorted(selected))
    y_train = encoder.transform(train_labels)
    y_test = encoder.transform(test_labels)
    params = dict(cfg["classifier"]["parameters"])
    params.update(
        {
            "device": requested_device,
            "random_state": seed,
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }
    )
    model = XGBClassifier(**params)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    predictions = np.asarray(model.predict(X_test))
    predict_seconds = time.perf_counter() - prediction_started
    metrics = _classification_metrics(
        y_test, predictions, list(encoder.classes_)
    )
    booster = model.get_booster()
    tree_dumps = booster.get_dump(dump_format="text")
    lines = [
        line
        for tree in tree_dumps
        for line in tree.splitlines()
        if line.strip()
    ]
    try:
        model_bytes = len(booster.save_raw(raw_format="ubj"))
    except TypeError:  # pragma: no cover - older XGBoost
        model_bytes = len(booster.save_raw())
    result = {
        "schema_version": 1,
        "task": "closed_set_class_count_scaling",
        "dataset": dataset,
        "class_count": class_count,
        "seed": seed,
        "model_order": order,
        "selected_agents": order[:class_count],
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "fit_seconds": fit_seconds,
        "predict_ms_per_trace": 1000.0 * predict_seconds / len(test_rows),
        "model_size_bytes": model_bytes,
        "n_trees": len(tree_dumps),
        "n_tree_nodes": len(lines),
        "n_tree_leaves": sum("leaf=" in line for line in lines),
        "metrics": metrics,
        "parameters": params,
        "feature_names": feature_names,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[SCALED] {dataset} classes={class_count} seed={seed} "
        f"macro_f1={metrics['macro_f1']:.3f} fit={fit_seconds:.2f}s"
    )
    return output


def summarize(cfg: dict[str, Any]) -> Path:
    root = cfg["experiment"]["artifact_root"]
    rows = []
    for path in sorted((root / "fits").rglob("results.json")):
        result = json.loads(path.read_text())
        rows.append(
            {
                "dataset": result["dataset"],
                "class_count": result["class_count"],
                "seed": result["seed"],
                "selected_agents": json.dumps(result["selected_agents"]),
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "macro_f1": result["metrics"]["macro_f1"],
                "accuracy": result["metrics"]["accuracy"],
                "fit_seconds": result["fit_seconds"],
                "predict_ms_per_trace": result["predict_ms_per_trace"],
                "model_size_bytes": result["model_size_bytes"],
                "n_trees": result["n_trees"],
                "n_tree_nodes": result["n_tree_nodes"],
                "n_tree_leaves": result["n_tree_leaves"],
                "results_path": str(path),
            }
        )
    summary_root = root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    results_csv = summary_root / "results.csv"
    fields = list(rows[0]) if rows else [
        "dataset", "class_count", "seed", "selected_agents", "n_train",
        "n_test", "macro_f1", "accuracy", "fit_seconds",
        "predict_ms_per_trace", "model_size_bytes", "n_trees",
        "n_tree_nodes", "n_tree_leaves", "results_path",
    ]
    with results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], int(row["class_count"]))].append(row)
    aggregate = []
    for (dataset, class_count), repetitions in sorted(grouped.items()):
        macro_values = [float(row["macro_f1"]) for row in repetitions]
        fit_values = [float(row["fit_seconds"]) for row in repetitions]
        prediction_values = [
            float(row["predict_ms_per_trace"]) for row in repetitions
        ]
        size_values = [
            float(row["model_size_bytes"]) / (1024 * 1024)
            for row in repetitions
        ]
        node_values = [float(row["n_tree_nodes"]) for row in repetitions]
        aggregate.append(
            {
                "dataset": dataset,
                "class_count": class_count,
                "n_repetitions": len(repetitions),
                "macro_f1_mean": float(np.mean(macro_values)),
                "macro_f1_sample_sd": _sample_std(macro_values),
                "accuracy_mean": float(
                    np.mean([float(row["accuracy"]) for row in repetitions])
                ),
                "fit_seconds_median": float(np.median(fit_values)),
                "fit_seconds_q25": float(np.quantile(fit_values, 0.25)),
                "fit_seconds_q75": float(np.quantile(fit_values, 0.75)),
                "predict_ms_per_trace_median": float(
                    np.median(prediction_values)
                ),
                "predict_ms_per_trace_q25": float(
                    np.quantile(prediction_values, 0.25)
                ),
                "predict_ms_per_trace_q75": float(
                    np.quantile(prediction_values, 0.75)
                ),
                "model_size_mib_median": float(np.median(size_values)),
                "model_size_mib_q25": float(np.quantile(size_values, 0.25)),
                "model_size_mib_q75": float(np.quantile(size_values, 0.75)),
                "n_trees_median": float(
                    np.median([float(row["n_trees"]) for row in repetitions])
                ),
                "n_tree_nodes_median": float(np.median(node_values)),
                "n_tree_nodes_q25": float(np.quantile(node_values, 0.25)),
                "n_tree_nodes_q75": float(np.quantile(node_values, 0.75)),
            }
        )
    curve_csv = summary_root / "class_count_curve.csv"
    aggregate_fields = list(aggregate[0]) if aggregate else []
    with curve_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        if aggregate_fields:
            writer.writeheader()
            writer.writerows(aggregate)
    plot_results(cfg)
    write_report(cfg)
    print(f"Summarized {len(rows)} class-count fits → {curve_csv}")
    return curve_csv


def plot_results(cfg: dict[str, Any]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = cfg["experiment"]["artifact_root"]
    rows = read_csv(root / "summaries" / "class_count_curve.csv")
    raw = read_csv(root / "summaries" / "results.csv")
    if not rows:
        return []
    display = {
        "2wikimultihop": "2WikiMultiHopQA",
        "webshop": "WebShop",
    }
    colors = {"2wikimultihop": "#2878B5", "webshop": "#D55E00"}
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    datasets = list(cfg["datasets"])
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    performance_axis, time_axis = axes
    for dataset in datasets:
        dataset_rows = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: int(row["class_count"]),
        )
        x = np.asarray([int(row["class_count"]) for row in dataset_rows])
        mean = np.asarray([float(row["macro_f1_mean"]) for row in dataset_rows])
        sd = np.asarray(
            [
                float(row["macro_f1_sample_sd"])
                if row["macro_f1_sample_sd"]
                else 0.0
                for row in dataset_rows
            ]
        )
        # Faint nested repetitions expose class-composition variability.
        seeds = sorted(
            {
                int(row["seed"])
                for row in raw
                if row["dataset"] == dataset
            }
        )
        for seed in seeds:
            seed_rows = sorted(
                (
                    row
                    for row in raw
                    if row["dataset"] == dataset
                    and int(row["seed"]) == seed
                ),
                key=lambda row: int(row["class_count"]),
            )
            performance_axis.plot(
                [int(row["class_count"]) for row in seed_rows],
                [float(row["macro_f1"]) for row in seed_rows],
                color=colors[dataset],
                alpha=0.15,
                linewidth=0.9,
            )
            time_axis.plot(
                [int(row["class_count"]) for row in seed_rows],
                [float(row["fit_seconds"]) for row in seed_rows],
                color=colors[dataset],
                alpha=0.15,
                linewidth=0.9,
            )
        performance_axis.fill_between(
            x,
            np.clip(mean - sd, 0, 1),
            np.clip(mean + sd, 0, 1),
            color=colors[dataset],
            alpha=0.16,
            linewidth=0,
        )
        performance_axis.plot(
            x,
            mean,
            color=colors[dataset],
            marker="o",
            linewidth=2.2,
            label=display.get(dataset, dataset),
        )
        time_axis.plot(
            x,
            [float(row["fit_seconds_median"]) for row in dataset_rows],
            color=colors[dataset],
            marker="o",
            linewidth=2.2,
            label=display.get(dataset, dataset),
        )
        time_axis.fill_between(
            x,
            [float(row["fit_seconds_q25"]) for row in dataset_rows],
            [float(row["fit_seconds_q75"]) for row in dataset_rows],
            color=colors[dataset],
            alpha=0.16,
            linewidth=0,
        )
    chance_x = np.asarray(cfg["scaling"]["class_counts"], dtype=float)
    performance_axis.plot(
        chance_x,
        1.0 / chance_x,
        color="#555555",
        linestyle="--",
        linewidth=1.4,
        label="Uniform random (1/k)",
    )
    performance_axis.set_title("Identification as the closed set grows")
    performance_axis.set_xlabel("Number of candidate models")
    performance_axis.set_ylabel("Macro-F1")
    performance_axis.set_ylim(0, 1.02)
    performance_axis.legend(frameon=False)
    time_axis.set_title("Classifier fitting cost")
    time_axis.set_xlabel("Number of candidate models")
    time_axis.set_ylabel("Median fit time (seconds)")
    time_axis.legend(frameon=False)
    for axis in axes:
        axis.set_xticks(cfg["scaling"]["class_counts"])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    figure.tight_layout()
    png = root / "closed_set_class_count_scaling.png"
    pdf = root / "closed_set_class_count_scaling.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    figure, (size_axis, nodes_axis) = plt.subplots(1, 2, figsize=(10.2, 3.7))
    for dataset in datasets:
        dataset_rows = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: int(row["class_count"]),
        )
        x = [int(row["class_count"]) for row in dataset_rows]
        label = display.get(dataset, dataset)
        size_axis.plot(
            x,
            [float(row["model_size_mib_median"]) for row in dataset_rows],
            color=colors[dataset],
            marker="o",
            linewidth=2.2,
            label=label,
        )
        size_axis.fill_between(
            x,
            [float(row["model_size_mib_q25"]) for row in dataset_rows],
            [float(row["model_size_mib_q75"]) for row in dataset_rows],
            color=colors[dataset],
            alpha=0.16,
            linewidth=0,
        )
        nodes_axis.plot(
            x,
            [float(row["n_tree_nodes_median"]) for row in dataset_rows],
            color=colors[dataset],
            marker="o",
            linewidth=2.2,
            label=label,
        )
        nodes_axis.fill_between(
            x,
            [float(row["n_tree_nodes_q25"]) for row in dataset_rows],
            [float(row["n_tree_nodes_q75"]) for row in dataset_rows],
            color=colors[dataset],
            alpha=0.16,
            linewidth=0,
        )
    size_axis.set_title("Serialized classifier size")
    size_axis.set_ylabel("Median booster size (MiB)")
    nodes_axis.set_title("Tree traversal complexity")
    nodes_axis.set_ylabel("Median total tree nodes")
    for axis in (size_axis, nodes_axis):
        axis.set_xlabel("Number of candidate models")
        axis.set_xticks(cfg["scaling"]["class_counts"])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        axis.legend(frameon=False)
    figure.tight_layout()
    complexity_png = root / "closed_set_classifier_complexity.png"
    complexity_pdf = root / "closed_set_classifier_complexity.pdf"
    figure.savefig(complexity_png, dpi=300, bbox_inches="tight")
    figure.savefig(complexity_pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf, complexity_png, complexity_pdf]


def write_report(cfg: dict[str, Any]) -> Path:
    root = cfg["experiment"]["artifact_root"]
    report = root / "REPORT.md"
    summary_root = root / "summaries"
    rows = read_csv(summary_root / "class_count_curve.csv")
    expected = (
        len(cfg["datasets"])
        * len(cfg["scaling"]["class_counts"])
        * len(cfg["scaling"]["seeds"])
    )
    selected_counts = {2, 4, 6, 8, 10, 12, 14}
    table = []
    for row in rows:
        if int(row["class_count"]) not in selected_counts:
            continue
        sd = (
            number(row["macro_f1_sample_sd"])
            if int(row["n_repetitions"]) >= 2
            else "—"
        )
        table.append(
            [
                row["dataset"],
                row["class_count"],
                row["n_repetitions"],
                f"{number(row['macro_f1_mean'])} ± {sd}",
                number(row["fit_seconds_median"], 2),
                number(row["model_size_mib_median"], 2),
                number(row["n_tree_nodes_median"], 0),
            ]
        )
    complete = sum(int(row["n_repetitions"]) for row in rows)
    lines = [
        "# Experiment report: closed-set class-count scaling",
        "",
        f"Experiment ID: `{cfg['experiment']['id']}`",
        "",
        "## Question",
        "",
        "How does model-identification performance and classifier complexity "
        "change as the closed set grows from 2 to 14 candidate models?",
        "",
        "The experiment uses the original task-matched MidScene train/test "
        "traces. Each of five seeds defines a different randomized ordering of "
        "the 14 models. Class sets are nested within a seed: the k+1 condition "
        "contains every model from k plus one additional model. XGBoost "
        "hyperparameters remain fixed.",
        "",
        f"Completed fits: **{complete}/{expected}**.",
        "",
        markdown_table(
            [
                "Dataset",
                "Candidate models",
                "Repetitions",
                "Macro-F1 mean ± sample SD",
                "Median fit seconds",
                "Median size (MiB)",
                "Median tree nodes",
            ],
            table,
        )
        if table
        else "_The scaling grid has not been run yet._",
        "",
        "The SD combines variation from the randomized nested model composition "
        "and the XGBoost training seed. It is not a trace-bootstrap confidence "
        "interval.",
        "",
        "## Scaling curves",
        "",
        "![Macro-F1 and fitting time as the closed set grows]"
        "(closed_set_class_count_scaling.png)",
        "",
        "Faint lines show the five nested model orders; thick lines show their "
        "mean and the band shows ±1 sample SD. The dashed 1/k curve is the "
        "expected macro-F1 of uniform random prediction on a balanced k-class "
        "test set. The fitting-time panel likewise shows each repetition as a "
        "faint line, with the thick line giving the median and the colored band "
        "showing the interquartile range.",
        "",
        "![Serialized model size and total tree nodes]"
        "(closed_set_classifier_complexity.png)",
        "",
        "Booster size and total tree nodes provide transparent storage and "
        "tree-traversal complexity measures; their bands show the interquartile "
        "range across model subsets. We do not report FLOPs because they are "
        "not a standardized cost measure for decision-tree inference.",
        "",
        "## Artifact map",
        "",
        f"- Every fit: {relative_link(summary_root / 'results.csv', report)}",
        f"- Class-count curve: "
        f"{relative_link(summary_root / 'class_count_curve.csv', report)}",
        "- Scaling figure: [PNG](closed_set_class_count_scaling.png), "
        "[PDF](closed_set_class_count_scaling.pdf).",
        "- Complexity figure: [PNG](closed_set_classifier_complexity.png), "
        "[PDF](closed_set_classifier_complexity.pdf).",
        "- Individual fits and selected model sets: `fits/`.",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(f"Wrote human-readable report → {report}")
    return report


def run_grid(
    cfg: dict[str, Any],
    *,
    xgb_device: str | None,
    force: bool,
    quick: bool,
) -> None:
    datasets = list(cfg["datasets"])
    class_counts = list(cfg["scaling"]["class_counts"])
    seeds = list(cfg["scaling"]["seeds"])
    if quick:
        datasets = datasets[:1]
        class_counts = [2, 4]
        seeds = [42]
    for dataset in datasets:
        for seed in seeds:
            for class_count in class_counts:
                run_one(
                    cfg,
                    dataset,
                    int(class_count),
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
        default=Path(__file__).with_name("configs")
        / "midscene_14model_class_count.yaml",
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
