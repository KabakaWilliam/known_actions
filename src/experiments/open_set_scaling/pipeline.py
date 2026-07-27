#!/usr/bin/env python3
"""Evaluate model identification when one to four models are entirely unknown."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
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


_DATA_CACHE: dict[
    tuple[str, str, str, str],
    tuple[np.ndarray, list[dict[str, Any]], list[str], list[str]],
] = {}
_SOURCE_HASH_CACHE: dict[str, str] = {}
_METRIC_FIELDS = (
    "open_set_auroc",
    "open_set_average_precision",
    "oscr",
    "known_macro_f1",
    "known_accuracy",
    "known_acceptance",
    "unknown_recall",
    "balanced_open_set_macro_f1",
)


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


def _source_hash(path: Path) -> str:
    key = str(path)
    if key not in _SOURCE_HASH_CACHE:
        _SOURCE_HASH_CACHE[key] = _sha256_file(path)
    return _SOURCE_HASH_CACHE[key]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(cfg: dict[str, Any]) -> None:
    expected_agents = set(cfg["agents"])
    for dataset in cfg["datasets"]:
        print(dataset)
        for split in ("train", "val", "test"):
            rows = _read_jsonl(_source_manifest(cfg, dataset, split))
            agents = {row["agent_id"] for row in rows}
            counts = Counter(row["agent_id"] for row in rows)
            if agents != expected_agents:
                raise RuntimeError(
                    f"{dataset}/{split}: model roster differs from configured agents"
                )
            if len(set(counts.values())) != 1:
                raise RuntimeError(
                    f"{dataset}/{split}: traces are not balanced by model: "
                    f"{dict(sorted(counts.items()))}"
                )
            print(
                f"  {split}: {len(rows)} traces, {len(agents)} models, "
                f"{next(iter(counts.values()))} traces/model"
            )


def _balanced_holdout_sets(
    agents: Iterable[str],
    unknown_count: int,
    requested: int | str,
    seed: int,
) -> list[tuple[str, ...]]:
    """Select deterministic combinations while balancing model/pair exposure."""
    agents = tuple(sorted(agents))
    candidates = list(itertools.combinations(agents, unknown_count))
    if requested == "all" or int(requested) >= len(candidates):
        return candidates
    target = int(requested)
    if target <= 0:
        raise ValueError("the number of holdout sets must be positive")
    rng = random.Random(seed + 1009 * unknown_count)
    rng.shuffle(candidates)
    model_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    selected: list[tuple[str, ...]] = []
    remaining = list(candidates)
    for _ in range(target):
        best_index = min(
            range(len(remaining)),
            key=lambda index: (
                sum(2 * model_counts[agent] + 1 for agent in remaining[index]),
                sum(
                    2 * pair_counts[pair] + 1
                    for pair in itertools.combinations(remaining[index], 2)
                ),
                index,
            ),
        )
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        model_counts.update(chosen)
        pair_counts.update(itertools.combinations(chosen, 2))
    return sorted(selected)


def _holdout_definitions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = []
    design_seed = int(cfg["open_set"]["holdout_design_seed"])
    requested_by_count = cfg["open_set"]["holdout_sets"]
    for unknown_count in cfg["open_set"]["unknown_counts"]:
        unknown_count = int(unknown_count)
        requested = requested_by_count.get(
            unknown_count, requested_by_count.get(str(unknown_count))
        )
        if requested is None:
            raise ValueError(f"missing holdout-set count for p={unknown_count}")
        sets = _balanced_holdout_sets(
            cfg["agents"], unknown_count, requested, design_seed
        )
        for index, agents in enumerate(sets):
            digest = hashlib.sha256("\0".join(agents).encode()).hexdigest()[:10]
            definitions.append(
                {
                    "unknown_count": unknown_count,
                    "holdout_index": index,
                    "holdout_id": f"p{unknown_count}_{index:03d}_{digest}",
                    "unknown_agents": list(agents),
                }
            )
    return definitions


def _holdouts_path(cfg: dict[str, Any]) -> Path:
    return cfg["experiment"]["artifact_root"] / "holdout_sets.json"


def prepare(cfg: dict[str, Any]) -> Path:
    audit(cfg)
    holdouts = _holdout_definitions(cfg)
    holdouts_content = json.dumps(holdouts, indent=2) + "\n"
    holdouts_path = _holdouts_path(cfg)
    holdouts_path.parent.mkdir(parents=True, exist_ok=True)
    if holdouts_path.exists() and holdouts_path.read_text() != holdouts_content:
        raise RuntimeError(f"refusing to replace frozen holdout design: {holdouts_path}")
    holdouts_path.write_text(holdouts_content)
    source_hashes = {}
    for dataset in cfg["datasets"]:
        for split in ("train", "val", "test"):
            path = _source_manifest(cfg, dataset, split)
            source_hashes[str(path)] = _source_hash(path)
    manifest = {
        "schema_version": 1,
        "experiment_id": cfg["experiment"]["id"],
        "config": str(cfg["_config_path"]),
        "protocol": "leave_p_models_out_open_set",
        "agents": cfg["agents"],
        "unknown_counts": cfg["open_set"]["unknown_counts"],
        "classifier_seeds": cfg["open_set"]["classifier_seeds"],
        "known_validation_acceptance": cfg["open_set"][
            "known_validation_acceptance"
        ],
        "n_holdout_sets": dict(Counter(row["unknown_count"] for row in holdouts)),
        "holdout_sets_sha256": _sha256_file(holdouts_path),
        "source_manifest_sha256": source_hashes,
    }
    output = cfg["experiment"]["artifact_root"] / "experiment_manifest.json"
    content = json.dumps(manifest, indent=2) + "\n"
    if output.exists() and output.read_text() != content:
        raise RuntimeError(f"refusing to replace frozen experiment manifest: {output}")
    output.write_text(content)
    print(
        f"Prepared {len(holdouts)} frozen leave-p-out sets → "
        f"{cfg['experiment']['artifact_root']}"
    )
    return output


def _load_holdouts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    path = _holdouts_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"holdout design not found; run prepare first: {path}")
    return json.loads(path.read_text())


def _dataset_arrays(
    cfg: dict[str, Any], dataset: str, split: str
) -> tuple[np.ndarray, list[dict[str, Any]], list[str], list[str]]:
    feature_group = cfg.get("feature_group", "full")
    key = (str(cfg["_config_path"]), dataset, split, feature_group)
    if key not in _DATA_CACHE:
        rows = _read_jsonl(_source_manifest(cfg, dataset, split))
        source_hash = _source_hash(_source_manifest(cfg, dataset, split))
        feature_spec = json.dumps(
            cfg.get("feature_groups", {}), sort_keys=True, separators=(",", ":")
        )
        cache_path = (
            cfg["experiment"]["artifact_root"]
            / "feature_cache"
            / dataset
            / f"{split}_{feature_group}.npz"
        )
        cached = None
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as archive:
                if (
                    str(archive["source_manifest_sha256"].item()) == source_hash
                    and str(archive["feature_spec"].item()) == feature_spec
                ):
                    cached = (
                        np.asarray(archive["X"], dtype=float),
                        [str(value) for value in archive["labels"]],
                        [str(value) for value in archive["feature_names"]],
                    )
        if cached is None:
            print(
                f"[FEATURES] extracting {dataset}/{split} from {len(rows)} traces",
                flush=True,
            )
            X, _, labels, feature_names = _load_examples(
                rows, False, cfg, feature_group
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    X=X,
                    labels=np.asarray(labels),
                    feature_names=np.asarray(feature_names),
                    source_manifest_sha256=np.asarray(source_hash),
                    feature_spec=np.asarray(feature_spec),
                )
            temporary.replace(cache_path)
            print(f"[FEATURES] cached → {cache_path}", flush=True)
        else:
            X, labels, feature_names = cached
            print(f"[FEATURES] loaded cache → {cache_path}", flush=True)
        _DATA_CACHE[key] = (X, rows, labels, feature_names)
    return _DATA_CACHE[key]


def _confidence_threshold(confidence: np.ndarray, acceptance: float) -> float:
    if not 0 < acceptance < 1:
        raise ValueError("known validation acceptance must be between zero and one")
    try:
        return float(np.quantile(confidence, 1.0 - acceptance, method="lower"))
    except TypeError:  # pragma: no cover - NumPy < 1.22
        return float(
            np.quantile(confidence, 1.0 - acceptance, interpolation="lower")
        )


def _oscr(
    known_confidence: np.ndarray,
    known_correct: np.ndarray,
    unknown_confidence: np.ndarray,
) -> float:
    """Area under correct-classification-rate vs unknown false-positive-rate."""
    grouped: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for score, correct in zip(known_confidence, known_correct, strict=True):
        grouped[float(score)][0] += int(bool(correct))
    for score in unknown_confidence:
        grouped[float(score)][1] += 1
    false_positive_rates = [0.0]
    correct_classification_rates = [0.0]
    cumulative_correct = 0
    cumulative_unknown = 0
    for score in sorted(grouped, reverse=True):
        correct_count, unknown_count = grouped[score]
        cumulative_correct += correct_count
        cumulative_unknown += unknown_count
        false_positive_rates.append(cumulative_unknown / len(unknown_confidence))
        correct_classification_rates.append(
            cumulative_correct / len(known_confidence)
        )
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(
        integrate(
            np.asarray(correct_classification_rates),
            np.asarray(false_positive_rates),
        )
    )


def _balanced_open_set_macro_f1(
    known_true: np.ndarray,
    known_predicted: np.ndarray,
    known_confidence: np.ndarray,
    unknown_predicted: np.ndarray,
    unknown_confidence: np.ndarray,
    threshold: float,
    n_known_classes: int,
    unknown_count: int,
) -> float:
    unknown_label = n_known_classes
    true = np.concatenate(
        [known_true, np.full(len(unknown_predicted), unknown_label, dtype=int)]
    )
    predicted = np.concatenate([known_predicted, unknown_predicted]).astype(int)
    confidence = np.concatenate([known_confidence, unknown_confidence])
    predicted[confidence < threshold] = unknown_label
    weights = np.concatenate(
        [
            np.ones(len(known_true), dtype=float),
            np.full(len(unknown_predicted), 1.0 / unknown_count, dtype=float),
        ]
    )
    return float(
        f1_score(
            true,
            predicted,
            labels=np.arange(n_known_classes + 1),
            average="macro",
            sample_weight=weights,
            zero_division=0,
        )
    )


def _result_path(
    cfg: dict[str, Any], dataset: str, holdout: dict[str, Any], seed: int
) -> Path:
    return (
        cfg["experiment"]["artifact_root"]
        / "fits"
        / dataset
        / f"unknown_count={holdout['unknown_count']}"
        / f"holdout={holdout['holdout_id']}"
        / f"seed={seed}"
        / "results.json"
    )


def run_one(
    cfg: dict[str, Any],
    dataset: str,
    holdout: dict[str, Any],
    seed: int,
    *,
    xgb_device: str | None,
    force: bool,
) -> Path:
    output = _result_path(cfg, dataset, holdout, seed)
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
    unknown_agents = set(holdout["unknown_agents"])
    known_agents = sorted(set(cfg["agents"]) - unknown_agents)
    known_agent_set = set(known_agents)
    arrays = {
        split: _dataset_arrays(cfg, dataset, split)
        for split in ("train", "val", "test")
    }
    feature_names = arrays["train"][3]
    if any(value[3] != feature_names for value in arrays.values()):
        raise RuntimeError("train/validation/test feature schema mismatch")
    encoder = LabelEncoder().fit(known_agents)

    def known_split(split: str) -> tuple[np.ndarray, np.ndarray]:
        X, _, labels, _ = arrays[split]
        indices = np.asarray(
            [index for index, label in enumerate(labels) if label in known_agents]
        )
        return X[indices], encoder.transform([labels[index] for index in indices])

    X_train, y_train = known_split("train")
    X_val, _ = known_split("val")
    X_test_all, test_rows, test_labels, _ = arrays["test"]
    known_indices = np.asarray(
        [
            index
            for index, label in enumerate(test_labels)
            if label in known_agent_set
        ]
    )
    unknown_indices = np.asarray(
        [
            index
            for index, label in enumerate(test_labels)
            if label in unknown_agents
        ]
    )
    X_known = X_test_all[known_indices]
    X_unknown = X_test_all[unknown_indices]
    y_known = encoder.transform([test_labels[index] for index in known_indices])

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
    val_probabilities = np.asarray(model.predict_proba(X_val))
    known_probabilities = np.asarray(model.predict_proba(X_known))
    unknown_probabilities = np.asarray(model.predict_proba(X_unknown))
    val_confidence = val_probabilities.max(axis=1)
    known_confidence = known_probabilities.max(axis=1)
    unknown_confidence = unknown_probabilities.max(axis=1)
    known_predicted = known_probabilities.argmax(axis=1)
    unknown_predicted = unknown_probabilities.argmax(axis=1)
    threshold = _confidence_threshold(
        val_confidence,
        float(cfg["open_set"]["known_validation_acceptance"]),
    )
    detection_truth = np.concatenate(
        [np.zeros(len(known_confidence)), np.ones(len(unknown_confidence))]
    )
    unknown_scores = 1.0 - np.concatenate(
        [known_confidence, unknown_confidence]
    )
    known_metrics = _classification_metrics(
        y_known, known_predicted, list(encoder.classes_)
    )
    unknown_count = int(holdout["unknown_count"])
    metrics = {
        "open_set_auroc": float(roc_auc_score(detection_truth, unknown_scores)),
        "open_set_average_precision": float(
            average_precision_score(detection_truth, unknown_scores)
        ),
        "oscr": _oscr(
            known_confidence,
            known_predicted == y_known,
            unknown_confidence,
        ),
        "known_macro_f1": known_metrics["macro_f1"],
        "known_accuracy": known_metrics["accuracy"],
        "known_acceptance": float(np.mean(known_confidence >= threshold)),
        "unknown_recall": float(np.mean(unknown_confidence < threshold)),
        "balanced_open_set_macro_f1": _balanced_open_set_macro_f1(
            y_known,
            known_predicted,
            known_confidence,
            unknown_predicted,
            unknown_confidence,
            threshold,
            len(known_agents),
            unknown_count,
        ),
    }
    per_unknown_agent = {}
    for agent in sorted(unknown_agents):
        positions = np.asarray(
            [
                position
                for position, test_index in enumerate(unknown_indices)
                if test_labels[test_index] == agent
            ]
        )
        per_unknown_agent[agent] = {
            "n_test": len(positions),
            "unknown_recall": float(
                np.mean(unknown_confidence[positions] < threshold)
            ),
            "mean_known_class_confidence": float(
                np.mean(unknown_confidence[positions])
            ),
        }
    result = {
        "schema_version": 1,
        "task": "leave_p_models_out_open_set_identification",
        "dataset": dataset,
        "unknown_count": unknown_count,
        "holdout_id": holdout["holdout_id"],
        "holdout_index": holdout["holdout_index"],
        "unknown_agents": sorted(unknown_agents),
        "known_agents": known_agents,
        "seed": seed,
        "feature_group": cfg.get("feature_group", "full"),
        "n_train_known": len(X_train),
        "n_validation_known": len(X_val),
        "n_test_known": len(X_known),
        "n_test_unknown": len(X_unknown),
        "known_validation_acceptance_target": cfg["open_set"][
            "known_validation_acceptance"
        ],
        "confidence_threshold": threshold,
        "fit_seconds": fit_seconds,
        "metrics": metrics,
        "known_classification": known_metrics,
        "per_unknown_agent": per_unknown_agent,
        "parameters": params,
        "feature_names": feature_names,
        "source_manifests": {
            split: str(_source_manifest(cfg, dataset, split))
            for split in ("train", "val", "test")
        },
        "source_manifest_sha256": {
            split: _source_hash(_source_manifest(cfg, dataset, split))
            for split in ("train", "val", "test")
        },
        "test_episode_ids": {
            "known": [test_rows[index]["episode_id"] for index in known_indices],
            "unknown": [test_rows[index]["episode_id"] for index in unknown_indices],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[OPEN SET] {dataset} p={unknown_count} "
        f"fold={holdout['holdout_index'] + 1} seed={seed} "
        f"AUROC={metrics['open_set_auroc']:.3f} "
        f"OSCR={metrics['oscr']:.3f}"
    )
    return output


def _bootstrap_mean_ci(
    values: list[float], samples: int, confidence: float, seed: int
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(data), size=(samples, len(data)))
    means = data[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def summarize(cfg: dict[str, Any]) -> Path:
    root = cfg["experiment"]["artifact_root"]
    raw_rows = []
    per_agent_rows = []
    per_known_agent_rows = []
    for path in sorted((root / "fits").rglob("results.json")):
        result = json.loads(path.read_text())
        row = {
            "dataset": result["dataset"],
            "unknown_count": result["unknown_count"],
            "holdout_id": result["holdout_id"],
            "holdout_index": result["holdout_index"],
            "unknown_agents": json.dumps(result["unknown_agents"]),
            "seed": result["seed"],
            "confidence_threshold": result["confidence_threshold"],
            "fit_seconds": result["fit_seconds"],
            **result["metrics"],
            "results_path": str(path),
        }
        raw_rows.append(row)
        for agent, metrics in result["per_unknown_agent"].items():
            per_agent_rows.append(
                {
                    "dataset": result["dataset"],
                    "unknown_count": result["unknown_count"],
                    "holdout_id": result["holdout_id"],
                    "seed": result["seed"],
                    "unknown_agent": agent,
                    **metrics,
                    "results_path": str(path),
                }
            )
        known_report = result.get("known_classification", {}).get(
            "classification_report", {}
        )
        for agent in result["known_agents"]:
            metrics = known_report.get(agent, {})
            per_known_agent_rows.append(
                {
                    "dataset": result["dataset"],
                    "unknown_count": result["unknown_count"],
                    "holdout_id": result["holdout_id"],
                    "seed": result["seed"],
                    "known_agent": agent,
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1-score"),
                    "support": metrics.get("support"),
                    "results_path": str(path),
                }
            )
    summary_root = root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    results_csv = summary_root / "results.csv"
    fields = list(raw_rows[0]) if raw_rows else [
        "dataset", "unknown_count", "holdout_id", "holdout_index",
        "unknown_agents", "seed", "confidence_threshold", "fit_seconds",
        *_METRIC_FIELDS, "results_path",
    ]
    with results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_rows)
    per_agent_csv = summary_root / "per_unknown_agent.csv"
    agent_fields = list(per_agent_rows[0]) if per_agent_rows else [
        "dataset", "unknown_count", "holdout_id", "seed", "unknown_agent",
        "n_test", "unknown_recall", "mean_known_class_confidence",
        "results_path",
    ]
    with per_agent_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=agent_fields)
        writer.writeheader()
        writer.writerows(per_agent_rows)
    per_known_agent_csv = summary_root / "per_known_agent.csv"
    known_agent_fields = list(per_known_agent_rows[0]) if per_known_agent_rows else [
        "dataset", "unknown_count", "holdout_id", "seed", "known_agent",
        "precision", "recall", "f1", "support", "results_path",
    ]
    with per_known_agent_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=known_agent_fields)
        writer.writeheader()
        writer.writerows(per_known_agent_rows)

    folds: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        folds[
            (row["dataset"], int(row["unknown_count"]), row["holdout_id"])
        ].append(row)
    fold_rows = []
    for (dataset, unknown_count, holdout_id), repetitions in sorted(folds.items()):
        fold_row = {
            "dataset": dataset,
            "unknown_count": unknown_count,
            "holdout_id": holdout_id,
            "unknown_agents": repetitions[0]["unknown_agents"],
            "n_seeds": len(repetitions),
        }
        for metric in _METRIC_FIELDS:
            values = [float(row[metric]) for row in repetitions]
            fold_row[f"{metric}_mean"] = float(np.mean(values))
            fold_row[f"{metric}_seed_sd"] = _sample_std(values)
        fold_rows.append(fold_row)
    fold_csv = summary_root / "holdout_set_means.csv"
    fold_fields = list(fold_rows[0]) if fold_rows else []
    with fold_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fold_fields)
        if fold_fields:
            writer.writeheader()
            writer.writerows(fold_rows)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[(row["dataset"], int(row["unknown_count"]))].append(row)
    aggregate = []
    bootstrap_samples = int(cfg["open_set"]["holdout_bootstrap_samples"])
    confidence = float(cfg["open_set"]["holdout_bootstrap_confidence"])
    for (dataset, unknown_count), holdout_rows in sorted(grouped.items()):
        item = {
            "dataset": dataset,
            "unknown_count": unknown_count,
            "n_holdout_sets": len(holdout_rows),
            "n_classifier_seeds": int(
                min(row["n_seeds"] for row in holdout_rows)
            ),
        }
        for metric in _METRIC_FIELDS:
            values = [float(row[f"{metric}_mean"]) for row in holdout_rows]
            digest = hashlib.sha256(
                f"{dataset}|{unknown_count}|{metric}|bootstrap".encode()
            ).digest()
            low, high = _bootstrap_mean_ci(
                values,
                bootstrap_samples,
                confidence,
                int.from_bytes(digest[:8], "big"),
            )
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_holdout_sd"] = _sample_std(values)
            item[f"{metric}_ci_low"] = low
            item[f"{metric}_ci_high"] = high
        aggregate.append(item)
    curve_csv = summary_root / "unknown_count_curve.csv"
    aggregate_fields = list(aggregate[0]) if aggregate else []
    with curve_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        if aggregate_fields:
            writer.writeheader()
            writer.writerows(aggregate)
    plot_results(cfg)
    write_report(cfg)
    print(f"Summarized {len(raw_rows)} open-set fits → {curve_csv}")
    return curve_csv


def plot_results(cfg: dict[str, Any]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = cfg["experiment"]["artifact_root"]
    curve = read_csv(root / "summaries" / "unknown_count_curve.csv")
    folds = read_csv(root / "summaries" / "holdout_set_means.csv")
    if not curve:
        return []
    display = {
        "2wikimultihop": "2WikiMultiHopQA",
        "webshop": "WebShop",
    }
    colors = {"2wikimultihop": "#2878B5", "webshop": "#D55E00"}
    panels = [
        ("open_set_auroc", "Known vs unknown AUROC", 0.5),
        ("oscr", "Open-set classification rate (OSCR)", 0.0),
        ("known_macro_f1", "Known-model macro-F1", 0.0),
        (
            "unknown_recall",
            "Unknown recall at 95% known-val acceptance",
            0.0,
        ),
    ]
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.5), sharex=True)
    for axis, (metric, title, lower) in zip(axes.flat, panels, strict=True):
        for dataset in cfg["datasets"]:
            dataset_curve = sorted(
                (row for row in curve if row["dataset"] == dataset),
                key=lambda row: int(row["unknown_count"]),
            )
            x = np.asarray([int(row["unknown_count"]) for row in dataset_curve])
            mean = np.asarray(
                [float(row[f"{metric}_mean"]) for row in dataset_curve]
            )
            low = np.asarray(
                [float(row[f"{metric}_ci_low"]) for row in dataset_curve]
            )
            high = np.asarray(
                [float(row[f"{metric}_ci_high"]) for row in dataset_curve]
            )
            dataset_folds = [
                row for row in folds if row["dataset"] == dataset
            ]
            for unknown_count in x:
                values = [
                    float(row[f"{metric}_mean"])
                    for row in dataset_folds
                    if int(row["unknown_count"]) == unknown_count
                ]
                jitter_rng = np.random.default_rng(
                    int.from_bytes(
                        hashlib.sha256(
                            f"{dataset}|{metric}|{unknown_count}".encode()
                        ).digest()[:8],
                        "big",
                    )
                )
                jitter = jitter_rng.uniform(-0.055, 0.055, size=len(values))
                axis.scatter(
                    unknown_count + jitter,
                    values,
                    color=colors[dataset],
                    alpha=0.08,
                    s=8,
                    linewidths=0,
                )
            axis.fill_between(
                x, low, high, color=colors[dataset], alpha=0.16, linewidth=0
            )
            axis.plot(
                x,
                mean,
                color=colors[dataset],
                marker="o",
                linewidth=2.0,
                label=display.get(dataset, dataset),
            )
        if metric == "open_set_auroc":
            axis.axhline(
                0.5,
                color="#555555",
                linestyle="--",
                linewidth=1.1,
                label="Random ranking",
            )
        axis.set_title(title)
        axis.set_ylim(lower, 1.02)
        axis.set_xticks(cfg["open_set"]["unknown_counts"])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    axes[1, 0].set_xlabel("Number of models absent from training")
    axes[1, 1].set_xlabel("Number of models absent from training")
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    png = root / "leave_p_models_out_open_set.png"
    pdf = root / "leave_p_models_out_open_set.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _metric_cell(row: dict[str, str], metric: str) -> str:
    return (
        f"{number(row[f'{metric}_mean'], 3)} "
        f"[{number(row[f'{metric}_ci_low'], 3)}, "
        f"{number(row[f'{metric}_ci_high'], 3)}]"
    )


def write_report(cfg: dict[str, Any]) -> Path:
    root = cfg["experiment"]["artifact_root"]
    curve = read_csv(root / "summaries" / "unknown_count_curve.csv")
    report = root / "REPORT.md"
    table = markdown_table(
        [
            "Dataset",
            "Unknown models",
            "Holdout sets",
            "Known/unknown AUROC",
            "OSCR",
            "Known macro-F1",
            "Unknown recall",
        ],
        [
            [
                row["dataset"],
                row["unknown_count"],
                row["n_holdout_sets"],
                _metric_cell(row, "open_set_auroc"),
                _metric_cell(row, "oscr"),
                _metric_cell(row, "known_macro_f1"),
                _metric_cell(row, "unknown_recall"),
            ]
            for row in curve
        ],
    )
    expected_holdouts = Counter(
        row["unknown_count"] for row in _load_holdouts(cfg)
    )
    completed = Counter(
        (row["dataset"], int(row["unknown_count"]))
        for row in read_csv(root / "summaries" / "holdout_set_means.csv")
    )
    completion_lines = []
    for dataset in cfg["datasets"]:
        for unknown_count in cfg["open_set"]["unknown_counts"]:
            completion_lines.append(
                f"- `{dataset}`, p={unknown_count}: "
                f"{completed[(dataset, int(unknown_count))]}/"
                f"{expected_holdouts[int(unknown_count)]} holdout sets summarized"
            )
    lines = [
        "# Leave-multiple-models-out open-set identification",
        "",
        "## Question and protocol",
        "",
        "This experiment asks whether a MidScene model-identity classifier can "
        "continue identifying known models while rejecting several models that "
        "are entirely absent from training. For each outer holdout set, all "
        "traces from p models are excluded from both training and validation. "
        "XGBoost is trained on the remaining 14-p models and evaluated on the "
        "unchanged test split containing both known and held-out models.",
        "",
        "The rejection score is one minus the maximum known-class probability. "
        "The decision threshold is calibrated using known validation traces "
        f"only, targeting {100 * float(cfg['open_set']['known_validation_acceptance']):.0f}% "
        "known validation acceptance. No held-out model affects model fitting, "
        "feature selection, or threshold calibration.",
        "",
        "The `UNKNOWN` class combines p unseen models. In the combined open-set "
        "macro-F1, each unknown trace receives weight 1/p so that UNKNOWN has "
        "the total weight of one known class. The headline table reports "
        "holdout-set means after first averaging the five classifier seeds; "
        "brackets are 95% bootstrap intervals over holdout sets, not test-trace "
        "confidence intervals.",
        "",
        "## Completion",
        "",
        *completion_lines,
        "",
        "## Headline results",
        "",
        table if table else "_The grid has not been run yet._",
        "",
        "OSCR is the area under the correct-known-classification-rate versus "
        "unknown false-positive-rate curve. Unknown recall uses the frozen "
        "known-validation threshold; AUROC is threshold-free.",
        "",
        "## Visualization",
        "",
        "![Leave-p-models-out open-set results]"
        "(leave_p_models_out_open_set.png)",
        "",
        "Faint points are individual holdout-set means after averaging "
        "classifier seeds. Thick lines are means across holdout sets and bands "
        "are 95% holdout-set bootstrap intervals.",
        "",
        "## Artifact map",
        "",
        f"- Frozen holdout design: {relative_link(_holdouts_path(cfg), report)}",
        f"- Every classifier fit: {relative_link(root / 'summaries' / 'results.csv', report)}",
        f"- Seed-averaged holdout sets: {relative_link(root / 'summaries' / 'holdout_set_means.csv', report)}",
        f"- Unknown-count curve: {relative_link(root / 'summaries' / 'unknown_count_curve.csv', report)}",
        f"- Per-unknown-model diagnostics: {relative_link(root / 'summaries' / 'per_unknown_agent.csv', report)}",
        f"- Per-known-model identification: {relative_link(root / 'summaries' / 'per_known_agent.csv', report)}",
        "- Figures: [PNG](leave_p_models_out_open_set.png), "
        "[PDF](leave_p_models_out_open_set.pdf).",
        "",
        "Raw traces and source manifests are read-only inputs and are never "
        "modified by this pipeline.",
        "",
    ]
    report.write_text("\n".join(lines))
    print(f"Wrote human-readable report → {report}")
    return report


def run_grid(
    cfg: dict[str, Any],
    *,
    datasets: list[str] | None,
    unknown_counts: list[int] | None,
    seeds: list[int] | None,
    limit_holdouts: int | None,
    xgb_device: str | None,
    force: bool,
) -> None:
    selected_datasets = datasets or list(cfg["datasets"])
    selected_counts = set(
        unknown_counts or [int(value) for value in cfg["open_set"]["unknown_counts"]]
    )
    selected_seeds = seeds or [
        int(value) for value in cfg["open_set"]["classifier_seeds"]
    ]
    holdouts = [
        row
        for row in _load_holdouts(cfg)
        if int(row["unknown_count"]) in selected_counts
    ]
    if limit_holdouts is not None:
        limited = []
        by_count: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in holdouts:
            by_count[int(row["unknown_count"])].append(row)
        for unknown_count in sorted(by_count):
            limited.extend(by_count[unknown_count][:limit_holdouts])
        holdouts = limited
    total = len(selected_datasets) * len(holdouts) * len(selected_seeds)
    print(
        f"Open-set grid: {total} fits across {len(selected_datasets)} datasets, "
        f"{len(holdouts)} holdout sets, and {len(selected_seeds)} seeds"
    )
    for dataset in selected_datasets:
        if dataset not in cfg["datasets"]:
            raise ValueError(f"unknown dataset: {dataset}")
        for holdout in holdouts:
            for seed in selected_seeds:
                run_one(
                    cfg,
                    dataset,
                    holdout,
                    seed,
                    xgb_device=xgb_device,
                    force=force,
                )
    summarize(cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent
        / "configs"
        / "midscene_14model_leave_p_out.yaml",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("prepare")
    run = sub.add_parser("run-grid")
    run.add_argument("--datasets", nargs="+")
    run.add_argument("--unknown-counts", nargs="+", type=int)
    run.add_argument("--seeds", nargs="+", type=int)
    run.add_argument(
        "--limit-holdouts",
        type=int,
        help="run only the first N frozen holdout sets per unknown count",
    )
    run.add_argument("--xgb-device", choices=["cpu", "cuda"])
    run.add_argument("--force", action="store_true")
    sub.add_parser("summarize")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "audit":
        audit(cfg)
    elif args.command == "prepare":
        prepare(cfg)
    elif args.command == "run-grid":
        run_grid(
            cfg,
            datasets=args.datasets,
            unknown_counts=args.unknown_counts,
            seeds=args.seeds,
            limit_holdouts=args.limit_holdouts,
            xgb_device=args.xgb_device,
            force=args.force,
        )
    elif args.command == "summarize":
        summarize(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
