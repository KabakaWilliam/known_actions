#!/usr/bin/env python3
"""Compute per-held-out-model open-set AUROC bootstrap intervals.

The original leave-one-model-out results contain only one AUROC value per
classifier.  Bootstrap intervals over evaluation traces require the underlying
known/unknown confidence scores, so this runner refits the selected classifier
with multiple seeds while reusing the hyperparameters already selected by the
original training CV.  It never overwrites ``results.json``.

By default, ten unique classifier seeds are generated with system randomness.
The same seeds are used for every held-out model in the batch and are recorded
in both the per-run and aggregate JSON outputs.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder

from closed_set_stats import (
    DEFAULT_CLASSIFIER_SEED_COUNT,
    generate_classifier_seeds,
    validate_classifier_seeds,
)
from open_set_stats import summarize_open_set_auroc
from trace_analyzer import load_dataset

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - exercised only in a missing dependency env
    raise SystemExit(
        "xgboost is required to compute open-set multi-seed statistics"
    ) from exc


DATASETS: dict[str, dict[str, Any]] = {
    "wiki": {
        "dataset": "2wikimultihop",
        "tag": "2wikimultihop_open_set",
        "resplit": False,
        "resplit_n_per_agent": None,
    },
    "frames": {
        "dataset": "frames",
        "tag": "frames_open_set",
        "resplit": True,
        "resplit_n_per_agent": 300,
    },
    "webshop": {
        "dataset": "webshop",
        "tag": "webshop_open_set",
        "resplit": False,
        "resplit_n_per_agent": None,
    },
    "deepshop": {
        "dataset": "deepshop",
        "tag": "deepshop_open_set",
        "resplit": True,
        "resplit_n_per_agent": None,
    },
}

BOOTSTRAP_STRATA = [
    "known_test_trace",
    "unknown_held_out_model_trace",
]
SCORE_DEFINITION = "max_predict_proba; higher_means_known"
CI_METHOD = (
    "paired_stratified_percentile_bootstrap_over_"
    "known_test_and_unknown_held_out_model_traces"
)


def _read_json(path: Path) -> dict:
    try:
        with path.open() as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict) -> None:
    """Write JSON through a same-directory temporary file and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_batch_seeds(
    manifest_path: Path,
    *,
    requested_seeds: list[int] | None,
    requested_seed_count: int,
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
    resume: bool,
) -> tuple[list[int], str]:
    """Create or recover one seed/configuration manifest for a resumable batch."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_name(f".{manifest_path.name}.lock")
    requested_bootstrap = {
        "replicates": bootstrap_replicates,
        "confidence_level": confidence_level,
        "seed": bootstrap_seed,
    }
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if resume and manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if manifest.get("schema_version") != 1:
                raise ValueError(
                    f"{manifest_path}: unsupported batch manifest schema"
                )
            try:
                manifest_seeds = validate_classifier_seeds(
                    manifest.get("classifier_seeds") or []
                )
            except ValueError as exc:
                raise ValueError(
                    f"{manifest_path}: invalid batch classifier seeds: {exc}"
                ) from exc
            if manifest.get("bootstrap") != requested_bootstrap:
                raise ValueError(
                    f"{manifest_path}: bootstrap settings differ from this "
                    "invocation; use --no-resume to begin a new batch"
                )
            if requested_seeds is not None:
                explicit = validate_classifier_seeds(requested_seeds)
                if explicit != manifest_seeds:
                    raise ValueError(
                        f"{manifest_path}: explicit classifier seeds differ "
                        "from the resumable batch; use --no-resume to replace it"
                    )
            elif len(manifest_seeds) != requested_seed_count:
                raise ValueError(
                    f"{manifest_path}: existing batch has "
                    f"{len(manifest_seeds)} seeds, but "
                    f"--classifier-seed-count is {requested_seed_count}; use "
                    "--no-resume to begin a new batch"
                )
            seed_source = manifest.get("classifier_seed_source")
            if seed_source not in {"explicit", "generated_system_random"}:
                raise ValueError(
                    f"{manifest_path}: invalid classifier_seed_source"
                )
            return manifest_seeds, seed_source

        if requested_seeds is None:
            seeds = generate_classifier_seeds(requested_seed_count)
            seed_source = "generated_system_random"
        else:
            seeds = validate_classifier_seeds(requested_seeds)
            seed_source = "explicit"
        manifest = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classifier_seeds": seeds,
            "n_classifier_seeds": len(seeds),
            "classifier_seed_source": seed_source,
            "bootstrap": requested_bootstrap,
            "score_definition": SCORE_DEFINITION,
        }
        _atomic_write_json(manifest_path, manifest)
        return seeds, seed_source


def _feature_matrix(features: list[dict], names: list[str]) -> np.ndarray:
    if not features:
        return np.empty((0, len(names)), dtype=float)
    return np.asarray(
        [[episode[name] for name in names] for episode in features],
        dtype=float,
    )


def _reference_leaf_dirs(
    traces_dir: Path,
    dataset_keys: list[str],
    agent_filter: set[str] | None,
) -> list[tuple[str, Path]]:
    leaves: list[tuple[str, Path]] = []
    classifiers_dir = traces_dir / "classifiers"
    for dataset_key in dataset_keys:
        tag = DATASETS[dataset_key]["tag"]
        tag_dir = classifiers_dir / tag
        if not tag_dir.is_dir():
            raise ValueError(f"missing leave-one-out results directory: {tag_dir}")
        dataset_leaves = []
        for path in sorted(tag_dir.glob("open_set_loo_*")):
            if not path.is_dir():
                continue
            agent = path.name.removeprefix("open_set_loo_")
            if agent_filter is not None and agent not in agent_filter:
                continue
            if not (path / "results.json").is_file():
                raise ValueError(f"missing reference results.json in {path}")
            dataset_leaves.append((dataset_key, path))
        if not dataset_leaves:
            raise ValueError(f"no matching leave-one-out runs found in {tag_dir}")
        leaves.extend(dataset_leaves)
    return leaves


def _load_leaf_if_compatible(
    path: Path,
    *,
    expected_tag: str | None = None,
    held_out_model: str | None = None,
    source_results_sha256: str | None = None,
    best_params: dict | None = None,
    n_known: int | None = None,
    n_unknown: int | None = None,
    seeds: list[int],
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> dict | None:
    if not path.is_file():
        return None
    value = _read_json(path)
    bootstrap = value.get("bootstrap") or {}
    model = ((value.get("models") or {}).get("XGBoost") or {})
    summary = model.get("auroc") or {}
    aggregation = value.get("aggregation") or {}
    confidence_interval = summary.get("confidence_interval") or {}
    per_seed = summary.get("per_seed")
    per_seed_ids = (
        [row.get("seed") for row in per_seed]
        if (
            isinstance(per_seed, list)
            and all(isinstance(row, dict) for row in per_seed)
        )
        else None
    )
    if (
        value.get("schema_version") == 1
        and (
            expected_tag is None
            or value.get("tag") == expected_tag
        )
        and (
            held_out_model is None
            or value.get("held_out_model") == held_out_model
        )
        and (
            source_results_sha256 is None
            or value.get("source_results_sha256") == source_results_sha256
        )
        and (
            best_params is None
            or model.get("best_params") == best_params
        )
        and (
            n_known is None
            or (
                value.get("n_known_traces") == n_known
                and summary.get("n_known") == n_known
            )
        )
        and (
            n_unknown is None
            or (
                value.get("n_unknown_traces") == n_unknown
                and summary.get("n_unknown") == n_unknown
            )
        )
        and value.get("classifier_seeds") == seeds
        and value.get("n_classifier_seeds") == len(seeds)
        and bootstrap.get("replicates") == bootstrap_replicates
        and bootstrap.get("confidence_level") == confidence_level
        and bootstrap.get("seed") == bootstrap_seed
        and bootstrap.get("unit") == "evaluation_trace"
        and bootstrap.get("strata") == BOOTSTRAP_STRATA
        and bootstrap.get("sampling")
        == "independent_nonparametric_with_replacement_within_stratum"
        and bootstrap.get("paired_across_classifier_seeds") is True
        and bootstrap.get("interval") == "percentile"
        and aggregation.get("metric") == "auroc"
        and aggregation.get("across_classifier_seeds") == "arithmetic_mean"
        and aggregation.get("score_definition") == SCORE_DEFINITION
        and aggregation.get("positive_class") == "known"
        and confidence_interval.get("method") == CI_METHOD
        and per_seed_ids == seeds
    ):
        return value
    return None


def compute_leaf(
    traces_dir: Path,
    dataset_key: str,
    leaf_dir: Path,
    *,
    classifier_seeds: list[int],
    classifier_seed_source: str,
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
    device: str,
    n_jobs: int,
    resume: bool,
) -> dict:
    """Compute and persist one held-out model's XGBoost AUROC summary."""
    stats_path = leaf_dir / "open_set_auroc.json"
    reference_path = leaf_dir / "results.json"
    reference = _read_json(reference_path)
    reference_sha256 = _sha256(reference_path)
    held_out = leaf_dir.name.removeprefix("open_set_loo_")
    expected_tag = f"{DATASETS[dataset_key]['tag']}/{leaf_dir.name}"
    if reference.get("tag") != expected_tag:
        raise ValueError(
            f"{reference_path}: tag is {reference.get('tag')!r}, "
            f"expected {expected_tag!r}"
        )

    known_agents = reference.get("class_names")
    if (
        not isinstance(known_agents, list)
        or not known_agents
        or any(not isinstance(agent, str) for agent in known_agents)
    ):
        raise ValueError(f"{reference_path}: invalid or missing class_names")
    if held_out in known_agents:
        raise ValueError(
            f"{reference_path}: held-out model {held_out!r} appears in class_names"
        )

    model_reference = (reference.get("models") or {}).get("XGBoost") or {}
    best_params = model_reference.get("best_params")
    if not isinstance(best_params, dict) or not best_params:
        raise ValueError(f"{reference_path}: missing XGBoost best_params")

    expected_known = (
        ((reference.get("open_set") or {}).get("XGBoost") or {}).get("n_known")
    )
    expected_unknown = (
        ((reference.get("open_set") or {}).get("XGBoost") or {}).get("n_unknown")
    )
    if expected_known is None or expected_unknown is None:
        raise ValueError(
            f"{reference_path}: missing XGBoost open-set trace counts"
        )
    expected_known = int(expected_known)
    expected_unknown = int(expected_unknown)

    if resume:
        cached = _load_leaf_if_compatible(
            stats_path,
            expected_tag=expected_tag,
            held_out_model=held_out,
            source_results_sha256=reference_sha256,
            best_params=best_params,
            n_known=expected_known,
            n_unknown=expected_unknown,
            seeds=classifier_seeds,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed,
        )
        if cached is not None:
            print(f"[resume] {dataset_key}/{leaf_dir.name}")
            return cached

    config = DATASETS[dataset_key]
    dataset_name = str(config["dataset"])
    splits, dataset_names = load_dataset(
        traces_dir,
        train_datasets=[dataset_name],
        agents=known_agents,
        open_set_agents=[held_out],
        resplit_datasets=[dataset_name] if config["resplit"] else None,
        resplit_n_per_agent=config["resplit_n_per_agent"],
        label_by="agent",
    )
    feat_train, _, lbl_train, _ = splits["train"]
    feat_val, _, lbl_val, _ = splits["val"]
    feat_test, _, lbl_test, _ = splits["test"]
    feat_unknown, _, lbl_unknown, _ = splits["open_set"]
    if not feat_train or not feat_test or not feat_unknown:
        raise ValueError(
            f"{dataset_key}/{held_out}: empty train, known-test, or unknown split"
        )
    if set(lbl_unknown) != {held_out}:
        raise ValueError(
            f"{dataset_key}/{held_out}: open-set split contains "
            f"{sorted(set(lbl_unknown))!r}"
        )

    if len(feat_test) != expected_known:
        raise ValueError(
            f"{dataset_key}/{held_out}: reconstructed {len(feat_test)} known "
            f"test traces, reference has {expected_known}"
        )
    if len(feat_unknown) != expected_unknown:
        raise ValueError(
            f"{dataset_key}/{held_out}: reconstructed {len(feat_unknown)} unknown "
            f"traces, reference has {expected_unknown}"
        )
    expected_episodes = reference.get("n_episodes") or {}
    reconstructed_counts = {
        "train": len(feat_train),
        "val": len(feat_val),
        "test": len(feat_test),
    }
    for split_name, reconstructed_count in reconstructed_counts.items():
        expected_count = expected_episodes.get(split_name)
        if expected_count is None or reconstructed_count != int(expected_count):
            raise ValueError(
                f"{dataset_key}/{held_out}: reconstructed {split_name} count "
                f"{reconstructed_count}, reference has {expected_count}"
            )
        dataset_key_name = f"{split_name}_datasets"
        reconstructed_names = sorted(dataset_names[split_name])
        expected_names = reference.get(dataset_key_name)
        if (
            not isinstance(expected_names, list)
            or reconstructed_names != sorted(expected_names)
        ):
            raise ValueError(
                f"{dataset_key}/{held_out}: reconstructed {dataset_key_name} "
                f"{reconstructed_names!r}, reference has {expected_names!r}"
            )

    encoder = LabelEncoder()
    encoder.fit(lbl_train + lbl_val + lbl_test)
    if list(encoder.classes_) != sorted(known_agents):
        raise ValueError(
            f"{dataset_key}/{held_out}: reconstructed labels do not match "
            "the reference class universe"
        )
    y_train = encoder.transform(lbl_train)
    feature_names = list(feat_train[0])
    X_train = _feature_matrix(feat_train, feature_names)
    X_known = _feature_matrix(feat_test, feature_names)
    X_unknown = _feature_matrix(feat_unknown, feature_names)

    known_scores_by_seed: dict[int, np.ndarray] = {}
    unknown_scores_by_seed: dict[int, np.ndarray] = {}
    print(
        f"[fit] {dataset_key}/{held_out}: train={len(X_train)}, "
        f"known={len(X_known)}, unknown={len(X_unknown)}, "
        f"seeds={len(classifier_seeds)}"
    )
    for index, seed in enumerate(classifier_seeds, start=1):
        model = XGBClassifier(
            tree_method="hist",
            device=device,
            eval_metric="mlogloss",
            random_state=seed,
            verbosity=0,
            n_jobs=n_jobs,
            **best_params,
        )
        model.fit(X_train, y_train)
        known_scores_by_seed[seed] = np.asarray(
            model.predict_proba(X_known).max(axis=1),
            dtype=float,
        )
        unknown_scores_by_seed[seed] = np.asarray(
            model.predict_proba(X_unknown).max(axis=1),
            dtype=float,
        )
        print(
            f"  seed {index:>2}/{len(classifier_seeds)}: {seed}",
            flush=True,
        )
        del model
        gc.collect()

    summary = summarize_open_set_auroc(
        known_scores_by_seed,
        unknown_scores_by_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
    )
    run_timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "schema_version": 1,
        "timestamp": run_timestamp,
        "tag": expected_tag,
        "source_results_file": "results.json",
        "source_results_sha256": reference_sha256,
        "train_datasets": sorted(dataset_names["train"]),
        "val_datasets": sorted(dataset_names["val"]),
        "test_datasets": sorted(dataset_names["test"]),
        "open_set_datasets": sorted(dataset_names["open_set"]),
        "held_out_model": held_out,
        "open_set_agents": [held_out],
        "n_known_traces": len(X_known),
        "n_unknown_traces": len(X_unknown),
        "classifier_seeds": classifier_seeds,
        "n_classifier_seeds": len(classifier_seeds),
        "classifier_seed_source": classifier_seed_source,
        "bootstrap": {
            "unit": "evaluation_trace",
            "strata": BOOTSTRAP_STRATA,
            "sampling": (
                "independent_nonparametric_with_replacement_within_stratum"
            ),
            "paired_across_classifier_seeds": True,
            "replicates": bootstrap_replicates,
            "confidence_level": confidence_level,
            "seed": bootstrap_seed,
            "interval": "percentile",
        },
        "aggregation": {
            "metric": "auroc",
            "across_classifier_seeds": "arithmetic_mean",
            "score_definition": SCORE_DEFINITION,
            "positive_class": "known",
            "hyperparameters": (
                "reused_from_results_json_then_fixed_across_seeds"
            ),
        },
        "models": {
            "XGBoost": {
                "best_params": best_params,
                "auroc": summary,
            }
        },
    }
    _atomic_write_json(stats_path, result)
    interval = summary["confidence_interval"]
    print(
        f"[saved] {stats_path}: AUROC={summary['estimate']:.4f}, "
        f"{confidence_level:.0%} CI "
        f"[{interval['lower']:.4f}, {interval['upper']:.4f}]"
    )
    return result


def build_aggregate(
    traces_dir: Path,
    dataset_keys: list[str],
    agent_filter: set[str] | None = None,
) -> dict:
    """Build an aggregate JSON object from compatible per-leaf output files."""
    datasets = {}
    common_seeds: list[int] | None = None
    common_bootstrap: dict | None = None
    for dataset_key in dataset_keys:
        tag = DATASETS[dataset_key]["tag"]
        held_out_models = {}
        tag_dir = traces_dir / "classifiers" / tag
        for leaf_dir in sorted(tag_dir.glob("open_set_loo_*")):
            if not leaf_dir.is_dir():
                continue
            agent = leaf_dir.name.removeprefix("open_set_loo_")
            if agent_filter is not None and agent not in agent_filter:
                continue
            stats_path = leaf_dir / "open_set_auroc.json"
            leaf = _read_json(stats_path)
            if leaf.get("schema_version") != 1:
                raise ValueError(f"{stats_path}: unsupported schema_version")
            if leaf.get("tag") != f"{tag}/{leaf_dir.name}":
                raise ValueError(f"{stats_path}: tag does not match its directory")
            if leaf.get("held_out_model") != agent:
                raise ValueError(
                    f"{stats_path}: held_out_model does not match its directory"
                )
            reference_path = leaf_dir / "results.json"
            if leaf.get("source_results_sha256") != _sha256(reference_path):
                raise ValueError(
                    f"{stats_path}: source results digest is missing or stale"
                )

            seeds_raw = leaf.get("classifier_seeds")
            if not isinstance(seeds_raw, list):
                raise ValueError(
                    f"{stats_path}: classifier_seeds must be a list"
                )
            try:
                seeds = validate_classifier_seeds(seeds_raw)
            except ValueError as exc:
                raise ValueError(
                    f"{stats_path}: invalid classifier_seeds: {exc}"
                ) from exc
            if leaf.get("n_classifier_seeds") != len(seeds):
                raise ValueError(
                    f"{stats_path}: n_classifier_seeds does not match "
                    "classifier_seeds"
                )
            if common_seeds is None:
                common_seeds = seeds
            elif seeds != common_seeds:
                raise ValueError(
                    f"{stats_path}: classifier seeds differ from other leaves"
                )

            bootstrap = leaf.get("bootstrap")
            expected_bootstrap_keys = {
                "unit",
                "strata",
                "sampling",
                "paired_across_classifier_seeds",
                "replicates",
                "confidence_level",
                "seed",
                "interval",
            }
            if (
                not isinstance(bootstrap, dict)
                or set(bootstrap) != expected_bootstrap_keys
                or bootstrap.get("unit") != "evaluation_trace"
                or bootstrap.get("strata") != BOOTSTRAP_STRATA
                or bootstrap.get("sampling")
                != "independent_nonparametric_with_replacement_within_stratum"
                or bootstrap.get("paired_across_classifier_seeds") is not True
                or isinstance(bootstrap.get("replicates"), bool)
                or not isinstance(bootstrap.get("replicates"), int)
                or bootstrap["replicates"] < 1
                or isinstance(bootstrap.get("confidence_level"), bool)
                or not isinstance(
                    bootstrap.get("confidence_level"),
                    (int, float),
                )
                or not 0.0 < float(bootstrap["confidence_level"]) < 1.0
                or isinstance(bootstrap.get("seed"), bool)
                or not isinstance(bootstrap.get("seed"), int)
                or bootstrap["seed"] < 0
                or bootstrap.get("interval") != "percentile"
            ):
                raise ValueError(f"{stats_path}: invalid bootstrap metadata")
            if common_bootstrap is None:
                common_bootstrap = bootstrap
            elif bootstrap != common_bootstrap:
                raise ValueError(
                    f"{stats_path}: bootstrap configuration differs from "
                    "other leaves"
                )

            aggregation = leaf.get("aggregation") or {}
            if (
                aggregation.get("metric") != "auroc"
                or aggregation.get("across_classifier_seeds")
                != "arithmetic_mean"
                or aggregation.get("score_definition") != SCORE_DEFINITION
                or aggregation.get("positive_class") != "known"
            ):
                raise ValueError(f"{stats_path}: invalid aggregation metadata")
            model = ((leaf.get("models") or {}).get("XGBoost") or {})
            summary = model.get("auroc") or {}
            interval = summary.get("confidence_interval") or {}
            if interval.get("method") != CI_METHOD:
                raise ValueError(
                    f"{stats_path}: unsupported AUROC confidence interval method"
                )
            per_seed = summary.get("per_seed")
            if (
                not isinstance(per_seed, list)
                or len(per_seed) != len(seeds)
                or any(not isinstance(row, dict) for row in per_seed)
                or [row.get("seed") for row in per_seed] != seeds
            ):
                raise ValueError(
                    f"{stats_path}: AUROC per_seed rows do not match "
                    "classifier_seeds"
                )
            if (
                summary.get("n_known") != leaf.get("n_known_traces")
                or summary.get("n_unknown") != leaf.get("n_unknown_traces")
            ):
                raise ValueError(
                    f"{stats_path}: summary trace counts do not match leaf "
                    "metadata"
                )
            held_out_models[agent] = leaf
        if not held_out_models:
            raise ValueError(f"no open_set_auroc.json files found in {tag_dir}")
        datasets[dataset_key] = {
            "tag": tag,
            "dataset": DATASETS[dataset_key]["dataset"],
            "held_out_models": held_out_models,
        }
    if common_seeds is None or common_bootstrap is None:
        raise ValueError("no compatible open-set statistics were found")
    return {
        "schema_version": 1,
        "description": (
            "Per-held-out-model open-set AUROC across classifier seeds, with "
            "paired stratified bootstrap confidence intervals over known test "
            "and unknown held-out-model traces."
        ),
        "classifier_seeds": common_seeds,
        "classifier_seed_count": len(common_seeds),
        "default_classifier_seed_count": len(common_seeds),
        "bootstrap": common_bootstrap,
        "score_definition": SCORE_DEFINITION,
        "datasets": datasets,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("./traces"),
        help="Root trace directory containing classifiers/<open-set tag>.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=list(DATASETS),
        help="Dataset keys to process (default: all four).",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="Optional held-out model IDs to process.",
    )
    parser.add_argument(
        "--classifier-seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help=(
            "At least five unique classifier seeds. If omitted, system-random "
            "seeds are generated once for the entire batch."
        ),
    )
    parser.add_argument(
        "--classifier-seed-count",
        type=int,
        default=DEFAULT_CLASSIFIER_SEED_COUNT,
        metavar="N",
        help="Number of random seeds to generate when none are passed (default: 10).",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
        metavar="N",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        metavar="LEVEL",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
        metavar="SEED",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="XGBoost device (default: cuda; use cpu for a local smoke run).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="CPU worker threads per XGBoost fit (default: 1).",
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=Path("open_set_auroc_results.json"),
        help="Aggregate JSON output (default: open_set_auroc_results.json).",
    )
    parser.add_argument(
        "--batch-manifest",
        type=Path,
        default=None,
        help=(
            "Seed/bootstrap manifest used for safe resume (default: "
            "<traces-dir>/classifiers/open_set_auroc_batch.json)."
        ),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only combine existing per-leaf open_set_auroc.json files.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Recompute leaves even when compatible per-leaf output exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n_jobs < 1:
        raise SystemExit("ERROR: --n-jobs must be positive")
    if args.bootstrap_replicates < 1:
        raise SystemExit("ERROR: --bootstrap-replicates must be positive")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise SystemExit("ERROR: --bootstrap-confidence must be in (0, 1)")
    if args.bootstrap_seed < 0:
        raise SystemExit("ERROR: --bootstrap-seed must be non-negative")

    agent_filter = set(args.agents) if args.agents else None
    if not args.aggregate_only:
        manifest_path = (
            args.batch_manifest
            if args.batch_manifest is not None
            else (
                args.traces_dir
                / "classifiers"
                / "open_set_auroc_batch.json"
            )
        )
        try:
            seeds, seed_source = _resolve_batch_seeds(
                manifest_path,
                requested_seeds=args.classifier_seeds,
                requested_seed_count=args.classifier_seed_count,
                bootstrap_replicates=args.bootstrap_replicates,
                confidence_level=args.bootstrap_confidence,
                bootstrap_seed=args.bootstrap_seed,
                resume=args.resume,
            )
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc
        print(f"classifier seeds: {','.join(map(str, seeds))}")
        print(f"batch manifest: {manifest_path}")
        try:
            leaves = _reference_leaf_dirs(
                args.traces_dir,
                args.datasets,
                agent_filter,
            )
            for dataset_key, leaf_dir in leaves:
                compute_leaf(
                    args.traces_dir,
                    dataset_key,
                    leaf_dir,
                    classifier_seeds=seeds,
                    classifier_seed_source=seed_source,
                    bootstrap_replicates=args.bootstrap_replicates,
                    confidence_level=args.bootstrap_confidence,
                    bootstrap_seed=args.bootstrap_seed,
                    device=args.device,
                    n_jobs=args.n_jobs,
                    resume=args.resume,
                )
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc

    try:
        aggregate = build_aggregate(
            args.traces_dir,
            args.datasets,
            agent_filter,
        )
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    _atomic_write_json(args.aggregate_output, aggregate)
    print(f"Saved aggregate: {args.aggregate_output}")


if __name__ == "__main__":
    main()
