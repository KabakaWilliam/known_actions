#!/usr/bin/env python3
"""Run pooled open-set experiments with one, two, or three held-out models.

Each held-out subset is treated as one binary unknown class.  The classifier
is trained only on the remaining known models, and AUROC is computed from the
maximum known-class probability on a fixed test population.  Every subset gets
its own pointwise trace-bootstrap confidence interval.

The runner deliberately uses one process and one XGBoost worker by default.
It preloads and featurizes each dataset once, checkpoints every subset
atomically, and can safely resume an interrupted batch.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep numerical libraries from silently consuming the whole host.  Explicit
# user settings still win.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler

from closed_set_stats import (
    DEFAULT_CLASSIFIER_SEED_COUNT,
    generate_classifier_seeds,
    validate_classifier_seeds,
)
from open_set_stats import summarize_pooled_open_set_auroc
from open_set_subset_design import (
    canonical_subset_id,
    select_holdout_subsets,
)
from trace_analyzer import (
    XGB_PARAM_DIST,
    _infer_split,
    _is_valid_trace,
    extract_features,
)


DATASETS: dict[str, dict[str, Any]] = {
    "wiki": {
        "dataset": "2wikimultihop",
        "display_name": "2WikiMultiHop",
        "split_strategy": "recorded_directory_split",
        "caps": None,
    },
    "frames": {
        "dataset": "frames",
        "display_name": "FRAMES",
        "split_strategy": "question_group_sha256",
        "caps": {"train": 150, "val": 75, "test": 75},
    },
    "webshop": {
        "dataset": "webshop",
        "display_name": "WebShop",
        "split_strategy": "recorded_directory_split",
        "caps": None,
    },
    "deepshop": {
        "dataset": "deepshop",
        "display_name": "DeepShop",
        "split_strategy": "question_group_sha256",
        "caps": None,
    },
}

PROTOCOL_NAME = "fixed_test_population_pooled_unknown_v1"
SCORE_DEFINITION = "max_predict_proba; higher_means_known"
BOOTSTRAP_STRATA = ["known_test_trace", "pooled_unknown_test_trace"]
BOOTSTRAP_SAMPLING = (
    "independent_nonparametric_with_replacement_within_stratum"
)
UNKNOWN_POOLING = "all_held_out_models_one_trace_weighted_binary_class"
POOLED_CI_METHOD = (
    "paired_stratified_percentile_bootstrap_over_"
    "known_test_and_pooled_unknown_test_traces"
)
SPLIT_FRACTIONS = {"train": 0.50, "val": 0.25, "test": 0.25}


@dataclass(frozen=True)
class TraceRecord:
    """One already-featurized, valid trace."""

    features: dict[str, float]
    episode_id: str
    question: str


@dataclass(frozen=True)
class DatasetCache:
    """Fixed per-agent matrices used by every held-out subset."""

    dataset_key: str
    feature_names: tuple[str, ...]
    matrices: dict[str, dict[str, np.ndarray]]
    episode_ids: dict[str, dict[str, tuple[str, ...]]]
    questions: dict[str, dict[str, tuple[str, ...]]]
    source_counts: dict[str, int]
    valid_counts: dict[str, int]
    cache_digest: str


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_digest() -> str:
    """Hash every local module that affects the experiment result."""
    paths = [
        Path(__file__),
        Path(__file__).with_name("closed_set_stats.py"),
        Path(__file__).with_name("open_set_stats.py"),
        Path(__file__).with_name("open_set_subset_design.py"),
        Path(__file__).with_name("trace_analyzer.py"),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _runtime_metadata(device: str, n_jobs: int) -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "scikit-learn", "xgboost"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "packages": packages,
        "xgboost_device": device,
        "xgboost_n_jobs": n_jobs,
    }


def _stable_digest(*parts: str) -> bytes:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.digest()


def _stable_unit_interval(*parts: str) -> float:
    numerator = int.from_bytes(_stable_digest(*parts)[:8], "big")
    return numerator / float(1 << 64)


def _question_group_split(dataset_key: str, question: str) -> str:
    """Assign the same question to the same split for every model."""
    value = _stable_unit_interval(PROTOCOL_NAME, dataset_key, question)
    if value < SPLIT_FRACTIONS["train"]:
        return "train"
    if value < SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]:
        return "val"
    return "test"


def _dataset_base(dataset_name: str) -> str:
    split = _infer_split(dataset_name)
    return (
        dataset_name.rsplit("_", 1)[0]
        if split is not None
        else dataset_name
    )


def _trace_inventory_digest(
    traces_dir: Path,
    dataset_keys: list[str],
    model_universe: list[str],
) -> str:
    """Fingerprint the selected raw trace inventory before resuming a batch."""
    if not traces_dir.is_dir():
        raise ValueError(f"missing trace directory: {traces_dir}")
    expected_bases = {
        str(DATASETS[dataset_key]["dataset"])
        for dataset_key in dataset_keys
    }
    agents = set(model_universe)
    digest = hashlib.sha256()
    matched = 0
    for path in sorted(traces_dir.rglob("*.json")):
        relative = path.relative_to(traces_dir)
        if len(relative.parts) < 3 or relative.parts[0] not in agents:
            continue
        if _dataset_base(relative.parts[1]) not in expected_bases:
            continue
        stat = path.stat()
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
        matched += 1
    if matched == 0:
        raise ValueError("no selected raw traces were found")
    digest.update(f"count={matched}".encode("ascii"))
    return digest.hexdigest()


def _trace_split(
    dataset_key: str,
    dataset_name: str,
    question: str,
) -> str | None:
    config = DATASETS[dataset_key]
    if config["split_strategy"] == "recorded_directory_split":
        split = _infer_split(dataset_name)
        return split if split in {"train", "val", "test"} else None
    return _question_group_split(dataset_key, question)


def _cap_records(
    dataset_key: str,
    agent: str,
    split: str,
    records: list[TraceRecord],
    cap: int | None,
) -> list[TraceRecord]:
    if cap is None or len(records) <= cap:
        return sorted(records, key=lambda record: record.episode_id)
    return sorted(
        records,
        key=lambda record: _stable_digest(
            PROTOCOL_NAME,
            dataset_key,
            agent,
            split,
            record.episode_id,
        ),
    )[:cap]


def load_fixed_dataset_cache(
    traces_dir: Path,
    dataset_key: str,
    model_universe: list[str],
) -> DatasetCache:
    """Load one dataset once and pin all per-model split assignments."""
    if dataset_key not in DATASETS:
        raise ValueError(f"unknown dataset key: {dataset_key}")
    if not traces_dir.is_dir():
        raise ValueError(f"missing trace directory: {traces_dir}")

    config = DATASETS[dataset_key]
    expected_base = str(config["dataset"])
    universe = set(model_universe)
    records: dict[str, dict[str, list[TraceRecord]]] = {
        agent: {"train": [], "val": [], "test": []}
        for agent in model_universe
    }
    source_counts = {agent: 0 for agent in model_universe}
    valid_counts = {agent: 0 for agent in model_universe}

    for path in sorted(traces_dir.rglob("*.json")):
        relative = path.relative_to(traces_dir)
        if len(relative.parts) < 3:
            continue
        agent = relative.parts[0]
        if agent == "classifiers" or agent not in universe:
            continue
        dataset_name = relative.parts[1]
        if _dataset_base(dataset_name) != expected_base:
            continue
        source_counts[agent] += 1
        try:
            episode = _read_json(path)
        except ValueError as exc:
            raise ValueError(f"could not load trace {path}: {exc}") from exc
        if not _is_valid_trace(episode):
            continue
        meta = episode.get("meta") or {}
        trace_agent = meta.get("agent_id")
        if trace_agent != agent:
            raise ValueError(
                f"{path}: meta.agent_id {trace_agent!r} does not match "
                f"directory agent {agent!r}"
            )
        episode_id = str(meta.get("episode_id") or path.stem)
        raw_question = meta.get("question")
        if config["split_strategy"] == "question_group_sha256" and (
            not isinstance(raw_question, str) or not raw_question.strip()
        ):
            raise ValueError(
                f"{path}: resplit dataset requires non-empty meta.question"
            )
        question = (
            raw_question.strip()
            if isinstance(raw_question, str) and raw_question.strip()
            else episode_id
        )
        split = _trace_split(dataset_key, dataset_name, question)
        if split is None:
            continue
        try:
            features = extract_features(episode)
        except Exception as exc:
            raise ValueError(f"feature extraction failed for {path}: {exc}") from exc
        records[agent][split].append(
            TraceRecord(
                features=features,
                episode_id=episode_id,
                question=question,
            )
        )
        valid_counts[agent] += 1

    missing_source = [
        agent for agent, count in source_counts.items() if count == 0
    ]
    if missing_source:
        raise ValueError(
            f"{dataset_key}: no source traces for model(s): {missing_source}"
        )

    caps = config["caps"] or {}
    feature_names: tuple[str, ...] | None = None
    matrices: dict[str, dict[str, np.ndarray]] = {}
    episode_ids: dict[str, dict[str, tuple[str, ...]]] = {}
    questions: dict[str, dict[str, tuple[str, ...]]] = {}
    for agent in model_universe:
        matrices[agent] = {}
        episode_ids[agent] = {}
        questions[agent] = {}
        for split in ("train", "val", "test"):
            selected = _cap_records(
                dataset_key,
                agent,
                split,
                records[agent][split],
                caps.get(split),
            )
            if not selected:
                raise ValueError(
                    f"{dataset_key}/{agent}: fixed {split} split is empty"
                )
            if feature_names is None:
                feature_names = tuple(selected[0].features)
            assert feature_names is not None
            for record in selected:
                if tuple(record.features) != feature_names:
                    raise ValueError(
                        f"{dataset_key}/{agent}: inconsistent feature names"
                    )
            matrices[agent][split] = np.asarray(
                [
                    [record.features[name] for name in feature_names]
                    for record in selected
                ],
                dtype=float,
            )
            episode_ids[agent][split] = tuple(
                record.episode_id for record in selected
            )
            questions[agent][split] = tuple(
                record.question for record in selected
            )

    assert feature_names is not None
    cache_digest_value = hashlib.sha256()
    cache_digest_value.update(PROTOCOL_NAME.encode("utf-8"))
    cache_digest_value.update(dataset_key.encode("utf-8"))
    cache_digest_value.update(
        json.dumps(feature_names, separators=(",", ":")).encode("utf-8")
    )
    for agent in model_universe:
        for split in ("train", "val", "test"):
            cache_digest_value.update(agent.encode("utf-8"))
            cache_digest_value.update(b"\0")
            cache_digest_value.update(split.encode("ascii"))
            cache_digest_value.update(b"\0")
            cache_digest_value.update(
                json.dumps(
                    episode_ids[agent][split],
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            cache_digest_value.update(
                np.ascontiguousarray(matrices[agent][split]).tobytes()
            )
    return DatasetCache(
        dataset_key=dataset_key,
        feature_names=feature_names,
        matrices=matrices,
        episode_ids=episode_ids,
        questions=questions,
        source_counts=source_counts,
        valid_counts=valid_counts,
        cache_digest=cache_digest_value.hexdigest(),
    )


def _load_model_universe(path: Path, dataset_keys: list[str]) -> list[str]:
    payload = _read_json(path)
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"{path}: missing datasets object")
    common: list[str] | None = None
    for dataset_key in dataset_keys:
        dataset = datasets.get(dataset_key)
        leaves = (
            dataset.get("held_out_models")
            if isinstance(dataset, dict)
            else None
        )
        if not isinstance(leaves, dict) or not leaves:
            raise ValueError(
                f"{path}: missing held_out_models for {dataset_key}"
            )
        agents = sorted(leaves)
        if common is None:
            common = agents
        elif agents != common:
            raise ValueError(
                f"{path}: model universe differs across selected datasets"
            )
    if common is None or len(common) < 4:
        raise ValueError(f"{path}: model universe must contain at least 4 models")
    return common


def _json_safe_params(params: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, np.integer):
            safe[key] = int(value)
        elif isinstance(value, np.floating):
            safe[key] = float(value)
        else:
            safe[key] = value
    return safe


def build_tuning_candidates(count: int, seed: int) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("tuning candidate count must be positive")
    candidates = list(
        ParameterSampler(
            XGB_PARAM_DIST,
            n_iter=count,
            random_state=seed,
        )
    )
    return [_json_safe_params(candidate) for candidate in candidates]


def _make_xgb(
    params: dict[str, Any],
    *,
    seed: int,
    device: str,
    n_jobs: int,
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - environment error
        raise ValueError(
            "xgboost is required for the multi-holdout experiment"
        ) from exc
    return XGBClassifier(
        tree_method="hist",
        device=device,
        eval_metric="mlogloss",
        random_state=seed,
        verbosity=0,
        n_jobs=n_jobs,
        **params,
    )


def _combine_known_training(
    cache: DatasetCache,
    known_models: list[str],
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    labels = []
    for label, agent in enumerate(known_models):
        matrix = cache.matrices[agent][split]
        matrices.append(matrix)
        labels.append(np.full(len(matrix), label, dtype=int))
    return np.concatenate(matrices), np.concatenate(labels)


def _combine_evaluation(
    cache: DatasetCache,
    models: Iterable[str],
) -> tuple[np.ndarray, dict[str, int]]:
    model_list = list(models)
    counts = {
        agent: len(cache.matrices[agent]["test"])
        for agent in model_list
    }
    return (
        np.concatenate(
            [cache.matrices[agent]["test"] for agent in model_list]
        ),
        counts,
    )


def select_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    tuning_seed: int,
    device: str,
    n_jobs: int,
) -> tuple[dict[str, Any], float, list[float]]:
    """Select a fixed candidate using held-in validation macro-F1 only."""
    scores: list[float] = []
    best_index = 0
    best_score = -math.inf
    for index, params in enumerate(candidates):
        model = _make_xgb(
            params,
            seed=tuning_seed,
            device=device,
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train)
        score = float(
            f1_score(
                y_val,
                model.predict(X_val),
                average="macro",
                zero_division=0,
            )
        )
        scores.append(score)
        if score > best_score:
            best_index = index
            best_score = score
        del model
        gc.collect()
    return dict(candidates[best_index]), best_score, scores


def _bootstrap_metadata(
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "unit": "evaluation_trace",
        "strata": list(BOOTSTRAP_STRATA),
        "sampling": BOOTSTRAP_SAMPLING,
        "paired_across_classifier_seeds": True,
        "replicates": replicates,
        "confidence_level": confidence_level,
        "seed": seed,
        "interval": "percentile",
        "scope": "pointwise_per_held_out_subset",
    }


def _protocol_metadata(tuning_candidates: list[dict[str, Any]]) -> dict:
    return {
        "name": PROTOCOL_NAME,
        "known_class_label": 1,
        "unknown_class_label": 0,
        "unknown_pooling": UNKNOWN_POOLING,
        "score_definition": SCORE_DEFINITION,
        "evaluation_population": (
            "fixed_valid_test_traces_for_both_known_and_unknown_models"
        ),
        "validity_filter": "trace_analyzer._is_valid_trace",
        "recorded_split_datasets": ["2wikimultihop", "webshop"],
        "resplit": {
            "datasets": ["frames", "deepshop"],
            "group": "exact_question_text",
            "assignment": "sha256_to_50_25_25_train_val_test",
            "fractions": dict(SPLIT_FRACTIONS),
            "frames_per_model_caps": {
                "train": 150,
                "val": 75,
                "test": 75,
            },
            "deepshop_per_model_caps": None,
        },
        "hyperparameter_selection": {
            "metric": "held_in_validation_macro_f1",
            "unknown_or_test_traces_used": False,
            "candidate_count": len(tuning_candidates),
            "candidates": tuning_candidates,
            "candidate_digest": _json_digest(tuning_candidates),
        },
    }


def _batch_settings(
    *,
    dataset_keys: list[str],
    holdout_sizes: list[int],
    max_subsets_per_size: int | None,
    subset_seed: int,
    tuning_seed: int,
    tuning_candidates: list[dict[str, Any]],
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
    model_universe: list[str],
    trace_inventory_digest: str,
    device: str,
    n_jobs: int,
) -> dict:
    return {
        "protocol": _protocol_metadata(tuning_candidates),
        "datasets": dataset_keys,
        "holdout_sizes": holdout_sizes,
        "max_subsets_per_size": max_subsets_per_size,
        "subset_seed": subset_seed,
        "tuning_seed": tuning_seed,
        "bootstrap": _bootstrap_metadata(
            bootstrap_replicates,
            confidence_level,
            bootstrap_seed,
        ),
        "model_universe": model_universe,
        "trace_inventory_digest": trace_inventory_digest,
        "implementation_digest": _implementation_digest(),
        "runtime": _runtime_metadata(device, n_jobs),
    }


def resolve_batch_manifest(
    path: Path,
    *,
    requested_seeds: list[int] | None,
    requested_seed_count: int,
    settings: dict,
    resume: bool,
) -> tuple[list[int], str, str]:
    """Create or recover the exact seeds and settings for a batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if resume and path.is_file():
            manifest = _read_json(path)
            if manifest.get("schema_version") != 1:
                raise ValueError(f"{path}: unsupported manifest schema")
            if manifest.get("settings") != settings:
                raise ValueError(
                    f"{path}: settings differ; use --no-resume with a new work dir"
                )
            seeds = validate_classifier_seeds(
                manifest.get("classifier_seeds") or []
            )
            if requested_seeds is not None:
                explicit = validate_classifier_seeds(requested_seeds)
                if explicit != seeds:
                    raise ValueError(
                        f"{path}: explicit seeds differ from manifest"
                    )
            elif len(seeds) != requested_seed_count:
                raise ValueError(
                    f"{path}: manifest contains {len(seeds)} seeds, requested "
                    f"count is {requested_seed_count}"
                )
            seed_source = manifest.get("classifier_seed_source")
            if seed_source not in {"explicit", "generated_system_random"}:
                raise ValueError(f"{path}: invalid classifier_seed_source")
            return seeds, seed_source, str(manifest["run_fingerprint"])

        if requested_seeds is None:
            seeds = generate_classifier_seeds(requested_seed_count)
            seed_source = "generated_system_random"
        else:
            seeds = validate_classifier_seeds(requested_seeds)
            seed_source = "explicit"
        fingerprint = _json_digest(
            {"settings": settings, "classifier_seeds": seeds}
        )
        manifest = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_fingerprint": fingerprint,
            "classifier_seeds": seeds,
            "n_classifier_seeds": len(seeds),
            "classifier_seed_source": seed_source,
            "settings": settings,
        }
        _atomic_write_json(path, manifest)
        return seeds, seed_source, fingerprint


def _compatible_checkpoint(
    path: Path,
    *,
    run_fingerprint: str,
    dataset_key: str,
    held_out_models: tuple[str, ...],
    dataset_cache_digest: str,
) -> dict | None:
    if not path.is_file():
        return None
    leaf = _read_json(path)
    if (
        leaf.get("schema_version") == 2
        and leaf.get("run_fingerprint") == run_fingerprint
        and leaf.get("dataset_key") == dataset_key
        and leaf.get("held_out_models") == list(held_out_models)
        and leaf.get("subset_id") == canonical_subset_id(held_out_models)
        and leaf.get("dataset_cache_digest") == dataset_cache_digest
    ):
        return leaf
    return None


def compute_subset(
    cache: DatasetCache,
    held_out_models: tuple[str, ...],
    *,
    model_universe: list[str],
    classifier_seeds: list[int],
    classifier_seed_source: str,
    tuning_candidates: list[dict[str, Any]],
    tuning_seed: int,
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
    device: str,
    n_jobs: int,
    run_fingerprint: str,
    checkpoint_path: Path,
    resume: bool,
) -> dict:
    """Fit, score, summarize, and checkpoint one held-out subset."""
    if resume:
        cached = _compatible_checkpoint(
            checkpoint_path,
            run_fingerprint=run_fingerprint,
            dataset_key=cache.dataset_key,
            held_out_models=held_out_models,
            dataset_cache_digest=cache.cache_digest,
        )
        if cached is not None:
            print(
                f"[resume] {cache.dataset_key}/"
                f"{cached['subset_id']}",
                flush=True,
            )
            return cached

    held_out_set = set(held_out_models)
    known_models = [
        agent for agent in model_universe if agent not in held_out_set
    ]
    if len(known_models) < 2:
        raise ValueError("at least two known models are required")

    X_train, y_train = _combine_known_training(
        cache,
        known_models,
        "train",
    )
    X_val, y_val = _combine_known_training(cache, known_models, "val")
    X_known, known_counts = _combine_evaluation(cache, known_models)
    X_unknown, unknown_counts = _combine_evaluation(
        cache,
        held_out_models,
    )

    best_params, best_validation_score, candidate_scores = (
        select_hyperparameters(
            X_train,
            y_train,
            X_val,
            y_val,
            tuning_candidates,
            tuning_seed=tuning_seed,
            device=device,
            n_jobs=n_jobs,
        )
    )

    known_scores_by_seed: dict[int, np.ndarray] = {}
    unknown_scores_by_seed: dict[int, np.ndarray] = {}
    for seed in classifier_seeds:
        model = _make_xgb(
            best_params,
            seed=seed,
            device=device,
            n_jobs=n_jobs,
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
        del model
        gc.collect()

    summary = summarize_pooled_open_set_auroc(
        known_scores_by_seed,
        unknown_scores_by_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
    )
    subset_id = canonical_subset_id(held_out_models)
    result = {
        "schema_version": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint": run_fingerprint,
        "dataset_cache_digest": cache.cache_digest,
        "dataset_key": cache.dataset_key,
        "dataset": DATASETS[cache.dataset_key]["dataset"],
        "subset_id": subset_id,
        "holdout_size": len(held_out_models),
        "held_out_models": list(held_out_models),
        "known_models": known_models,
        "unknown_pooling": UNKNOWN_POOLING,
        "n_train_traces": len(X_train),
        "n_val_traces": len(X_val),
        "n_known_traces": len(X_known),
        "n_unknown_traces": len(X_unknown),
        "n_known_traces_by_model": known_counts,
        "n_unknown_traces_by_model": unknown_counts,
        "classifier_seeds": classifier_seeds,
        "classifier_seed_count": len(classifier_seeds),
        "n_classifier_seeds": len(classifier_seeds),
        "classifier_seed_source": classifier_seed_source,
        "bootstrap": _bootstrap_metadata(
            bootstrap_replicates,
            confidence_level,
            bootstrap_seed,
        ),
        "models": {
            "XGBoost": {
                "best_params": best_params,
                "hyperparameter_selection": {
                    "metric": "held_in_validation_macro_f1",
                    "best_score": best_validation_score,
                    "candidate_scores": candidate_scores,
                    "tuning_seed": tuning_seed,
                },
                "auroc": summary,
            }
        },
    }
    _atomic_write_json(checkpoint_path, result)
    interval = summary["confidence_interval"]
    print(
        f"[saved] {cache.dataset_key}/{subset_id} "
        f"k={len(held_out_models)} AUROC={summary['estimate']:.4f} "
        f"CI=[{interval['lower']:.4f}, {interval['upper']:.4f}]",
        flush=True,
    )
    return result


def build_aggregate(
    work_dir: Path,
    *,
    dataset_keys: list[str],
    model_universe: list[str],
    holdout_sizes: list[int],
    max_subsets_per_size: int | None,
    subset_seed: int,
    classifier_seeds: list[int],
    classifier_seed_source: str,
    tuning_candidates: list[dict[str, Any]],
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
    run_fingerprint: str,
) -> dict:
    """Combine the exact selected subset checkpoints into schema v2."""
    datasets: dict[str, Any] = {}
    possible_counts = {
        str(size): math.comb(len(model_universe), size)
        for size in holdout_sizes
    }
    for dataset_key in dataset_keys:
        by_size: dict[str, Any] = {}
        common_cache_digest: str | None = None
        for size in holdout_sizes:
            selection = select_holdout_subsets(
                model_universe,
                size,
                max_subsets=max_subsets_per_size,
                seed=subset_seed,
            )
            subsets: dict[str, dict] = {}
            for held_out_models in selection.subsets:
                subset_id = canonical_subset_id(held_out_models)
                path = (
                    work_dir
                    / dataset_key
                    / f"k{size}"
                    / f"{subset_id}.json"
                )
                leaf = _read_json(path)
                if (
                    leaf.get("schema_version") != 2
                    or leaf.get("run_fingerprint") != run_fingerprint
                    or leaf.get("dataset_key") != dataset_key
                    or leaf.get("subset_id") != subset_id
                    or leaf.get("holdout_size") != size
                    or leaf.get("held_out_models")
                    != list(held_out_models)
                    or leaf.get("known_models")
                    != [
                        model
                        for model in model_universe
                        if model not in held_out_models
                    ]
                    or leaf.get("unknown_pooling") != UNKNOWN_POOLING
                    or leaf.get("classifier_seeds") != classifier_seeds
                    or leaf.get("classifier_seed_count")
                    != len(classifier_seeds)
                    or leaf.get("bootstrap")
                    != _bootstrap_metadata(
                        bootstrap_replicates,
                        confidence_level,
                        bootstrap_seed,
                    )
                ):
                    raise ValueError(f"{path}: incompatible checkpoint")
                cache_digest = leaf.get("dataset_cache_digest")
                if not isinstance(cache_digest, str) or not cache_digest:
                    raise ValueError(
                        f"{path}: missing dataset_cache_digest"
                    )
                if common_cache_digest is None:
                    common_cache_digest = cache_digest
                elif cache_digest != common_cache_digest:
                    raise ValueError(
                        f"{path}: dataset cache digest differs from other leaves"
                    )
                summary = (
                    ((leaf.get("models") or {}).get("XGBoost") or {}).get(
                        "auroc"
                    )
                    or {}
                )
                interval = summary.get("confidence_interval") or {}
                if (
                    summary.get("n_known") != leaf.get("n_known_traces")
                    or summary.get("n_unknown") != leaf.get("n_unknown_traces")
                    or interval.get("method") != POOLED_CI_METHOD
                    or [row.get("seed") for row in summary.get("per_seed", [])]
                    != classifier_seeds
                ):
                    raise ValueError(f"{path}: invalid AUROC summary")
                subsets[subset_id] = leaf
            by_size[str(size)] = {
                "n_possible_subsets": selection.possible_count,
                "n_evaluated_subsets": len(selection.subsets),
                "selection_mode": selection.selection_mode,
                "model_inclusion_counts": selection.model_inclusion_counts,
                "subsets": subsets,
            }
        datasets[dataset_key] = {
            "dataset": DATASETS[dataset_key]["dataset"],
            "display_name": DATASETS[dataset_key]["display_name"],
            "dataset_cache_digest": common_cache_digest,
            "holdout_sizes": by_size,
        }

    return {
        "schema_version": 2,
        "description": (
            "Per-subset pooled open-set XGBoost AUROC for progressively "
            "larger held-out model sets. Every interval is a pointwise "
            "paired trace-bootstrap interval for one subset."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint": run_fingerprint,
        "classifier_seeds": classifier_seeds,
        "classifier_seed_count": len(classifier_seeds),
        "classifier_seed_source": classifier_seed_source,
        "bootstrap": _bootstrap_metadata(
            bootstrap_replicates,
            confidence_level,
            bootstrap_seed,
        ),
        "protocol": _protocol_metadata(tuning_candidates),
        "subset_design": {
            "model_universe": model_universe,
            "holdout_sizes": holdout_sizes,
            "possible_counts": possible_counts,
            "max_subsets_per_size": max_subsets_per_size,
            "subset_seed": subset_seed,
        },
        "datasets": datasets,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("./traces"),
    )
    parser.add_argument(
        "--model-universe-stats",
        type=Path,
        default=Path(__file__).with_name("open_set_auroc_results.json"),
        help="Existing singleton aggregate used only to define model IDs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument(
        "--holdout-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--max-subsets-per-size",
        type=int,
        default=100,
        help=(
            "Evaluate all combinations up to this count, otherwise take a "
            "deterministic balanced sample (default: 100)."
        ),
    )
    parser.add_argument("--subset-seed", type=int, default=2026)
    parser.add_argument(
        "--classifier-seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
    )
    parser.add_argument(
        "--classifier-seed-count",
        type=int,
        default=DEFAULT_CLASSIFIER_SEED_COUNT,
    )
    parser.add_argument("--tuning-candidates", type=int, default=8)
    parser.add_argument("--tuning-seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        default="cuda",
        help="XGBoost device (default: cuda; use cpu for local smoke tests).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="XGBoost CPU workers per fit (default: 1).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("open_set_multi_checkpoints"),
    )
    parser.add_argument(
        "--batch-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=Path("open_set_multi_holdout_auroc_results.json"),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help=(
            "Run selected dataset workers without building the final "
            "aggregate. Use with dataset sharding, then run --aggregate-only."
        ),
    )
    parser.add_argument(
        "--dataset-shard-count",
        type=int,
        default=1,
        help="Number of disjoint dataset workers (default: 1).",
    )
    parser.add_argument(
        "--dataset-shard-index",
        type=int,
        default=0,
        help="Zero-based dataset worker index (default: 0).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> list[int]:
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive")
    if args.dataset_shard_count < 1:
        raise ValueError("--dataset-shard-count must be positive")
    if not 0 <= args.dataset_shard_index < args.dataset_shard_count:
        raise ValueError(
            "--dataset-shard-index must be in [0, dataset-shard-count)"
        )
    if args.aggregate_only and args.no_aggregate:
        raise ValueError("--aggregate-only and --no-aggregate conflict")
    if args.tuning_candidates < 1:
        raise ValueError("--tuning-candidates must be positive")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("--bootstrap-confidence must be in (0, 1)")
    if args.bootstrap_seed < 0 or args.subset_seed < 0 or args.tuning_seed < 0:
        raise ValueError("seed values must be non-negative")
    if args.max_subsets_per_size is not None:
        if args.max_subsets_per_size < 1:
            raise ValueError("--max-subsets-per-size must be positive")
    sizes = sorted(set(args.holdout_sizes))
    if not sizes or any(size not in {1, 2, 3} for size in sizes):
        raise ValueError("--holdout-sizes must contain only 1, 2, and/or 3")
    return sizes


def main() -> None:
    args = _parse_args()
    try:
        holdout_sizes = _validate_args(args)
        dataset_keys = list(dict.fromkeys(args.datasets))
        model_universe = _load_model_universe(
            args.model_universe_stats,
            dataset_keys,
        )
        if max(holdout_sizes) >= len(model_universe) - 1:
            raise ValueError("holdout sizes leave fewer than two known models")
        tuning_candidates = build_tuning_candidates(
            args.tuning_candidates,
            args.tuning_seed,
        )
        trace_inventory_digest = _trace_inventory_digest(
            args.traces_dir,
            dataset_keys,
            model_universe,
        )
        settings = _batch_settings(
            dataset_keys=dataset_keys,
            holdout_sizes=holdout_sizes,
            max_subsets_per_size=args.max_subsets_per_size,
            subset_seed=args.subset_seed,
            tuning_seed=args.tuning_seed,
            tuning_candidates=tuning_candidates,
            bootstrap_replicates=args.bootstrap_replicates,
            confidence_level=args.bootstrap_confidence,
            bootstrap_seed=args.bootstrap_seed,
            model_universe=model_universe,
            trace_inventory_digest=trace_inventory_digest,
            device=args.device,
            n_jobs=args.n_jobs,
        )
        manifest_path = (
            args.batch_manifest
            if args.batch_manifest is not None
            else args.work_dir / "batch_manifest.json"
        )
        seeds, seed_source, run_fingerprint = resolve_batch_manifest(
            manifest_path,
            requested_seeds=args.classifier_seeds,
            requested_seed_count=args.classifier_seed_count,
            settings=settings,
            resume=args.resume,
        )

        worker_dataset_keys = dataset_keys[
            args.dataset_shard_index :: args.dataset_shard_count
        ]
        if not worker_dataset_keys and not args.aggregate_only:
            raise ValueError(
                "dataset shard selects no datasets; reduce "
                "--dataset-shard-count"
            )
        total_subsets = sum(
            min(
                math.comb(len(model_universe), size),
                args.max_subsets_per_size
                if args.max_subsets_per_size is not None
                else math.inf,
            )
            for size in holdout_sizes
        ) * len(worker_dataset_keys)
        print(f"classifier seeds: {','.join(map(str, seeds))}")
        print(f"run fingerprint: {run_fingerprint}")
        print(
            f"dataset worker: {args.dataset_shard_index}/"
            f"{args.dataset_shard_count} -> "
            f"{','.join(worker_dataset_keys) or 'aggregate-only'}"
        )
        print(f"planned subset evaluations: {total_subsets}")

        if not args.aggregate_only:
            completed = 0
            batch_start = time.monotonic()
            for dataset_key in worker_dataset_keys:
                load_start = time.monotonic()
                print(f"[load] {dataset_key}", flush=True)
                cache = load_fixed_dataset_cache(
                    args.traces_dir,
                    dataset_key,
                    model_universe,
                )
                counts = {
                    split: sum(
                        len(cache.matrices[agent][split])
                        for agent in model_universe
                    )
                    for split in ("train", "val", "test")
                }
                print(
                    f"[loaded] {dataset_key} in "
                    f"{time.monotonic() - load_start:.1f}s: {counts}",
                    flush=True,
                )
                for size in holdout_sizes:
                    selection = select_holdout_subsets(
                        model_universe,
                        size,
                        max_subsets=args.max_subsets_per_size,
                        seed=args.subset_seed,
                    )
                    print(
                        f"[design] {dataset_key} k={size}: "
                        f"{len(selection.subsets)}/"
                        f"{selection.possible_count} "
                        f"({selection.selection_mode})",
                        flush=True,
                    )
                    for held_out_models in selection.subsets:
                        subset_id = canonical_subset_id(held_out_models)
                        checkpoint_path = (
                            args.work_dir
                            / dataset_key
                            / f"k{size}"
                            / f"{subset_id}.json"
                        )
                        leaf_start = time.monotonic()
                        compute_subset(
                            cache,
                            held_out_models,
                            model_universe=model_universe,
                            classifier_seeds=seeds,
                            classifier_seed_source=seed_source,
                            tuning_candidates=tuning_candidates,
                            tuning_seed=args.tuning_seed,
                            bootstrap_replicates=args.bootstrap_replicates,
                            confidence_level=args.bootstrap_confidence,
                            bootstrap_seed=args.bootstrap_seed,
                            device=args.device,
                            n_jobs=args.n_jobs,
                            run_fingerprint=run_fingerprint,
                            checkpoint_path=checkpoint_path,
                            resume=args.resume,
                        )
                        completed += 1
                        elapsed = time.monotonic() - batch_start
                        rate = elapsed / completed
                        print(
                            f"[progress] {completed}/{total_subsets}; "
                            f"leaf={time.monotonic() - leaf_start:.1f}s; "
                            f"ETA={(total_subsets - completed) * rate / 60:.1f}m",
                            flush=True,
                        )
                del cache
                gc.collect()

        if args.no_aggregate:
            print("Dataset worker complete; aggregate intentionally deferred.")
            return
        aggregate = build_aggregate(
            args.work_dir,
            dataset_keys=dataset_keys,
            model_universe=model_universe,
            holdout_sizes=holdout_sizes,
            max_subsets_per_size=args.max_subsets_per_size,
            subset_seed=args.subset_seed,
            classifier_seeds=seeds,
            classifier_seed_source=seed_source,
            tuning_candidates=tuning_candidates,
            bootstrap_replicates=args.bootstrap_replicates,
            confidence_level=args.bootstrap_confidence,
            bootstrap_seed=args.bootstrap_seed,
            run_fingerprint=run_fingerprint,
        )
        _atomic_write_json(args.aggregate_output, aggregate)
        print(f"Saved aggregate: {args.aggregate_output}")
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
