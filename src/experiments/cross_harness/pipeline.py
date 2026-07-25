#!/usr/bin/env python3
"""Manifest-driven closed-set experiments across MidScene and browser-use.

Raw traces are only read.  ``prepare`` freezes exact trace membership before
training, and ``evaluate`` never refits a model or preprocessing object.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

from trace_analyzer import (
    AgentLSTM,
    EVENT_VOCAB,
    RF_PARAM_GRID,
    XGB_PARAM_DIST,
    _LSTM_BATCH_SIZE,
    _LSTM_EMBED_DIM,
    _LSTM_GRID,
    _LSTM_LR,
    _LSTM_N_LAYERS,
    _LSTM_WEIGHT_DECAY,
    _N_CONTINUOUS,
    SequenceDataset,
    collate_fn,
    extract_features,
    extract_sequence,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - surfaced cleanly at training time
    XGBClassifier = None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


HARNESSES = ("midscene", "browser_use")
POLICIES = ("midscene", "browser_use", "mixed50")
SPLITS = ("train", "val", "test")
INVALID_ERROR_PATTERNS = (
    "credit balance is too low",
    "insufficient credits",
    "insufficient_quota",
    "invalid_api_key",
    "401 unauthorized",
    "402 payment",
    "failed to call ai model service",
)
TIMING_FEATURES = (
    "total_duration_s",
    "t_first_action_ms",
    "mean_iei_ms",
    "std_iei_ms",
    "median_iei_ms",
    "p10_iei_ms",
    "p90_iei_ms",
    "iei_trend",
    "mean_click_iei_ms",
    "std_click_iei_ms",
    "mean_nav_iei_ms",
    "std_nav_iei_ms",
    "max_page_dwell_ms",
    "mean_key_iei_ms",
    "std_key_iei_ms",
)


@dataclass(frozen=True)
class TraceRecord:
    trace_path: str
    episode_id: str
    agent_id: str
    dataset: str
    split: str
    harness: str
    task_id: str
    collection_run_id: str
    task_success: bool | None
    valid_trace: bool
    n_events: int
    error: str | None
    timestamp: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_task(value: str) -> str:
    return " ".join(value.casefold().split())


def _task_id(dataset: str, question: str) -> str:
    payload = f"{dataset}\0{_normalise_task(question)}".encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _resolve_path(config_path: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else config_path.parent / path


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    cfg = yaml.safe_load(path.read_text())
    exp = cfg["experiment"]
    exp["traces_dir"] = _resolve_path(path, exp["traces_dir"]).resolve()
    exp["artifact_root"] = _resolve_path(path, exp["artifact_root"]).resolve()
    cfg["_config_path"] = path
    return cfg


def _infer_harness(relative: Path, episode: dict[str, Any]) -> tuple[str, str | None]:
    explicit = relative.parts[2] if len(relative.parts) >= 5 else None
    path_harness = explicit if explicit in HARNESSES else None
    meta_harness = (episode.get("meta") or {}).get("harness")
    if path_harness and meta_harness and path_harness != meta_harness:
        return path_harness, (
            f"harness mismatch: path={path_harness}, metadata={meta_harness}"
        )
    if path_harness:
        return path_harness, None
    if meta_harness in HARNESSES:
        return meta_harness, None
    # Traces collected before harness directories/metadata existed are MidScene.
    return "midscene", None


def _trace_valid(episode: dict[str, Any], harness_error: str | None) -> bool:
    if harness_error:
        return False
    error = str(episode.get("error") or "").casefold()
    if any(pattern in error for pattern in INVALID_ERROR_PATTERNS):
        return False
    return bool((episode.get("dom_trace") or {}).get("events"))


def scan_inventory(cfg: dict[str, Any]) -> list[TraceRecord]:
    traces_dir: Path = cfg["experiment"]["traces_dir"]
    records: list[TraceRecord] = []
    for agent_id in cfg["agents"]:
        for dataset, _dataset_cfg in cfg["datasets"].items():
            for split in SPLITS:
                dataset_name = f"{dataset}_{split}"
                dataset_dir = traces_dir / agent_id / dataset_name
                if not dataset_dir.exists():
                    continue
                for path in sorted(dataset_dir.rglob("*.json")):
                    try:
                        episode = json.loads(path.read_text())
                    except Exception as exc:
                        records.append(
                            TraceRecord(
                                trace_path=str(path.resolve()),
                                episode_id=path.stem,
                                agent_id=agent_id,
                                dataset=dataset,
                                split=split,
                                harness="midscene",
                                task_id="",
                                collection_run_id=path.parent.name,
                                task_success=None,
                                valid_trace=False,
                                n_events=0,
                                error=f"{type(exc).__name__}: {exc}",
                                timestamp="",
                            )
                        )
                        continue
                    relative = path.relative_to(traces_dir)
                    harness, harness_error = _infer_harness(relative, episode)
                    meta = episode.get("meta") or {}
                    question = str(meta.get("question") or "")
                    events = (episode.get("dom_trace") or {}).get("events") or []
                    verification = episode.get("verification") or {}
                    success = verification.get("correct")
                    if success is None:
                        success = verification.get("task_success")
                    records.append(
                        TraceRecord(
                            trace_path=str(path.resolve()),
                            episode_id=str(meta.get("episode_id") or path.stem),
                            agent_id=agent_id,
                            dataset=dataset,
                            split=split,
                            harness=harness,
                            task_id=_task_id(dataset, question) if question else "",
                            collection_run_id=path.parent.name,
                            task_success=bool(success) if success is not None else None,
                            valid_trace=_trace_valid(episode, harness_error) and bool(question),
                            n_events=len(events),
                            error=harness_error or episode.get("error"),
                            timestamp=str(meta.get("timestamp") or ""),
                        )
                    )
    return records


def _coverage(
    cfg: dict[str, Any], records: list[TraceRecord]
) -> dict[str, Any]:
    valid = [record for record in records if record.valid_trace]
    valid_counts: Counter[tuple[str, str, str, str]] = Counter(
        (r.dataset, r.split, r.harness, r.agent_id) for r in valid
    )
    task_sets: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for record in valid:
        task_sets[(record.dataset, record.split, record.harness, record.agent_id)].add(
            record.task_id
        )

    common: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for dataset in cfg["datasets"]:
        for split in SPLITS:
            required_sets = [
                task_sets[(dataset, split, harness, agent)]
                for harness in HARNESSES
                for agent in cfg["agents"]
            ]
            common[dataset][split] = sorted(
                set.intersection(*required_sets) if required_sets else set()
            )
    return {
        "n_json_files": len(records),
        "n_valid_traces": len(valid),
        "valid_counts": {
            "|".join(key): count for key, count in sorted(valid_counts.items())
        },
        "common_task_ids": common,
    }


def print_audit(cfg: dict[str, Any], records: list[TraceRecord]) -> dict[str, Any]:
    coverage = _coverage(cfg, records)
    print(f"Inventory: {coverage['n_json_files']} JSON files, "
          f"{coverage['n_valid_traces']} valid traces")
    for dataset, dataset_cfg in cfg["datasets"].items():
        print(f"\n{dataset}")
        for split in SPLITS:
            common_n = len(coverage["common_task_ids"][dataset][split])
            expected = int(dataset_cfg["expected_tasks"][split])
            print(f"  {split:5s}: common={common_n}/{expected}")
            for harness in HARNESSES:
                by_agent = {
                    agent: coverage["valid_counts"].get(
                        f"{dataset}|{split}|{harness}|{agent}", 0
                    )
                    for agent in cfg["agents"]
                }
                counts = list(by_agent.values())
                print(
                    f"    {harness:11s} min={min(counts, default=0):3d} "
                    f"max={max(counts, default=0):3d}"
                )
                incomplete = {
                    agent: count
                    for agent, count in by_agent.items()
                    if count < expected
                }
                if incomplete:
                    details = ", ".join(
                        f"{agent}={count}" for agent, count in incomplete.items()
                    )
                    print(f"      incomplete: {details}")
    return coverage


def _selected_record_index(
    records: list[TraceRecord],
) -> dict[tuple[str, str, str, str, str], TraceRecord]:
    grouped: dict[tuple[str, str, str, str, str], list[TraceRecord]] = defaultdict(list)
    for record in records:
        if record.valid_trace:
            grouped[
                (
                    record.dataset,
                    record.split,
                    record.harness,
                    record.agent_id,
                    record.task_id,
                )
            ].append(record)
    selected = {}
    for key, candidates in grouped.items():
        selected[key] = min(
            candidates, key=lambda r: (r.timestamp, r.trace_path)
        )
    return selected


def _mixed_assignment(task_ids: list[str], seed: int) -> dict[str, str]:
    ranked = sorted(
        task_ids,
        key=lambda task: hashlib.sha256(f"{seed}\0{task}".encode()).hexdigest(),
    )
    n_midscene = len(ranked) // 2
    return {
        task: ("midscene" if index < n_midscene else "browser_use")
        for index, task in enumerate(ranked)
    }


def _manifest_rows(
    cfg: dict[str, Any],
    selected: dict[tuple[str, str, str, str, str], TraceRecord],
    dataset: str,
    split: str,
    policy: str,
    task_ids: list[str],
) -> list[dict[str, Any]]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    assignment = (
        _mixed_assignment(task_ids, int(cfg["experiment"]["sampling_seed"]))
        if policy == "mixed50"
        else {task: policy for task in task_ids}
    )
    rows = []
    for task_id in sorted(task_ids):
        harness = assignment[task_id]
        for agent_id in cfg["agents"]:
            record = selected[(dataset, split, harness, agent_id, task_id)]
            rows.append(asdict(record))
    return rows


def _write_frozen_jsonl(path: Path, rows: list[dict[str, Any]], force: bool) -> str:
    content = b"".join(_json_bytes(row) for row in rows)
    digest = _sha256_bytes(content)
    if path.exists() and path.read_bytes() != content and not force:
        raise RuntimeError(
            f"refusing to replace frozen manifest with different content: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return digest


def prepare_manifests(
    cfg: dict[str, Any], records: list[TraceRecord], force: bool = False
) -> dict[str, Any]:
    coverage = _coverage(cfg, records)
    artifact_root: Path = cfg["experiment"]["artifact_root"]
    manifests_root = artifact_root / "splits"
    selected = _selected_record_index(records)
    manifest_hashes: dict[str, str] = {}
    selected_task_counts: dict[str, dict[str, int]] = defaultdict(dict)

    for dataset, dataset_cfg in cfg["datasets"].items():
        all_split_tasks: dict[str, set[str]] = {}
        for split in SPLITS:
            task_ids = coverage["common_task_ids"][dataset][split]
            expected = int(dataset_cfg["expected_tasks"][split])
            minimum = int(
                (dataset_cfg.get("minimum_common_tasks") or {}).get(split, 1)
            )
            if len(task_ids) < minimum:
                raise RuntimeError(
                    f"{dataset}/{split} has {len(task_ids)} tasks complete across "
                    f"all six agents and both harnesses; minimum is {minimum} "
                    f"(nominal dataset size is {expected})"
                )
            if len(task_ids) < expected:
                print(
                    f"[WARN] {dataset}/{split}: freezing {len(task_ids)} paired "
                    f"tasks rather than nominal {expected}"
                )
            selected_task_counts[dataset][split] = len(task_ids)
            all_split_tasks[split] = set(task_ids)
            for policy in POLICIES:
                rows = _manifest_rows(
                    cfg, selected, dataset, split, policy, task_ids
                )
                path = (
                    manifests_root
                    / dataset
                    / f"seed={cfg['experiment']['sampling_seed']}"
                    / f"{split}_{policy}.jsonl"
                )
                manifest_hashes[str(path.relative_to(artifact_root))] = (
                    _write_frozen_jsonl(path, rows, force)
                )
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                overlap = all_split_tasks[left] & all_split_tasks[right]
                if overlap:
                    raise RuntimeError(
                        f"{dataset}: {len(overlap)} task IDs overlap {left}/{right}"
                    )

    inventory_rows = [asdict(record) for record in sorted(
        records,
        key=lambda r: (
            r.dataset, r.split, r.harness, r.agent_id, r.task_id, r.trace_path
        ),
    )]
    inventory_hash = _write_frozen_jsonl(
        artifact_root / "trace_inventory.jsonl", inventory_rows, force
    )
    experiment_manifest = {
        "schema_version": 1,
        "experiment_id": cfg["experiment"]["id"],
        "created_at": _utc_now(),
        "config_path": str(cfg["_config_path"]),
        "agents": list(cfg["agents"]),
        "datasets": list(cfg["datasets"]),
        "harnesses": list(HARNESSES),
        "policies": list(POLICIES),
        "sampling_seed": cfg["experiment"]["sampling_seed"],
        "selected_task_counts": selected_task_counts,
        "inventory_sha256": inventory_hash,
        "manifest_sha256": manifest_hashes,
    }
    manifest_path = artifact_root / "experiment_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not force:
        old = json.loads(manifest_path.read_text())
        comparable = {
            key: value for key, value in experiment_manifest.items()
            if key != "created_at"
        }
        old_comparable = {key: value for key, value in old.items() if key != "created_at"}
        if old_comparable != comparable:
            raise RuntimeError(
                f"refusing to replace frozen experiment manifest: {manifest_path}"
            )
    manifest_path.write_text(json.dumps(experiment_manifest, indent=2) + "\n")
    print(f"Prepared frozen manifests under {manifests_root}")
    return experiment_manifest


def _manifest_path(cfg: dict[str, Any], dataset: str, split: str, policy: str) -> Path:
    return (
        cfg["experiment"]["artifact_root"]
        / "splits"
        / dataset
        / f"seed={cfg['experiment']['sampling_seed']}"
        / f"{split}_{policy}.jsonl"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found; run prepare first: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _load_examples(
    rows: list[dict[str, Any]],
    include_sequences: bool,
    cfg: dict[str, Any] | None = None,
    feature_group: str = "full",
) -> tuple[np.ndarray, list, list[str], list[str]]:
    features: list[dict[str, float]] = []
    sequences = []
    labels = []
    feature_names: list[str] = []
    for row in rows:
        episode = json.loads(Path(row["trace_path"]).read_text())
        feature = extract_features(episode)
        if not feature_names:
            feature_names = list(feature)
        if list(feature) != feature_names:
            raise RuntimeError(f"feature schema mismatch in {row['trace_path']}")
        features.append(feature)
        if include_sequences:
            sequences.append(extract_sequence(episode))
        labels.append(row["agent_id"])
    feature_names = _resolve_feature_group(
        cfg or {}, feature_group, feature_names
    )
    X = np.asarray(
        [[feature[name] for name in feature_names] for feature in features],
        dtype=float,
    )
    if not np.isfinite(X).all():
        raise RuntimeError("feature extraction produced non-finite values")
    return X, sequences, labels, feature_names


def _resolve_feature_group(
    cfg: dict[str, Any],
    feature_group: str,
    available_names: list[str],
) -> list[str]:
    """Resolve a feature view without modifying the underlying trace."""
    configured = cfg.get("feature_groups", {})
    default_groups: dict[str, dict[str, Any]] = {
        "full": {"include": "*"},
        "timing_only": {"include": list(TIMING_FEATURES)},
        "non_timing": {"exclude": list(TIMING_FEATURES)},
    }
    groups = {**default_groups, **configured}
    if feature_group not in groups:
        raise ValueError(
            f"unknown feature group {feature_group!r}; "
            f"choose from {sorted(groups)}"
        )
    spec = groups[feature_group] or {}
    include = spec.get("include")
    exclude = set(spec.get("exclude", []))
    if include == "*" or include is None:
        selected = [name for name in available_names if name not in exclude]
    else:
        requested = list(include)
        missing = sorted(set(requested) - set(available_names))
        if missing:
            raise RuntimeError(
                f"feature group {feature_group!r} references missing features: "
                f"{missing}"
            )
        selected = [name for name in available_names if name in requested]
        selected = [name for name in selected if name not in exclude]
    if not selected:
        raise RuntimeError(f"feature group {feature_group!r} selects no features")
    return selected


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_dir(
    cfg: dict[str, Any],
    dataset: str,
    train_policy: str,
    seed: int,
    classifier: str,
    feature_group: str = "full",
) -> Path:
    return (
        cfg["experiment"]["artifact_root"]
        / "models"
        / dataset
        / f"train={train_policy}"
        / f"features={feature_group}"
        / f"seed={seed}"
        / classifier
    )


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]
) -> dict[str, Any]:
    labels = np.arange(len(classes))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
    }


def _binary_ranking_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    positive_class: str = "browser_use",
) -> dict[str, float | str]:
    if positive_class not in classes:
        raise RuntimeError(f"positive class {positive_class!r} is unavailable")
    positive_index = classes.index(positive_class)
    binary_true = (y_true == positive_index).astype(int)
    scores = probabilities[:, positive_index]
    return {
        "positive_class": positive_class,
        "auroc": float(roc_auc_score(binary_true, scores)),
        "average_precision": float(
            average_precision_score(binary_true, scores)
        ),
    }


def _make_lstm_tensors(sequences):
    token_tensors = [
        torch.tensor([event[0] for event in sequence], dtype=torch.long)
        for sequence in sequences
    ]
    time_tensors = [
        torch.tensor(
            [[event[1], event[2], event[3], event[4], event[5]]
             for event in sequence],
            dtype=torch.float,
        ).reshape(-1, _N_CONTINUOUS)
        for sequence in sequences
    ]
    return token_tensors, time_tensors


def _fit_lstm_once(
    sequences,
    X,
    y,
    n_classes: int,
    params: dict[str, Any],
    epochs: int,
    seed: int,
    device: torch.device,
) -> AgentLSTM:
    _set_seed(seed)
    tokens, times = _make_lstm_tensors(sequences)
    rf_features = [torch.tensor(X[index], dtype=torch.float) for index in range(len(X))]
    labels = torch.tensor(y, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        SequenceDataset(tokens, times, rf_features, labels),
        batch_size=_LSTM_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )
    model = AgentLSTM(
        len(EVENT_VOCAB),
        _LSTM_EMBED_DIM,
        int(params["hidden_dim"]),
        _LSTM_N_LAYERS,
        n_classes,
        n_rf_features=X.shape[1],
        dropout=float(params["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=_LSTM_LR, weight_decay=_LSTM_WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch_tokens, batch_times, lengths, batch_rf, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(
                batch_tokens.to(device),
                batch_times.to(device),
                lengths.to(device),
                batch_rf.to(device),
            )
            criterion(logits, batch_labels.to(device)).backward()
            optimizer.step()
    return model


def _predict_lstm(
    model: AgentLSTM, sequences, X: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    tokens, times = _make_lstm_tensors(sequences)
    rf_features = [torch.tensor(X[index], dtype=torch.float) for index in range(len(X))]
    dummy_labels = torch.zeros(len(X), dtype=torch.long)
    loader = DataLoader(
        SequenceDataset(tokens, times, rf_features, dummy_labels),
        batch_size=_LSTM_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    predictions = []
    probabilities = []
    model.eval()
    with torch.no_grad():
        for batch_tokens, batch_times, lengths, batch_rf, _ in loader:
            logits = model(
                batch_tokens.to(device),
                batch_times.to(device),
                lengths.to(device),
                batch_rf.to(device),
            )
            probs = torch.softmax(logits, dim=1)
            predictions.extend(probs.argmax(dim=1).cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    return np.asarray(predictions), np.asarray(probabilities)


def train_model(
    cfg: dict[str, Any],
    dataset: str,
    train_policy: str,
    classifier: str,
    seed: int,
    *,
    feature_group: str = "full",
    quick: bool = False,
    xgb_device: str | None = None,
    force: bool = False,
) -> Path:
    if classifier not in cfg["classifiers"]["enabled"]:
        raise ValueError(f"{classifier} is not enabled by the experiment config")
    if classifier == "LSTM" and feature_group != "full":
        raise ValueError(
            "timing/non-timing ablations are currently tabular-only; the LSTM "
            "event sequence contains timing even when tabular timing is removed"
        )
    model_dir = _model_dir(
        cfg, dataset, train_policy, seed, classifier, feature_group
    )
    bundle_path = model_dir / "model.pkl"
    if bundle_path.exists() and not force:
        print(f"[SKIP] trained model exists: {bundle_path}")
        return model_dir

    train_rows = _read_jsonl(_manifest_path(cfg, dataset, "train", train_policy))
    val_rows = _read_jsonl(_manifest_path(cfg, dataset, "val", train_policy))
    include_sequences = classifier == "LSTM"
    X_train, seq_train, labels_train, feature_names = _load_examples(
        train_rows, include_sequences, cfg, feature_group
    )
    X_val, seq_val, labels_val, val_feature_names = _load_examples(
        val_rows, include_sequences, cfg, feature_group
    )
    if feature_names != val_feature_names:
        raise RuntimeError("train/validation feature schema mismatch")

    encoder = LabelEncoder().fit(labels_train)
    expected_classes = sorted(cfg["agents"])
    if list(encoder.classes_) != expected_classes:
        raise RuntimeError(
            f"training classes {list(encoder.classes_)} != configured {expected_classes}"
        )
    if set(labels_val) != set(expected_classes):
        raise RuntimeError("validation class set differs from training class set")
    y_train = encoder.transform(labels_train)
    y_val = encoder.transform(labels_val)
    _set_seed(seed)
    best_params: dict[str, Any]
    scaler = None
    model = None
    if classifier == "RandomForest":
        if quick or not cfg["classifiers"]["random_forest"].get("search", True):
            best_params = {"n_estimators": 100, "max_depth": 15}
            model = RandomForestClassifier(
                **best_params, random_state=seed,
                n_jobs=int(cfg["classifiers"].get("cpu_jobs", 8)),
            ).fit(X_train, y_train)
        else:
            search = GridSearchCV(
                RandomForestClassifier(random_state=seed),
                RF_PARAM_GRID,
                cv=3,
                scoring="f1_macro",
                n_jobs=int(cfg["classifiers"].get("cpu_jobs", 8)),
                refit=True,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
        y_val_pred = model.predict(X_val)
    elif classifier == "XGBoost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed")
        device = xgb_device or cfg["classifiers"]["xgboost"].get("device", "cuda")
        base = XGBClassifier(
            tree_method="hist",
            device=device,
            eval_metric="mlogloss",
            random_state=seed,
            verbosity=0,
        )
        if quick:
            best_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
            model = base.set_params(**best_params).fit(X_train, y_train)
        else:
            search = RandomizedSearchCV(
                base,
                XGB_PARAM_DIST,
                n_iter=int(cfg["classifiers"]["xgboost"].get(
                    "search_iterations", 40
                )),
                cv=3,
                scoring="f1_macro",
                n_jobs=1,
                refit=True,
                random_state=seed,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
        y_val_pred = model.predict(X_val)
    elif classifier == "LSTM":
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        device_name = (
            "cpu" if quick else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        device = torch.device(device_name)
        epochs = 2 if quick else int(cfg["classifiers"]["lstm"].get("epochs", 50))
        candidates = [_LSTM_GRID[0]] if quick else _LSTM_GRID
        best_score = -1.0
        best_params = dict(candidates[0])
        for candidate_index, candidate in enumerate(candidates):
            candidate_model = _fit_lstm_once(
                seq_train,
                X_train_scaled,
                y_train,
                len(encoder.classes_),
                candidate,
                epochs,
                seed + candidate_index,
                device,
            )
            candidate_pred, _ = _predict_lstm(
                candidate_model, seq_val, X_val_scaled, device
            )
            score = f1_score(
                y_val,
                candidate_pred,
                labels=np.arange(len(encoder.classes_)),
                average="macro",
                zero_division=0,
            )
            if score > best_score:
                best_score = score
                best_params = dict(candidate)
        model = _fit_lstm_once(
            seq_train,
            X_train_scaled,
            y_train,
            len(encoder.classes_),
            best_params,
            epochs,
            seed,
            device,
        )
        y_val_pred, _ = _predict_lstm(
            model, seq_val, X_val_scaled, device
        )
    else:
        raise ValueError(f"unsupported classifier: {classifier}")

    validation_metrics = _classification_metrics(
        y_val, np.asarray(y_val_pred), list(encoder.classes_)
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": 1,
        "classifier": classifier,
        "dataset": dataset,
        "train_policy": train_policy,
        "feature_group": feature_group,
        "seed": seed,
        "classes": list(encoder.classes_),
        "feature_names": feature_names,
        "feature_schema_sha256": _sha256_bytes(_json_bytes(feature_names)),
        "label_encoder": encoder,
        "scaler": scaler,
        "best_params": best_params,
        "model": None if classifier == "LSTM" else model,
        "lstm_device": "cpu" if classifier == "LSTM" and quick else None,
        "lstm_epochs": (
            2 if classifier == "LSTM" and quick
            else int(cfg["classifiers"]["lstm"].get("epochs", 50))
        ),
        "train_manifest": str(_manifest_path(cfg, dataset, "train", train_policy)),
        "val_manifest": str(_manifest_path(cfg, dataset, "val", train_policy)),
        "train_manifest_sha256": _sha256_file(
            _manifest_path(cfg, dataset, "train", train_policy)
        ),
        "val_manifest_sha256": _sha256_file(
            _manifest_path(cfg, dataset, "val", train_policy)
        ),
    }
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    if classifier == "LSTM":
        torch.save(model.state_dict(), model_dir / "model.pt")
    training_result = {
        "timestamp": _utc_now(),
        "classifier": classifier,
        "dataset": dataset,
        "train_policy": train_policy,
        "feature_group": feature_group,
        "seed": seed,
        "quick": quick,
        "best_params": best_params,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "validation": validation_metrics,
    }
    (model_dir / "training_results.json").write_text(
        json.dumps(training_result, indent=2) + "\n"
    )
    print(
        f"[TRAINED] {classifier} {dataset} train={train_policy} "
        f"features={feature_group} "
        f"val_macro_f1={validation_metrics['macro_f1']:.3f}"
    )
    return model_dir


def evaluate_model(
    cfg: dict[str, Any],
    dataset: str,
    train_policy: str,
    eval_policy: str,
    classifier: str,
    seed: int,
    *,
    feature_group: str = "full",
    force: bool = False,
) -> Path:
    model_dir = _model_dir(
        cfg, dataset, train_policy, seed, classifier, feature_group
    )
    bundle_path = model_dir / "model.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(f"trained model not found: {bundle_path}")
    eval_dir = model_dir / "evaluations" / f"test={eval_policy}"
    results_path = eval_dir / "results.json"
    if results_path.exists() and not force:
        print(f"[SKIP] evaluation exists: {results_path}")
        return results_path

    with bundle_path.open("rb") as handle:
        bundle = pickle.load(handle)
    rows = _read_jsonl(_manifest_path(cfg, dataset, "test", eval_policy))
    include_sequences = classifier == "LSTM"
    X, sequences, labels, feature_names = _load_examples(
        rows, include_sequences, cfg, feature_group
    )
    if feature_names != bundle["feature_names"]:
        raise RuntimeError("evaluation feature schema differs from trained model")
    if bundle.get("feature_group", "full") != feature_group:
        raise RuntimeError("evaluation feature group differs from trained model")
    encoder: LabelEncoder = bundle["label_encoder"]
    if set(labels) != set(encoder.classes_):
        raise RuntimeError("evaluation class set differs from trained model")
    y_true = encoder.transform(labels)

    if classifier == "LSTM":
        scaler: StandardScaler = bundle["scaler"]
        X_model = scaler.transform(X)
        requested_device = bundle.get("lstm_device")
        device = torch.device(
            requested_device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        params = bundle["best_params"]
        model = AgentLSTM(
            len(EVENT_VOCAB),
            _LSTM_EMBED_DIM,
            int(params["hidden_dim"]),
            _LSTM_N_LAYERS,
            len(encoder.classes_),
            n_rf_features=X.shape[1],
            dropout=float(params["dropout"]),
        ).to(device)
        model.load_state_dict(
            torch.load(model_dir / "model.pt", map_location=device)
        )
        y_pred, probabilities = _predict_lstm(model, sequences, X_model, device)
    else:
        model = bundle["model"]
        y_pred = model.predict(X)
        probabilities = (
            model.predict_proba(X) if hasattr(model, "predict_proba") else None
        )

    metrics = _classification_metrics(
        y_true, np.asarray(y_pred), list(encoder.classes_)
    )
    predictions = []
    for index, row in enumerate(rows):
        prediction = {
            "episode_id": row["episode_id"],
            "task_id": row["task_id"],
            "agent_id": row["agent_id"],
            "true_label": labels[index],
            "predicted_label": encoder.inverse_transform(
                [int(y_pred[index])]
            )[0],
            "dataset": dataset,
            "split": "test",
            "harness": row["harness"],
            "train_policy": train_policy,
            "eval_policy": eval_policy,
            "feature_group": feature_group,
            "classifier": classifier,
            "classifier_seed": seed,
        }
        if probabilities is not None:
            prediction["probabilities"] = {
                class_name: float(probabilities[index][class_index])
                for class_index, class_name in enumerate(encoder.classes_)
            }
        predictions.append(prediction)

    eval_dir.mkdir(parents=True, exist_ok=True)
    _write_frozen_jsonl(eval_dir / "predictions.jsonl", predictions, force)
    result = {
        "timestamp": _utc_now(),
        "classifier": classifier,
        "dataset": dataset,
        "train_policy": train_policy,
        "eval_policy": eval_policy,
        "feature_group": feature_group,
        "seed": seed,
        "n_test": len(rows),
        "class_names": list(encoder.classes_),
        "test_manifest": str(_manifest_path(cfg, dataset, "test", eval_policy)),
        "test_manifest_sha256": _sha256_file(
            _manifest_path(cfg, dataset, "test", eval_policy)
        ),
        "metrics": metrics,
    }
    results_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[EVALUATED] {classifier} {dataset} {train_policy}->{eval_policy} "
        f"features={feature_group} "
        f"macro_f1={metrics['macro_f1']:.3f}"
    )
    return results_path


GRID = {
    "midscene": ("midscene", "browser_use"),
    "browser_use": ("browser_use", "midscene"),
    "mixed50": ("mixed50", "midscene", "browser_use"),
}


def summarize_results(cfg: dict[str, Any]) -> Path:
    artifact_root: Path = cfg["experiment"]["artifact_root"]
    rows = []
    models_root = artifact_root / "models"
    if models_root.exists():
        for path in sorted(models_root.rglob("evaluations/test=*/results.json")):
            result = json.loads(path.read_text())
            metrics = result["metrics"]
            rows.append(
                {
                    "dataset": result["dataset"],
                    "train_policy": result["train_policy"],
                    "eval_policy": result["eval_policy"],
                    "feature_group": result.get("feature_group", "full"),
                    "classifier": result["classifier"],
                    "seed": result["seed"],
                    "n_test": result["n_test"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "results_path": str(path),
                }
            )
    summary_root = artifact_root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    json_path = summary_root / "results.json"
    csv_path = summary_root / "results.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    fields = [
        "dataset",
        "train_policy",
        "eval_policy",
        "feature_group",
        "classifier",
        "seed",
        "n_test",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "results_path",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summarized {len(rows)} evaluations → {csv_path}")
    return csv_path


def _harness_detector_dir(
    cfg: dict[str, Any],
    dataset: str,
    classifier: str,
    seed: int,
    feature_group: str = "full",
) -> Path:
    return (
        cfg["experiment"]["artifact_root"]
        / "harness_detector"
        / dataset
        / f"features={feature_group}"
        / f"seed={seed}"
        / classifier
    )


def _harness_rows(
    cfg: dict[str, Any],
    dataset: str,
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for harness in HARNESSES:
        rows.extend(_read_jsonl(_manifest_path(cfg, dataset, split, harness)))
    return rows


def run_harness_detector_lomo(
    cfg: dict[str, Any],
    dataset: str,
    classifier: str,
    seed: int,
    *,
    feature_group: str = "full",
    quick: bool = False,
    xgb_device: str | None = None,
    force: bool = False,
) -> Path:
    """Predict the harness while holding out every trace from one model.

    Training and validation use only non-held-out models and their respective
    dataset splits. Testing uses only the held-out model's unseen test tasks,
    under both harnesses.
    """
    if classifier not in {"XGBoost", "RandomForest"}:
        raise ValueError(
            "the harness detector currently supports XGBoost or RandomForest"
        )
    detector_dir = _harness_detector_dir(
        cfg, dataset, classifier, seed, feature_group
    )
    summary_path = detector_dir / "lomo_results.json"
    if summary_path.exists() and not force:
        print(f"[SKIP] harness detector exists: {summary_path}")
        return summary_path

    split_rows = {
        split: _harness_rows(cfg, dataset, split)
        for split in SPLITS
    }
    harness_encoder = LabelEncoder().fit(list(HARNESSES))
    fold_results = []
    all_true: list[int] = []
    all_pred: list[int] = []
    all_probabilities: list[list[float]] = []
    all_predictions: list[dict[str, Any]] = []

    for fold_index, held_out_agent in enumerate(cfg["agents"]):
        train_rows = [
            row for row in split_rows["train"]
            if row["agent_id"] != held_out_agent
        ]
        val_rows = [
            row for row in split_rows["val"]
            if row["agent_id"] != held_out_agent
        ]
        test_rows = [
            row for row in split_rows["test"]
            if row["agent_id"] == held_out_agent
        ]
        X_train, _, _, feature_names = _load_examples(
            train_rows, False, cfg, feature_group
        )
        X_val, _, _, val_feature_names = _load_examples(
            val_rows, False, cfg, feature_group
        )
        X_test, _, _, test_feature_names = _load_examples(
            test_rows, False, cfg, feature_group
        )
        if feature_names != val_feature_names or feature_names != test_feature_names:
            raise RuntimeError("harness-detector feature schema mismatch")
        y_train = harness_encoder.transform(
            [row["harness"] for row in train_rows]
        )
        y_val = harness_encoder.transform(
            [row["harness"] for row in val_rows]
        )
        y_test = harness_encoder.transform(
            [row["harness"] for row in test_rows]
        )
        if set(y_train) != {0, 1} or set(y_val) != {0, 1} or set(y_test) != {0, 1}:
            raise RuntimeError(
                f"{held_out_agent}: each split must contain both harness labels"
            )

        fold_seed = seed + fold_index
        _set_seed(fold_seed)
        if classifier == "XGBoost":
            if XGBClassifier is None:
                raise RuntimeError("xgboost is not installed")
            device = xgb_device or cfg["classifiers"]["xgboost"].get(
                "device", "cuda"
            )
            base = XGBClassifier(
                tree_method="hist",
                device=device,
                eval_metric="logloss",
                random_state=fold_seed,
                verbosity=0,
            )
            if quick:
                best_params = {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 4,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                }
                model = base.set_params(**best_params).fit(X_train, y_train)
            else:
                search = RandomizedSearchCV(
                    base,
                    XGB_PARAM_DIST,
                    n_iter=int(cfg["classifiers"]["xgboost"].get(
                        "search_iterations", 40
                    )),
                    cv=3,
                    scoring="f1_macro",
                    n_jobs=1,
                    refit=True,
                    random_state=fold_seed,
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
        else:
            if quick or not cfg["classifiers"]["random_forest"].get(
                "search", True
            ):
                best_params = {"n_estimators": 100, "max_depth": 15}
                model = RandomForestClassifier(
                    **best_params,
                    random_state=fold_seed,
                    n_jobs=int(cfg["classifiers"].get("cpu_jobs", 8)),
                ).fit(X_train, y_train)
            else:
                search = GridSearchCV(
                    RandomForestClassifier(random_state=fold_seed),
                    RF_PARAM_GRID,
                    cv=3,
                    scoring="f1_macro",
                    n_jobs=int(cfg["classifiers"].get("cpu_jobs", 8)),
                    refit=True,
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_

        y_val_pred = np.asarray(model.predict(X_val))
        y_test_pred = np.asarray(model.predict(X_test))
        val_probabilities = model.predict_proba(X_val)
        probabilities = (
            model.predict_proba(X_test)
            if hasattr(model, "predict_proba")
            else None
        )
        fold_metrics = _classification_metrics(
            y_test, y_test_pred, list(harness_encoder.classes_)
        )
        fold_metrics.update(
            _binary_ranking_metrics(
                y_test, probabilities, list(harness_encoder.classes_)
            )
        )
        fold_dir = detector_dir / f"held_out={held_out_agent}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with (fold_dir / "model.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "schema_version": 1,
                    "task": "binary_harness_detection",
                    "classifier": classifier,
                    "dataset": dataset,
                    "held_out_agent": held_out_agent,
                    "seed": fold_seed,
                    "feature_group": feature_group,
                    "classes": list(harness_encoder.classes_),
                    "feature_names": feature_names,
                    "best_params": best_params,
                    "model": model,
                },
                handle,
            )
        fold_predictions = []
        for index, row in enumerate(test_rows):
            prediction = {
                "episode_id": row["episode_id"],
                "task_id": row["task_id"],
                "agent_id": row["agent_id"],
                "held_out_agent": held_out_agent,
                "true_harness": row["harness"],
                "predicted_harness": harness_encoder.inverse_transform(
                    [int(y_test_pred[index])]
                )[0],
                "dataset": dataset,
                "split": "test",
                "classifier": classifier,
                "classifier_seed": seed,
                "feature_group": feature_group,
            }
            if probabilities is not None:
                prediction["probabilities"] = {
                    class_name: float(probabilities[index][class_index])
                    for class_index, class_name in enumerate(
                        harness_encoder.classes_
                    )
                }
            fold_predictions.append(prediction)
        _write_frozen_jsonl(
            fold_dir / "predictions.jsonl", fold_predictions, force
        )
        fold_result = {
            "held_out_agent": held_out_agent,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_test": len(test_rows),
            "best_params": best_params,
            "validation": _classification_metrics(
                y_val, y_val_pred, list(harness_encoder.classes_)
            ),
            "test": fold_metrics,
        }
        fold_result["validation"].update(
            _binary_ranking_metrics(
                y_val, val_probabilities, list(harness_encoder.classes_)
            )
        )
        (fold_dir / "results.json").write_text(
            json.dumps(fold_result, indent=2) + "\n"
        )
        fold_results.append(fold_result)
        all_true.extend(int(value) for value in y_test)
        all_pred.extend(int(value) for value in y_test_pred)
        all_probabilities.extend(probabilities.tolist())
        all_predictions.extend(fold_predictions)
        print(
            f"[HARNESS LOMO] {classifier} {dataset} "
            f"held_out={held_out_agent} macro_f1={fold_metrics['macro_f1']:.3f}"
        )

    pooled_metrics = _classification_metrics(
        np.asarray(all_true),
        np.asarray(all_pred),
        list(harness_encoder.classes_),
    )
    pooled_metrics.update(
        _binary_ranking_metrics(
            np.asarray(all_true),
            np.asarray(all_probabilities),
            list(harness_encoder.classes_),
        )
    )
    macro_f1s = [fold["test"]["macro_f1"] for fold in fold_results]
    aurocs = [fold["test"]["auroc"] for fold in fold_results]
    summary = {
        "timestamp": _utc_now(),
        "task": "binary_harness_detection",
        "protocol": "leave_one_model_out",
        "classifier": classifier,
        "dataset": dataset,
        "seed": seed,
        "feature_group": feature_group,
        "quick": quick,
        "class_names": list(harness_encoder.classes_),
        "folds": fold_results,
        "fold_macro_f1_mean": float(np.mean(macro_f1s)),
        "fold_macro_f1_std": float(np.std(macro_f1s)),
        "fold_auroc_mean": float(np.mean(aurocs)),
        "fold_auroc_std": float(np.std(aurocs)),
        "pooled_test": pooled_metrics,
    }
    detector_dir.mkdir(parents=True, exist_ok=True)
    _write_frozen_jsonl(
        detector_dir / "predictions.jsonl", all_predictions, force
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[HARNESS LOMO] {classifier} {dataset} "
        f"mean_macro_f1={summary['fold_macro_f1_mean']:.3f} "
        f"mean_auroc={summary['fold_auroc_mean']:.3f} "
        f"pooled_macro_f1={pooled_metrics['macro_f1']:.3f}"
    )
    return summary_path


def run_grid(
    cfg: dict[str, Any],
    datasets: Iterable[str],
    classifier: str,
    seed: int,
    *,
    feature_group: str = "full",
    quick: bool,
    xgb_device: str | None,
    force: bool,
) -> None:
    for dataset in datasets:
        for train_policy, eval_policies in GRID.items():
            train_model(
                cfg,
                dataset,
                train_policy,
                classifier,
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
                    classifier,
                    seed,
                    feature_group=feature_group,
                    force=force,
                )
    summarize_results(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and run frozen cross-harness classifier experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "final_6model.yaml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--force", action="store_true")
    subparsers.add_parser("summarize")

    def add_model_arguments(subparser, include_eval: bool = False):
        subparser.add_argument("--dataset", required=True)
        subparser.add_argument("--train-policy", choices=POLICIES, required=True)
        if include_eval:
            subparser.add_argument("--eval-policy", choices=POLICIES, required=True)
        subparser.add_argument(
            "--classifier",
            choices=["XGBoost", "RandomForest", "LSTM"],
            default=None,
        )
        subparser.add_argument("--seed", type=int, default=None)
        subparser.add_argument("--feature-group", default="full")
        subparser.add_argument("--force", action="store_true")

    train_parser = subparsers.add_parser("train")
    add_model_arguments(train_parser)
    train_parser.add_argument("--quick", action="store_true")
    train_parser.add_argument("--xgb-device", choices=["cpu", "cuda"], default=None)

    evaluate_parser = subparsers.add_parser("evaluate")
    add_model_arguments(evaluate_parser, include_eval=True)

    grid = subparsers.add_parser("run-grid")
    grid.add_argument("--datasets", nargs="+", default=None)
    grid.add_argument(
        "--classifier",
        choices=["XGBoost", "RandomForest", "LSTM"],
        default=None,
    )
    grid.add_argument("--seed", type=int, default=None)
    grid.add_argument("--feature-group", default="full")
    grid.add_argument("--quick", action="store_true")
    grid.add_argument("--xgb-device", choices=["cpu", "cuda"], default=None)
    grid.add_argument("--force", action="store_true")
    ablation = subparsers.add_parser("run-ablation")
    ablation.add_argument("--datasets", nargs="+", default=None)
    ablation.add_argument(
        "--classifier",
        choices=["XGBoost", "RandomForest"],
        default=None,
    )
    ablation.add_argument("--seed", type=int, default=None)
    ablation.add_argument(
        "--feature-groups",
        nargs="+",
        default=["full", "timing_only", "non_timing"],
    )
    ablation.add_argument("--quick", action="store_true")
    ablation.add_argument(
        "--xgb-device", choices=["cpu", "cuda"], default=None
    )
    ablation.add_argument("--force", action="store_true")
    detector = subparsers.add_parser("harness-detector")
    detector.add_argument("--datasets", nargs="+", default=None)
    detector.add_argument(
        "--classifier",
        choices=["XGBoost", "RandomForest"],
        default="XGBoost",
    )
    detector.add_argument("--seed", type=int, default=None)
    detector.add_argument("--feature-group", default="full")
    detector.add_argument("--quick", action="store_true")
    detector.add_argument(
        "--xgb-device", choices=["cpu", "cuda"], default=None
    )
    detector.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.command in {"audit", "prepare"}:
        records = scan_inventory(cfg)
        print_audit(cfg, records)
        if args.command == "prepare":
            prepare_manifests(cfg, records, force=args.force)
        return 0
    if args.command == "summarize":
        summarize_results(cfg)
        return 0

    classifier = args.classifier or cfg["classifiers"]["primary"]
    seed = (
        args.seed
        if args.seed is not None
        else int(cfg["experiment"]["classifier_seed"])
    )
    if args.command == "train":
        train_model(
            cfg,
            args.dataset,
            args.train_policy,
            classifier,
            seed,
            feature_group=args.feature_group,
            quick=args.quick,
            xgb_device=args.xgb_device,
            force=args.force,
        )
    elif args.command == "evaluate":
        evaluate_model(
            cfg,
            args.dataset,
            args.train_policy,
            args.eval_policy,
            classifier,
            seed,
            feature_group=args.feature_group,
            force=args.force,
        )
    elif args.command == "run-grid":
        run_grid(
            cfg,
            args.datasets or cfg["datasets"].keys(),
            classifier,
            seed,
            feature_group=args.feature_group,
            quick=args.quick,
            xgb_device=args.xgb_device,
            force=args.force,
        )
    elif args.command == "run-ablation":
        for feature_group in args.feature_groups:
            run_grid(
                cfg,
                args.datasets or cfg["datasets"].keys(),
                classifier,
                seed,
                feature_group=feature_group,
                quick=args.quick,
                xgb_device=args.xgb_device,
                force=args.force,
            )
    elif args.command == "harness-detector":
        for dataset in args.datasets or cfg["datasets"].keys():
            run_harness_detector_lomo(
                cfg,
                dataset,
                classifier,
                seed,
                feature_group=args.feature_group,
                quick=args.quick,
                xgb_device=args.xgb_device,
                force=args.force,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
