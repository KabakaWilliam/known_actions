"""Load validated per-held-out-model open-set AUROC intervals for figures."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from pathlib import Path


def _format_names(names: Collection[str]) -> str:
    return ", ".join(repr(name) for name in sorted(names)) or "(none)"


def load_open_set_intervals(
    path: Path,
    classifier: str,
    expected_agents_by_tag: Mapping[str, Collection[str]] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return AUROC intervals keyed by open-set experiment tag and agent.

    Dataset-to-figure mappings come from each aggregate dataset's ``tag`` field,
    rather than from a second hard-coded registry. When ``expected_agents_by_tag``
    is supplied, only those tags are returned and each aggregate agent set must
    exactly match the corresponding set represented by the original figure data.
    """
    try:
        with path.open() as f:
            payload = json.load(f)
    except OSError as exc:
        raise ValueError(f"{path}: could not read open-set stats: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported or missing schema_version")
    default_seed_count = payload.get("default_classifier_seed_count")
    if (
        isinstance(default_seed_count, bool)
        or not isinstance(default_seed_count, int)
        or default_seed_count < 5
    ):
        raise ValueError(
            f"{path}: default_classifier_seed_count must be an integer >= 5"
        )
    classifier_seed_count = payload.get("classifier_seed_count")
    aggregate_seeds = payload.get("classifier_seeds")
    if (
        isinstance(classifier_seed_count, bool)
        or classifier_seed_count != default_seed_count
    ):
        raise ValueError(
            f"{path}: classifier_seed_count must match "
            "default_classifier_seed_count"
        )
    if (
        not isinstance(aggregate_seeds, list)
        or len(aggregate_seeds) != classifier_seed_count
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in aggregate_seeds
        )
        or len(set(aggregate_seeds)) != classifier_seed_count
    ):
        raise ValueError(
            f"{path}: classifier_seeds must contain exactly "
            f"{classifier_seed_count} unique integers"
        )
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"{path}: missing non-empty datasets object")

    datasets_by_tag: dict[str, tuple[str, dict]] = {}
    for dataset_key, dataset in datasets.items():
        location = f"{path}: dataset '{dataset_key}'"
        if not isinstance(dataset_key, str) or not dataset_key.strip():
            raise ValueError(f"{path}: dataset keys must be non-empty strings")
        if not isinstance(dataset, dict):
            raise ValueError(f"{location} must be an object")
        tag = dataset.get("tag")
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"{location}.tag must be a non-empty string")
        if tag in datasets_by_tag:
            other_key = datasets_by_tag[tag][0]
            raise ValueError(
                f"{path}: duplicate dataset tag '{tag}' in datasets "
                f"'{other_key}' and '{dataset_key}'"
            )
        datasets_by_tag[tag] = (dataset_key, dataset)

    requested_tags = (
        list(expected_agents_by_tag)
        if expected_agents_by_tag is not None
        else list(datasets_by_tag)
    )
    missing_tags = set(requested_tags) - set(datasets_by_tag)
    if missing_tags:
        raise ValueError(
            f"{path}: no open-set dataset entry for figure tag(s): "
            f"{_format_names(missing_tags)}"
        )

    intervals: dict[str, dict[str, dict[str, float]]] = {}
    for tag in requested_tags:
        dataset_key, dataset = datasets_by_tag[tag]
        held_out_models = dataset.get("held_out_models")
        if not isinstance(held_out_models, dict) or not held_out_models:
            raise ValueError(
                f"{path}: dataset '{dataset_key}' is missing non-empty "
                "held_out_models"
            )

        by_agent: dict[str, dict[str, float]] = {}
        for agent, leaf in held_out_models.items():
            location = (
                f"{path}: dataset '{dataset_key}', held-out model '{agent}'"
            )
            if not isinstance(agent, str) or not agent.strip():
                raise ValueError(
                    f"{path}: held_out_models keys must be non-empty strings"
                )
            if not isinstance(leaf, dict):
                raise ValueError(f"{location} must be an object")
            seed_count = leaf.get("n_classifier_seeds")
            seeds = leaf.get("classifier_seeds")
            if (
                isinstance(seed_count, bool)
                or not isinstance(seed_count, int)
                or seed_count < 5
            ):
                raise ValueError(
                    f"{location}.n_classifier_seeds must be an integer >= 5"
                )
            if seed_count != default_seed_count:
                raise ValueError(
                    f"{location}.n_classifier_seeds does not match the "
                    "aggregate default_classifier_seed_count"
                )
            if (
                not isinstance(seeds, list)
                or len(seeds) != seed_count
                or any(
                    isinstance(seed, bool) or not isinstance(seed, int)
                    for seed in seeds
                )
                or len(set(seeds)) != seed_count
            ):
                raise ValueError(
                    f"{location}.classifier_seeds must contain exactly "
                    f"{seed_count} unique integers"
                )
            if seeds != aggregate_seeds:
                raise ValueError(
                    f"{location}.classifier_seeds do not match the aggregate "
                    "classifier_seeds"
                )
            models = leaf.get("models")
            if not isinstance(models, dict):
                raise ValueError(f"{location} is missing models")
            model = models.get(classifier)
            if not isinstance(model, dict):
                raise ValueError(
                    f"{location} has no '{classifier}' results"
                )
            auroc = model.get("auroc")
            if not isinstance(auroc, dict):
                raise ValueError(
                    f"{location}, classifier '{classifier}' is missing auroc"
                )
            confidence_interval = auroc.get("confidence_interval")
            try:
                estimate = float(auroc["estimate"])
                lower = float(confidence_interval["lower"])
                upper = float(confidence_interval["upper"])
                level = float(confidence_interval["level"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{location}, classifier '{classifier}' has an incomplete "
                    "AUROC estimate or confidence interval"
                ) from exc
            method = (
                confidence_interval.get("method")
                if isinstance(confidence_interval, dict)
                else None
            )
            if not isinstance(method, str) or not method.strip():
                raise ValueError(
                    f"{location}, classifier '{classifier}' confidence "
                    "interval is missing method"
                )

            values = (lower, estimate, upper, level)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"{location}, classifier '{classifier}' contains a "
                    "non-finite interval value"
                )
            if not 0.0 <= lower <= estimate <= upper <= 1.0:
                raise ValueError(
                    f"{location}, classifier '{classifier}' interval must "
                    "satisfy 0 <= lower <= estimate <= upper <= 1"
                )
            if not 0.0 < level < 1.0:
                raise ValueError(
                    f"{location}, classifier '{classifier}' confidence level "
                    "must be in (0, 1)"
                )

            per_seed = auroc.get("per_seed")
            if not isinstance(per_seed, list) or len(per_seed) != seed_count:
                raise ValueError(
                    f"{location}, classifier '{classifier}' auroc.per_seed "
                    f"must contain exactly {seed_count} rows"
                )
            per_seed_values: list[float] = []
            per_seed_ids: list[int] = []
            for seed_position, seed_row in enumerate(per_seed):
                seed_location = (
                    f"{location}, classifier '{classifier}' "
                    f"auroc.per_seed[{seed_position}]"
                )
                if not isinstance(seed_row, dict):
                    raise ValueError(f"{seed_location} must be an object")
                seed = seed_row.get("seed")
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise ValueError(f"{seed_location}.seed must be an integer")
                try:
                    seed_auroc = float(seed_row["auroc"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{seed_location}.auroc must be numeric"
                    ) from exc
                if not math.isfinite(seed_auroc) or not 0.0 <= seed_auroc <= 1.0:
                    raise ValueError(
                        f"{seed_location}.auroc must be finite and in [0, 1]"
                    )
                per_seed_ids.append(seed)
                per_seed_values.append(seed_auroc)
            if per_seed_ids != seeds:
                raise ValueError(
                    f"{location}, classifier '{classifier}' per-seed IDs do "
                    "not match classifier_seeds"
                )
            seed_mean = math.fsum(per_seed_values) / len(per_seed_values)
            if not math.isclose(seed_mean, estimate, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    f"{location}, classifier '{classifier}' estimate is not "
                    "the arithmetic mean of per_seed AUROCs"
                )

            for count_key in ("n_known", "n_unknown"):
                count = auroc.get(count_key)
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 1
                ):
                    raise ValueError(
                        f"{location}, classifier '{classifier}' "
                        f"auroc.{count_key} must be a positive integer"
                    )

            by_agent[agent] = {
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
                "level": level,
            }

        if expected_agents_by_tag is not None:
            expected_names = set(expected_agents_by_tag[tag])
            actual_names = set(by_agent)
            if actual_names != expected_names:
                missing = expected_names - actual_names
                unexpected = actual_names - expected_names
                raise ValueError(
                    f"{path}: held-out model names for figure tag '{tag}' do "
                    f"not match; missing from stats: {_format_names(missing)}; "
                    f"unexpected in stats: {_format_names(unexpected)}"
                )

        intervals[tag] = by_agent

    return intervals
