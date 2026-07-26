"""Load validated per-class closed-set intervals for figure error bars."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from pathlib import Path


DATASET_KEY_BY_EXPERIMENT_TAG = {
    "wiki_ood_all": "wiki",
    "frames_ood_all": "frames",
    "webshop_ood_all": "webshop",
    "deepshop_ood_all": "deepshop",
}


def _format_names(names: Collection[str]) -> str:
    return ", ".join(repr(name) for name in sorted(names)) or "(none)"


def load_closed_set_intervals(
    path: Path,
    classifier: str,
    expected_classes_by_tag: Mapping[str, Collection[str]] | None = None,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Return validated per-class intervals keyed by experiment tag and class.

    When ``expected_classes_by_tag`` is supplied, only those experiment tags are
    loaded and each stats class set must exactly match the corresponding figure
    class set. This prevents a missing or misspelled class from being rendered
    as a zero-valued bar.
    """
    try:
        with path.open() as f:
            payload = json.load(f)
    except OSError as exc:
        raise ValueError(f"{path}: could not read closed-set stats: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported or missing schema_version")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"{path}: missing datasets object")

    requested_tags = (
        list(expected_classes_by_tag)
        if expected_classes_by_tag is not None
        else list(DATASET_KEY_BY_EXPERIMENT_TAG)
    )
    unsupported_tags = [
        tag for tag in requested_tags
        if tag not in DATASET_KEY_BY_EXPERIMENT_TAG
    ]
    if unsupported_tags:
        raise ValueError(
            f"{path}: no closed-set dataset mapping for figure tag(s): "
            f"{_format_names(unsupported_tags)}"
        )

    intervals: dict[str, dict[str, dict[str, float | int]]] = {}
    for tag in requested_tags:
        dataset_key = DATASET_KEY_BY_EXPERIMENT_TAG[tag]
        dataset = datasets.get(dataset_key)
        if not isinstance(dataset, dict):
            raise ValueError(
                f"{path}: figure tag '{tag}' is missing dataset '{dataset_key}'"
            )

        models = dataset.get("models")
        if not isinstance(models, dict):
            raise ValueError(f"{path}: dataset '{dataset_key}' is missing models")
        model = models.get(classifier)
        if not isinstance(model, dict):
            raise ValueError(
                f"{path}: dataset '{dataset_key}' has no '{classifier}' results"
            )
        macro_f1 = model.get("macro_f1")
        if not isinstance(macro_f1, dict):
            raise ValueError(
                f"{path}: dataset '{dataset_key}', classifier '{classifier}' "
                "is missing macro_f1"
            )
        per_class = macro_f1.get("per_class")
        if not isinstance(per_class, list) or not per_class:
            raise ValueError(
                f"{path}: missing non-empty macro_f1.per_class list for "
                f"dataset '{dataset_key}', classifier '{classifier}'"
            )

        by_class: dict[str, dict[str, float | int]] = {}
        seen_indices: set[int] = set()
        for position, row in enumerate(per_class):
            location = (
                f"{path}: dataset '{dataset_key}' macro_f1.per_class[{position}]"
            )
            if not isinstance(row, dict):
                raise ValueError(f"{location} must be an object")

            class_index = row.get("class_index")
            class_name = row.get("class_name")
            if (
                isinstance(class_index, bool)
                or not isinstance(class_index, int)
                or class_index < 0
            ):
                raise ValueError(
                    f"{location}.class_index must be a non-negative integer"
                )
            if not isinstance(class_name, str) or not class_name.strip():
                raise ValueError(
                    f"{location}.class_name must be a non-empty string"
                )
            if class_name in by_class:
                raise ValueError(
                    f"{path}: duplicate class_name '{class_name}' in "
                    f"dataset '{dataset_key}'"
                )
            if class_index in seen_indices:
                raise ValueError(
                    f"{path}: duplicate class_index {class_index} in "
                    f"dataset '{dataset_key}'"
                )

            ci = row.get("confidence_interval")
            try:
                estimate = float(row["estimate"])
                lower = float(ci["lower"])
                upper = float(ci["upper"])
                level = float(ci["level"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{location} has an incomplete estimate or confidence interval"
                ) from exc

            values = (lower, estimate, upper, level)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{location} contains a non-finite interval value")
            if not 0.0 <= lower <= estimate <= upper <= 1.0:
                raise ValueError(
                    f"{location} interval must satisfy "
                    "0 <= lower <= estimate <= upper <= 1"
                )
            if not 0.0 < level < 1.0:
                raise ValueError(
                    f"{location} confidence level must be in (0, 1)"
                )

            seen_indices.add(class_index)
            by_class[class_name] = {
                "class_index": class_index,
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
                "level": level,
            }

        expected_indices = set(range(len(per_class)))
        if seen_indices != expected_indices:
            raise ValueError(
                f"{path}: dataset '{dataset_key}' class_index values must be "
                f"contiguous from 0 through {len(per_class) - 1}"
            )

        if expected_classes_by_tag is not None:
            expected_names = set(expected_classes_by_tag[tag])
            actual_names = set(by_class)
            if actual_names != expected_names:
                missing = expected_names - actual_names
                unexpected = actual_names - expected_names
                raise ValueError(
                    f"{path}: class names for figure tag '{tag}' do not match; "
                    f"missing from stats: {_format_names(missing)}; "
                    f"unexpected in stats: {_format_names(unexpected)}"
                )

        intervals[tag] = by_class

    return intervals
