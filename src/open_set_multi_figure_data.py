"""Load validated multi-model open-set AUROC intervals for figures.

Schema version 2 stores one result for every evaluated held-out subset.  This
loader deliberately preserves that granularity: it never computes an interval
across subsets or replaces the per-subset trace-bootstrap intervals with an
interval for a mean.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from open_set_subset_design import canonical_subset_id


_SELECTION_MODES = {"exhaustive", "balanced_sample"}
_PROTOCOL_NAME = "fixed_test_population_pooled_unknown_v1"
_UNKNOWN_POOLING = "all_held_out_models_one_trace_weighted_binary_class"
_BOOTSTRAP_STRATA = [
    "known_test_trace",
    "pooled_unknown_test_trace",
]
_BOOTSTRAP_SAMPLING = (
    "independent_nonparametric_with_replacement_within_stratum"
)
_CI_METHOD = (
    "paired_stratified_percentile_bootstrap_over_"
    "known_test_and_pooled_unknown_test_traces"
)


def _positive_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _nonnegative_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return value


def _unique_names(
    value: object,
    location: str,
    *,
    require_sorted: bool,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    if not allow_empty and not value:
        raise ValueError(f"{location} must not be empty")
    if any(
        not isinstance(name, str)
        or not name.strip()
        or name != name.strip()
        for name in value
    ):
        raise ValueError(
            f"{location} must contain non-empty strings without surrounding "
            "whitespace"
        )
    if len(set(value)) != len(value):
        raise ValueError(f"{location} must contain unique model IDs")
    if require_sorted and value != sorted(value):
        raise ValueError(f"{location} must be sorted")
    return list(value)


def _size_mapping(
    value: object,
    location: str,
) -> dict[int, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object keyed by holdout size")
    normalized: dict[int, object] = {}
    for raw_size, item in value.items():
        try:
            size = int(raw_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{location} has invalid holdout-size key {raw_size!r}"
            ) from exc
        if size < 1 or str(size) != str(raw_size):
            raise ValueError(
                f"{location} has invalid holdout-size key {raw_size!r}"
            )
        if size in normalized:
            raise ValueError(f"{location} contains duplicate size {size}")
        normalized[size] = item
    return normalized


def _validate_seeds(
    raw_seeds: object,
    raw_count: object,
    location: str,
) -> list[int]:
    count = _positive_int(raw_count, f"{location}.classifier_seed_count")
    if count < 5:
        raise ValueError(
            f"{location}.classifier_seed_count must be at least 5"
        )
    if (
        not isinstance(raw_seeds, list)
        or len(raw_seeds) != count
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in raw_seeds
        )
        or len(set(raw_seeds)) != count
    ):
        raise ValueError(
            f"{location}.classifier_seeds must contain exactly {count} "
            "unique integers"
        )
    return list(raw_seeds)


def _validate_interval(
    auroc: object,
    *,
    seeds: list[int],
    confidence_level: float,
    location: str,
) -> dict[str, object]:
    if not isinstance(auroc, dict):
        raise ValueError(f"{location} must be an object")
    interval = auroc.get("confidence_interval")
    if not isinstance(interval, dict):
        raise ValueError(f"{location}.confidence_interval must be an object")
    try:
        estimate = float(auroc["estimate"])
        lower = float(interval["lower"])
        upper = float(interval["upper"])
        level = float(interval["level"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{location} has an incomplete estimate or confidence interval"
        ) from exc
    if not all(math.isfinite(item) for item in (lower, estimate, upper, level)):
        raise ValueError(f"{location} interval values must be finite")
    if not 0.0 <= estimate <= 1.0:
        raise ValueError(f"{location}.estimate must be in [0, 1]")
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError(
            f"{location} interval must satisfy 0 <= lower <= upper <= 1"
        )
    if not 0.0 < level < 1.0:
        raise ValueError(f"{location} confidence level must be in (0, 1)")
    if not math.isclose(
        level,
        confidence_level,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{location} confidence level does not match bootstrap metadata"
        )
    method = interval.get("method")
    if method != _CI_METHOD:
        raise ValueError(
            f"{location}.confidence_interval.method must be {_CI_METHOD!r}"
        )

    per_seed = auroc.get("per_seed")
    if not isinstance(per_seed, list) or len(per_seed) != len(seeds):
        raise ValueError(
            f"{location}.per_seed must contain exactly {len(seeds)} rows"
        )
    per_seed_ids: list[int] = []
    per_seed_values: list[float] = []
    for index, row in enumerate(per_seed):
        row_location = f"{location}.per_seed[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{row_location} must be an object")
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{row_location}.seed must be an integer")
        try:
            value = float(row["auroc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{row_location}.auroc must be numeric") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{row_location}.auroc must be finite and in [0, 1]"
            )
        per_seed_ids.append(seed)
        per_seed_values.append(value)
    if per_seed_ids != seeds:
        raise ValueError(
            f"{location}.per_seed IDs do not match classifier_seeds"
        )
    seed_mean = math.fsum(per_seed_values) / len(per_seed_values)
    if not math.isclose(seed_mean, estimate, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"{location}.estimate is not the arithmetic mean of per-seed "
            "AUROCs"
        )

    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "level": level,
        "method": method,
    }


def load_open_set_multi_intervals(
    path: Path,
    classifier: str = "XGBoost",
) -> dict[str, object]:
    """Return normalized per-subset AUROC intervals from schema version 2.

    The returned object keeps datasets keyed by their aggregate dataset key and
    holdout sizes keyed by integers.  Within each size, ``subsets`` is a list
    sorted by the canonical held-out-model tuple.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise ValueError(f"{path}: could not read open-set stats: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    if payload.get("schema_version") != 2:
        raise ValueError(f"{path}: unsupported or missing schema_version")
    if not isinstance(classifier, str) or not classifier.strip():
        raise ValueError("classifier must be a non-empty string")

    seeds = _validate_seeds(
        payload.get("classifier_seeds"),
        payload.get("classifier_seed_count"),
        str(path),
    )

    bootstrap = payload.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError(f"{path}: bootstrap must be an object")
    _positive_int(
        bootstrap.get("replicates"),
        f"{path}: bootstrap.replicates",
    )
    try:
        confidence_level = float(bootstrap["confidence_level"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: bootstrap.confidence_level must be numeric"
        ) from exc
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"{path}: bootstrap.confidence_level must be in (0, 1)"
        )
    if (
        bootstrap.get("unit") != "evaluation_trace"
        or bootstrap.get("strata") != _BOOTSTRAP_STRATA
        or bootstrap.get("sampling") != _BOOTSTRAP_SAMPLING
        or bootstrap.get("paired_across_classifier_seeds") is not True
        or bootstrap.get("interval") != "percentile"
        or bootstrap.get("scope") != "pointwise_per_held_out_subset"
    ):
        raise ValueError(
            f"{path}: bootstrap metadata does not describe the expected "
            "pointwise pooled-unknown trace bootstrap"
        )
    bootstrap_seed = _nonnegative_int(
        bootstrap.get("seed"),
        f"{path}: bootstrap.seed",
    )
    run_fingerprint = payload.get("run_fingerprint")
    if not isinstance(run_fingerprint, str) or not run_fingerprint:
        raise ValueError(f"{path}: run_fingerprint must be a non-empty string")
    protocol = payload.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("name") != _PROTOCOL_NAME
        or protocol.get("unknown_pooling") != _UNKNOWN_POOLING
        or protocol.get("evaluation_population")
        != "fixed_valid_test_traces_for_both_known_and_unknown_models"
    ):
        raise ValueError(f"{path}: invalid pooled open-set protocol metadata")

    design = payload.get("subset_design")
    if not isinstance(design, dict):
        raise ValueError(f"{path}: subset_design must be an object")
    universe = _unique_names(
        design.get("model_universe"),
        f"{path}: subset_design.model_universe",
        require_sorted=True,
    )
    raw_sizes = design.get("holdout_sizes")
    if (
        not isinstance(raw_sizes, list)
        or not raw_sizes
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in raw_sizes
        )
        or len(set(raw_sizes)) != len(raw_sizes)
        or raw_sizes != sorted(raw_sizes)
    ):
        raise ValueError(
            f"{path}: subset_design.holdout_sizes must be a sorted list of "
            "unique positive integers"
        )
    if any(size >= len(universe) for size in raw_sizes):
        raise ValueError(
            f"{path}: every holdout size must be smaller than the model universe"
        )
    holdout_sizes = list(raw_sizes)

    possible_raw = _size_mapping(
        design.get("possible_counts"),
        f"{path}: subset_design.possible_counts",
    )
    if set(possible_raw) != set(holdout_sizes):
        raise ValueError(
            f"{path}: subset_design.possible_counts keys must match "
            "holdout_sizes"
        )
    possible_counts: dict[int, int] = {}
    for size in holdout_sizes:
        count = _positive_int(
            possible_raw[size],
            f"{path}: subset_design.possible_counts[{size}]",
        )
        expected = math.comb(len(universe), size)
        if count != expected:
            raise ValueError(
                f"{path}: subset_design.possible_counts[{size}] must equal "
                f"C({len(universe)}, {size}) = {expected}"
            )
        possible_counts[size] = count

    max_subsets = design.get("max_subsets_per_size")
    if max_subsets is not None:
        _positive_int(
            max_subsets,
            f"{path}: subset_design.max_subsets_per_size",
        )
    subset_seed = _nonnegative_int(
        design.get("subset_seed"),
        f"{path}: subset_design.subset_seed",
    )

    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise ValueError(f"{path}: datasets must be a non-empty object")

    normalized_datasets: dict[str, dict[str, object]] = {}
    universe_set = set(universe)
    for dataset_key, dataset in raw_datasets.items():
        dataset_location = f"{path}: dataset {dataset_key!r}"
        if (
            not isinstance(dataset_key, str)
            or not dataset_key.strip()
            or not isinstance(dataset, dict)
        ):
            raise ValueError(
                f"{path}: dataset keys must be non-empty strings and values "
                "must be objects"
            )
        dataset_cache_digest = dataset.get("dataset_cache_digest")
        if (
            not isinstance(dataset_cache_digest, str)
            or not dataset_cache_digest
        ):
            raise ValueError(
                f"{dataset_location}.dataset_cache_digest must be a "
                "non-empty string"
            )
        groups = _size_mapping(
            dataset.get("holdout_sizes"),
            f"{dataset_location}.holdout_sizes",
        )
        if set(groups) != set(holdout_sizes):
            raise ValueError(
                f"{dataset_location}.holdout_sizes keys must match "
                "subset_design.holdout_sizes"
            )

        normalized_groups: dict[int, dict[str, object]] = {}
        for size in holdout_sizes:
            group = groups[size]
            group_location = f"{dataset_location}.holdout_sizes[{size}]"
            if not isinstance(group, dict):
                raise ValueError(f"{group_location} must be an object")
            n_possible = _positive_int(
                group.get("n_possible_subsets"),
                f"{group_location}.n_possible_subsets",
            )
            if n_possible != possible_counts[size]:
                raise ValueError(
                    f"{group_location}.n_possible_subsets does not match "
                    "subset_design.possible_counts"
                )
            n_evaluated = _positive_int(
                group.get("n_evaluated_subsets"),
                f"{group_location}.n_evaluated_subsets",
            )
            if n_evaluated > n_possible:
                raise ValueError(
                    f"{group_location}.n_evaluated_subsets cannot exceed "
                    "n_possible_subsets"
                )
            selection_mode = group.get("selection_mode")
            if selection_mode not in _SELECTION_MODES:
                raise ValueError(
                    f"{group_location}.selection_mode must be one of "
                    f"{sorted(_SELECTION_MODES)!r}"
                )
            if selection_mode == "exhaustive" and n_evaluated != n_possible:
                raise ValueError(
                    f"{group_location} marked exhaustive but evaluated "
                    f"{n_evaluated} of {n_possible} subsets"
                )

            inclusion = group.get("model_inclusion_counts")
            if not isinstance(inclusion, dict) or set(inclusion) != universe_set:
                raise ValueError(
                    f"{group_location}.model_inclusion_counts must have "
                    "exactly the model-universe keys"
                )
            normalized_inclusion = {
                model: _nonnegative_int(
                    inclusion[model],
                    (
                        f"{group_location}.model_inclusion_counts"
                        f"[{model!r}]"
                    ),
                )
                for model in universe
            }
            if sum(normalized_inclusion.values()) != size * n_evaluated:
                raise ValueError(
                    f"{group_location}.model_inclusion_counts must sum to "
                    "holdout_size * n_evaluated_subsets"
                )
            if selection_mode == "exhaustive":
                expected_inclusion = math.comb(len(universe) - 1, size - 1)
                if any(
                    count != expected_inclusion
                    for count in normalized_inclusion.values()
                ):
                    raise ValueError(
                        f"{group_location}.model_inclusion_counts are "
                        "inconsistent with exhaustive subset coverage"
                    )

            raw_subsets = group.get("subsets")
            if not isinstance(raw_subsets, dict):
                raise ValueError(f"{group_location}.subsets must be an object")
            if len(raw_subsets) != n_evaluated:
                raise ValueError(
                    f"{group_location}.subsets must contain exactly "
                    f"{n_evaluated} entries"
                )

            seen_model_sets: set[tuple[str, ...]] = set()
            counted_inclusion = {model: 0 for model in universe}
            normalized_subsets: list[dict[str, object]] = []
            for subset_key, leaf in raw_subsets.items():
                leaf_location = f"{group_location}.subsets[{subset_key!r}]"
                if not isinstance(subset_key, str) or not subset_key.strip():
                    raise ValueError(
                        f"{group_location}.subsets keys must be non-empty strings"
                    )
                if not isinstance(leaf, dict):
                    raise ValueError(f"{leaf_location} must be an object")
                if leaf.get("schema_version") != 2:
                    raise ValueError(
                        f"{leaf_location}.schema_version must equal 2"
                    )
                if leaf.get("run_fingerprint") != run_fingerprint:
                    raise ValueError(
                        f"{leaf_location}.run_fingerprint does not match "
                        "the aggregate"
                    )
                if leaf.get("dataset_cache_digest") != dataset_cache_digest:
                    raise ValueError(
                        f"{leaf_location}.dataset_cache_digest does not "
                        "match its dataset"
                    )
                if leaf.get("dataset_key") != dataset_key:
                    raise ValueError(
                        f"{leaf_location}.dataset_key does not match its key"
                    )
                if leaf.get("holdout_size") != size:
                    raise ValueError(
                        f"{leaf_location}.holdout_size must equal {size}"
                    )
                if leaf.get("unknown_pooling") != _UNKNOWN_POOLING:
                    raise ValueError(
                        f"{leaf_location}.unknown_pooling is invalid"
                    )
                if leaf.get("bootstrap") != bootstrap:
                    raise ValueError(
                        f"{leaf_location}.bootstrap does not match the aggregate"
                    )
                subset_id = leaf.get("subset_id")
                if subset_id != subset_key:
                    raise ValueError(
                        f"{leaf_location}.subset_id must match its object key"
                    )
                held_out = _unique_names(
                    leaf.get("held_out_models"),
                    f"{leaf_location}.held_out_models",
                    require_sorted=True,
                )
                if len(held_out) != size:
                    raise ValueError(
                        f"{leaf_location}.held_out_models length must equal "
                        f"holdout size {size}"
                    )
                if not set(held_out) <= universe_set:
                    raise ValueError(
                        f"{leaf_location}.held_out_models contains a model "
                        "outside subset_design.model_universe"
                    )
                expected_subset_id = canonical_subset_id(held_out)
                if subset_id != expected_subset_id:
                    raise ValueError(
                        f"{leaf_location}.subset_id is not the canonical ID "
                        "for held_out_models"
                    )
                held_out_tuple = tuple(held_out)
                if held_out_tuple in seen_model_sets:
                    raise ValueError(
                        f"{group_location} contains duplicate held-out model "
                        f"subset {held_out!r}"
                    )
                seen_model_sets.add(held_out_tuple)
                for model in held_out:
                    counted_inclusion[model] += 1

                known = _unique_names(
                    leaf.get("known_models"),
                    f"{leaf_location}.known_models",
                    require_sorted=True,
                )
                expected_known = sorted(universe_set - set(held_out))
                if known != expected_known:
                    raise ValueError(
                        f"{leaf_location}.known_models must be the sorted "
                        "complement of held_out_models"
                    )
                n_known = _positive_int(
                    leaf.get("n_known_traces"),
                    f"{leaf_location}.n_known_traces",
                )
                n_unknown = _positive_int(
                    leaf.get("n_unknown_traces"),
                    f"{leaf_location}.n_unknown_traces",
                )
                leaf_seeds = _validate_seeds(
                    leaf.get("classifier_seeds"),
                    leaf.get("classifier_seed_count"),
                    leaf_location,
                )
                alternate_seed_count = leaf.get("n_classifier_seeds")
                if (
                    alternate_seed_count is not None
                    and alternate_seed_count != len(leaf_seeds)
                ):
                    raise ValueError(
                        f"{leaf_location}.n_classifier_seeds does not match "
                        "classifier_seed_count"
                    )
                if leaf_seeds != seeds:
                    raise ValueError(
                        f"{leaf_location}.classifier_seeds do not match the "
                        "aggregate classifier_seeds"
                    )
                models = leaf.get("models")
                if not isinstance(models, dict):
                    raise ValueError(f"{leaf_location}.models must be an object")
                model = models.get(classifier)
                if not isinstance(model, dict):
                    raise ValueError(
                        f"{leaf_location} has no {classifier!r} results"
                    )
                interval = _validate_interval(
                    model.get("auroc"),
                    seeds=seeds,
                    confidence_level=confidence_level,
                    location=f"{leaf_location}.models[{classifier!r}].auroc",
                )
                summary = model["auroc"]
                if (
                    summary.get("n_known") != n_known
                    or summary.get("n_unknown") != n_unknown
                ):
                    raise ValueError(
                        f"{leaf_location} AUROC trace counts do not match "
                        "leaf metadata"
                    )
                normalized_subsets.append(
                    {
                        "subset_id": subset_id,
                        "held_out_models": held_out,
                        "known_models": known,
                        "n_known": n_known,
                        "n_unknown": n_unknown,
                        **interval,
                    }
                )

            if counted_inclusion != normalized_inclusion:
                raise ValueError(
                    f"{group_location}.model_inclusion_counts do not match "
                    "the listed subsets"
                )

            normalized_subsets.sort(
                key=lambda record: (
                    tuple(record["held_out_models"]),
                    record["subset_id"],
                )
            )
            normalized_groups[size] = {
                "n_possible": n_possible,
                "n_evaluated": n_evaluated,
                "selection_mode": selection_mode,
                "model_inclusion_counts": normalized_inclusion,
                "subsets": normalized_subsets,
            }

        normalized_datasets[dataset_key] = {
            "dataset": dataset.get("dataset", dataset_key),
            "display_name": dataset.get("display_name"),
            "dataset_cache_digest": dataset_cache_digest,
            "tag": dataset.get("tag"),
            "holdout_sizes": normalized_groups,
        }

    return {
        "schema_version": 2,
        "classifier": classifier,
        "classifier_seeds": seeds,
        "classifier_seed_count": len(seeds),
        "confidence_level": confidence_level,
        "model_universe": universe,
        "holdout_sizes": holdout_sizes,
        "possible_counts": possible_counts,
        "max_subsets_per_size": max_subsets,
        "subset_seed": subset_seed,
        "datasets": normalized_datasets,
    }


__all__ = ["load_open_set_multi_intervals"]
