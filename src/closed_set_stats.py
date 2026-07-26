"""Statistical summaries for closed-set classifier predictions.

The confidence interval implemented here is a paired, non-parametric percentile
bootstrap over held-out test traces.  Each bootstrap replicate resamples one set
of trace indices and applies it to every classifier seed, then averages macro-F1
across seeds.  This keeps classifier-seed variation separate from test-sample
uncertainty while preserving the pairing between seeds.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence

import numpy as np


MIN_CLASSIFIER_SEEDS = 5
DEFAULT_CLASSIFIER_SEED_COUNT = 10
MAX_CLASSIFIER_SEED = 2**31 - 1


def generate_classifier_seeds(
    count: int = DEFAULT_CLASSIFIER_SEED_COUNT,
) -> list[int]:
    """Generate unique classifier seeds with system-provided randomness."""
    if count < MIN_CLASSIFIER_SEEDS:
        raise ValueError(
            f"at least {MIN_CLASSIFIER_SEEDS} classifier seeds are required; "
            f"received {count}"
        )
    return secrets.SystemRandom().sample(range(MAX_CLASSIFIER_SEED + 1), count)


def validate_classifier_seeds(
    seeds: Sequence[int],
    minimum: int = MIN_CLASSIFIER_SEEDS,
) -> list[int]:
    """Return normalized seeds, requiring at least ``minimum`` unique values."""
    normalized = [int(seed) for seed in seeds]
    if any(seed < 0 or seed > MAX_CLASSIFIER_SEED for seed in normalized):
        raise ValueError(
            f"classifier seeds must be in [0, {MAX_CLASSIFIER_SEED}]"
        )
    if len(normalized) != len(set(normalized)):
        raise ValueError("classifier seeds must be unique")
    if len(normalized) < minimum:
        raise ValueError(
            f"at least {minimum} classifier seeds are required; "
            f"received {len(normalized)}"
        )
    return normalized


def macro_f1_from_encoded(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> float:
    """Compute macro-F1 over a fixed encoded class universe.

    Classes absent from a resample remain in the average with F1=0, matching
    ``sklearn.metrics.f1_score(..., labels=all_classes, zero_division=0)``.
    """
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.ndim != 1 or pred.ndim != 1 or true.shape != pred.shape:
        raise ValueError("y_true and y_pred must be one-dimensional and equal length")
    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    if true.size == 0:
        raise ValueError("at least one test trace is required")
    if (
        np.any(true < 0)
        or np.any(pred < 0)
        or np.any(true >= n_classes)
        or np.any(pred >= n_classes)
    ):
        raise ValueError("encoded labels must be in [0, n_classes)")

    confusion = np.bincount(
        true * n_classes + pred,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)
    true_count = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    denominator = true_count + pred_count
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(n_classes, dtype=float),
        where=denominator != 0,
    )
    return float(per_class_f1.mean())


def _validate_class_names(
    class_names: Sequence[str] | None,
    n_classes: int,
) -> list[str] | None:
    """Normalize optional display names for the encoded class universe."""
    if class_names is None:
        return None
    if isinstance(class_names, (str, bytes)):
        raise ValueError("class_names must be a sequence with one name per class")

    normalized = list(class_names)
    if len(normalized) != n_classes:
        raise ValueError(
            f"class_names must contain exactly {n_classes} values; "
            f"received {len(normalized)}"
        )
    if any(not isinstance(name, str) or not name for name in normalized):
        raise ValueError("class_names must contain non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError("class_names must be unique")
    return normalized


def summarize_macro_f1(
    y_true: Sequence[int] | np.ndarray,
    predictions_by_seed: Mapping[int, Sequence[int] | np.ndarray],
    n_classes: int,
    *,
    class_names: Sequence[str] | None = None,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 2026,
) -> dict:
    """Summarize macro- and per-class F1 with paired trace bootstrap CIs.

    ``per_class`` reports one row for every encoded class.  Its estimate is the
    mean class F1 across classifier seeds.  Each bootstrap replicate uses the
    same resampled held-out trace indices for every seed and every class.
    """
    seeds = validate_classifier_seeds(list(predictions_by_seed))
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be non-negative")

    true = np.asarray(y_true, dtype=np.int64)
    if true.ndim != 1 or true.size == 0:
        raise ValueError("y_true must contain at least one test trace")
    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    if np.any(true < 0) or np.any(true >= n_classes):
        raise ValueError("encoded labels must be in [0, n_classes)")
    names = _validate_class_names(class_names, n_classes)

    predictions = {}
    for seed in seeds:
        pred = np.asarray(predictions_by_seed[seed], dtype=np.int64)
        if pred.shape != true.shape:
            raise ValueError(
                f"seed {seed} has {pred.size} predictions for "
                f"{true.size} test traces"
            )
        if np.any(pred < 0) or np.any(pred >= n_classes):
            raise ValueError("encoded labels must be in [0, n_classes)")
        predictions[seed] = pred

    per_seed_class_values: dict[int, np.ndarray] = {}
    per_seed_values = {}
    for seed in seeds:
        confusion = np.bincount(
            true * n_classes + predictions[seed],
            minlength=n_classes * n_classes,
        ).reshape(n_classes, n_classes)
        denominator = confusion.sum(axis=1) + confusion.sum(axis=0)
        class_values = np.divide(
            2.0 * np.diag(confusion),
            denominator,
            out=np.zeros(n_classes, dtype=float),
            where=denominator != 0,
        )
        per_seed_class_values[seed] = class_values
        per_seed_values[seed] = float(class_values.mean())
    seed_scores = np.asarray(list(per_seed_values.values()), dtype=float)
    seed_class_scores = np.stack(
        [per_seed_class_values[seed] for seed in seeds],
        axis=0,
    )

    # Use one resampled trace-index vector across all seeds in each replicate.
    # Reusing pre-encoded confusion-matrix cells avoids repeated metric-library
    # overhead without changing the bootstrap statistic.
    encoded_cells = {
        seed: true * n_classes + predictions[seed]
        for seed in seeds
    }
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_scores = np.empty(bootstrap_replicates, dtype=float)
    bootstrap_class_scores = np.empty(
        (bootstrap_replicates, n_classes),
        dtype=float,
    )
    for replicate in range(bootstrap_replicates):
        indices = rng.integers(0, true.size, size=true.size)
        seed_total = 0.0
        class_total = np.zeros(n_classes, dtype=float)
        for seed in seeds:
            confusion = np.bincount(
                encoded_cells[seed][indices],
                minlength=n_classes * n_classes,
            ).reshape(n_classes, n_classes)
            true_count = confusion.sum(axis=1)
            pred_count = confusion.sum(axis=0)
            denominator = true_count + pred_count
            per_class_f1 = np.divide(
                2.0 * np.diag(confusion),
                denominator,
                out=np.zeros(n_classes, dtype=float),
                where=denominator != 0,
            )
            seed_total += float(per_class_f1.mean())
            class_total += per_class_f1
        bootstrap_scores[replicate] = seed_total / len(seeds)
        bootstrap_class_scores[replicate] = class_total / len(seeds)

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        bootstrap_scores,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    estimate = float(seed_scores.mean())
    class_lower, class_upper = np.quantile(
        bootstrap_class_scores,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        axis=0,
    )
    per_class = []
    for class_index in range(n_classes):
        class_seed_scores = seed_class_scores[:, class_index]
        class_estimate = float(class_seed_scores.mean())
        row = {
            "class_index": class_index,
            "estimate": class_estimate,
            "confidence_interval": {
                "level": float(confidence_level),
                "lower": float(class_lower[class_index]),
                "upper": float(class_upper[class_index]),
                "method": "paired_percentile_bootstrap_over_test_traces",
            },
            "seed_variability": {
                "mean": class_estimate,
                "sample_std": float(class_seed_scores.std(ddof=1)),
                "min": float(class_seed_scores.min()),
                "max": float(class_seed_scores.max()),
            },
            "per_seed": [
                {
                    "seed": seed,
                    "f1": float(per_seed_class_values[seed][class_index]),
                }
                for seed in seeds
            ],
        }
        if names is not None:
            row["class_name"] = names[class_index]
        per_class.append(row)

    return {
        "estimate": estimate,
        "confidence_interval": {
            "level": float(confidence_level),
            "lower": float(lower),
            "upper": float(upper),
            "method": "paired_percentile_bootstrap_over_test_traces",
        },
        "seed_variability": {
            "mean": estimate,
            "sample_std": float(seed_scores.std(ddof=1)),
            "min": float(seed_scores.min()),
            "max": float(seed_scores.max()),
        },
        "per_seed": [
            {"seed": seed, "macro_f1": float(per_seed_values[seed])}
            for seed in seeds
        ],
        "per_class": per_class,
    }
