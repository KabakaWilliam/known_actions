"""Statistical summaries for open-set confidence scores.

The confidence interval implemented here is a paired, stratified,
non-parametric percentile bootstrap over held-out evaluation traces.  Known
and unknown traces are resampled independently with replacement, and each
bootstrap replicate applies the same two index samples to every classifier
seed before averaging AUROC across seeds.  This keeps classifier-seed
variation separate from evaluation-sample uncertainty while preserving the
pairing between seeds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from closed_set_stats import validate_classifier_seeds


_BOOTSTRAP_BATCH_SIZE = 128
_CI_METHOD = (
    "paired_stratified_percentile_bootstrap_over_"
    "known_test_and_unknown_held_out_model_traces"
)
_POOLED_UNKNOWN_CI_METHOD = (
    "paired_stratified_percentile_bootstrap_over_"
    "known_test_and_pooled_unknown_test_traces"
)


def _normalize_score_mapping(
    scores_by_seed: Mapping[int, Sequence[float] | np.ndarray],
    group_name: str,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Normalize seed keys and one-dimensional finite score arrays."""
    normalized: dict[int, np.ndarray] = {}
    seeds: list[int] = []
    for raw_seed, raw_scores in scores_by_seed.items():
        seed = int(raw_seed)
        if seed in normalized:
            raise ValueError(f"{group_name} classifier seeds must be unique")
        scores = np.asarray(raw_scores, dtype=float)
        if scores.ndim != 1:
            raise ValueError(f"{group_name} scores must be one-dimensional")
        if scores.size == 0:
            raise ValueError(
                f"at least one {group_name} test trace is required"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{group_name} scores must be finite")
        seeds.append(seed)
        normalized[seed] = scores

    validate_classifier_seeds(seeds)
    return seeds, normalized


def _auc_plan(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute score order and tie-group starts for weighted AUROC."""
    scores = np.concatenate((known_scores, unknown_scores))
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    group_starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1,
        )
    )
    return order, group_starts


def _auc_from_bootstrap_counts(
    positive_counts: np.ndarray,
    negative_counts: np.ndarray,
    order: np.ndarray,
    group_starts: np.ndarray,
    denominator: float,
) -> np.ndarray:
    """Compute tie-aware AUROC for a batch of weighted trace samples."""
    positive_by_group = np.add.reduceat(
        positive_counts[:, order],
        group_starts,
        axis=1,
    )
    negative_by_group = np.add.reduceat(
        negative_counts[:, order],
        group_starts,
        axis=1,
    )
    negative_before = (
        np.cumsum(negative_by_group, axis=1) - negative_by_group
    )
    numerator = np.sum(
        positive_by_group
        * (negative_before + 0.5 * negative_by_group),
        axis=1,
    )
    return numerator / denominator


def summarize_open_set_auroc(
    known_scores_by_seed: Mapping[int, Sequence[float] | np.ndarray],
    unknown_scores_by_seed: Mapping[int, Sequence[float] | np.ndarray],
    *,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 2026,
) -> dict:
    """Summarize seed-averaged AUROC with a paired trace bootstrap CI.

    Higher scores are interpreted as stronger evidence that a trace belongs
    to the known (positive) set.  Every seed must score the same known traces
    and the same unknown traces in the same order.

    The two bootstrap strata are the known and unknown evaluation traces.
    Within a replicate, each stratum is sampled independently with replacement
    and the resulting trace multiplicities are reused for every classifier
    seed.  The replicate statistic is the mean AUROC across seeds.
    """
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be non-negative")

    seeds, known = _normalize_score_mapping(
        known_scores_by_seed,
        "known",
    )
    unknown_seeds, unknown = _normalize_score_mapping(
        unknown_scores_by_seed,
        "unknown",
    )
    if set(seeds) != set(unknown_seeds):
        raise ValueError(
            "known and unknown scores must contain the same classifier seeds"
        )

    n_known = known[seeds[0]].size
    n_unknown = unknown[seeds[0]].size
    for seed in seeds:
        if known[seed].size != n_known:
            raise ValueError(
                "every classifier seed must score the same number of known "
                "test traces"
            )
        if unknown[seed].size != n_unknown:
            raise ValueError(
                "every classifier seed must score the same number of unknown "
                "test traces"
            )

    plans = {
        seed: _auc_plan(known[seed], unknown[seed])
        for seed in seeds
    }
    denominator = float(n_known * n_unknown)

    # Unit weights recover each fitted seed's AUROC without depending on a
    # metric library.  The rank calculation includes half credit for ties.
    unit_positive = np.zeros((1, n_known + n_unknown), dtype=np.int64)
    unit_negative = np.zeros_like(unit_positive)
    unit_positive[:, :n_known] = 1
    unit_negative[:, n_known:] = 1
    per_seed_values = {
        seed: float(
            _auc_from_bootstrap_counts(
                unit_positive,
                unit_negative,
                *plans[seed],
                denominator,
            )[0]
        )
        for seed in seeds
    }
    seed_scores = np.asarray(
        [per_seed_values[seed] for seed in seeds],
        dtype=float,
    )

    # Multinomial multiplicities are exactly the count representation of
    # ordinary sampling with replacement.  Generating them in modest batches
    # keeps memory bounded and lets the tie-aware rank calculation stay
    # vectorized.  These count matrices are shared by all classifier seeds.
    rng = np.random.default_rng(bootstrap_seed)
    known_probabilities = np.full(n_known, 1.0 / n_known)
    unknown_probabilities = np.full(n_unknown, 1.0 / n_unknown)
    bootstrap_scores = np.empty(bootstrap_replicates, dtype=float)
    for start in range(0, bootstrap_replicates, _BOOTSTRAP_BATCH_SIZE):
        stop = min(start + _BOOTSTRAP_BATCH_SIZE, bootstrap_replicates)
        batch_size = stop - start
        known_counts = rng.multinomial(
            n_known,
            known_probabilities,
            size=batch_size,
        )
        unknown_counts = rng.multinomial(
            n_unknown,
            unknown_probabilities,
            size=batch_size,
        )

        positive_counts = np.zeros(
            (batch_size, n_known + n_unknown),
            dtype=np.int64,
        )
        negative_counts = np.zeros_like(positive_counts)
        positive_counts[:, :n_known] = known_counts
        negative_counts[:, n_known:] = unknown_counts

        batch_scores = np.zeros(batch_size, dtype=float)
        for seed in seeds:
            batch_scores += _auc_from_bootstrap_counts(
                positive_counts,
                negative_counts,
                *plans[seed],
                denominator,
            )
        bootstrap_scores[start:stop] = batch_scores / len(seeds)

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        bootstrap_scores,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    estimate = float(seed_scores.mean())

    return {
        "estimate": estimate,
        "confidence_interval": {
            "level": float(confidence_level),
            "lower": float(lower),
            "upper": float(upper),
            "method": _CI_METHOD,
        },
        "seed_variability": {
            "mean": estimate,
            "sample_std": float(seed_scores.std(ddof=1)),
            "min": float(seed_scores.min()),
            "max": float(seed_scores.max()),
        },
        "per_seed": [
            {"seed": seed, "auroc": float(per_seed_values[seed])}
            for seed in seeds
        ],
        "n_known": int(n_known),
        "n_unknown": int(n_unknown),
    }


def summarize_pooled_open_set_auroc(
    known_scores_by_seed: Mapping[int, Sequence[float] | np.ndarray],
    pooled_unknown_scores_by_seed: Mapping[
        int,
        Sequence[float] | np.ndarray,
    ],
    *,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 2026,
) -> dict:
    """Summarize AUROC against one class of pooled unknown test traces.

    The pooled unknown arrays may contain test traces from multiple held-out
    models.  They form one negative-class bootstrap stratum: known test traces
    and pooled unknown test traces are independently resampled with
    replacement, while each pair of resamples is shared across classifier
    seeds.

    Apart from the confidence-interval method metadata, the result schema and
    calculation are identical to :func:`summarize_open_set_auroc`.
    """
    result = summarize_open_set_auroc(
        known_scores_by_seed,
        pooled_unknown_scores_by_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
    )
    result["confidence_interval"]["method"] = _POOLED_UNKNOWN_CI_METHOD
    return result
