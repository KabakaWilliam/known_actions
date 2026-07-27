"""Deterministic subset designs for multi-model open-set experiments.

The public entry point, :func:`build_subset_design`, either returns every
possible holdout subset or a deterministic, balanced sample of them.  Model
identifiers within every subset are canonicalized by requiring a sorted,
unique model universe and by using :func:`itertools.combinations`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Literal


SelectionMode = Literal["exhaustive", "balanced_sample"]


@dataclass(frozen=True, slots=True)
class HoldoutSubset:
    """One selected holdout subset and its stable filesystem identifier."""

    models: tuple[str, ...]
    subset_id: str


@dataclass(frozen=True, slots=True)
class SubsetDesign:
    """A complete or capped design for one holdout size."""

    model_universe: tuple[str, ...]
    holdout_size: int
    subsets: tuple[HoldoutSubset, ...]
    possible_count: int
    evaluated_count: int
    selection_mode: SelectionMode
    inclusion_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class HoldoutSelection:
    """Compatibility view used by the multi-holdout experiment runner."""

    subsets: tuple[tuple[str, ...], ...]
    possible_count: int
    selection_mode: SelectionMode
    model_inclusion_counts: dict[str, int]


def validate_model_universe(model_universe: Iterable[str]) -> tuple[str, ...]:
    """Return a validated, already-sorted tuple of unique model identifiers."""
    if isinstance(model_universe, (str, bytes)):
        raise ValueError("model_universe must be an iterable of model identifiers")
    try:
        models = tuple(model_universe)
    except TypeError as exc:
        raise ValueError(
            "model_universe must be an iterable of model identifiers"
        ) from exc

    if not models:
        raise ValueError("model_universe must not be empty")
    if any(not isinstance(model, str) for model in models):
        raise ValueError("every model identifier must be a string")
    if any(not model or model != model.strip() for model in models):
        raise ValueError(
            "model identifiers must be non-empty and have no surrounding whitespace"
        )
    if len(set(models)) != len(models):
        raise ValueError("model_universe must contain unique model identifiers")
    if models != tuple(sorted(models)):
        raise ValueError("model_universe must be sorted")
    return models


def _validate_holdout_size(holdout_size: int, universe_size: int) -> int:
    if isinstance(holdout_size, bool) or not isinstance(holdout_size, int):
        raise ValueError("holdout_size must be an integer")
    if not 1 <= holdout_size <= universe_size:
        raise ValueError(
            "holdout_size must be between 1 and the model universe size"
        )
    return holdout_size


def _canonical_subset(models: Iterable[str]) -> tuple[str, ...]:
    if isinstance(models, (str, bytes)):
        raise ValueError("models must be an iterable of model identifiers")
    try:
        canonical = tuple(sorted(models))
    except TypeError as exc:
        raise ValueError("models must contain only string identifiers") from exc
    if not canonical:
        raise ValueError("a holdout subset must not be empty")
    if any(not isinstance(model, str) for model in canonical):
        raise ValueError("models must contain only string identifiers")
    if any(not model or model != model.strip() for model in canonical):
        raise ValueError(
            "model identifiers must be non-empty and have no surrounding whitespace"
        )
    if len(set(canonical)) != len(canonical):
        raise ValueError("a holdout subset must contain unique model identifiers")
    return canonical


def _canonical_payload(models: tuple[str, ...]) -> bytes:
    return json.dumps(
        models,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def make_subset_id(models: Iterable[str]) -> str:
    """Return an order-independent, filesystem-safe ID for a model subset."""
    canonical = _canonical_subset(models)
    digest = hashlib.sha256(_canonical_payload(canonical)).hexdigest()[:20]
    return f"h{len(canonical)}-{digest}"


def canonical_subset_id(models: Iterable[str]) -> str:
    """Compatibility alias for :func:`make_subset_id`."""
    return make_subset_id(models)


def enumerate_subsets(
    model_universe: Iterable[str],
    holdout_size: int,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate all canonical holdout combinations in lexicographic order."""
    models = validate_model_universe(model_universe)
    size = _validate_holdout_size(holdout_size, len(models))
    return tuple(combinations(models, size))


def _selection_key(
    seed: int,
    subset: tuple[str, ...],
) -> bytes:
    """Stable pseudo-random ordering key independent of Python hash randomization."""
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_payload(subset))
    return digest.digest()


def _inclusion_counts(
    universe: tuple[str, ...],
    subsets: Iterable[tuple[str, ...]],
) -> dict[str, int]:
    counts = dict.fromkeys(universe, 0)
    for subset in subsets:
        for model in subset:
            counts[model] += 1
    return counts


def _balanced_sample(
    all_subsets: tuple[tuple[str, ...], ...],
    universe: tuple[str, ...],
    sample_size: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Select a deterministic sample whose model counts differ by at most one.

    A SHA-256 ordering supplies the initial without-replacement sample.  The
    repair step swaps an overrepresented model for an underrepresented model
    in one selected subset at a time.  Whenever two inclusion counts differ by
    at least two, such a swap must exist: replacing the high-count model with
    the low-count model is a bijection between the corresponding families of
    candidate subsets, and the selected family on the high side is larger.
    Each swap strictly decreases the sum of squared inclusion counts, so the
    procedure terminates.
    """
    ranked = sorted(
        all_subsets,
        key=lambda subset: _selection_key(seed, subset),
    )
    selected = set(ranked[:sample_size])
    counts = _inclusion_counts(universe, selected)

    while max(counts.values()) - min(counts.values()) > 1:
        high_models = sorted(
            universe,
            key=lambda model: (
                -counts[model],
                hashlib.sha256(
                    f"{seed}\0{model}".encode("utf-8")
                ).digest(),
            ),
        )
        low_models = sorted(
            universe,
            key=lambda model: (
                counts[model],
                hashlib.sha256(
                    f"{seed}\0{model}".encode("utf-8")
                ).digest(),
            ),
        )

        repaired = False
        for high in high_models:
            for low in low_models:
                if counts[high] - counts[low] <= 1:
                    continue
                replacements = []
                for old_subset in selected:
                    if high not in old_subset or low in old_subset:
                        continue
                    new_subset = tuple(
                        sorted((set(old_subset) - {high}) | {low})
                    )
                    if new_subset not in selected:
                        replacements.append((old_subset, new_subset))
                if not replacements:
                    continue

                old_subset, new_subset = min(
                    replacements,
                    key=lambda pair: (
                        _selection_key(seed, pair[1]),
                        pair[0],
                    ),
                )
                selected.remove(old_subset)
                selected.add(new_subset)
                counts[high] -= 1
                counts[low] += 1
                repaired = True
                break
            if repaired:
                break

        if not repaired:  # Defensive: the counting argument above says unreachable.
            raise RuntimeError("could not balance the sampled holdout subsets")

    return tuple(sorted(selected))


def build_subset_design(
    model_universe: Iterable[str],
    holdout_size: int,
    *,
    cap: int | None = None,
    seed: int = 2026,
) -> SubsetDesign:
    """Build an exhaustive or deterministically capped holdout design.

    ``cap=None`` and caps at least as large as the possible combination count
    produce the exhaustive design.  Smaller caps select without replacement
    and balance model inclusion counts to differ by at most one.
    """
    universe = validate_model_universe(model_universe)
    size = _validate_holdout_size(holdout_size, len(universe))
    if cap is not None and (
        isinstance(cap, bool) or not isinstance(cap, int) or cap < 1
    ):
        raise ValueError("cap must be a positive integer or None")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    possible_count = math.comb(len(universe), size)
    all_subsets = tuple(combinations(universe, size))
    if len(all_subsets) != possible_count:  # pragma: no cover - defensive
        raise RuntimeError("combination enumeration count mismatch")

    if cap is None or cap >= possible_count:
        selected = all_subsets
        selection_mode: SelectionMode = "exhaustive"
    else:
        selected = _balanced_sample(
            all_subsets,
            universe,
            cap,
            seed,
        )
        selection_mode = "balanced_sample"

    records = tuple(
        HoldoutSubset(models=subset, subset_id=make_subset_id(subset))
        for subset in selected
    )
    subset_ids = [record.subset_id for record in records]
    if len(set(subset_ids)) != len(subset_ids):
        raise RuntimeError("subset ID collision")

    inclusion_counts = _inclusion_counts(universe, selected)
    return SubsetDesign(
        model_universe=universe,
        holdout_size=size,
        subsets=records,
        possible_count=possible_count,
        evaluated_count=len(records),
        selection_mode=selection_mode,
        inclusion_counts=inclusion_counts,
    )


def select_holdout_subsets(
    model_universe: Iterable[str],
    holdout_size: int,
    *,
    max_subsets: int | None,
    seed: int,
) -> HoldoutSelection:
    """Return the compact subset-selection interface used by the runner."""
    design = build_subset_design(
        model_universe,
        holdout_size,
        cap=max_subsets,
        seed=seed,
    )
    return HoldoutSelection(
        subsets=tuple(record.models for record in design.subsets),
        possible_count=design.possible_count,
        selection_mode=design.selection_mode,
        model_inclusion_counts=dict(design.inclusion_counts),
    )
