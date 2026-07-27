#!/usr/bin/env python3
"""Plot per-subset open-set AUROC intervals for 1--3 model holdouts.

Two complementary figures are produced:

* ``open_set_holdout_progression_bootstrap_ci`` compares the distribution of
  individual held-out subsets as the holdout size grows.  Every point and
  whisker is one subset's estimate and trace-bootstrap confidence interval.
  The overlaid median and IQR describe the estimates across subsets; they are
  not confidence bounds.
* ``open_set_holdout_ranked_bootstrap_ci`` sorts subsets within each
  dataset/holdout-size panel and draws every pointwise interval, which remains
  legible when there are hundreds of possible subsets.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from open_set_multi_figure_data import load_open_set_multi_intervals


DATASET_LABELS = {
    "wiki": "2WikiMultiHop",
    "frames": "FRAMES",
    "webshop": "WebShop",
    "deepshop": "DeepShop",
}
DATASET_ORDER = tuple(DATASET_LABELS)
HOLDOUT_COLOURS = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")
CHANCE_COLOUR = "#888888"
SUMMARY_COLOUR = "#252525"

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
    }
)


def _dataset_items(data: dict[str, object]) -> list[tuple[str, dict]]:
    datasets = data["datasets"]
    preferred = [
        key
        for key in DATASET_ORDER
        if key in datasets
    ]
    remaining = sorted(set(datasets) - set(preferred))
    return [(key, datasets[key]) for key in preferred + remaining]


def _dataset_label(key: str, dataset: dict) -> str:
    configured = dataset.get("display_name")
    if isinstance(configured, str) and configured.strip():
        return configured
    return DATASET_LABELS.get(key, str(dataset.get("dataset") or key))


def _colour_by_size(holdout_sizes: list[int]) -> dict[int, str]:
    return {
        size: HOLDOUT_COLOURS[index % len(HOLDOUT_COLOURS)]
        for index, size in enumerate(holdout_sizes)
    }


def _stable_jitter(
    count: int,
    *,
    width: float,
    dataset_index: int,
    holdout_size: int,
) -> np.ndarray:
    """Return deterministic, well-spread offsets without Python hash state."""
    if count < 1:
        return np.empty(0, dtype=float)
    golden_ratio_conjugate = 0.6180339887498949
    phase = (dataset_index * 0.173205 + holdout_size * 0.113) % 1.0
    unit = (
        np.arange(count, dtype=float) * golden_ratio_conjugate + phase
    ) % 1.0
    return (unit - 0.5) * 2.0 * width


def _point_style(count: int) -> tuple[float, float, float, float]:
    """Marker size, point alpha, interval alpha, and interval width."""
    if count <= 20:
        return 22.0, 0.82, 0.72, 0.9
    if count <= 120:
        return 12.0, 0.62, 0.46, 0.65
    return 6.0, 0.42, 0.26, 0.42


def _common_ylim(data: dict[str, object]) -> tuple[float, float]:
    intervals = [
        (float(record["lower"]), float(record["upper"]))
        for _, dataset in _dataset_items(data)
        for group in dataset["holdout_sizes"].values()
        for record in group["subsets"]
    ]
    lower = min(item[0] for item in intervals)
    upper = max(item[1] for item in intervals)
    lower = min(lower, 0.5)
    upper = max(upper, 0.5)
    y_min = max(0.0, math.floor((lower - 0.025) * 20.0) / 20.0)
    y_max = min(1.0, math.ceil((upper + 0.025) * 20.0) / 20.0)
    if y_max - y_min < 0.2:
        centre = (y_min + y_max) / 2.0
        y_min = max(0.0, centre - 0.1)
        y_max = min(1.0, centre + 0.1)
    return y_min, y_max


def _style_axis(ax, y_limits: tuple[float, float]) -> None:
    ax.set_ylim(*y_limits)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.30)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _coverage_label(size: int, group: dict) -> str:
    noun = "model" if size == 1 else "models"
    evaluated = int(group["n_evaluated"])
    possible = int(group["n_possible"])
    suffix = (
        "\nbalanced sample"
        if group["selection_mode"] == "balanced_sample"
        else ""
    )
    return f"{size} {noun}\n{evaluated}/{possible} subsets{suffix}"


def make_progression_figure(data: dict[str, object]):
    """Create the dataset-faceted holdout-size progression figure."""
    dataset_items = _dataset_items(data)
    holdout_sizes = list(data["holdout_sizes"])
    colours = _colour_by_size(holdout_sizes)
    n_columns = min(2, len(dataset_items))
    n_rows = math.ceil(len(dataset_items) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(6.1 * n_columns, 3.8 * n_rows),
        sharey=True,
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    y_limits = _common_ylim(data)

    for dataset_index, ((dataset_key, dataset), ax) in enumerate(
        zip(dataset_items, flat_axes)
    ):
        tick_labels: list[str] = []
        for x_position, size in enumerate(holdout_sizes):
            group = dataset["holdout_sizes"][size]
            records = group["subsets"]
            offsets = _stable_jitter(
                len(records),
                width=0.29,
                dataset_index=dataset_index,
                holdout_size=size,
            )
            x_values = x_position + offsets
            estimates = np.asarray(
                [float(record["estimate"]) for record in records]
            )
            segments = [
                [
                    (x_value, float(record["lower"])),
                    (x_value, float(record["upper"])),
                ]
                for x_value, record in zip(x_values, records)
            ]
            marker_size, point_alpha, interval_alpha, interval_width = (
                _point_style(len(records))
            )
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=colours[size],
                    linewidths=interval_width,
                    alpha=interval_alpha,
                    zorder=2,
                )
            )
            ax.scatter(
                x_values,
                estimates,
                s=marker_size,
                color=colours[size],
                alpha=point_alpha,
                edgecolors="white" if len(records) <= 120 else "none",
                linewidths=0.35,
                zorder=3,
            )

            q1, median, q3 = np.quantile(estimates, [0.25, 0.5, 0.75])
            box_width = 0.18
            ax.add_patch(
                Rectangle(
                    (x_position - box_width / 2.0, q1),
                    box_width,
                    q3 - q1,
                    facecolor="white",
                    edgecolor=SUMMARY_COLOUR,
                    linewidth=1.0,
                    alpha=0.90,
                    zorder=4,
                )
            )
            ax.plot(
                [
                    x_position - box_width / 2.0,
                    x_position + box_width / 2.0,
                ],
                [median, median],
                color=SUMMARY_COLOUR,
                linewidth=2.0,
                zorder=5,
            )
            tick_labels.append(_coverage_label(size, group))

        ax.axhline(
            0.5,
            color=CHANCE_COLOUR,
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        ax.set_xlim(-0.48, len(holdout_sizes) - 0.52)
        ax.set_xticks(range(len(holdout_sizes)))
        ax.set_xticklabels(tick_labels)
        ax.set_title(
            _dataset_label(dataset_key, dataset),
            fontsize=10,
            fontweight="bold",
            loc="left",
        )
        ax.set_ylabel("Open-set AUROC")
        _style_axis(ax, y_limits)

    for ax in flat_axes[len(dataset_items):]:
        ax.set_visible(False)

    level = float(data["confidence_level"])
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=colours[size],
            linewidth=1.0,
            markersize=5,
            label=(
                f"{size}-model subset: estimate + "
                f"{level:.0%} trace-bootstrap CI"
            ),
        )
        for size in holdout_sizes
    ]
    handles.extend(
        [
            Patch(
                facecolor="white",
                edgecolor=SUMMARY_COLOUR,
                label="Across-subset IQR (descriptive, not a CI)",
            ),
            Line2D(
                [0],
                [0],
                color=CHANCE_COLOUR,
                linestyle="--",
                label="Chance (0.5)",
            ),
        ]
    )
    fig.suptitle(
        "Open-set detection with progressively larger held-out model sets",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.955,
        (
            "Unknown-model traces are pooled into one class. Every colored "
            "point and whisker is one held-out subset; m/N gives evaluated/"
            "possible subsets."
        ),
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=min(3, len(handles)),
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(
        top=0.88,
        bottom=0.18,
        hspace=0.32,
        wspace=0.16,
    )
    return fig


def make_ranked_figure(data: dict[str, object]):
    """Create ranked panels containing every individual subset interval."""
    dataset_items = _dataset_items(data)
    holdout_sizes = list(data["holdout_sizes"])
    colours = _colour_by_size(holdout_sizes)
    fig, axes = plt.subplots(
        len(holdout_sizes),
        len(dataset_items),
        figsize=(3.25 * len(dataset_items), 2.55 * len(holdout_sizes)),
        sharey=True,
        squeeze=False,
    )
    y_limits = _common_ylim(data)

    for row_index, size in enumerate(holdout_sizes):
        for column_index, (dataset_key, dataset) in enumerate(dataset_items):
            ax = axes[row_index, column_index]
            group = dataset["holdout_sizes"][size]
            records = sorted(
                group["subsets"],
                key=lambda record: (
                    float(record["estimate"]),
                    tuple(record["held_out_models"]),
                    record["subset_id"],
                ),
            )
            if len(records) == 1:
                x_values = np.asarray([50.0])
            else:
                x_values = np.linspace(0.0, 100.0, len(records))
            estimates = np.asarray(
                [float(record["estimate"]) for record in records]
            )
            segments = [
                [
                    (x_value, float(record["lower"])),
                    (x_value, float(record["upper"])),
                ]
                for x_value, record in zip(x_values, records)
            ]
            marker_size, point_alpha, interval_alpha, interval_width = (
                _point_style(len(records))
            )
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=colours[size],
                    linewidths=interval_width,
                    alpha=interval_alpha,
                    zorder=2,
                )
            )
            ax.scatter(
                x_values,
                estimates,
                s=max(3.0, marker_size * 0.65),
                color=colours[size],
                alpha=max(0.45, point_alpha),
                edgecolors="none",
                zorder=3,
            )
            ax.axhline(
                0.5,
                color=CHANCE_COLOUR,
                linestyle="--",
                linewidth=0.9,
                zorder=1,
            )
            ax.set_xlim(-2.5, 102.5)
            ax.set_xticks([0, 50, 100])
            if row_index == len(holdout_sizes) - 1:
                ax.set_xlabel("AUROC rank percentile (low → high)")
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if row_index == 0:
                ax.set_title(
                    _dataset_label(dataset_key, dataset),
                    fontsize=10,
                    fontweight="bold",
                )
            if column_index == 0:
                noun = "model" if size == 1 else "models"
                ax.set_ylabel(f"{size} held-out {noun}\nAUROC")
            ax.text(
                0.025,
                0.965,
                (
                    f"{group['n_evaluated']}/{group['n_possible']} subsets"
                    + (
                        " (sampled)"
                        if group["selection_mode"] == "balanced_sample"
                        else ""
                    )
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="#444444",
            )
            _style_axis(ax, y_limits)

    level = float(data["confidence_level"])
    fig.suptitle(
        "Ranked pointwise intervals for held-out model subsets",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.965,
        (
            f"Subsets are independently sorted by AUROC within each panel; "
            f"each hairline is that subset's {level:.0%} trace-bootstrap CI, "
            "not an interval for a mean. Dashed line: chance (0.5)."
        ),
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.subplots_adjust(
        top=0.91,
        bottom=0.09,
        left=0.075,
        right=0.985,
        hspace=0.16,
        wspace=0.12,
    )
    return fig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("open_set_multi_holdout_auroc_results.json"),
        help=(
            "Schema-v2 aggregate JSON (default: "
            "open_set_multi_holdout_auroc_results.json)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        "--output-dir",
        dest="out_dir",
        type=Path,
        default=Path("figures"),
        help="Directory for both figures (default: figures).",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "both"),
        default="both",
        help="Output format (default: both PNG and PDF).",
    )
    parser.add_argument(
        "--classifier",
        default="XGBoost",
        help="Classifier key to plot (default: XGBoost).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution (default: 300).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.dpi < 1:
        raise SystemExit("ERROR: --dpi must be positive")
    try:
        data = load_open_set_multi_intervals(
            args.stats,
            classifier=args.classifier,
        )
    except ValueError as exc:
        raise SystemExit(f"ERROR: invalid --stats: {exc}") from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    formats = ("png", "pdf") if args.format == "both" else (args.format,)
    figures = (
        (
            "open_set_holdout_progression_bootstrap_ci",
            make_progression_figure(data),
        ),
        (
            "open_set_holdout_ranked_bootstrap_ci",
            make_ranked_figure(data),
        ),
    )
    try:
        for stem, figure in figures:
            for output_format in formats:
                output = args.out_dir / f"{stem}.{output_format}"
                figure.savefig(
                    output,
                    format=output_format,
                    dpi=args.dpi,
                    bbox_inches="tight",
                )
                print(f"Saved: {output}")
    finally:
        for _, figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
