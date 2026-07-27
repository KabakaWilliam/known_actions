import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from open_set_multi_figure_data import load_open_set_multi_intervals
from plot_open_set_holdout_progression import (
    main,
    make_progression_figure,
    make_ranked_figure,
)
from test_open_set_multi_figure_data import (
    retain_balanced_sample,
    synthetic_multi_payload,
)


class OpenSetHoldoutProgressionPlotTests(unittest.TestCase):
    def _load(self, payload):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "open_set_multi.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_open_set_multi_intervals(path)

    @staticmethod
    def _interval_segment_count(figure):
        return sum(
            len(collection.get_segments())
            for axis in figure.axes
            for collection in axis.collections
            if isinstance(collection, LineCollection)
        )

    def test_progression_draws_one_pointwise_interval_per_subset(self):
        data = self._load(synthetic_multi_payload())
        expected_count = 2 * (4 + 6 + 4)
        figure = make_progression_figure(data)
        self.addCleanup(plt.close, figure)

        self.assertEqual(
            self._interval_segment_count(figure),
            expected_count,
        )
        tick_text = {
            label.get_text()
            for axis in figure.axes
            for label in axis.get_xticklabels()
        }
        self.assertIn("1 model\n4/4 subsets", tick_text)
        self.assertIn("2 models\n6/6 subsets", tick_text)
        legend_text = {
            text.get_text()
            for legend in figure.legends
            for text in legend.get_texts()
        }
        self.assertIn(
            "Across-subset IQR (descriptive, not a CI)",
            legend_text,
        )

    def test_progression_labels_sampled_coverage_without_hiding_denominator(self):
        payload = synthetic_multi_payload(("wiki",))
        retain_balanced_sample(payload, "wiki", 3, 2)
        data = self._load(payload)
        figure = make_progression_figure(data)
        self.addCleanup(plt.close, figure)

        tick_text = {
            label.get_text()
            for axis in figure.axes
            for label in axis.get_xticklabels()
        }
        self.assertIn(
            "3 models\n2/4 subsets\nbalanced sample",
            tick_text,
        )

    def test_ranked_draws_every_interval_and_reports_coverage(self):
        data = self._load(synthetic_multi_payload())
        expected_count = 2 * (4 + 6 + 4)
        figure = make_ranked_figure(data)
        self.addCleanup(plt.close, figure)

        self.assertEqual(
            self._interval_segment_count(figure),
            expected_count,
        )
        annotations = {
            text.get_text()
            for axis in figure.axes
            for text in axis.texts
        }
        self.assertIn("4/4 subsets", annotations)
        self.assertIn("6/6 subsets", annotations)
        for axis in figure.axes:
            y_min, y_max = axis.get_ylim()
            self.assertLessEqual(y_min, 0.48)
            self.assertGreaterEqual(y_max, 0.74)

    def test_cli_writes_both_png_and_pdf_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stats = root / "open_set_multi.json"
            out_dir = root / "figures"
            stats.write_text(
                json.dumps(synthetic_multi_payload(("wiki",))),
                encoding="utf-8",
            )

            main(
                [
                    "--stats",
                    str(stats),
                    "--out-dir",
                    str(out_dir),
                    "--format",
                    "both",
                    "--dpi",
                    "72",
                ]
            )

            expected = {
                "open_set_holdout_progression_bootstrap_ci.png",
                "open_set_holdout_progression_bootstrap_ci.pdf",
                "open_set_holdout_ranked_bootstrap_ci.png",
                "open_set_holdout_ranked_bootstrap_ci.pdf",
            }
            self.assertEqual(
                {path.name for path in out_dir.iterdir()},
                expected,
            )
            for filename in expected:
                self.assertGreater((out_dir / filename).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
