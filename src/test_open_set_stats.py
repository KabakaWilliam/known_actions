import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from open_set_stats import summarize_open_set_auroc


SEEDS = [11, 12, 13, 14, 15]


def _same_scores(values):
    return {seed: np.asarray(values, dtype=float) for seed in SEEDS}


class OpenSetStatsTests(unittest.TestCase):
    def test_estimate_matches_mean_of_per_seed_sklearn_aurocs(self):
        known = {
            11: [0.9, 0.8, 0.2],
            12: [0.7, 0.5, 0.4],
            13: [0.6, 0.6, 0.1],
            14: [0.9, 0.3, 0.3],
            15: [0.8, 0.7, 0.6],
        }
        unknown = {
            11: [0.1, 0.4],
            12: [0.6, 0.2],
            13: [0.6, 0.2],
            14: [0.3, 0.1],
            15: [0.5, 0.4],
        }
        result = summarize_open_set_auroc(
            known,
            unknown,
            bootstrap_replicates=25,
            bootstrap_seed=7,
        )

        expected = []
        labels = np.array([1, 1, 1, 0, 0])
        for seed in SEEDS:
            expected.append(
                roc_auc_score(labels, known[seed] + unknown[seed])
            )

        self.assertTrue(
            np.allclose(
                [row["auroc"] for row in result["per_seed"]],
                expected,
            )
        )
        self.assertAlmostEqual(result["estimate"], np.mean(expected))
        self.assertEqual(result["n_known"], 3)
        self.assertEqual(result["n_unknown"], 2)

    def test_perfect_scores_have_degenerate_interval(self):
        result = summarize_open_set_auroc(
            _same_scores([0.9, 0.8, 0.7]),
            _same_scores([0.3, 0.2]),
            bootstrap_replicates=100,
            bootstrap_seed=3,
        )

        self.assertEqual(result["estimate"], 1.0)
        self.assertEqual(result["confidence_interval"]["lower"], 1.0)
        self.assertEqual(result["confidence_interval"]["upper"], 1.0)
        self.assertEqual(result["seed_variability"]["sample_std"], 0.0)

    def test_bootstrap_is_reproducible(self):
        known = _same_scores([0.9, 0.4, 0.3])
        unknown = _same_scores([0.8, 0.2])
        first = summarize_open_set_auroc(
            known,
            unknown,
            bootstrap_replicates=200,
            bootstrap_seed=99,
        )
        second = summarize_open_set_auroc(
            known,
            unknown,
            bootstrap_replicates=200,
            bootstrap_seed=99,
        )

        self.assertEqual(
            first["confidence_interval"],
            second["confidence_interval"],
        )
        self.assertEqual(
            first["confidence_interval"]["method"],
            (
                "paired_stratified_percentile_bootstrap_over_"
                "known_test_and_unknown_held_out_model_traces"
            ),
        )

    def test_known_trace_resamples_are_paired_across_seeds(self):
        # Relative to the singleton unknown score, the two seed families have
        # complementary outcomes for each known trace.  Sharing each known
        # resample across seeds therefore makes every seed-averaged replicate
        # exactly 0.5.
        known = {
            seed: ([1.0, 0.0] if index < 3 else [0.0, 1.0])
            for index, seed in enumerate([1, 2, 3, 4, 5, 6])
        }
        unknown = {
            seed: [0.5]
            for seed in known
        }
        result = summarize_open_set_auroc(
            known,
            unknown,
            bootstrap_replicates=200,
            bootstrap_seed=123,
        )

        self.assertEqual(result["estimate"], 0.5)
        self.assertEqual(result["confidence_interval"]["lower"], 0.5)
        self.assertEqual(result["confidence_interval"]["upper"], 0.5)

    def test_unknown_trace_resamples_are_paired_across_seeds(self):
        # The same construction for the unknown stratum verifies that its
        # resampled indices are also shared across seeds.
        known = {
            seed: [0.5]
            for seed in [1, 2, 3, 4, 5, 6]
        }
        unknown = {
            seed: ([0.0, 1.0] if index < 3 else [1.0, 0.0])
            for index, seed in enumerate(known)
        }
        result = summarize_open_set_auroc(
            known,
            unknown,
            bootstrap_replicates=200,
            bootstrap_seed=123,
        )

        self.assertEqual(result["estimate"], 0.5)
        self.assertEqual(result["confidence_interval"]["lower"], 0.5)
        self.assertEqual(result["confidence_interval"]["upper"], 0.5)

    def test_requires_at_least_five_unique_matching_seeds(self):
        four = {
            seed: [0.9]
            for seed in [1, 2, 3, 4]
        }
        with self.assertRaisesRegex(ValueError, "at least 5"):
            summarize_open_set_auroc(
                four,
                four,
                bootstrap_replicates=1,
            )

        known = _same_scores([0.9])
        unknown = {
            seed: [0.1]
            for seed in [11, 12, 13, 14, 16]
        }
        with self.assertRaisesRegex(ValueError, "same classifier seeds"):
            summarize_open_set_auroc(
                known,
                unknown,
                bootstrap_replicates=1,
            )

    def test_rejects_inconsistent_counts_and_nonfinite_scores(self):
        known = _same_scores([0.9, 0.8])
        known[15] = np.array([0.9])
        with self.assertRaisesRegex(ValueError, "same number of known"):
            summarize_open_set_auroc(
                known,
                _same_scores([0.1]),
                bootstrap_replicates=1,
            )

        unknown = _same_scores([0.1])
        unknown[15] = np.array([np.nan])
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_open_set_auroc(
                _same_scores([0.9]),
                unknown,
                bootstrap_replicates=1,
            )

    def test_validates_bootstrap_arguments(self):
        known = _same_scores([0.9])
        unknown = _same_scores([0.1])
        invalid = (
            ({"bootstrap_replicates": 0}, "positive"),
            ({"confidence_level": 1.0}, "between 0 and 1"),
            ({"bootstrap_seed": -1}, "non-negative"),
        )
        for kwargs, message in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    summarize_open_set_auroc(
                        known,
                        unknown,
                        **kwargs,
                    )


if __name__ == "__main__":
    unittest.main()
