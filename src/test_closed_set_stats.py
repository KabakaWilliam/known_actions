import unittest

import numpy as np
from sklearn.metrics import f1_score

from closed_set_stats import (
    DEFAULT_CLASSIFIER_SEED_COUNT,
    generate_classifier_seeds,
    macro_f1_from_encoded,
    summarize_macro_f1,
    validate_classifier_seeds,
)


class ClosedSetStatsTests(unittest.TestCase):
    def test_generates_ten_unique_seeds_by_default(self):
        seeds = generate_classifier_seeds()
        self.assertEqual(len(seeds), DEFAULT_CLASSIFIER_SEED_COUNT)
        self.assertEqual(len(set(seeds)), DEFAULT_CLASSIFIER_SEED_COUNT)
        self.assertTrue(all(0 <= seed < 2**31 for seed in seeds))

    def test_generated_seed_count_still_requires_at_least_five(self):
        with self.assertRaisesRegex(ValueError, "at least 5"):
            generate_classifier_seeds(4)

    def test_requires_five_unique_classifier_seeds(self):
        with self.assertRaisesRegex(ValueError, "at least 5"):
            validate_classifier_seeds([1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_classifier_seeds([1, 2, 3, 4, 4])
        self.assertEqual(
            validate_classifier_seeds([10, 11, 12, 13, 14]),
            [10, 11, 12, 13, 14],
        )

    def test_explicit_classifier_seeds_stay_in_supported_range(self):
        with self.assertRaisesRegex(ValueError, r"\[0,"):
            validate_classifier_seeds([-1, 1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, r"\[0,"):
            validate_classifier_seeds([1, 2, 3, 4, 2**31])

    def test_macro_f1_uses_fixed_class_universe(self):
        # Class 2 is absent and therefore contributes zero to macro-F1.
        score = macro_f1_from_encoded(
            np.array([0, 0, 1, 1]),
            np.array([0, 0, 1, 1]),
            n_classes=3,
        )
        self.assertAlmostEqual(score, 2.0 / 3.0)

    def test_macro_f1_matches_sklearn(self):
        y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 1, 0, 1, 2, 2, 0, 2])
        expected = f1_score(
            y_true,
            y_pred,
            labels=np.arange(3),
            average="macro",
            zero_division=0,
        )
        self.assertAlmostEqual(
            macro_f1_from_encoded(y_true, y_pred, n_classes=3),
            expected,
        )

    def test_perfect_predictions_have_degenerate_interval(self):
        y_true = np.array([0, 0, 0, 0, 0, 0])
        predictions = {
            seed: y_true.copy()
            for seed in [40, 41, 42, 43, 44]
        }
        result = summarize_macro_f1(
            y_true,
            predictions,
            n_classes=1,
            bootstrap_replicates=100,
            bootstrap_seed=7,
        )
        self.assertEqual(result["estimate"], 1.0)
        self.assertEqual(result["confidence_interval"]["lower"], 1.0)
        self.assertEqual(result["confidence_interval"]["upper"], 1.0)
        self.assertEqual(result["seed_variability"]["sample_std"], 0.0)

    def test_rejects_negative_bootstrap_seed_before_sampling(self):
        y_true = np.array([0, 1])
        predictions = {
            seed: y_true.copy()
            for seed in [40, 41, 42, 43, 44]
        }
        with self.assertRaisesRegex(ValueError, "non-negative"):
            summarize_macro_f1(
                y_true,
                predictions,
                n_classes=2,
                bootstrap_replicates=1,
                bootstrap_seed=-1,
            )

    def test_bootstrap_is_reproducible_and_seed_averaged(self):
        y_true = np.array([0, 0, 1, 1])
        predictions = {
            1: np.array([0, 0, 1, 1]),
            2: np.array([0, 1, 1, 1]),
            3: np.array([0, 0, 0, 1]),
            4: np.array([1, 0, 1, 1]),
            5: np.array([0, 0, 1, 0]),
        }
        first = summarize_macro_f1(
            y_true,
            predictions,
            n_classes=2,
            bootstrap_replicates=200,
            bootstrap_seed=99,
        )
        second = summarize_macro_f1(
            y_true,
            predictions,
            n_classes=2,
            bootstrap_replicates=200,
            bootstrap_seed=99,
        )
        per_seed_mean = np.mean(
            [row["macro_f1"] for row in first["per_seed"]]
        )
        self.assertAlmostEqual(first["estimate"], per_seed_mean)
        self.assertEqual(
            first["confidence_interval"],
            second["confidence_interval"],
        )

    def test_per_class_summary_is_seed_averaged_and_named(self):
        y_true = np.array([0, 0, 1, 1])
        predictions = {
            1: np.array([0, 0, 1, 1]),
            2: np.array([0, 1, 1, 1]),
            3: np.array([0, 0, 0, 1]),
            4: np.array([1, 0, 1, 1]),
            5: np.array([0, 0, 1, 0]),
        }
        result = summarize_macro_f1(
            y_true,
            predictions,
            n_classes=2,
            class_names=["model-a", "model-b"],
            bootstrap_replicates=200,
            bootstrap_seed=99,
        )

        self.assertEqual(
            [row["class_index"] for row in result["per_class"]],
            [0, 1],
        )
        self.assertEqual(
            [row["class_name"] for row in result["per_class"]],
            ["model-a", "model-b"],
        )
        for class_index, row in enumerate(result["per_class"]):
            expected_per_seed = [
                f1_score(
                    y_true == class_index,
                    prediction == class_index,
                    zero_division=0,
                )
                for prediction in predictions.values()
            ]
            self.assertAlmostEqual(row["estimate"], np.mean(expected_per_seed))
            self.assertAlmostEqual(
                row["estimate"],
                np.mean([item["f1"] for item in row["per_seed"]]),
            )
            self.assertEqual(
                [item["seed"] for item in row["per_seed"]],
                list(predictions),
            )

    def test_per_class_interval_uses_paired_trace_resamples(self):
        # With one trace and opposite seed predictions, every paired replicate
        # has the same seed-averaged class scores: 0.5 for class 0 and 0 for 1.
        y_true = np.array([0])
        predictions = {
            1: np.array([0]),
            2: np.array([0]),
            3: np.array([0]),
            4: np.array([1]),
            5: np.array([1]),
            6: np.array([1]),
        }
        result = summarize_macro_f1(
            y_true,
            predictions,
            n_classes=2,
            bootstrap_replicates=20,
            bootstrap_seed=123,
        )

        class_zero, class_one = result["per_class"]
        self.assertEqual(class_zero["estimate"], 0.5)
        self.assertEqual(class_zero["confidence_interval"]["lower"], 0.5)
        self.assertEqual(class_zero["confidence_interval"]["upper"], 0.5)
        self.assertEqual(class_one["estimate"], 0.0)
        self.assertEqual(class_one["confidence_interval"]["lower"], 0.0)
        self.assertEqual(class_one["confidence_interval"]["upper"], 0.0)

    def test_class_names_are_optional_and_validated(self):
        y_true = np.array([0, 1])
        predictions = {
            seed: y_true.copy()
            for seed in [40, 41, 42, 43, 44]
        }
        unnamed = summarize_macro_f1(
            y_true,
            predictions,
            n_classes=2,
            bootstrap_replicates=1,
        )
        self.assertNotIn("class_name", unnamed["per_class"][0])

        invalid_names = (
            ["only-one"],
            ["a", ""],
            ["duplicate", "duplicate"],
        )
        for names in invalid_names:
            with self.subTest(names=names):
                with self.assertRaisesRegex(ValueError, "class_names"):
                    summarize_macro_f1(
                        y_true,
                        predictions,
                        n_classes=2,
                        class_names=names,
                        bootstrap_replicates=1,
                    )


if __name__ == "__main__":
    unittest.main()
