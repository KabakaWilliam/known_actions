import re
import unittest

from open_set_subset_design import (
    build_subset_design,
    canonical_subset_id,
    enumerate_subsets,
    make_subset_id,
    select_holdout_subsets,
    validate_model_universe,
)


class ModelUniverseValidationTests(unittest.TestCase):
    def test_accepts_a_sorted_unique_universe(self):
        self.assertEqual(
            validate_model_universe(["alpha", "beta", "gamma"]),
            ("alpha", "beta", "gamma"),
        )

    def test_rejects_invalid_universes(self):
        invalid_values = [
            [],
            "alpha",
            ["beta", "alpha"],
            ["alpha", "alpha"],
            ["alpha", ""],
            ["alpha", " beta"],
            ["alpha", 2],
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_model_universe(value)


class SubsetEnumerationTests(unittest.TestCase):
    def test_enumerates_all_combinations_in_canonical_order(self):
        self.assertEqual(
            enumerate_subsets(("a", "b", "c", "d"), 2),
            (
                ("a", "b"),
                ("a", "c"),
                ("a", "d"),
                ("b", "c"),
                ("b", "d"),
                ("c", "d"),
            ),
        )

    def test_rejects_invalid_holdout_sizes(self):
        universe = ("a", "b", "c")
        for size in (True, 0, -1, 4, 1.5):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    enumerate_subsets(universe, size)


class SubsetIdentifierTests(unittest.TestCase):
    def test_id_is_order_independent_stable_and_filesystem_safe(self):
        expected = make_subset_id(("alpha", "beta"))
        self.assertEqual(make_subset_id(("beta", "alpha")), expected)
        self.assertEqual(make_subset_id(("alpha", "beta")), expected)
        self.assertEqual(canonical_subset_id(("beta", "alpha")), expected)
        self.assertRegex(expected, re.compile(r"^h2-[0-9a-f]{20}$"))
        self.assertNotEqual(expected, make_subset_id(("alpha", "gamma")))

    def test_rejects_invalid_subset_members(self):
        for models in ([], ["a", "a"], ["a", ""], ["a", 2], "a"):
            with self.subTest(models=models):
                with self.assertRaises(ValueError):
                    make_subset_id(models)


class SubsetDesignTests(unittest.TestCase):
    def setUp(self):
        self.universe = tuple(f"model-{index:02d}" for index in range(14))

    def test_none_or_large_cap_returns_exhaustive_design(self):
        uncapped = build_subset_design(self.universe, 2, cap=None, seed=11)
        large_cap = build_subset_design(self.universe, 2, cap=100, seed=99)

        self.assertEqual(uncapped.selection_mode, "exhaustive")
        self.assertEqual(uncapped.possible_count, 91)
        self.assertEqual(uncapped.evaluated_count, 91)
        self.assertEqual(uncapped, large_cap)
        self.assertEqual(set(uncapped.inclusion_counts.values()), {13})
        self.assertEqual(
            [record.models for record in uncapped.subsets],
            sorted(record.models for record in uncapped.subsets),
        )

    def test_capped_design_is_deterministic_unique_and_balanced(self):
        first = build_subset_design(self.universe, 3, cap=100, seed=1234)
        again = build_subset_design(self.universe, 3, cap=100, seed=1234)

        self.assertEqual(first, again)
        self.assertEqual(first.selection_mode, "balanced_sample")
        self.assertEqual(first.possible_count, 364)
        self.assertEqual(first.evaluated_count, 100)
        self.assertEqual(len({item.models for item in first.subsets}), 100)
        self.assertEqual(len({item.subset_id for item in first.subsets}), 100)
        self.assertTrue(
            all(item.models == tuple(sorted(item.models)) for item in first.subsets)
        )
        counts = list(first.inclusion_counts.values())
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(sum(counts), 3 * first.evaluated_count)

    def test_seed_changes_a_capped_selection(self):
        first = build_subset_design(self.universe, 3, cap=40, seed=1)
        second = build_subset_design(self.universe, 3, cap=40, seed=2)
        self.assertNotEqual(
            {item.models for item in first.subsets},
            {item.models for item in second.subsets},
        )

    def test_small_cap_remains_balanced(self):
        design = build_subset_design(self.universe, 2, cap=5, seed=42)
        counts = list(design.inclusion_counts.values())
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(sum(counts), 10)

    def test_rejects_invalid_cap_and_seed(self):
        invalid_caps = (0, -1, True, 2.5)
        for cap in invalid_caps:
            with self.subTest(cap=cap):
                with self.assertRaises(ValueError):
                    build_subset_design(self.universe, 2, cap=cap)

        invalid_seeds = (-1, True, 2.5)
        for seed in invalid_seeds:
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    build_subset_design(self.universe, 2, seed=seed)

    def test_runner_compatibility_selection_exposes_plain_tuples(self):
        selection = select_holdout_subsets(
            self.universe,
            3,
            max_subsets=25,
            seed=77,
        )
        self.assertEqual(selection.possible_count, 364)
        self.assertEqual(selection.selection_mode, "balanced_sample")
        self.assertEqual(len(selection.subsets), 25)
        self.assertTrue(
            all(
                isinstance(subset, tuple)
                and subset == tuple(sorted(subset))
                and len(subset) == 3
                for subset in selection.subsets
            )
        )
        counts = list(selection.model_inclusion_counts.values())
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(sum(counts), 75)


if __name__ == "__main__":
    unittest.main()
