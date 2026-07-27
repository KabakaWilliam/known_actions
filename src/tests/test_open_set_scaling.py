import unittest

import numpy as np

from experiments.open_set_scaling.pipeline import (
    _balanced_holdout_sets,
    _balanced_open_set_macro_f1,
    _confidence_threshold,
    _oscr,
)


class OpenSetScalingTests(unittest.TestCase):
    def test_balanced_holdouts_are_deterministic_unique_and_balanced(self):
        agents = [f"model-{index:02d}" for index in range(14)]
        first = _balanced_holdout_sets(agents, 4, 100, 20260727)
        second = _balanced_holdout_sets(agents, 4, 100, 20260727)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertEqual(len(set(first)), 100)
        counts = {
            agent: sum(agent in holdout for holdout in first) for agent in agents
        }
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_all_requested_returns_every_combination(self):
        agents = ["a", "b", "c", "d"]
        sets = _balanced_holdout_sets(agents, 2, "all", 1)
        self.assertEqual(len(sets), 6)

    def test_confidence_threshold_targets_lower_tail(self):
        confidence = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(_confidence_threshold(confidence, 0.8), 0.1)

    def test_oscr_is_one_for_perfect_separation_and_identification(self):
        score = _oscr(
            np.asarray([0.8, 0.9]),
            np.asarray([True, True]),
            np.asarray([0.1, 0.2]),
        )
        self.assertAlmostEqual(score, 1.0)

    def test_balanced_open_set_macro_f1_is_one_when_perfect(self):
        score = _balanced_open_set_macro_f1(
            known_true=np.asarray([0, 1]),
            known_predicted=np.asarray([0, 1]),
            known_confidence=np.asarray([0.9, 0.8]),
            unknown_predicted=np.asarray([0, 1, 0, 1]),
            unknown_confidence=np.asarray([0.1, 0.2, 0.1, 0.2]),
            threshold=0.5,
            n_known_classes=2,
            unknown_count=2,
        )
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
