import unittest

from experiments.closed_set_scaling.pipeline import _ordered_agents


class ClosedSetScalingTests(unittest.TestCase):
    def test_seeded_model_orders_are_permutations_and_nested(self):
        cfg = {"agents": [f"model-{index}" for index in range(14)]}
        first = _ordered_agents(cfg, "webshop", 40)
        second = _ordered_agents(cfg, "webshop", 41)
        self.assertEqual(set(first), set(cfg["agents"]))
        self.assertEqual(len(first), len(set(first)))
        self.assertNotEqual(first, second)
        for class_count in range(2, 14):
            self.assertEqual(
                set(first[:class_count]),
                set(first[: class_count + 1]) - {first[class_count]},
            )


if __name__ == "__main__":
    unittest.main()
