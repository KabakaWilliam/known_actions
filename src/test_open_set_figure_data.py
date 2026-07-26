import json
import tempfile
import unittest
from pathlib import Path

from open_set_figure_data import load_open_set_intervals


class OpenSetFigureDataTests(unittest.TestCase):
    @staticmethod
    def _held_out_leaf(lower=0.60, estimate=0.70, upper=0.80):
        seeds = list(range(5))
        return {
            "classifier_seeds": seeds,
            "n_classifier_seeds": len(seeds),
            "models": {
                "XGBoost": {
                    "auroc": {
                        "estimate": estimate,
                        "confidence_interval": {
                            "lower": lower,
                            "upper": upper,
                            "level": 0.95,
                            "method": (
                                "paired_stratified_percentile_bootstrap_"
                                "over_evaluation_traces"
                            ),
                        },
                        "per_seed": [
                            {"seed": seed, "auroc": estimate}
                            for seed in seeds
                        ],
                        "n_known": 130,
                        "n_unknown": 10,
                    }
                }
            },
        }

    @classmethod
    def _payload(cls):
        return {
            "schema_version": 1,
            "default_classifier_seed_count": 5,
            "classifier_seed_count": 5,
            "classifier_seeds": list(range(5)),
            "datasets": {
                "wiki": {
                    "tag": "2wikimultihop_open_set",
                    "held_out_models": {
                        "agent_a": cls._held_out_leaf(0.50, 0.60, 0.70),
                        "agent_b": cls._held_out_leaf(0.70, 0.80, 0.90),
                    },
                },
                "frames": {
                    "tag": "frames_open_set",
                    "held_out_models": {
                        "agent_a": cls._held_out_leaf(0.55, 0.65, 0.75),
                        "agent_b": cls._held_out_leaf(0.65, 0.75, 0.85),
                    },
                },
            },
        }

    def _load(self, payload, expected_agents_by_tag=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "open_set.json"
            path.write_text(json.dumps(payload))
            return load_open_set_intervals(
                path,
                "XGBoost",
                expected_agents_by_tag=expected_agents_by_tag,
            )

    def test_maps_intervals_by_dataset_tag(self):
        intervals = self._load(self._payload())
        self.assertEqual(
            set(intervals),
            {"2wikimultihop_open_set", "frames_open_set"},
        )
        self.assertEqual(
            intervals["2wikimultihop_open_set"]["agent_a"],
            {
                "estimate": 0.60,
                "lower": 0.50,
                "upper": 0.70,
                "level": 0.95,
            },
        )

    def test_can_load_and_validate_only_requested_tag(self):
        intervals = self._load(
            self._payload(),
            {"frames_open_set": {"agent_a", "agent_b"}},
        )
        self.assertEqual(set(intervals), {"frames_open_set"})

    def test_rejects_missing_requested_tag(self):
        with self.assertRaisesRegex(
            ValueError, "no open-set dataset entry.*'webshop_open_set'"
        ):
            self._load(
                self._payload(),
                {"webshop_open_set": {"agent_a", "agent_b"}},
            )

    def test_rejects_duplicate_dataset_tag(self):
        payload = self._payload()
        payload["datasets"]["frames"]["tag"] = "2wikimultihop_open_set"
        with self.assertRaisesRegex(ValueError, "duplicate dataset tag"):
            self._load(payload)

    def test_rejects_missing_classifier_statistics(self):
        payload = self._payload()
        del payload["datasets"]["wiki"]["held_out_models"]["agent_a"][
            "models"
        ]["XGBoost"]
        with self.assertRaisesRegex(ValueError, "has no 'XGBoost' results"):
            self._load(payload)

    def test_rejects_interval_that_does_not_contain_estimate(self):
        payload = self._payload()
        payload["datasets"]["wiki"]["held_out_models"]["agent_a"]["models"][
            "XGBoost"
        ]["auroc"]["confidence_interval"]["lower"] = 0.61
        with self.assertRaisesRegex(ValueError, "must satisfy"):
            self._load(payload)

    def test_rejects_mismatched_held_out_model_names(self):
        with self.assertRaisesRegex(
            ValueError,
            r"missing from stats: 'agent_c'.*unexpected in stats: 'agent_b'",
        ):
            self._load(
                self._payload(),
                {"2wikimultihop_open_set": {"agent_a", "agent_c"}},
            )

    def test_rejects_fewer_than_five_classifier_seeds(self):
        payload = self._payload()
        payload["default_classifier_seed_count"] = 4
        with self.assertRaisesRegex(
            ValueError, "default_classifier_seed_count.*>= 5"
        ):
            self._load(payload)

    def test_rejects_leaf_seeds_that_differ_from_aggregate(self):
        payload = self._payload()
        leaf = payload["datasets"]["wiki"]["held_out_models"]["agent_a"]
        leaf["classifier_seeds"] = [5, 6, 7, 8, 9]
        leaf["models"]["XGBoost"]["auroc"]["per_seed"] = [
            {"seed": seed, "auroc": 0.60}
            for seed in leaf["classifier_seeds"]
        ]
        with self.assertRaisesRegex(
            ValueError, "do not match the aggregate classifier_seeds"
        ):
            self._load(payload)


if __name__ == "__main__":
    unittest.main()
