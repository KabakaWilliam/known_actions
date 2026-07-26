import json
import tempfile
import unittest
from pathlib import Path

from closed_set_figure_data import load_closed_set_intervals


class ClosedSetFigureDataTests(unittest.TestCase):
    @staticmethod
    def _class_row(
        class_index,
        class_name,
        lower=0.70,
        estimate=0.75,
        upper=0.80,
    ):
        return {
            "class_index": class_index,
            "class_name": class_name,
            "estimate": estimate,
            "confidence_interval": {
                "lower": lower,
                "upper": upper,
                "level": 0.95,
                "method": "paired_percentile_bootstrap_over_test_traces",
            },
            "seed_variability": {
                "mean": estimate,
                "sample_std": 0.01,
                "min": estimate - 0.01,
                "max": estimate + 0.01,
            },
            "per_seed": [
                {"seed": seed, "f1": estimate}
                for seed in range(5)
            ],
        }

    @classmethod
    def _payload(cls):
        dataset = {
            "models": {
                "XGBoost": {
                    "macro_f1": {
                        "estimate": 0.75,
                        "confidence_interval": {
                            "lower": 0.70,
                            "upper": 0.80,
                            "level": 0.95,
                        },
                        "per_class": [
                            cls._class_row(0, "agent_a", 0.60, 0.70, 0.80),
                            cls._class_row(1, "agent_b", 0.75, 0.80, 0.90),
                        ],
                    }
                }
            }
        }
        return {
            "schema_version": 1,
            "datasets": {
                key: dataset
                for key in ("wiki", "frames", "webshop", "deepshop")
            },
        }

    def _load(self, payload, expected_classes_by_tag=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "closed_set.json"
            path.write_text(json.dumps(payload))
            return load_closed_set_intervals(
                path,
                "XGBoost",
                expected_classes_by_tag=expected_classes_by_tag,
            )

    def test_maps_per_class_intervals_to_existing_figure_tags(self):
        intervals = self._load(self._payload())
        self.assertEqual(
            set(intervals),
            {
                "wiki_ood_all",
                "frames_ood_all",
                "webshop_ood_all",
                "deepshop_ood_all",
            },
        )
        self.assertEqual(
            intervals["wiki_ood_all"]["agent_a"],
            {
                "class_index": 0,
                "estimate": 0.70,
                "lower": 0.60,
                "upper": 0.80,
                "level": 0.95,
            },
        )

    def test_can_load_and_validate_only_requested_figure_tags(self):
        intervals = self._load(
            self._payload(),
            {"frames_ood_all": {"agent_a", "agent_b"}},
        )
        self.assertEqual(set(intervals), {"frames_ood_all"})

    def test_rejects_missing_per_class_statistics(self):
        payload = self._payload()
        del payload["datasets"]["wiki"]["models"]["XGBoost"]["macro_f1"][
            "per_class"
        ]
        with self.assertRaisesRegex(ValueError, r"macro_f1\.per_class"):
            self._load(payload)

    def test_rejects_interval_that_does_not_contain_estimate(self):
        payload = self._payload()
        payload["datasets"]["wiki"]["models"]["XGBoost"]["macro_f1"][
            "per_class"
        ][0]["confidence_interval"]["lower"] = 0.71
        with self.assertRaisesRegex(ValueError, "must satisfy"):
            self._load(payload)

    def test_rejects_duplicate_class_names(self):
        payload = self._payload()
        payload["datasets"]["wiki"]["models"]["XGBoost"]["macro_f1"][
            "per_class"
        ][1]["class_name"] = "agent_a"
        with self.assertRaisesRegex(ValueError, "duplicate class_name"):
            self._load(payload)

    def test_rejects_noncontiguous_class_indices(self):
        payload = self._payload()
        payload["datasets"]["wiki"]["models"]["XGBoost"]["macro_f1"][
            "per_class"
        ][1]["class_index"] = 2
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self._load(payload)

    def test_rejects_mismatched_figure_class_names(self):
        with self.assertRaisesRegex(
            ValueError,
            r"missing from stats: 'agent_c'.*unexpected in stats: 'agent_b'",
        ):
            self._load(
                self._payload(),
                {"wiki_ood_all": {"agent_a", "agent_c"}},
            )

    def test_rejects_unknown_figure_tag(self):
        with self.assertRaisesRegex(ValueError, "no closed-set dataset mapping"):
            self._load(
                self._payload(),
                {"unknown_ood_all": {"agent_a", "agent_b"}},
            )


if __name__ == "__main__":
    unittest.main()
