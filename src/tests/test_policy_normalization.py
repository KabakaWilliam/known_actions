import json
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.policy_normalization.pipeline import (
    _common_tasks,
    _latest_valid_records,
    load_config,
    prepare,
)


class PolicyNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.canonical = self.root / "canonical"
        self.normalized = self.root / "normalized"
        self.artifacts = self.root / "artifacts"
        self.agents = ["agent_a", "agent_b"]
        config = {
            "experiment": {
                "id": "policy_test",
                "artifact_root": str(self.artifacts),
                "sampling_seed": 42,
                "classifier_seed": 42,
            },
            "agents": self.agents,
            "conditions": {
                "canonical": {
                    "traces_dir": str(self.canonical),
                    "harness": "browser_use",
                },
                "normalized_policy": {
                    "traces_dir": str(self.normalized),
                    "harness": "browser_use",
                },
            },
            "datasets": {
                "webshop": {
                    "expected_tasks": {
                        "train": 4,
                        "val": 2,
                        "test": 2,
                    },
                    "minimum_common_tasks": {
                        "train": 4,
                        "val": 2,
                        "test": 2,
                    },
                }
            },
            "classifiers": {
                "primary": "XGBoost",
                "enabled": ["XGBoost"],
                "cpu_jobs": 1,
                "xgboost": {"device": "cpu", "search_iterations": 1},
                "random_forest": {"search": False},
                "lstm": {"epochs": 1},
            },
            "evaluation": {
                "classifier_seeds": [1, 2],
                "bootstrap_samples": 20,
                "bootstrap_confidence": 0.95,
            },
        }
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(config))
        self.cfg = load_config(self.config_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _populate(self):
        counts = {"train": 4, "val": 2, "test": 2}
        for condition, root in (
            ("canonical", self.canonical),
            ("normalized_policy", self.normalized),
        ):
            for agent in self.agents:
                for split, count in counts.items():
                    run = root / agent / f"webshop_{split}" / "browser_use" / "run"
                    run.mkdir(parents=True, exist_ok=True)
                    for index in range(count):
                        episode = {
                            "meta": {
                                "episode_id": (f"{condition}_{agent}_{split}_{index}"),
                                "agent_id": agent,
                                "question": f"{split} question {index}",
                                "timestamp": (f"2026-01-01T00:00:{index:02d}Z"),
                            },
                            "dom_trace": {
                                "events": [
                                    {
                                        "type": "click",
                                        "t_episode": index + 1,
                                    }
                                ]
                            },
                            "verification": {"correct": index % 2 == 0},
                            "error": None,
                        }
                        (run / f"{episode['meta']['episode_id']}.json").write_text(
                            json.dumps(episode)
                        )

    def test_condition_roots_are_matched_and_frozen_separately(self):
        self._populate()
        records = _latest_valid_records(self.cfg)
        common = _common_tasks(self.cfg, records)
        self.assertEqual(len(common["webshop"]["train"]), 4)
        prepare(self.cfg)
        canonical_manifest = (
            self.artifacts / "splits" / "webshop" / "seed=42" / "train_canonical.jsonl"
        )
        normalized_manifest = canonical_manifest.with_name(
            "train_normalized_policy.jsonl"
        )
        canonical_rows = [
            json.loads(line) for line in canonical_manifest.read_text().splitlines()
        ]
        normalized_rows = [
            json.loads(line) for line in normalized_manifest.read_text().splitlines()
        ]
        self.assertTrue(
            all(str(self.canonical) in row["trace_path"] for row in canonical_rows)
        )
        self.assertTrue(
            all(str(self.normalized) in row["trace_path"] for row in normalized_rows)
        )
        self.assertEqual(
            {row["task_id"] for row in canonical_rows},
            {row["task_id"] for row in normalized_rows},
        )
        utility = json.loads(
            (self.artifacts / "summaries" / "task_success.json").read_text()
        )
        self.assertTrue(utility)
        self.assertEqual(
            {row["condition"] for row in utility},
            {"canonical", "normalized_policy"},
        )
        self.assertTrue(
            all(
                row["success_ci_lower"]
                <= row["success_rate"]
                <= row["success_ci_upper"]
                for row in utility
            )
        )


if __name__ == "__main__":
    unittest.main()
