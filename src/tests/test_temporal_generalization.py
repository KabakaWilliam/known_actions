import json
import tempfile
import unittest
from pathlib import Path

from experiments.temporal_generalization.pipeline import (
    _read_jsonl,
    inventory,
    prepare,
    scan,
)
from experiments.cross_harness.pipeline import _manifest_path


class TemporalGeneralizationManifestTests(unittest.TestCase):
    def _write_trace(
        self,
        root: Path,
        agent: str,
        split: str,
        question: str,
        model: str,
        timestamp: str,
        *,
        explicit_harness: bool = False,
    ) -> None:
        directory = root / agent / f"webshop_{split}"
        if explicit_harness:
            directory = directory / "midscene"
        path = directory / "run" / f"{question}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "meta": {
                        "agent_id": agent,
                        "episode_id": f"{agent}-{split}-{question}",
                        "model_name": model,
                        "question": question,
                        "timestamp": timestamp,
                    },
                    "dom_trace": {
                        "events": [{"type": "click", "timestamp": 1}]
                    },
                    "error": None,
                    "task_success": False,
                }
            )
        )

    def test_prepare_uses_only_original_train_val_and_matched_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            future = root / "future"
            aliases = {"a": ["model-a"], "b": ["model-b"]}
            for agent, models in aliases.items():
                model = models[0]
                self._write_trace(
                    original, agent, "train", "train-task", model, "2026-04-01T00:00:00Z"
                )
                self._write_trace(
                    original, agent, "val", "val-task", model, "2026-04-02T00:00:00Z"
                )
                self._write_trace(
                    original, agent, "test", "test-task", model, "2026-04-03T00:00:00Z"
                )
                self._write_trace(
                    future,
                    agent,
                    "test",
                    "test-task",
                    model,
                    "2026-07-26T00:00:00Z",
                    explicit_harness=True,
                )
            cfg = {
                "experiment": {
                    "id": "test",
                    "artifact_root": root / "artifacts",
                    "original_traces_dir": original,
                    "future_traces_dir": future,
                    "sampling_seed": 42,
                },
                "artifact_layout": {
                    "frozen_manifests": "frozen",
                    "model_identity": "models",
                    "identity_summaries": "summaries",
                },
                "agents": ["a", "b"],
                "model_aliases": aliases,
                "datasets": {
                    "webshop": {
                        "expected_tasks": {"train": 1, "val": 1, "test": 1},
                        "minimum_original_tasks": {"train": 1, "val": 1},
                        "minimum_matched_test_tasks": 1,
                    }
                },
                "_config_path": root / "config.yaml",
            }
            records = scan(cfg)
            common = inventory(cfg, records)
            self.assertEqual(
                {split: len(tasks) for split, tasks in common["webshop"].items()},
                {"train": 1, "val": 1, "test": 1},
            )
            prepare(cfg)

            train = _read_jsonl(
                _manifest_path(cfg, "webshop", "train", "original")
            )
            old_test = _read_jsonl(
                _manifest_path(cfg, "webshop", "test", "original")
            )
            future_test = _read_jsonl(
                _manifest_path(cfg, "webshop", "test", "future")
            )
            self.assertEqual({row["wave"] for row in train}, {"original"})
            self.assertEqual({row["wave"] for row in old_test}, {"original"})
            self.assertEqual({row["wave"] for row in future_test}, {"future"})
            self.assertEqual(
                {(row["agent_id"], row["task_id"]) for row in old_test},
                {(row["agent_id"], row["task_id"]) for row in future_test},
            )


if __name__ == "__main__":
    unittest.main()
