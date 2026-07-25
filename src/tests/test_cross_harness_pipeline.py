import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cross_harness_pipeline import (
    TIMING_FEATURES,
    XGBClassifier,
    _coverage,
    _load_examples,
    _manifest_rows,
    _manifest_path,
    _mixed_assignment,
    _read_jsonl,
    _selected_record_index,
    evaluate_model,
    load_config,
    prepare_manifests,
    run_harness_detector_lomo,
    scan_inventory,
    summarize_results,
    train_model,
)


class CrossHarnessManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.traces = self.root / "traces"
        self.agents = ["agent_a", "agent_b"]
        self.datasets = {"wiki": {"expected_tasks": {
            "train": 4, "val": 4, "test": 4
        }}}
        config = {
            "experiment": {
                "id": "test",
                "traces_dir": "traces",
                "artifact_root": "artifacts",
                "sampling_seed": 42,
                "classifier_seed": 42,
            },
            "agents": self.agents,
            "datasets": self.datasets,
            "classifiers": {
                "primary": "RandomForest",
                "enabled": ["RandomForest", "XGBoost", "LSTM"],
                "cpu_jobs": 1,
                "random_forest": {"search": False},
                "xgboost": {"device": "cpu", "search_iterations": 1},
                "lstm": {"epochs": 1},
            },
        }
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(config))
        self.cfg = load_config(self.config_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_trace(self, agent, split, harness, task_index, legacy=False):
        dataset_name = f"wiki_{split}"
        run_dir = (
            self.traces / agent / dataset_name / "legacy_run"
            if legacy
            else self.traces / agent / dataset_name / harness / "run"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        episode = {
            "meta": {
                "episode_id": f"{agent}_{split}_{harness}_{task_index}",
                "agent_id": agent,
                "question": f"question {split} {task_index}",
                "timestamp": f"2026-01-01T00:00:{task_index:02d}Z",
            },
            "dom_trace": {
                "events": [{"type": "click", "t_episode": task_index + 1}]
            },
            "verification": {"correct": task_index % 2 == 0},
            "error": None,
        }
        if not legacy:
            episode["meta"]["harness"] = harness
        path = run_dir / f"{episode['meta']['episode_id']}.json"
        path.write_text(json.dumps(episode))

    def _populate(self):
        for agent in self.agents:
            for split in ("train", "val", "test"):
                for task_index in range(4):
                    self._write_trace(
                        agent, split, "midscene", task_index, legacy=True
                    )
                    self._write_trace(
                        agent, split, "browser_use", task_index
                    )

    def test_legacy_midscene_and_browser_use_are_distinct(self):
        self._populate()
        records = scan_inventory(self.cfg)
        harness_counts = {}
        for harness in ("midscene", "browser_use"):
            harness_counts[harness] = sum(
                record.valid_trace and record.harness == harness
                for record in records
            )
        self.assertEqual(harness_counts["midscene"], 24)
        self.assertEqual(harness_counts["browser_use"], 24)

    def test_common_task_universe_and_mixed_balance(self):
        self._populate()
        records = scan_inventory(self.cfg)
        coverage = _coverage(self.cfg, records)
        self.assertEqual(len(coverage["common_task_ids"]["wiki"]["train"]), 4)
        selected = _selected_record_index(records)
        task_ids = coverage["common_task_ids"]["wiki"]["train"]
        rows = _manifest_rows(
            self.cfg, selected, "wiki", "train", "mixed50", task_ids
        )
        harnesses_by_task = {}
        for row in rows:
            harnesses_by_task.setdefault(row["task_id"], set()).add(row["harness"])
        self.assertTrue(all(len(value) == 1 for value in harnesses_by_task.values()))
        assignment = _mixed_assignment(task_ids, 42)
        self.assertEqual(
            list(assignment.values()).count("midscene"), 2
        )
        self.assertEqual(
            list(assignment.values()).count("browser_use"), 2
        )

    def test_prepare_train_and_cross_harness_evaluate_on_cpu(self):
        self._populate()
        records = scan_inventory(self.cfg)
        prepare_manifests(self.cfg, records)
        model_dir = train_model(
            self.cfg,
            "wiki",
            "midscene",
            "RandomForest",
            42,
            quick=True,
        )
        result_path = evaluate_model(
            self.cfg,
            "wiki",
            "midscene",
            "browser_use",
            "RandomForest",
            42,
        )
        self.assertTrue((model_dir / "model.pkl").exists())
        result = json.loads(result_path.read_text())
        self.assertEqual(result["n_test"], 8)
        self.assertEqual(result["class_names"], self.agents)
        summary = summarize_results(self.cfg)
        self.assertIn(
            "midscene,browser_use,full,RandomForest",
            summary.read_text(),
        )

    def test_feature_groups_partition_full_schema_without_touching_traces(self):
        self._populate()
        records = scan_inventory(self.cfg)
        prepare_manifests(self.cfg, records)
        rows = _read_jsonl(
            _manifest_path(self.cfg, "wiki", "train", "midscene")
        )
        _, _, _, full = _load_examples(rows, False, self.cfg, "full")
        _, _, _, timing = _load_examples(
            rows, False, self.cfg, "timing_only"
        )
        _, _, _, non_timing = _load_examples(
            rows, False, self.cfg, "non_timing"
        )
        self.assertEqual(set(timing), set(TIMING_FEATURES))
        self.assertFalse(set(timing) & set(non_timing))
        self.assertEqual(set(full), set(timing) | set(non_timing))

    @unittest.skipUnless(XGBClassifier is not None, "xgboost is not installed")
    def test_xgboost_quick_path_runs_on_cpu(self):
        self._populate()
        records = scan_inventory(self.cfg)
        prepare_manifests(self.cfg, records)
        train_model(
            self.cfg,
            "wiki",
            "browser_use",
            "XGBoost",
            7,
            quick=True,
            xgb_device="cpu",
        )
        result_path = evaluate_model(
            self.cfg,
            "wiki",
            "browser_use",
            "midscene",
            "XGBoost",
            7,
        )
        result = json.loads(result_path.read_text())
        self.assertEqual(result["classifier"], "XGBoost")
        self.assertEqual(result["n_test"], 8)

    def test_lstm_quick_path_stays_on_cpu(self):
        self._populate()
        records = scan_inventory(self.cfg)
        prepare_manifests(self.cfg, records)
        model_dir = train_model(
            self.cfg,
            "wiki",
            "mixed50",
            "LSTM",
            9,
            quick=True,
        )
        result_path = evaluate_model(
            self.cfg,
            "wiki",
            "mixed50",
            "browser_use",
            "LSTM",
            9,
        )
        with (model_dir / "model.pkl").open("rb") as handle:
            import pickle
            bundle = pickle.load(handle)
        self.assertEqual(bundle["lstm_device"], "cpu")
        self.assertEqual(json.loads(result_path.read_text())["n_test"], 8)

    @unittest.skipUnless(XGBClassifier is not None, "xgboost is not installed")
    def test_binary_harness_detector_leave_one_model_out(self):
        self._populate()
        records = scan_inventory(self.cfg)
        prepare_manifests(self.cfg, records)
        result_path = run_harness_detector_lomo(
            self.cfg,
            "wiki",
            "XGBoost",
            11,
            quick=True,
            xgb_device="cpu",
        )
        result = json.loads(result_path.read_text())
        self.assertEqual(result["protocol"], "leave_one_model_out")
        self.assertEqual(len(result["folds"]), 2)
        self.assertEqual(
            {fold["held_out_agent"] for fold in result["folds"]},
            set(self.agents),
        )
        self.assertTrue(all(fold["n_test"] == 8 for fold in result["folds"]))
        self.assertIn("fold_auroc_mean", result)
        self.assertTrue(
            all("auroc" in fold["test"] for fold in result["folds"])
        )


if __name__ == "__main__":
    unittest.main()
