import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from open_set_subset_design import (
    canonical_subset_id,
    select_holdout_subsets,
)


_trace_analyzer_stub = types.ModuleType("trace_analyzer")
_trace_analyzer_stub.XGB_PARAM_DIST = {
    "max_depth": [3, 4],
    "n_estimators": [10, 20],
}
_trace_analyzer_stub._infer_split = lambda name: next(
    (
        split
        for split in ("train", "val", "test", "ood")
        if name.endswith(f"_{split}")
    ),
    None,
)
_trace_analyzer_stub._is_valid_trace = lambda episode: True
_trace_analyzer_stub.extract_features = lambda episode: {"x": 1.0}
_runner_path = Path(__file__).with_name("run_open_set_multi_stats.py")
_runner_spec = importlib.util.spec_from_file_location(
    "_run_open_set_multi_stats_under_test",
    _runner_path,
)
assert _runner_spec is not None and _runner_spec.loader is not None
runner = importlib.util.module_from_spec(_runner_spec)
with patch.dict(
    sys.modules,
    {
        "trace_analyzer": _trace_analyzer_stub,
        _runner_spec.name: runner,
    },
):
    _runner_spec.loader.exec_module(runner)


SEEDS = [101, 102, 103, 104, 105]


class FixedSplitTests(unittest.TestCase):
    def test_protocol_names_the_validity_filter_that_the_runner_imports(self):
        metadata = runner._protocol_metadata([{"max_depth": 3}])

        self.assertEqual(
            metadata["validity_filter"],
            "trace_analyzer._is_valid_trace",
        )

    def test_question_group_split_is_stable_and_shared_across_models(self):
        question = "Which event happened first?"
        first = runner._question_group_split("frames", question)
        second = runner._question_group_split("frames", question)

        self.assertEqual(first, second)
        self.assertIn(first, {"train", "val", "test"})

    def test_dataset_key_is_part_of_question_split(self):
        # The exact split may coincide, so compare the underlying stable values.
        question = "same task text"
        self.assertNotEqual(
            runner._stable_unit_interval(
                runner.PROTOCOL_NAME,
                "frames",
                question,
            ),
            runner._stable_unit_interval(
                runner.PROTOCOL_NAME,
                "deepshop",
                question,
            ),
        )

    def test_fixed_cap_selection_is_deterministic(self):
        records = [
            runner.TraceRecord(
                features={"x": float(index)},
                episode_id=f"episode-{index}",
                question=f"question-{index}",
            )
            for index in range(20)
        ]
        first = runner._cap_records(
            "frames",
            "model-a",
            "test",
            records,
            7,
        )
        second = runner._cap_records(
            "frames",
            "model-a",
            "test",
            list(reversed(records)),
            7,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)


class BatchManifestTests(unittest.TestCase):
    def test_generated_seeds_are_reused_on_resume(self):
        settings = {"protocol": {"name": "synthetic"}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            with patch.object(
                runner,
                "generate_classifier_seeds",
                return_value=list(SEEDS),
            ):
                first = runner.resolve_batch_manifest(
                    path,
                    requested_seeds=None,
                    requested_seed_count=len(SEEDS),
                    settings=settings,
                    resume=True,
                )
            with patch.object(
                runner,
                "generate_classifier_seeds",
                side_effect=AssertionError("must reuse persisted seeds"),
            ):
                resumed = runner.resolve_batch_manifest(
                    path,
                    requested_seeds=None,
                    requested_seed_count=len(SEEDS),
                    settings=settings,
                    resume=True,
                )

        self.assertEqual(first, resumed)
        self.assertEqual(first[0], SEEDS)
        self.assertEqual(first[1], "generated_system_random")

    def test_resume_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            runner.resolve_batch_manifest(
                path,
                requested_seeds=SEEDS,
                requested_seed_count=len(SEEDS),
                settings={"version": 1},
                resume=True,
            )
            with self.assertRaisesRegex(ValueError, "settings differ"):
                runner.resolve_batch_manifest(
                    path,
                    requested_seeds=SEEDS,
                    requested_seed_count=len(SEEDS),
                    settings={"version": 2},
                    resume=True,
                )


class ComputeSubsetTests(unittest.TestCase):
    def test_two_held_out_models_are_one_test_only_unknown_pool(self):
        models = ["known-a", "known-b", "unknown-a", "unknown-b"]

        def matrix(*values):
            return np.asarray(values, dtype=float).reshape(-1, 1)

        cache = runner.DatasetCache(
            dataset_key="wiki",
            feature_names=("x",),
            matrices={
                "known-a": {
                    "train": matrix(0.11, 0.12),
                    "val": matrix(0.21),
                    "test": matrix(0.61),
                },
                "known-b": {
                    "train": matrix(0.31, 0.32),
                    "val": matrix(0.41),
                    "test": matrix(0.62, 0.63),
                },
                "unknown-a": {
                    "train": matrix(0.51),
                    "val": matrix(0.52),
                    "test": matrix(0.81),
                },
                "unknown-b": {
                    "train": matrix(0.53),
                    "val": matrix(0.54),
                    "test": matrix(0.82, 0.83),
                },
            },
            episode_ids={},
            questions={},
            source_counts={},
            valid_counts={},
            cache_digest="synthetic-cache",
        )
        observed: dict[str, object] = {}

        class FakeModel:
            def fit(self, X, y):
                observed.setdefault("fit_features", []).append(
                    np.asarray(X)[:, 0].tolist()
                )
                return self

            def predict_proba(self, X):
                scores = np.asarray(X, dtype=float)[:, 0]
                return np.column_stack((scores, 1.0 - scores))

        def summarize(known, unknown, **kwargs):
            observed["known_scores"] = known
            observed["unknown_scores"] = unknown
            return {
                "estimate": 0.75,
                "confidence_interval": {
                    "method": runner.POOLED_CI_METHOD,
                    "lower": 0.70,
                    "upper": 0.80,
                },
                "per_seed": [],
            }

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            runner,
            "select_hyperparameters",
            return_value=({}, 0.5, []),
        ), patch.object(
            runner,
            "_make_xgb",
            side_effect=lambda *args, **kwargs: FakeModel(),
        ), patch.object(
            runner,
            "summarize_pooled_open_set_auroc",
            side_effect=summarize,
        ):
            result = runner.compute_subset(
                cache,
                ("unknown-a", "unknown-b"),
                model_universe=models,
                classifier_seeds=SEEDS,
                classifier_seed_source="explicit",
                tuning_candidates=[{}],
                tuning_seed=42,
                bootstrap_replicates=100,
                confidence_level=0.95,
                bootstrap_seed=2026,
                device="cpu",
                n_jobs=1,
                run_fingerprint="synthetic-run",
                checkpoint_path=Path(temporary) / "leaf.json",
                resume=False,
            )

        self.assertEqual(result["known_models"], ["known-a", "known-b"])
        self.assertEqual(result["n_known_traces"], 3)
        self.assertEqual(result["n_unknown_traces"], 3)
        self.assertEqual(
            result["n_unknown_traces_by_model"],
            {"unknown-a": 1, "unknown-b": 2},
        )
        for fit_features in observed["fit_features"]:
            self.assertEqual(fit_features, [0.11, 0.12, 0.31, 0.32])
        for scores in observed["unknown_scores"].values():
            np.testing.assert_allclose(scores, [0.81, 0.82, 0.83])
        for scores in observed["known_scores"].values():
            np.testing.assert_allclose(scores, [0.61, 0.62, 0.63])


class AggregateTests(unittest.TestCase):
    def test_aggregate_records_exhaustive_and_sampled_coverage(self):
        universe = [f"model-{index:02d}" for index in range(6)]
        sizes = [1, 2, 3]
        cap = 8
        fingerprint = "synthetic-run"
        candidates = [{"max_depth": 3}]

        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            for size in sizes:
                selection = select_holdout_subsets(
                    universe,
                    size,
                    max_subsets=cap,
                    seed=2026,
                )
                for held_out in selection.subsets:
                    subset_id = canonical_subset_id(held_out)
                    path = (
                        work_dir
                        / "wiki"
                        / f"k{size}"
                        / f"{subset_id}.json"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(
                            {
                                "schema_version": 2,
                                "run_fingerprint": fingerprint,
                                "dataset_cache_digest": "synthetic-cache",
                                "dataset_key": "wiki",
                                "subset_id": subset_id,
                                "holdout_size": size,
                                "held_out_models": list(held_out),
                                "known_models": [
                                    model
                                    for model in universe
                                    if model not in held_out
                                ],
                                "unknown_pooling": runner.UNKNOWN_POOLING,
                                "classifier_seeds": SEEDS,
                                "classifier_seed_count": len(SEEDS),
                                "bootstrap": runner._bootstrap_metadata(
                                    100,
                                    0.95,
                                    2026,
                                ),
                                "n_known_traces": 10,
                                "n_unknown_traces": 2,
                                "models": {
                                    "XGBoost": {
                                        "auroc": {
                                            "n_known": 10,
                                            "n_unknown": 2,
                                            "confidence_interval": {
                                                "method": (
                                                    runner.POOLED_CI_METHOD
                                                )
                                            },
                                            "per_seed": [
                                                {
                                                    "seed": seed,
                                                    "auroc": 0.7,
                                                }
                                                for seed in SEEDS
                                            ],
                                        }
                                    }
                                },
                            }
                        ),
                        encoding="utf-8",
                    )

            aggregate = runner.build_aggregate(
                work_dir,
                dataset_keys=["wiki"],
                model_universe=universe,
                holdout_sizes=sizes,
                max_subsets_per_size=cap,
                subset_seed=2026,
                classifier_seeds=SEEDS,
                classifier_seed_source="explicit",
                tuning_candidates=candidates,
                bootstrap_replicates=100,
                confidence_level=0.95,
                bootstrap_seed=2026,
                run_fingerprint=fingerprint,
            )

        groups = aggregate["datasets"]["wiki"]["holdout_sizes"]
        self.assertEqual(groups["1"]["selection_mode"], "exhaustive")
        self.assertEqual(groups["1"]["n_evaluated_subsets"], 6)
        self.assertEqual(groups["2"]["selection_mode"], "balanced_sample")
        self.assertEqual(groups["2"]["n_evaluated_subsets"], 8)
        self.assertEqual(groups["3"]["n_possible_subsets"], 20)
        self.assertEqual(
            aggregate["subset_design"]["possible_counts"],
            {"1": 6, "2": 15, "3": 20},
        )


if __name__ == "__main__":
    unittest.main()
