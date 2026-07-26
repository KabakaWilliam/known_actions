import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


# The functions under test only organize JSON files.  Stub the training-only
# imports so this unit test cannot accidentally load traces or construct a
# classifier.
def _unexpected_training_call(*args, **kwargs):
    raise AssertionError("the heavy training path must not run in this test")


class _UnexpectedXGBClassifier:
    def __init__(self, *args, **kwargs):
        _unexpected_training_call()


_trace_analyzer_stub = types.ModuleType("trace_analyzer")
_trace_analyzer_stub.load_dataset = _unexpected_training_call
_xgboost_stub = types.ModuleType("xgboost")
_xgboost_stub.XGBClassifier = _UnexpectedXGBClassifier
_runner_path = Path(__file__).with_name("run_open_set_stats.py")
_runner_spec = importlib.util.spec_from_file_location(
    "_run_open_set_stats_under_test",
    _runner_path,
)
assert _runner_spec is not None and _runner_spec.loader is not None
runner = importlib.util.module_from_spec(_runner_spec)
with patch.dict(
    sys.modules,
    {
        "trace_analyzer": _trace_analyzer_stub,
        "xgboost": _xgboost_stub,
    },
):
    _runner_spec.loader.exec_module(runner)


SEEDS = [101, 102, 103, 104, 105]
ALT_SEEDS = [201, 202, 203, 204, 205]


def _auroc_summary(seeds, *, n_known=20, n_unknown=4):
    per_seed_values = [
        0.70 + 0.01 * index
        for index in range(len(seeds))
    ]
    estimate = sum(per_seed_values) / len(per_seed_values)
    return {
        "estimate": estimate,
        "confidence_interval": {
            "level": 0.95,
            "lower": 0.65,
            "upper": 0.84,
            "method": runner.CI_METHOD,
        },
        "seed_variability": {
            "mean": estimate,
            "sample_std": 0.02,
            "min": min(per_seed_values),
            "max": max(per_seed_values),
        },
        "per_seed": [
            {"seed": seed, "auroc": value}
            for seed, value in zip(seeds, per_seed_values)
        ],
        "n_known": n_known,
        "n_unknown": n_unknown,
    }


def _bootstrap_metadata(
    *,
    replicates=10_000,
    confidence_level=0.95,
    seed=2026,
):
    return {
        "unit": "evaluation_trace",
        "strata": list(runner.BOOTSTRAP_STRATA),
        "sampling": (
            "independent_nonparametric_with_replacement_within_stratum"
        ),
        "paired_across_classifier_seeds": True,
        "replicates": replicates,
        "confidence_level": confidence_level,
        "seed": seed,
        "interval": "percentile",
    }


def _reference_result(dataset_key, agent):
    tag = runner.DATASETS[dataset_key]["tag"]
    leaf_name = f"open_set_loo_{agent}"
    return {
        "tag": f"{tag}/{leaf_name}",
        "class_names": ["known-a", "known-b"],
        "models": {
            "XGBoost": {
                "best_params": {"max_depth": 3},
            }
        },
        "open_set": {
            "XGBoost": {
                "n_known": 20,
                "n_unknown": 4,
            }
        },
    }


def _leaf_result(
    dataset_key,
    agent,
    *,
    seeds=SEEDS,
    source_results_sha256="synthetic-reference-digest",
    bootstrap=None,
):
    tag = runner.DATASETS[dataset_key]["tag"]
    leaf_name = f"open_set_loo_{agent}"
    bootstrap = (
        _bootstrap_metadata()
        if bootstrap is None
        else copy.deepcopy(bootstrap)
    )
    return {
        "schema_version": 1,
        "timestamp": "2026-07-25T00:00:00+00:00",
        "tag": f"{tag}/{leaf_name}",
        "source_results_file": "results.json",
        "source_results_sha256": source_results_sha256,
        "train_datasets": [runner.DATASETS[dataset_key]["dataset"]],
        "val_datasets": [runner.DATASETS[dataset_key]["dataset"]],
        "test_datasets": [runner.DATASETS[dataset_key]["dataset"]],
        "open_set_datasets": [runner.DATASETS[dataset_key]["dataset"]],
        "held_out_model": agent,
        "open_set_agents": [agent],
        "n_known_traces": 20,
        "n_unknown_traces": 4,
        "classifier_seeds": list(seeds),
        "n_classifier_seeds": len(seeds),
        "classifier_seed_source": "explicit",
        "bootstrap": bootstrap,
        "aggregation": {
            "metric": "auroc",
            "across_classifier_seeds": "arithmetic_mean",
            "score_definition": runner.SCORE_DEFINITION,
            "positive_class": "known",
        },
        "models": {
            "XGBoost": {
                "best_params": {"max_depth": 3},
                "auroc": _auroc_summary(seeds),
            }
        },
    }


def _write_leaf(
    traces_dir,
    dataset_key,
    agent,
    *,
    seeds=SEEDS,
    bootstrap=None,
):
    tag = runner.DATASETS[dataset_key]["tag"]
    leaf_dir = (
        traces_dir
        / "classifiers"
        / tag
        / f"open_set_loo_{agent}"
    )
    leaf_dir.mkdir(parents=True)
    reference = _reference_result(dataset_key, agent)
    reference_path = leaf_dir / "results.json"
    reference_path.write_text(
        json.dumps(reference, sort_keys=True),
        encoding="utf-8",
    )
    result = _leaf_result(
        dataset_key,
        agent,
        seeds=seeds,
        source_results_sha256=runner._sha256(reference_path),
        bootstrap=bootstrap,
    )
    (leaf_dir / "open_set_auroc.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )
    return leaf_dir, result


class RunOpenSetStatsTests(unittest.TestCase):
    def test_build_aggregate_collects_sorted_leaves_and_dataset_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            traces_dir = Path(temporary)
            _, wiki_b = _write_leaf(traces_dir, "wiki", "model-b")
            _, wiki_a = _write_leaf(traces_dir, "wiki", "model-a")
            _, frames = _write_leaf(traces_dir, "frames", "model-c")

            aggregate = runner.build_aggregate(
                traces_dir,
                ["wiki", "frames"],
            )

        self.assertEqual(aggregate["schema_version"], 1)
        self.assertEqual(
            aggregate["default_classifier_seed_count"],
            len(SEEDS),
        )
        self.assertEqual(aggregate["classifier_seeds"], SEEDS)
        self.assertEqual(
            aggregate["classifier_seed_count"],
            len(SEEDS),
        )
        self.assertEqual(
            aggregate["bootstrap"],
            _bootstrap_metadata(),
        )
        self.assertEqual(
            aggregate["score_definition"],
            runner.SCORE_DEFINITION,
        )
        self.assertEqual(
            list(aggregate["datasets"]),
            ["wiki", "frames"],
        )
        self.assertEqual(
            aggregate["datasets"]["wiki"]["tag"],
            "2wikimultihop_open_set",
        )
        self.assertEqual(
            aggregate["datasets"]["wiki"]["dataset"],
            "2wikimultihop",
        )
        self.assertEqual(
            list(
                aggregate["datasets"]["wiki"]["held_out_models"]
            ),
            ["model-a", "model-b"],
        )
        self.assertEqual(
            aggregate["datasets"]["wiki"]["held_out_models"]["model-a"],
            wiki_a,
        )
        self.assertEqual(
            aggregate["datasets"]["wiki"]["held_out_models"]["model-b"],
            wiki_b,
        )
        self.assertEqual(
            aggregate["datasets"]["frames"]["held_out_models"]["model-c"],
            frames,
        )

    def test_build_aggregate_honors_agent_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            traces_dir = Path(temporary)
            _write_leaf(traces_dir, "wiki", "keep")
            _write_leaf(traces_dir, "wiki", "skip")

            aggregate = runner.build_aggregate(
                traces_dir,
                ["wiki"],
                {"keep"},
            )
            self.assertEqual(
                list(
                    aggregate["datasets"]["wiki"]["held_out_models"]
                ),
                ["keep"],
            )

            with self.assertRaisesRegex(
                ValueError,
                "no open_set_auroc.json files",
            ):
                runner.build_aggregate(
                    traces_dir,
                    ["wiki"],
                    {"missing"},
                )

    def test_build_aggregate_rejects_tag_and_held_out_model_mismatches(self):
        mutations = (
            ("tag", "wrong/tag", "tag does not match"),
            ("held_out_model", "different-agent", "held_out_model"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    traces_dir = Path(temporary)
                    leaf_dir, leaf = _write_leaf(
                        traces_dir,
                        "wiki",
                        "model-a",
                    )
                    leaf[field] = value
                    (leaf_dir / "open_set_auroc.json").write_text(
                        json.dumps(leaf),
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(ValueError, message):
                        runner.build_aggregate(
                            traces_dir,
                            ["wiki"],
                        )

    def test_build_aggregate_requires_the_exact_same_seed_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            traces_dir = Path(temporary)
            _write_leaf(
                traces_dir,
                "wiki",
                "model-a",
                seeds=SEEDS,
            )
            _write_leaf(
                traces_dir,
                "wiki",
                "model-b",
                seeds=ALT_SEEDS,
            )

            with self.assertRaisesRegex(
                ValueError,
                "classifier seeds differ",
            ):
                runner.build_aggregate(traces_dir, ["wiki"])

    def test_build_aggregate_requires_the_exact_same_bootstrap_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            traces_dir = Path(temporary)
            _write_leaf(traces_dir, "wiki", "model-a")
            _write_leaf(
                traces_dir,
                "wiki",
                "model-b",
                bootstrap=_bootstrap_metadata(replicates=9_999),
            )

            with self.assertRaisesRegex(
                ValueError,
                "bootstrap configuration differs",
            ):
                runner.build_aggregate(traces_dir, ["wiki"])

    def test_build_aggregate_rejects_a_leaf_without_stats_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            traces_dir = Path(temporary)
            missing_dir = (
                traces_dir
                / "classifiers"
                / runner.DATASETS["wiki"]["tag"]
                / "open_set_loo_model-a"
            )
            missing_dir.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "missing required file"):
                runner.build_aggregate(traces_dir, ["wiki"])

    def test_compatible_resume_returns_leaf_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            traces_dir = Path(temporary)
            leaf_dir, expected = _write_leaf(
                traces_dir,
                "wiki",
                "model-a",
            )

            actual = runner.compute_leaf(
                traces_dir,
                "wiki",
                leaf_dir,
                classifier_seeds=SEEDS,
                classifier_seed_source="explicit",
                bootstrap_replicates=10_000,
                confidence_level=0.95,
                bootstrap_seed=2026,
                device="cpu",
                n_jobs=1,
                resume=True,
            )

        self.assertEqual(actual, expected)

    def test_resume_compatibility_checks_every_statistical_setting(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "open_set_auroc.json"
            compatible = _leaf_result("wiki", "model-a")
            path.write_text(json.dumps(compatible), encoding="utf-8")

            loaded = runner._load_leaf_if_compatible(
                path,
                seeds=SEEDS,
                bootstrap_replicates=10_000,
                confidence_level=0.95,
                bootstrap_seed=2026,
            )
            self.assertEqual(loaded, compatible)

            variants = (
                {"expected_tag": "wrong/tag"},
                {"held_out_model": "different-model"},
                {"source_results_sha256": "stale-digest"},
                {"best_params": {"max_depth": 99}},
                {"n_known": 19},
                {"n_unknown": 3},
                {"seeds": list(reversed(SEEDS))},
                {"bootstrap_replicates": 9999},
                {"confidence_level": 0.90},
                {"bootstrap_seed": 7},
            )
            defaults = {
                "expected_tag": (
                    "2wikimultihop_open_set/open_set_loo_model-a"
                ),
                "held_out_model": "model-a",
                "source_results_sha256": "synthetic-reference-digest",
                "best_params": {"max_depth": 3},
                "n_known": 20,
                "n_unknown": 4,
                "seeds": SEEDS,
                "bootstrap_replicates": 10_000,
                "confidence_level": 0.95,
                "bootstrap_seed": 2026,
            }
            for changes in variants:
                with self.subTest(changes=changes):
                    arguments = copy.deepcopy(defaults)
                    arguments.update(changes)
                    self.assertIsNone(
                        runner._load_leaf_if_compatible(
                            path,
                            **arguments,
                        )
                    )

            self.assertIsNone(
                runner._load_leaf_if_compatible(
                    Path(temporary) / "missing.json",
                    **defaults,
                )
            )

    def test_batch_manifest_resumes_exact_seeds_and_bootstrap_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "batch.json"
            defaults = {
                "manifest_path": manifest_path,
                "requested_seed_count": len(SEEDS),
                "bootstrap_replicates": 10_000,
                "confidence_level": 0.95,
                "bootstrap_seed": 2026,
            }

            with patch.object(
                runner,
                "generate_classifier_seeds",
                return_value=SEEDS,
            ) as generate:
                created = runner._resolve_batch_seeds(
                    requested_seeds=None,
                    resume=False,
                    **defaults,
                )
                count_resume = runner._resolve_batch_seeds(
                    requested_seeds=None,
                    resume=True,
                    **defaults,
                )
            generate.assert_called_once_with(len(SEEDS))
            self.assertEqual(
                created,
                (SEEDS, "generated_system_random"),
            )
            self.assertEqual(count_resume, created)
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["classifier_seeds"], SEEDS)
            self.assertEqual(
                manifest["bootstrap"],
                {
                    "replicates": 10_000,
                    "confidence_level": 0.95,
                    "seed": 2026,
                },
            )
            self.assertEqual(
                manifest["score_definition"],
                runner.SCORE_DEFINITION,
            )

            explicit_resume = runner._resolve_batch_seeds(
                requested_seeds=SEEDS,
                resume=True,
                **defaults,
            )
            self.assertEqual(explicit_resume, created)

    def test_atomic_json_write_replaces_target_without_temporary_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "value.json"
            runner._atomic_write_json(path, {"version": 1})
            runner._atomic_write_json(path, {"version": 2})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"version": 2},
            )
            self.assertEqual(
                list(path.parent.glob(f".{path.name}.tmp-*")),
                [],
            )

    def test_batch_manifest_resume_rejects_seed_and_bootstrap_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "batch.json"
            defaults = {
                "manifest_path": manifest_path,
                "requested_seed_count": len(SEEDS),
                "bootstrap_replicates": 10_000,
                "confidence_level": 0.95,
                "bootstrap_seed": 2026,
            }
            runner._resolve_batch_seeds(
                requested_seeds=SEEDS,
                resume=False,
                **defaults,
            )

            with self.assertRaisesRegex(
                ValueError,
                "explicit classifier seeds differ",
            ):
                runner._resolve_batch_seeds(
                    requested_seeds=list(reversed(SEEDS)),
                    resume=True,
                    **defaults,
                )

            with self.assertRaisesRegex(
                ValueError,
                "bootstrap settings differ",
            ):
                runner._resolve_batch_seeds(
                    requested_seeds=SEEDS,
                    resume=True,
                    **{
                        **defaults,
                        "bootstrap_replicates": 9_999,
                    },
                )

            with self.assertRaisesRegex(
                ValueError,
                "classifier-seed-count",
            ):
                runner._resolve_batch_seeds(
                    requested_seeds=None,
                    resume=True,
                    **{
                        **defaults,
                        "requested_seed_count": len(SEEDS) + 1,
                    },
                )

    def test_reference_leaf_discovery_is_sorted_filtered_and_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            traces_dir = Path(temporary)
            tag_dir = (
                traces_dir
                / "classifiers"
                / runner.DATASETS["wiki"]["tag"]
            )
            for agent in ("model-b", "model-a"):
                leaf_dir = tag_dir / f"open_set_loo_{agent}"
                leaf_dir.mkdir(parents=True)
                (leaf_dir / "results.json").write_text(
                    "{}",
                    encoding="utf-8",
                )

            leaves = runner._reference_leaf_dirs(
                traces_dir,
                ["wiki"],
                None,
            )
            self.assertEqual(
                [path.name for _, path in leaves],
                [
                    "open_set_loo_model-a",
                    "open_set_loo_model-b",
                ],
            )
            filtered = runner._reference_leaf_dirs(
                traces_dir,
                ["wiki"],
                {"model-b"},
            )
            self.assertEqual(
                [path.name for _, path in filtered],
                ["open_set_loo_model-b"],
            )

            (tag_dir / "open_set_loo_model-b" / "results.json").unlink()
            with self.assertRaisesRegex(
                ValueError,
                "missing reference results.json",
            ):
                runner._reference_leaf_dirs(
                    traces_dir,
                    ["wiki"],
                    None,
                )


if __name__ == "__main__":
    unittest.main()
