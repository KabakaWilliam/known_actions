import copy
import json
import math
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

from open_set_multi_figure_data import load_open_set_multi_intervals
from open_set_subset_design import canonical_subset_id


SEEDS = [101, 102, 103, 104, 105]
MODELS = ["agent_a", "agent_b", "agent_c", "agent_d"]
RUN_FINGERPRINT = "synthetic-run-fingerprint"
DATASET_CACHE_DIGEST = "synthetic-dataset-cache"
UNKNOWN_POOLING = "all_held_out_models_one_trace_weighted_binary_class"
CI_METHOD = (
    "paired_stratified_percentile_bootstrap_over_"
    "known_test_and_pooled_unknown_test_traces"
)


def synthetic_bootstrap():
    return {
        "unit": "evaluation_trace",
        "strata": [
            "known_test_trace",
            "pooled_unknown_test_trace",
        ],
        "sampling": (
            "independent_nonparametric_with_replacement_within_stratum"
        ),
        "paired_across_classifier_seeds": True,
        "replicates": 100,
        "confidence_level": 0.95,
        "seed": 17,
        "interval": "percentile",
        "scope": "pointwise_per_held_out_subset",
    }


def synthetic_multi_payload(dataset_keys=("wiki", "frames")):
    possible_counts = {
        str(size): math.comb(len(MODELS), size)
        for size in (1, 2, 3)
    }
    datasets = {}
    for dataset_index, dataset_key in enumerate(dataset_keys):
        groups = {}
        for size in (1, 2, 3):
            subset_rows = {}
            inclusion_counts = {model: 0 for model in MODELS}
            for subset_index, held_out_tuple in enumerate(
                combinations(MODELS, size)
            ):
                held_out = list(held_out_tuple)
                for model in held_out:
                    inclusion_counts[model] += 1
                subset_id = canonical_subset_id(held_out)
                estimate = (
                    0.52
                    + 0.05 * size
                    + 0.01 * subset_index
                    + 0.01 * dataset_index
                )
                subset_rows[subset_id] = {
                    "schema_version": 2,
                    "run_fingerprint": RUN_FINGERPRINT,
                    "dataset_cache_digest": DATASET_CACHE_DIGEST,
                    "dataset_key": dataset_key,
                    "subset_id": subset_id,
                    "holdout_size": size,
                    "held_out_models": held_out,
                    "known_models": sorted(set(MODELS) - set(held_out)),
                    "unknown_pooling": UNKNOWN_POOLING,
                    "n_known_traces": (len(MODELS) - size) * 20,
                    "n_unknown_traces": size * 10,
                    "classifier_seeds": list(SEEDS),
                    "classifier_seed_count": len(SEEDS),
                    "n_classifier_seeds": len(SEEDS),
                    "bootstrap": synthetic_bootstrap(),
                    "models": {
                        "XGBoost": {
                            "auroc": {
                                "estimate": estimate,
                                "confidence_interval": {
                                    "level": 0.95,
                                    "lower": estimate - 0.04,
                                    "upper": estimate + 0.04,
                                    "method": CI_METHOD,
                                },
                                "per_seed": [
                                    {"seed": seed, "auroc": estimate}
                                    for seed in SEEDS
                                ],
                                "n_known": (len(MODELS) - size) * 20,
                                "n_unknown": size * 10,
                            }
                        }
                    },
                }
            groups[str(size)] = {
                "n_possible_subsets": possible_counts[str(size)],
                "n_evaluated_subsets": len(subset_rows),
                "selection_mode": "exhaustive",
                "model_inclusion_counts": inclusion_counts,
                "subsets": subset_rows,
            }
        datasets[dataset_key] = {
            "dataset": dataset_key,
            "display_name": {
                "wiki": "Synthetic Wiki",
                "frames": "Synthetic FRAMES",
            }.get(dataset_key, dataset_key),
            "tag": f"{dataset_key}_open_set_multi",
            "dataset_cache_digest": DATASET_CACHE_DIGEST,
            "holdout_sizes": groups,
        }
    return {
        "schema_version": 2,
        "run_fingerprint": RUN_FINGERPRINT,
        "classifier_seeds": list(SEEDS),
        "classifier_seed_count": len(SEEDS),
        "bootstrap": synthetic_bootstrap(),
        "protocol": {
            "name": "fixed_test_population_pooled_unknown_v1",
            "unknown_pooling": UNKNOWN_POOLING,
            "evaluation_population": (
                "fixed_valid_test_traces_for_both_known_and_unknown_models"
            ),
        },
        "subset_design": {
            "model_universe": list(MODELS),
            "holdout_sizes": [1, 2, 3],
            "possible_counts": possible_counts,
            "max_subsets_per_size": None,
            "subset_seed": 17,
        },
        "datasets": datasets,
    }


def retain_balanced_sample(payload, dataset_key, size, count):
    group = payload["datasets"][dataset_key]["holdout_sizes"][str(size)]
    kept_items = list(group["subsets"].items())[:count]
    group["subsets"] = dict(kept_items)
    group["n_evaluated_subsets"] = count
    group["selection_mode"] = "balanced_sample"
    inclusion = {model: 0 for model in MODELS}
    for leaf in group["subsets"].values():
        for model in leaf["held_out_models"]:
            inclusion[model] += 1
    group["model_inclusion_counts"] = inclusion


class OpenSetMultiFigureDataTests(unittest.TestCase):
    def _load(self, payload):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "open_set_multi.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_open_set_multi_intervals(path)

    def test_loads_normalized_per_subset_intervals_and_coverage(self):
        data = self._load(synthetic_multi_payload())

        self.assertEqual(data["schema_version"], 2)
        self.assertEqual(data["holdout_sizes"], [1, 2, 3])
        self.assertEqual(data["possible_counts"], {1: 4, 2: 6, 3: 4})
        self.assertEqual(
            data["datasets"]["wiki"]["display_name"],
            "Synthetic Wiki",
        )
        wiki_pairs = data["datasets"]["wiki"]["holdout_sizes"][2]
        self.assertEqual(wiki_pairs["n_evaluated"], 6)
        self.assertEqual(wiki_pairs["selection_mode"], "exhaustive")
        self.assertEqual(len(wiki_pairs["subsets"]), 6)
        first = wiki_pairs["subsets"][0]
        self.assertEqual(first["held_out_models"], ["agent_a", "agent_b"])
        self.assertEqual(first["known_models"], ["agent_c", "agent_d"])
        self.assertAlmostEqual(first["estimate"], 0.62)
        self.assertAlmostEqual(first["lower"], 0.58)
        self.assertAlmostEqual(first["upper"], 0.66)

    def test_accepts_balanced_sample_and_preserves_possible_count(self):
        payload = synthetic_multi_payload(("wiki",))
        retain_balanced_sample(payload, "wiki", 3, 2)

        data = self._load(payload)

        triples = data["datasets"]["wiki"]["holdout_sizes"][3]
        self.assertEqual(triples["n_evaluated"], 2)
        self.assertEqual(triples["n_possible"], 4)
        self.assertEqual(triples["selection_mode"], "balanced_sample")

    def test_rejects_combinatorially_incorrect_possible_count(self):
        payload = synthetic_multi_payload(("wiki",))
        payload["subset_design"]["possible_counts"]["2"] = 5
        with self.assertRaisesRegex(ValueError, r"must equal C\(4, 2\) = 6"):
            self._load(payload)

    def test_rejects_unsorted_or_duplicate_held_out_models(self):
        mutations = (
            (["agent_b", "agent_a"], "must be sorted"),
            (["agent_a", "agent_a"], "unique model IDs"),
        )
        for held_out, message in mutations:
            with self.subTest(held_out=held_out):
                payload = synthetic_multi_payload(("wiki",))
                group = payload["datasets"]["wiki"]["holdout_sizes"]["2"]
                first_leaf = next(iter(group["subsets"].values()))
                first_leaf["held_out_models"] = held_out
                with self.assertRaisesRegex(ValueError, message):
                    self._load(payload)

    def test_rejects_subset_id_that_is_not_canonical_for_models(self):
        payload = synthetic_multi_payload(("wiki",))
        group = payload["datasets"]["wiki"]["holdout_sizes"]["2"]
        leaves = list(group["subsets"].values())
        leaves[1]["held_out_models"] = list(leaves[0]["held_out_models"])
        leaves[1]["known_models"] = list(leaves[0]["known_models"])
        with self.assertRaisesRegex(ValueError, "not the canonical ID"):
            self._load(payload)

    def test_rejects_inclusion_counts_that_do_not_match_subsets(self):
        payload = synthetic_multi_payload(("wiki",))
        group = payload["datasets"]["wiki"]["holdout_sizes"]["2"]
        group["model_inclusion_counts"]["agent_a"] += 1
        group["model_inclusion_counts"]["agent_b"] -= 1
        with self.assertRaisesRegex(
            ValueError, "model_inclusion_counts"
        ):
            self._load(payload)

    def test_rejects_leaf_seeds_that_differ_from_aggregate(self):
        payload = synthetic_multi_payload(("wiki",))
        group = payload["datasets"]["wiki"]["holdout_sizes"]["1"]
        leaf = next(iter(group["subsets"].values()))
        leaf["classifier_seeds"] = [201, 202, 203, 204, 205]
        leaf["models"]["XGBoost"]["auroc"]["per_seed"] = [
            {"seed": seed, "auroc": leaf["models"]["XGBoost"]["auroc"]["estimate"]}
            for seed in leaf["classifier_seeds"]
        ]
        with self.assertRaisesRegex(
            ValueError, "do not match the aggregate classifier_seeds"
        ):
            self._load(payload)

    def test_accepts_percentile_interval_that_does_not_contain_estimate(self):
        payload = synthetic_multi_payload(("wiki",))
        group = payload["datasets"]["wiki"]["holdout_sizes"]["1"]
        leaf = next(iter(group["subsets"].values()))
        leaf["models"]["XGBoost"]["auroc"]["confidence_interval"][
            "lower"
        ] = leaf["models"]["XGBoost"]["auroc"]["estimate"] + 0.01

        data = self._load(payload)

        first = data["datasets"]["wiki"]["holdout_sizes"][1]["subsets"][0]
        self.assertGreater(first["lower"], first["estimate"])

    def test_rejects_partial_group_marked_exhaustive(self):
        payload = synthetic_multi_payload(("wiki",))
        group = payload["datasets"]["wiki"]["holdout_sizes"]["3"]
        group["subsets"].pop(next(iter(group["subsets"])))
        group["n_evaluated_subsets"] -= 1
        with self.assertRaisesRegex(ValueError, "marked exhaustive"):
            self._load(payload)


if __name__ == "__main__":
    unittest.main()
