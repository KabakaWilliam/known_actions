"""
prep_datasets.py — Download and standardize QA datasets for agent tracing.

Reads dataset specs from an experiment config, downloads from HuggingFace,
and saves each as datasets/{name}.json — a list of {"question", "answer"} dicts.

Usage:
    python prep_datasets.py --config custom_config.yaml
    python prep_datasets.py --config my_experiment.yaml
"""

import argparse, json, random, sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

PROJECT_DIR  = Path(__file__).parent.resolve()
REGISTRY     = PROJECT_DIR / "config.yaml"
DEFAULT_CFG  = PROJECT_DIR / "custom_config.yaml"
DATASETS_DIR = PROJECT_DIR / "datasets"


# ---------------------------------------------------------------------------
# HuggingFace loaders — one function per logical dataset
# ---------------------------------------------------------------------------

def _hf_load(hf_repo: str, split: str, n_questions=None, seed=42,
             question_field="question", answer_field="answer") -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("HuggingFace `datasets` not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"  Downloading {hf_repo} ({split} split) ...")
    ds = load_dataset(hf_repo, split=split)

    rows = []
    for item in ds:
        q = item.get(question_field, "").strip()
        a = item.get(answer_field, "")
        if not q or not isinstance(a, str) or not a.strip() or len(a) > 200:
            continue
        rows.append({"question": q, "answer": a.strip()})

    if n_questions is not None and len(rows) > n_questions:
        rows = random.Random(seed).sample(rows, n_questions)

    return rows


# Maps loader keys (from config.yaml dataset_loaders) to loader callables.
# Each callable accepts: hf_repo, split, n_questions, seed → list[dict]
LOADERS = {
    "2wikimultihop": _hf_load,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and prep QA datasets defined in an experiment config.",
        epilog="Example: python prep_datasets.py --config custom_config.yaml",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG,
                        help=f"Experiment config to read datasets from (default: {DEFAULT_CFG.name})")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    with open(REGISTRY) as f:
        registry = yaml.safe_load(f)
    with open(args.config) as f:
        exp = yaml.safe_load(f)

    loader_registry = registry.get("dataset_loaders", {})

    DATASETS_DIR.mkdir(exist_ok=True)

    for dcfg in exp.get("datasets", []):
        source     = dcfg.get("source", "builtin")
        hf_key     = dcfg.get("hf_dataset")
        name       = dcfg["name"]

        if source == "builtin" or not hf_key:
            print(f"[SKIP] {name}: builtin dataset, no download needed.")
            continue

        out_path = PROJECT_DIR / source
        if out_path.exists():
            print(f"[SKIP] {name}: {out_path.relative_to(PROJECT_DIR)} already exists. Delete to re-download.")
            continue

        if hf_key not in loader_registry:
            print(f"[SKIP] {name}: loader key '{hf_key}' not found in config.yaml dataset_loaders.")
            continue

        if hf_key not in LOADERS:
            print(f"[SKIP] {name}: no loader function implemented for '{hf_key}'.")
            continue

        hf_repo     = loader_registry[hf_key]["hf_repo"]
        split       = dcfg.get("split", "validation")
        n_questions = dcfg.get("n_questions")
        seed        = dcfg.get("seed", 42)

        print(f"\n[{name}]  hf={hf_repo}  split={split}  n={n_questions or 'all'}  seed={seed}")
        rows = LOADERS[hf_key](hf_repo, split, n_questions=n_questions, seed=seed)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  Saved {len(rows)} questions → {out_path.relative_to(PROJECT_DIR)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
