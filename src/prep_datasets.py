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
        if not q:
            continue
        if answer_field is not None:
            a = item.get(answer_field, "")
            if not isinstance(a, str):
                continue
            a_clean = a.strip() or "NA"
        else:
            a_clean = ""
        rows.append({"question": q, "answer": a_clean})

    if n_questions is not None and len(rows) > n_questions:
        rows = random.Random(seed).sample(rows, n_questions)

    return rows


_WEBGAMES_JSONL = (
    "https://raw.githubusercontent.com/convergence-ai/webgames"
    "/main/evals/browseruse_webgames/webgames_tasks.jsonl"
)


def _webgames_load(cfg: dict, n: int | None = None, offset: int = 0) -> list[dict]:
    """Fetch WebGames task list from GitHub, sort by path for determinism, then slice.

    Split strategy (153 tasks total):
      train: offset=0,   n=100
      val:   offset=100, n=25
      test:  offset=125, n=28
    """
    import urllib.request
    print(f"  Fetching {_WEBGAMES_JSONL} ...")
    with urllib.request.urlopen(_WEBGAMES_JSONL) as resp:
        lines = resp.read().decode().strip().splitlines()
    tasks = sorted(
        [json.loads(l) for l in lines if l.strip()],
        key=lambda t: t["path"],
    )
    end = (offset + n) if n is not None else None
    tasks = tasks[offset:end]
    return [
        {
            "question":  t["description"],
            "answer":    t["password"],
            "start_url": f"https://webgames.convergence.ai/{t['path']}",
        }
        for t in tasks
    ]


def _webshop_load(local_file: str, split: str, n_questions=None, seed=42, offset=0) -> list[dict]:
    """Load a deterministic non-overlapping slice of a local goals JSON file.

    The full list is shuffled once with `seed`, then sliced [offset:offset+n_questions].
    Use offset to ensure non-overlapping train/val/test splits from the same pool.
    """
    path = PROJECT_DIR / local_file
    if not path.exists():
        print(f"  [ERROR] Local file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        goals = json.load(f)
    rng = random.Random(seed)
    rng.shuffle(goals)
    end = (offset + n_questions) if n_questions is not None else None
    slice_ = goals[offset:end]
    return [{"question": g, "answer": ""} for g in slice_]


# Maps loader keys (from config.yaml dataset_loaders) to loader callables.
LOADERS = {
    "2wikimultihop": _hf_load,
    "webshop_goals": _webshop_load,
    "deepshop":      _hf_load,     # same loader — question_field/answer_field from registry
    "webgames":      _webgames_load,
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

        loader_reg  = loader_registry[hf_key]
        split       = dcfg.get("split", "validation")
        n_questions = dcfg.get("n_questions")
        seed        = dcfg.get("seed", 42)

        if loader_reg.get("loader") == "webgames":
            # WebGames: fetch JSONL from GitHub, slice by offset
            offset = dcfg.get("offset", 0)
            print(f"\n[{name}]  webgames  offset={offset}  n={n_questions or 'all'}")
            rows = LOADERS["webgames"](loader_reg, n=n_questions, offset=offset)
        elif "local_file" in loader_reg:
            # Local file loader (e.g. webshop_goals): uses offset-based slicing
            local_file = loader_reg["local_file"]
            offset     = dcfg.get("offset", 0)
            print(f"\n[{name}]  local={local_file}  offset={offset}  n={n_questions or 'all'}  seed={seed}")
            rows = LOADERS[hf_key](local_file, split, n_questions=n_questions, seed=seed, offset=offset)
        else:
            # HuggingFace loader: question_field / answer_field from registry
            hf_repo      = loader_reg["hf_repo"]
            q_field      = loader_reg.get("question_field", "question")
            a_field      = loader_reg.get("answer_field", "answer")  # None → no answer column
            print(f"\n[{name}]  hf={hf_repo}  split={split}  n={n_questions or 'all'}  seed={seed}")
            rows = LOADERS[hf_key](hf_repo, split, n_questions=n_questions, seed=seed,
                                   question_field=q_field, answer_field=a_field)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  Saved {len(rows)} questions → {out_path.relative_to(PROJECT_DIR)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
