#!/usr/bin/env python3
"""Re-run exact-match verification on all existing trace files in-place.

Applies the current logic from agent_runner.ts:
    correct = ground_truth.lower() in predicted.lower()

Also backfills missing ground_truth from local datasets/ JSON files for traces
that were run without EXPECTED_ANSWER set (e.g. older 2wikimultihop runs).

Traces where neither verification nor a dataset match exists (e.g. webshop) are skipped.

Usage:
    python reverify.py                          # dry-run, print summary
    python reverify.py --apply                  # write changes back to files
    python reverify.py --traces-dir ./traces
    python reverify.py --datasets-dir ./datasets
"""
import argparse
import json
import sys
from pathlib import Path


def _load_qa_map(datasets_dir: Path) -> dict[str, str]:
    """Build {question: answer} from all datasets/*.json files."""
    qa: dict[str, str] = {}
    for f in sorted(datasets_dir.glob("*.json")):
        try:
            rows = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if q and a and a.upper() != "NA":
                qa[q] = a
    return qa


def _reverify(ground_truth: str, predicted: str) -> bool:
    return ground_truth.lower().strip() in (predicted or "").lower().strip()


def process_file(path: Path, qa_map: dict[str, str], apply: bool) -> tuple[str, bool]:
    """Returns (status, changed).

    status: 'skipped' | 'existing' | 'backfilled'
    """
    try:
        with open(path) as f:
            episode = json.load(f)
    except Exception as e:
        print(f"  WARN: could not read {path.name}: {e}", file=sys.stderr)
        return "skipped", False

    verification = episode.get("verification")
    result       = episode.get("result")
    predicted    = (result.get("answer") if isinstance(result, dict) else None) or ""
    question     = (episode.get("meta") or {}).get("question", "").strip()

    # Case 1: has ground_truth already — validate it isn't a placeholder 'NA'
    if verification and verification.get("ground_truth"):
        gt = verification["ground_truth"]
        if gt.strip().upper() == "NA":
            # Invalid ground truth — nullify the verification
            if apply:
                episode["verification"] = None
                with open(path, "w") as f:
                    json.dump(episode, f, indent=2)
            return "backfilled", True  # counts as a change to fix

        old_correct = verification.get("correct")
        new_correct = _reverify(gt, predicted)
        if old_correct == new_correct:
            return "existing", False
        if apply:
            episode["verification"]["correct"] = new_correct
            with open(path, "w") as f:
                json.dump(episode, f, indent=2)
        return "existing", True

    # Case 2: no ground_truth — try backfill from dataset files
    gt = qa_map.get(question)
    if not gt:
        return "skipped", False

    new_correct = _reverify(gt, predicted)
    if apply:
        episode["verification"] = {
            "correct":      new_correct,
            "predicted":    predicted,
            "ground_truth": gt,
        }
        with open(path, "w") as f:
            json.dump(episode, f, indent=2)
    return "backfilled", True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces)")
    parser.add_argument("--datasets-dir", type=Path, default=Path("./datasets"),
                        help="Directory with dataset JSON files (default: ./datasets)")
    parser.add_argument("--apply", action="store_true",
                        help="Write changes back to files (default: dry-run only)")
    cli = parser.parse_args()

    qa_map = _load_qa_map(cli.datasets_dir)
    print(f"Loaded {len(qa_map):,} Q→A pairs from {cli.datasets_dir}")

    paths = sorted(cli.traces_dir.rglob("*.json"))
    paths = [p for p in paths
             if not p.relative_to(cli.traces_dir).parts[0].startswith("classifiers")]

    counts = {"skipped": 0, "existing_changed": 0, "existing_unchanged": 0, "backfilled": 0}

    for path in paths:
        status, changed = process_file(path, qa_map, cli.apply)
        if status == "skipped":
            counts["skipped"] += 1
        elif status == "existing":
            counts["existing_changed" if changed else "existing_unchanged"] += 1
        elif status == "backfilled":
            counts["backfilled"] += 1

    mode = "Applied" if cli.apply else "Dry-run"
    print(f"\n{mode}:")
    print(f"  {counts['backfilled']:>6,}  verification backfilled from dataset files")
    print(f"  {counts['existing_changed']:>6,}  existing verification corrected")
    print(f"  {counts['existing_unchanged']:>6,}  existing verification unchanged")
    print(f"  {counts['skipped']:>6,}  skipped (no ground truth available)")
    if not cli.apply and (counts["backfilled"] or counts["existing_changed"]):
        print("\nRe-run with --apply to write changes.")


if __name__ == "__main__":
    main()
