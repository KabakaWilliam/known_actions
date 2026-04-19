#!/usr/bin/env python3
"""Re-run exact-match verification on all existing trace files in-place.

Applies the current logic from agent_runner.ts:
    correct = ground_truth.lower() in predicted.lower()

Traces where verification is null (no expected answer, e.g. webshop) are skipped.

Usage:
    python reverify.py                          # dry-run, print summary
    python reverify.py --apply                  # write changes back to files
    python reverify.py --traces-dir ./traces    # explicit directory
"""
import argparse
import json
import sys
from pathlib import Path


def _reverify(verification: dict) -> bool:
    """Return the correct value under the current case-insensitive contains logic."""
    ground_truth = (verification.get("ground_truth") or "").lower().strip()
    predicted    = (verification.get("predicted") or "").lower().strip()
    if not ground_truth:
        return False
    return ground_truth in predicted


def process_file(path: Path, apply: bool) -> tuple[bool, bool]:
    """Returns (had_verification, changed)."""
    try:
        with open(path) as f:
            episode = json.load(f)
    except Exception as e:
        print(f"  WARN: could not read {path.name}: {e}", file=sys.stderr)
        return False, False

    verification = episode.get("verification")
    if not verification or not verification.get("ground_truth"):
        return False, False

    old_correct = verification.get("correct")
    new_correct = _reverify(verification)

    if old_correct == new_correct:
        return True, False

    if apply:
        episode["verification"]["correct"] = new_correct
        with open(path, "w") as f:
            json.dump(episode, f, indent=2)

    return True, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces)")
    parser.add_argument("--apply", action="store_true",
                        help="Write corrected verification back to files (default: dry-run only)")
    cli = parser.parse_args()

    paths = sorted(cli.traces_dir.rglob("*.json"))
    paths = [p for p in paths
             if not p.relative_to(cli.traces_dir).parts[0].startswith("classifiers")]

    total = 0
    with_verif = 0
    changed = 0

    for path in paths:
        total += 1
        had_v, was_changed = process_file(path, cli.apply)
        if had_v:
            with_verif += 1
        if was_changed:
            changed += 1
            if not cli.apply:
                print(f"  would change: {path.relative_to(cli.traces_dir)}")

    mode = "Updated" if cli.apply else "Would update"
    print(f"\nScanned {total:,} traces | {with_verif:,} with verification | "
          f"{mode} {changed:,} files")
    if not cli.apply and changed:
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
