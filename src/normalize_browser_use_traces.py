"""Normalize task-success semantics in existing browser-use traces.

For labeled tasks, ground-truth verification is authoritative. The original
browser-use completion judgement is preserved as
browser_use_log.agent_reported_success.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

UNKNOWN_ANSWER_SENTINELS = {"", "na", "n/a", "none", "null", "unknown"}


def normalize_trace(path: Path, dry_run: bool = False) -> tuple[bool, bool]:
    episode = json.loads(path.read_text())
    if (episode.get("meta") or {}).get("harness") != "browser_use":
        return False, False

    changed = False
    browser_log = episode.get("browser_use_log")
    if not isinstance(browser_log, dict):
        browser_log = {}
        episode["browser_use_log"] = browser_log
        changed = True

    old_success = episode.get("task_success")
    if (
        "agent_reported_success" not in browser_log
        and isinstance(old_success, bool)
    ):
        browser_log["agent_reported_success"] = old_success
        changed = True

    verification = episode.get("verification")
    ground_truth = (
        str(verification.get("ground_truth", "")).strip().lower()
        if isinstance(verification, dict)
        else ""
    )
    is_labeled = (
        isinstance(verification, dict)
        and isinstance(verification.get("correct"), bool)
        and ground_truth not in UNKNOWN_ANSWER_SENTINELS
    )
    agent_reported_success = browser_log.get(
        "agent_reported_success", old_success
    )
    normalized_success = (
        bool(verification["correct"])
        if is_labeled
        else agent_reported_success
    )
    source = "ground_truth" if is_labeled else "agent_reported"

    if episode.get("task_success") != normalized_success:
        episode["task_success"] = normalized_success
        changed = True
    if episode.get("task_success_source") != source:
        episode["task_success_source"] = source
        changed = True
    if browser_log.get("task_success") != normalized_success:
        browser_log["task_success"] = normalized_success
        changed = True

    if changed and not dry_run:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(episode, indent=2) + "\n")
        os.replace(temp_path, path)
    return True, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path(__file__).parent / "traces",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scanned = browser_use = changed = failures = 0
    for path in args.trace_dir.rglob("*.json"):
        scanned += 1
        try:
            is_browser_use, was_changed = normalize_trace(path, args.dry_run)
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {path}: {exc}")
            continue
        browser_use += int(is_browser_use)
        changed += int(was_changed)

    mode = "would update" if args.dry_run else "updated"
    print(
        f"Scanned {scanned} JSON files; found {browser_use} browser-use traces; "
        f"{mode} {changed}; failures={failures}"
    )


if __name__ == "__main__":
    main()
