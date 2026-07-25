#!/usr/bin/env python3
"""Compatibility entry point for the cross-harness experiment.

New code should import or run ``experiments.cross_harness.pipeline``.
"""

import sys
from pathlib import Path

from experiments.cross_harness import pipeline as _pipeline

globals().update(
    {
        name: getattr(_pipeline, name)
        for name in dir(_pipeline)
        if not name.startswith("__")
    }
)


def _rewrite_legacy_config_args() -> None:
    aliases = {
        "cross_harness_config.yaml": "final_6model.yaml",
        "cross_harness_config.provisional.yaml": "provisional_5model.yaml",
    }
    for index, value in enumerate(sys.argv[:-1]):
        if value != "--config":
            continue
        old_name = Path(sys.argv[index + 1]).name
        if old_name in aliases and not Path(sys.argv[index + 1]).exists():
            sys.argv[index + 1] = str(
                Path(__file__).parent
                / "experiments"
                / "cross_harness"
                / "configs"
                / aliases[old_name]
            )


if __name__ == "__main__":
    _rewrite_legacy_config_args()
    raise SystemExit(_pipeline.main())
