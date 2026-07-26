"""Small helpers for human-readable experiment reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | float | None, digits: int = 3) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    header_list = [str(value) for value in headers]

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(map(clean, header_list)) + " |",
        "| " + " | ".join("---" for _ in header_list) + " |",
    ]
    lines.extend("| " + " | ".join(map(clean, row)) + " |" for row in rows)
    return "\n".join(lines)


def relative_link(target: Path, report_path: Path, label: str | None = None) -> str:
    relative = target.resolve().relative_to(report_path.parent.resolve())
    return f"[{label or target.name}]({relative.as_posix()})"
