#!/usr/bin/env python3
"""Prevent experiment milestones from becoming CVSplit vocabulary."""

from __future__ import annotations

from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
MILESTONE_PATTERN = re.compile(
    r"stage(?:[\s._-]*[0-9])",
    re.IGNORECASE,
)

OWNED_PATHS = (
    Path("third_party/ascend/backend/compiler.py"),
    Path("third_party/ascend/include/CVSplitScheduling"),
    Path("third_party/ascend/lib/CVSplitScheduling"),
    Path("third_party/ascend/triton_ascend.cc"),
    Path("third_party/ascend/unittest/Conversion/General/CVSplitScheduling"),
)


def source_files() -> list[Path]:
    result = []
    for relative in OWNED_PATHS:
        path = REPOSITORY_ROOT / relative
        if path.is_file():
            result.append(path)
            continue
        result.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
    return sorted(result)


def test_no_development_milestone_names() -> None:
    violations = []
    for path in source_files():
        relative = path.relative_to(REPOSITORY_ROOT)
        if MILESTONE_PATTERN.search(path.name):
            violations.append(f"filename: {relative}")
        try:
            contents = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        match = MILESTONE_PATTERN.search(contents)
        if match:
            line = contents.count("\n", 0, match.start()) + 1
            violations.append(f"content: {relative}:{line}")

    assert not violations, (
        "Use behavior or compiler-phase names instead of experiment "
        "milestones:\n" + "\n".join(violations)
    )


def main() -> None:
    test_no_development_milestone_names()
    print("Milestone naming source contract: PASS")


if __name__ == "__main__":
    main()
