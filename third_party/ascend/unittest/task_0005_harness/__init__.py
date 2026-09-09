"""Reusable Stage-00 fixtures and runners for the Juhe rule work.

The package deliberately keeps the frozen before/after DSL beside the test
code.  Nothing in here imports the task-book checkout by an absolute path, so
later stages can select a test file or invoke the runner from any worktree.
"""

from .contracts import (
    COMPILE_MODE,
    LOGITS_PRIMARY,
    MERGE_PRIMARY_CASES,
    NORM_PRIMARY_TOKENS,
    TARGET_ARCH,
)

__all__ = (
    "COMPILE_MODE",
    "LOGITS_PRIMARY",
    "MERGE_PRIMARY_CASES",
    "NORM_PRIMARY_TOKENS",
    "TARGET_ARCH",
)
