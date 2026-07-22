from __future__ import annotations

from typing import Optional, Protocol


class PipelineStagesHook(Protocol):

    def __call__(self, stages, options, language, capability):
        ...


class CompilationListener(Protocol):

    def __call__(self, *, src, metadata, metadata_group, times, cache_hit) -> None:
        ...


class _runtime:

    add_stages_inspection_hook: Optional[PipelineStagesHook] = None


class _compilation:

    listener: Optional[CompilationListener] = None


runtime = _runtime()
compilation = _compilation()
