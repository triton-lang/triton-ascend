# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Per-candidate execution state shared by vector and matrix batching."""
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ExecutionContract:
    _factor: int
    _axis: int
    preserve_grid: bool
    positions: object
    candidate_hash: str
    candidate: object = field(repr=False, compare=False)
    guard: object = field(repr=False, compare=False)
    _legacy_report: object = field(default=None, repr=False, compare=False)
    _legacy_request: object = field(default=None, repr=False, compare=False)

    @property
    def factor(self):
        return self._legacy_request.factor if self._legacy_report is not None else self._factor

    @property
    def axis(self):
        return self._legacy_report.get('selected_axis') if self._legacy_report is not None else self._axis

    @classmethod
    def from_validated_candidate(cls, candidate, request, report, positions, preserve_grid):
        from .block_batching_worker import BatchCompileError
        from .block_batching_guard import RuntimeGuard
        owned = report.get('cv_mode') == 'disabled'
        axis = report.get('selected_axis')
        if owned:
            positions = tuple(positions)
            if (report.get('owner') != 'block_batching' or type(axis) is not int or axis not in (0, 1, 2)
                    or report.get('lane_width') != request.factor or type(preserve_grid) is not bool
                    or any(type(p) is not int or p < 0 for p in positions)):
                raise BatchCompileError('VV_EXECUTION_CONTRACT_MISMATCH', kind='InvalidArtifact', stage='bind')
        # RuntimeGuard owns a deep copy and mutable evaluation caches. The
        # existing CV dispatch policy intentionally reads its live public report;
        # moving state into this class must not change that compatibility rule.
        return cls(request.factor, axis, preserve_grid, positions, candidate.hash, candidate,
                   RuntimeGuard(report['guard'], request.factor), None if owned else report, None if owned else request)
