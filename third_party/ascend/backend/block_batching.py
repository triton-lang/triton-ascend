# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Block batching runtime: launch preparation, handle reuse and guarded dispatch.

Compilation and artifact validation live in the worker and report modules.
This module always launches a complete baseline or candidate executable.
"""
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import threading

IMPLEMENTATION_VERSION = 56


# Requests and the execution state owned by each compiled candidate.
def uses_grouping_policy(arch):
    """Whether the target selects on TTIR legality and occupancy alone.

    The A5 model prices individual instructions against an A5 register file
    and unified buffer, so it does not transfer to the earlier generation.
    A2 and A3 share that generation and therefore share the simpler policy.
    """
    return isinstance(arch, str) and arch.startswith(('Ascend910_93', 'Ascend910B'))


def supports_target(arch):
    return isinstance(arch, str) and (arch.startswith(('Ascend950', 'Ascend910_95')) or uses_grouping_policy(arch))


def _is_cv_report(report):
    return report.get("cv_mode") in ("independent_batch_v1", "spatial_concat_v1")


@dataclass(frozen=True)
class BatchRequest:
    factor: int = 1
    axis: int = -1

    def __post_init__(self):
        if type(self.factor) is not int or self.factor <= 0:
            raise ValueError("block_batching_factor must be a positive integer (not bool)")
        if type(self.axis) is not int or self.axis not in (-1, 0, 1, 2):
            raise ValueError("block_batching_axis must be -1, 0, 1 or 2 (not bool)")


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


# Prepare launch options before the ordinary JIT cache lookup.
def _capture_state():
    import sys
    torch = sys.modules.get("torch")
    query = getattr(getattr(torch, "npu", None), "is_current_stream_capturing", None)
    return query() if query is not None else None


def _argument(value):
    if value is None or type(value) in (bool, int) or type(value) is float and math.isfinite(value):
        return dict(kind="scalar", value=value)
    if (hasattr(value, "dtype") and hasattr(value, "shape") and callable(getattr(value, "stride", None))
            and callable(getattr(value, "storage_offset", None))):
        shape, stride, offset = list(value.shape), list(value.stride()), value.storage_offset()
        if all(type(n) is int for n in (*shape, *stride, offset)):
            return dict(kind="tensor", dtype=str(value.dtype), shape=shape, stride=stride, storage_offset=offset)
    return dict(kind="unsupported")


def prepare_launch(fn, args, kwargs, grid, warmup, device, backend, binding=None):
    return _prepare_launch(fn, args, kwargs, grid, warmup, device, backend, binding=binding)[:2]


def _generalize_data_arguments(arguments, names):
    for name in names:
        argument = arguments.get(name, {})
        value = argument.get("value")
        # Preserve common constant-folding cases; other data-only integers
        # are scored as runtime operands, independent of first launch order.
        if type(value) is int and value not in (-1, 0, 1):
            del argument["value"]


def _prepare_launch(fn, args, kwargs, grid, warmup, device, backend, binding=None):
    """Resolve the automatic launch contract before the ordinary JIT lookup.

    ``binding`` optionally collects the argument binding performed here. It is
    taken before any option is injected, so it carries the ordinary cache
    identity, and it stays empty on paths that must not bind, such as capture.
    """
    if "block_batching_factor" in fn.arg_names or kwargs.get("block_batching_factor", "auto") != "auto":
        return kwargs, grid, None
    from .reuse.config import enabled as reuse_enabled

    def baseline():
        # Only an omitted or automatic factor reaches this point, and both mean
        # the ordinary compilation. Dropping the key keeps that launch on the
        # ordinary JIT cache identity instead of splitting it, while an omitted
        # and an explicit "auto" factor still resolve to the same options.
        result = dict(kwargs)
        result.pop("block_auto_contract", None)
        result.pop("block_batching_factor", None)
        return result, grid, None

    from triton.runtime.autotuner import _autotuning_kernel
    if "block_batching_factor" not in kwargs and _autotuning_kernel.get() is fn.fn:
        return baseline()

    # Dynamic reuse owns constexpr/tile adaptation and must receive the original
    # callable grid. A concrete auto contract would split its variant families.
    # Compile-only runs have no device to resolve the selection inputs from, so
    # an automatic launch there is an ordinary one rather than a hard error.
    from triton import knobs

    if (reuse_enabled() or getattr(getattr(fn, "ASTSource", None), "frontend_kind", "triton") != "triton"
            or not supports_target(backend.target.arch) or _capture_state() is not False or knobs.ascend.compile_only):
        return baseline()
    bound, specialization, raw_options = fn.device_caches[device][4](*args, **kwargs)
    if binding is not None:
        binding.update(specialization=specialization, options=raw_options)
    if callable(grid):
        grid = grid(bound)
    if not isinstance(grid, (tuple, list)) or not 1 <= len(grid) <= 3 or any(
            type(n) is not int or not 0 < n <= 2147483647 for n in grid):
        return baseline()
    extents = tuple(grid) + (1, ) * (3 - len(grid))
    axis = kwargs.get("block_batching_axis", -1)
    if max(extents) == 1 or type(axis) is int and 0 <= axis < 3 and extents[axis] == 1:
        # No two programs can share a group. Keep the ordinary JIT cache key,
        # including its runtime scalar/pointer specialization policy.
        return baseline()
    arguments = {
        name: _argument(value)
        for (name, value), spec in zip(bound.items(), specialization)
        if spec[0] != "constexpr"
    }
    # dtype-only warmup uses MockTensor, which has no real storage/layout.
    if warmup and any(value["kind"] == "unsupported" for value in arguments.values()):
        return baseline()
    family = None
    if any(type(arg.get("value")) is int for arg in arguments.values()):
        # The input TTIR depends on the ordinary JIT specialization and
        # compiler options. The decision below still includes grid and layout.
        key = ("block_auto_data_arguments", fn.cache_key, tuple(specialization), str(raw_options))
        known = fn.device_caches[device][1].get(key)
        if known is None:
            family = key
        else:
            _generalize_data_arguments(arguments, known)
    contract = dict(grid=extents, arguments=arguments, argument_order=list(arguments),
                    core_floor=os.getenv("TA_BLOCK_BATCHING_CORE_FLOOR"), device_limit=os.getenv("NPU_DEVICE_LIMIT"))
    if uses_grouping_policy(str(backend.target.arch)):
        contract["cv_empirical_short_batch"] = False
    # Inject the public default before JIT cache selection and the bundle hook.
    # Concrete compilation stages still receive numeric factors only.
    # Raw option order participates in the in-memory JIT key. Canonicalize our
    # keys so an omitted factor and an explicit "auto" share the same bundle.
    result = {
        key: value
        for key, value in kwargs.items()
        if key not in ("block_batching_factor", "block_batching_grid", "block_auto_contract")
    }
    result.update(block_batching_factor="auto", block_batching_grid=extents,
                  block_auto_contract=json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return result, grid, family


def _annotate_launch_grid(kwargs, grid, *, bind=None):
    """Attach launch extents before cache selection; return the launch grid.

    Automatic axis selection needs callable extents as well as static ones.
    The batching continuation binds arguments only for actual launches, keeping
    warmup callbacks unevaluated. Returning the resolved value also keeps
    reuse and launch from invoking the callback a second time.
    """
    factor = kwargs.get("block_batching_factor", 1)
    # An explicit axis is already selected; launch extents must not split
    # the candidate or its baseline cache, for static or callable grids.
    if (factor == 1 or "block_batching_grid" in kwargs or kwargs.get("block_batching_axis", -1) != -1):
        return grid
    if callable(grid):
        if bind is None or type(factor) is not int or factor <= 1:
            return grid
        grid = grid(bind())
    if not isinstance(grid, (tuple, list)) or not 1 <= len(grid) <= 3:
        return grid
    extents = [*grid, *((1, ) * (3 - len(grid)))]
    if all(type(extent) is int and extent > 0 for extent in extents):
        kwargs["block_batching_grid"] = tuple(extents)
    return grid


class _BatchLaunchHandler:
    """Block-batching policy behind the existing backend JIT continuation."""

    def run(self, fn, args, kwargs, grid, warmup, device, stream, backend, jit_tail):
        binding = {}
        kwargs, grid, family = _prepare_launch(fn, args, kwargs, grid, warmup, device, backend, binding=binding)
        # Preserve pre-run hooks, cache identity and single callback evaluation
        # without adding a hook to the common JIT frontend.
        bind = None if warmup else lambda: fn.device_caches[device][4](*args, **kwargs)[0]
        grid = _annotate_launch_grid(kwargs, grid, bind=bind)
        # Both memos below are bookkeeping, not compiled kernels, so they live in
        # the key cache rather than the kernel cache. Both device slots share one
        # lifetime, while the kernel cache stays one entry per compiled kernel.
        if "block_batching_factor" not in fn.arg_names and kwargs.get("block_batching_factor", 1) != 1:
            kwargs = backend.prepare_block_compile_options(kwargs, cache=fn.device_caches[device][1])
        from .reuse.config import enabled

        if enabled():
            from .reuse.bridge import get_handler

            return get_handler().run(fn, args, kwargs, grid, warmup, device, stream, backend, jit_tail)
        if kwargs.get("block_batching_factor") == "auto":
            cache = fn.device_caches[device][1]
            # The contract reaches compilation through the options, but it
            # describes one launch, not the artifact. Keeping it out of the
            # cache identity stops every new shape from re-entering compile().
            memory_key = None
            if binding:
                from triton.runtime.jit import compute_cache_key

                memory_key = compute_cache_key(fn.device_caches[device][1], binding["specialization"],
                                               binding["options"])
            kernel = jit_tail(args, kwargs, grid, warmup, device, stream, memory_key=memory_key,
                              initialize=lambda kernel: initialize_auto_kernel(cache, kernel))
            if family is not None and isinstance(kernel, BatchKernel) and hasattr(kernel, "block_auto_data_arguments"):
                fn.device_caches[device][1][family] = kernel.block_auto_data_arguments
            return kernel
        return jit_tail(args, kwargs, grid, warmup, device, stream)


_launch_handler = _BatchLaunchHandler()


def get_launch_handler():
    return _launch_handler


def initialize_auto_kernel(cache, kernel):
    """Share identical automatic decisions within the native per-device cache.

    Launch contracts may differ in runtime scalars while producing the same
    complete artifact and decision. Keep those contracts separate for selection,
    but do not load an identical executable again. Clearing the JIT cache also
    clears this identity; no process-global device handles are retained.
    """
    if isinstance(kernel, BatchKernel) and hasattr(kernel, "block_auto_selection"):
        initialized_key = ("block_auto_initialized", kernel)
        initialized = cache.get(initialized_key)
        if initialized is not None:
            return initialized
        key = ("block_auto_kernel", kernel.hash, repr(kernel.block_batching), repr(kernel.block_auto_selection))
        kernel = cache.setdefault(key, kernel)
        kernel._init_handles()
        # Initialization may fail or replace a candidate with its baseline.
        # Remember the resolved object only after initialization succeeds.
        cache[initialized_key] = kernel
    else:
        kernel._init_handles()
    return kernel


# Select a complete executable after grid, capture and guard checks.
def _core_floor():
    """Minimum merged task count for the supported AIV-only candidates.

    Batching validates final execution mode as AIV, so its scheduling floor
    must use available vector cores rather than cube cores. Device limits
    remain owned and validated by the device interface. An explicit override
    is honored only after that validation; zero disables the floor.
    """
    from triton.runtime import driver
    available_aiv = driver.active.utils.get_aivector_core_num()
    override = os.getenv("TA_BLOCK_BATCHING_CORE_FLOOR")
    return int(override) if override is not None else available_aiv


def _batched_blocks(grid, axis, factor):
    batched = list(grid)
    batched[axis] = (grid[axis] + factor - 1) // factor
    return batched[0] * batched[1] * batched[2]


def _reduces_blocks(grid, axis, factor):
    """Whether batching this axis reduces the flattened block count."""
    return _batched_blocks(grid, axis, factor) < _batched_blocks(grid, axis, 1)


def candidate_launch_refusal(request, report, grid, preserve_grid, capture, *, cv, auto_selected=False):
    """Return a refusal or None, preserving lazy queries and refusal precedence.

    The caller queries capture only after finding a candidate. The core query
    remains lazy: earlier refusals must not read device/environment state.
    Report reads intentionally retain their existing timing; isolating public
    reports from execution is a separate behavior change.
    """
    if cv and grid[report["selected_axis"]] % request.factor:
        # VV uses lane masks for tails; CV still requires complete groups.
        return "CV_GRID_NOT_DIVISIBLE"
    if not preserve_grid and not _reduces_blocks(grid, report["selected_axis"], request.factor):
        return "GRID_NOT_REDUCED"
    # The native selector scores small CV grids. Manual CV and VV keep the floor.
    if (not (cv and auto_selected)
            and _batched_blocks(grid, report["selected_axis"], 1 if preserve_grid else request.factor) < _core_floor()):
        return "GRID_BELOW_CORE_FLOOR"
    if capture is not False:
        return "CAPTURE_BASELINE" if capture else "CAPTURE_QUERY_UNAVAILABLE"
    return None


class BatchKernel:
    """CompiledKernel-compatible facade selecting a complete variant per call."""

    def __init__(self, baseline, candidate, request, report, positions, *, publish_manifest=True):
        self.baseline, self.candidate = baseline, candidate
        self._vv_mode = report.get("cv_mode") == "disabled"
        if self._vv_mode:
            report = deepcopy(report)
        self.request, self.block_batching = request, report
        self.positions = positions
        from .block_batching_report import validate_vv_loop_contract
        self._preserve_grid = (validate_vv_loop_contract(candidate.metadata._asdict())
                               if candidate is not None else False)
        self._contract = (ExecutionContract.from_validated_candidate(
            candidate, request, report, positions, self._preserve_grid) if candidate is not None else None)
        # Legacy performance diagnostics inspect this evaluator. Dispatch and
        # ownership stay in the common contract; this is an alias, not a copy.
        self._guard = self._contract.guard if self._contract is not None and not self._vv_mode else None
        self._lock = threading.RLock()
        self._ready = False
        self._local = threading.local()
        self._set_primary(candidate if candidate is not None else baseline)
        if publish_manifest:
            self.publish_manifest()

    def _set_primary(self, kernel):
        self._primary = kernel
        self.hash = kernel.hash
        fields = dict(kernel.metadata._asdict(), block_batching=self.block_batching)
        self.metadata = namedtuple("BatchKernelMetadata", sorted(fields))(**fields)

    def publish_manifest(self):
        from triton.runtime.cache import get_cache_manager
        manifest = {
            "version": 1, "baseline_hash": self.baseline.hash, "candidate_hash":
            self.candidate.hash if self.candidate is not None else None, "request":
            {"factor": self.request.factor, "axis": self.request.axis}, "report": self.block_batching
        }
        identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        get_cache_manager(identity).put(json.dumps(manifest, sort_keys=True), "block_batching.bundle.json")

    def __getattr__(self, name):
        return getattr(self._primary, name)

    @property
    def last_launch(self):
        result = getattr(self._local, "launch", {"runtime_status": "NOT_LAUNCHED"})
        if self._vv_mode:
            return deepcopy(result)
        return result

    def _init_handles(self):
        with self._lock:
            if self._ready:
                return
            # Capture cannot trigger loading a new guarded variant. A normal
            # eager warmup initializes both complete handles beforehand.
            if _capture_state() and getattr(self.baseline, "_run", None) is None:
                raise RuntimeError("warm up the block batching baseline before graph capture")
            self.baseline._init_handles()
            if self.candidate is not None:
                from triton.runtime.errors import OutOfResources
                try:
                    self.candidate._init_handles()
                except (OutOfResources, NotImplementedError) as error:
                    if isinstance(error,
                                  NotImplementedError) and not getattr(error, 'ascend_launcher_abi_mismatch', False):
                        raise
                    self.block_batching.update(compile_status="REJECTED", reason=str(error), owner="none")
                    self.candidate = None
                    self._contract = None
                    self._set_primary(self.baseline)
            self._ready = True

    @property
    def run(self):
        self._init_handles()
        return self._dispatch

    def launch_metadata(self, grid, stream, *arguments):
        # Hooks are constructed in _dispatch, after actual variant selection.
        return None

    def _dispatch(self, *launch):
        from triton import knobs
        from .block_batching_guard import integer, I32_MAX
        grid = tuple(integer(x) for x in launch[:3])
        if any(x < 0 or x > I32_MAX for x in grid):
            raise ValueError("original grid must contain nonnegative i32 extents")
        if any(x == 0 for x in grid):
            self._local.launch = {"runtime_status": "EMPTY_GRID", "applied_factor": 1, "original_grid": grid}
            return
        selected = self.baseline
        reason = self.block_batching.get("reason", "CANDIDATE_UNAVAILABLE")
        accepted = False
        arguments = launch[9:]
        contract = self._contract
        if self.candidate is not None:
            capture = _capture_state()
            live = contract._legacy_report
            request = contract._legacy_request if live is not None else contract
            report = live if live is not None else {'selected_axis': contract.axis}
            reason = candidate_launch_refusal(request, report, grid, contract.preserve_grid, capture,
                                              cv=_is_cv_report(live) if live is not None else False,
                                              auto_selected=hasattr(self, "block_auto_selection"))
            if reason is None:
                accepted, reason = contract.guard([arguments[index] for index in contract.positions], grid)
                if accepted:
                    selected = contract.candidate
        # Never mix baseline function/metadata/workspace/layout with candidate.
        forwarded = list(launch)
        forwarded[4], forwarded[5] = selected.function, selected.packed_metadata
        forwarded[6] = selected.launch_metadata(grid, launch[3], *arguments)
        result = selected.run(*forwarded)
        requested_factor = contract.factor if contract is not None else self.request.factor
        preserve_grid = contract.preserve_grid if contract is not None else self._preserve_grid
        axis = contract.axis if contract is not None else self.block_batching.get("selected_axis")
        factor = requested_factor if accepted else 1
        batch_grid = list(grid)
        if accepted and not preserve_grid:
            batch_grid[axis] = (grid[axis] + factor - 1) // factor
        self._local.launch = {
            "runtime_status": "APPLIED" if accepted else "BASELINE_SELECTED", "owner":
            "block_batching" if accepted else "baseline", "requested_factor": requested_factor, "applied_factor":
            factor, "selected_axis": axis, "original_grid": grid, "batch_grid": tuple(batch_grid), "reason": reason,
            "plan_id": self.hash
        }
        self._local.launch['grouping'] = 'loop_iterations' if accepted and preserve_grid else 'programs'
        if knobs.ascend.compile_only:
            self._local.launch.update(runtime_status="NOT_LAUNCHED", reason="COMPILE_ONLY")
        return result

    def __getitem__(self, grid):

        def runner(*args, stream=None):
            from triton import knobs
            from triton.runtime import driver
            self._init_handles()
            if stream is None:
                stream = driver.active.get_current_stream(driver.active.get_current_device())
            return self.run(*grid, stream, self.function, self.packed_metadata, None, knobs.runtime.launch_enter_hook,
                            knobs.runtime.launch_exit_hook, *args)

        return runner
