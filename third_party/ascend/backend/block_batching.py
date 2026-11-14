# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Block batching request and analysis contracts.

Candidates require a proved runtime memory domain, a matching native ABI and
a successfully compiled backend artifact under the ordinary launch contract. Baseline and candidate are separate complete
compiler artifacts; rejected or guarded calls select the complete baseline.
"""
from dataclasses import dataclass

IMPLEMENTATION_VERSION = 56


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
        from .block_auto_dispatch import prepare_launch
        kwargs, grid = prepare_launch(fn, args, kwargs, grid, warmup, device, backend)
        # Preserve pre-run hooks, cache identity and single callback evaluation
        # without adding a hook to the common JIT frontend.
        bind = None if warmup else lambda: fn.device_caches[device][4](*args, **kwargs)[0]
        grid = _annotate_launch_grid(kwargs, grid, bind=bind)
        from .reuse.config import enabled

        if enabled():
            from .reuse.bridge import get_handler

            return get_handler().run(fn, args, kwargs, grid, warmup, device, stream, backend, jit_tail)
        return jit_tail(args, kwargs, grid, warmup, device, stream)


_launch_handler = _BatchLaunchHandler()


def get_launch_handler():
    return _launch_handler


def _has_independent_row_layout(report):
    """Recognize top-level row accesses already covered by the source proof.

    This does not certify the backend memory form. The subsequent lowering
    check and per-launch byte-domain proof remain mandatory.
    """
    if any(report.get("operations", {}).get(op) for op in ("scf.for", "scf.if")):
        return False
    accesses = report.get("layout", {}).get("accesses", [])
    return bool(accesses) and all(
        access.get("control_flow_independent") is True and access.get("proof_scope") == "existing_guard_no_wrap"
        and access.get("action") == "Keep" and access.get("rewritten") is False
        and access.get("reason") == "ALREADY_STRUCTURED" and access.get("mask_class") == "independent_axes" and
        (shape := access.get("original_shape", [])) and all(type(extent) is int and extent > 0
                                                            for extent in shape) and sum(extent != 1
                                                                                         for extent in shape) <= 1
        for access in accesses)


def vv_layout_refusal(report):
    """Conservative structural matching for independent row pointwise bodies.

    Proven top-level rows may use the structured backend path. Incomplete
    layout evidence, shared accesses and control-flow bodies retain the
    existing refusal. No kernel names or measured timings enter this policy.
    """
    if (report.get("owner") != "block_batching" or report.get("cv_mode") != "disabled"
            or report.get("lane_width", 1) <= 1):
        return None
    if _has_independent_row_layout(report):
        return None
    operations = report.get("operations", {})
    if (not operations.get("tt.load") or not operations.get("tt.store")
            or any(operations.get(name) for name in ("tt.reduce", "tt.scan", "tt.dot"))):
        return None
    accesses = report.get("layout", {}).get("accesses", [])
    if (any(operations.get(name) for name in ("scf.for", "scf.if"))
            and not all(access.get("control_flow_independent") is True for access in accesses)):
        return None

    def row_rank(access):
        shape = access.get("original_shape", [])
        if any(type(extent) is not int or extent <= 0 for extent in shape):
            return None
        # Retain the existing distinction between scalar accesses and a
        # one-element row while ignoring singleton coordinate dimensions.
        return max(1, sum(extent != 1 for extent in shape)) if shape else 0

    independent = [access for access in accesses if access.get("mask_class") == "independent_axes"]
    if independent and all(
            access.get("action") == "Keep" and access.get("rewritten") is False and (
                (access.get("mask_class") == "independent_axes" and row_rank(access) == 1) or
                (access.get("mask_class") == "shared" and row_rank(access) in (0, 1))) for access in accesses):
        return "VV_POINTWISE_MEMORY_VIEW_REQUIRED"
    return None


def vv_computation_refusal(report):
    """Require the native original-domain contract, independently of memory.

    The native verifier checks combiners, effects and loop-carried values.
    This boundary checks that their domain contract reached the compiler;
    neither operation counts nor a successful memory conversion replace it.
    """
    if (report.get("owner") != "block_batching" or report.get("cv_mode") != "disabled"
            or report.get("lane_width", 1) <= 1):
        return None
    computation = report.get("computation", {})
    schedule = "sequential_iterations" if report.get("vv_lowering") == "sequential_iterations_v1" else "program_lanes"
    if (computation.get("version") != 2 or computation.get("scope") != "normalized_original_program"
            or computation.get("stage") != "semantic_ttir_before_memory_views"
            or computation.get("schedule") != schedule or computation.get("lane_policy") != "separate_original_domains"
            or not isinstance(computation.get("aggregations"), list)):
        return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
    expected = {"reduce": "original_program_reduction", "scan": "independent_row_scan"}
    counts = {kind: 0 for kind in expected}
    target_ids = set()
    for source_id, aggregation in enumerate(computation["aggregations"]):
        if (not isinstance(aggregation, dict) or type(aggregation.get("id")) is not int
                or aggregation["id"] != source_id):
            return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
        kind = aggregation.get("kind")
        if kind not in expected or aggregation.get("domain") != expected[kind]:
            return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
        axis = aggregation.get("original_axis")
        shape = aggregation.get("original_shape")
        targets = aggregation.get("mapped_operations")
        if (type(axis) is not int or not isinstance(shape, list) or not 0 <= axis < len(shape)
                or any(type(d) is not int or d <= 0 for d in shape) or not isinstance(targets, list) or not targets
                or (schedule == "program_lanes" and len(targets) != 1)):
            return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
        for target in targets:
            if not isinstance(target, dict):
                return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
            prefix = target.get("batch_prefix")
            target_id = target.get("id")
            if (type(prefix) is not bool or (schedule == "sequential_iterations" and prefix)
                    or type(target_id) is not int or target_id < 0 or target_id in target_ids
                    or type(target.get("axis")) is not int or type(aggregation.get("lifted_axis")) is not int
                    or target.get("axis") != axis + int(prefix) or aggregation.get("lifted_axis") != target.get("axis")
                    or target.get("shape") != ([report["lane_width"]] + shape if prefix else shape)):
                return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
            target_ids.add(target_id)
        counts[kind] += 1
    if target_ids != set(range(len(target_ids))):
        return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
    if any(count != report.get("operations", {}).get("tt." + kind, 0) for kind, count in counts.items()):
        return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
    return None


def vv_memory_lowering_refusal(report):
    """Keep VV candidates on supported memory paths after layout lowering.

    Independent row reductions and scans remain supported. Their presence
    does not prove that a batched access became a structured memory view:
    reject residual indirect GM helpers regardless of the compute body.
    Internal tensor gathers and SIMT arithmetic are not memory refusals.
    """
    if (report.get("owner") != "block_batching" or report.get("cv_mode") != "disabled"
            or report.get("lane_width", 1) <= 1):
        return None
    observation = report.get("layout", {}).get("lowering_observation")
    if (not isinstance(observation, dict) or observation.get("stage") != "ttadapter"
            or observation.get("scope") != "module" or any(
                type(observation.get(key)) is not int or observation[key] < 0
                for key in ("indirect_load_ops", "indirect_store_ops"))):
        return "VV_MEMORY_LOWERING_NOT_PROVEN"
    if observation.get("indirect_load_ops", 0) or observation.get("indirect_store_ops", 0):
        return "VV_INDIRECT_MEMORY_NOT_SUPPORTED"
    return None


def memory_compile_options(metadata, compiler_path, supports_inlining):
    """Inline contiguous or proved independent-row DMA in VV candidates.

    This is a lowering policy, not a factor selector or a profitability test.
    Unsupported compiler versions retain their normal candidate compilation.
    """
    report = metadata.get("block_batching", {})
    lowering = report.get("memory_lowering", {})
    observation = report.get("layout", {}).get("lowering_observation", {})
    row_copy = (_has_independent_row_layout(report) and observation.get("stage") == "ttadapter"
                and observation.get("scope") == "module" and observation.get("indirect_load_ops") == 0
                and observation.get("indirect_store_ops") == 0 and type(observation.get("memref_copy_ops")) is int
                and observation["memref_copy_ops"] > 0)
    if (metadata.get("block_batching_factor", 1) <= 1 or report.get("owner") != "block_batching"
            or report.get("cv_mode") != "disabled" or not (lowering.get("dynamic_contiguous_copy") is True or row_copy)
            or not metadata["target"].arch.startswith("Ascend950") or metadata.get("mix_mode") != "aiv"
            or metadata.get("parallel_mode") != "simd"):
        return []
    lowering["structured_row_copy"] = row_copy
    lowering["libcall_inlining"] = bool(supports_inlining(compiler_path))
    return ["--enable-lib-call-no-inline=false"] if lowering["libcall_inlining"] else []


@dataclass(frozen=True)
class BatchRequest:
    factor: int = 1
    axis: int = -1

    def __post_init__(self):
        if type(self.factor) is not int or self.factor <= 0:
            raise ValueError("block_batching_factor must be a positive integer (not bool)")
        if type(self.axis) is not int or self.axis not in (-1, 0, 1, 2):
            raise ValueError("block_batching_axis must be -1, 0, 1 or 2 (not bool)")


def _has_dot(mod):
    """Classify actual IR operations, including dots inside nested regions."""
    found = False

    def visit(op):
        nonlocal found
        if op.get_name() == "tt.dot":
            found = True

    mod.walk(visit)
    return found


def _is_cv_report(report):
    return report.get("cv_mode") in ("independent_batch_v1", "spatial_concat_v1")


def analyze_request(mod, request, binding=None, *, report_listener=None):
    """Run native preflight in a worker. Observe its completed report in the caller."""
    if request.factor == 1:
        return {
            "report_version": 1, "compile_status": "DISABLED", "requested_factor": 1, "requested_axis": request.axis,
            "applied_factor": 1, "owner": "none", "reason": "DISABLED"
        }
    # MLIR shapes and the future native launch contract use signed i32 extents.
    # A positive Python integer outside that capability is not a config error.
    if request.factor > 2147483647:
        return {
            "report_version": 1, "compile_status": "REJECTED", "requested_factor": request.factor, "requested_axis":
            request.axis, "applied_factor": 1, "owner": "none", "reason": "FACTOR_OUT_OF_RANGE"
        }
    from .block_batching_worker import BatchCompileError, isolated_report
    if binding is not None:
        from types import BuiltinFunctionType
        analyze = getattr(binding, 'analyze_block_batching', None)
        if analyze is None:
            raise RuntimeError("block_batching requires a matching rebuilt Ascend C++ extension")
        # An arbitrary Python wrapper may itself call native MLIR. Only the
        # canonical pybind function can be reconstructed in the new interpreter.
        native_analyze = None
        if isinstance(analyze, BuiltinFunctionType):
            from triton._C.libtriton import ascend
            native_analyze = getattr(ascend.passes.ttir, 'analyze_block_batching', None)
        if analyze is not native_analyze:
            raise BatchCompileError(
                'custom analysis bindings cannot be reconstructed in the worker. '
                'use report_listener to observe native reports', kind='UnsupportedBinding', stage='analyze')
    report = isolated_report('analyze', str(mod), request.factor, request.axis)
    if report_listener is not None:
        from copy import deepcopy
        report_listener(deepcopy(report))
    return report


def probe_request(mod, request, layout_policy='semantic_only', cv_batch_matmul=False):
    if request.factor == 1 or request.factor > 2147483647:
        return analyze_request(mod, request)
    from .block_batching_worker import isolated_report
    return isolated_report('probe', str(mod), request.factor, request.axis, layout_policy, cv_batch_matmul)


def _runtime_positions(src):
    from .block_batching_worker import BatchDecline
    constants = getattr(src, "constants", {})
    result = []
    for index, name in enumerate(src.fn.arg_names):
        signature = src.signature.get(name, src.signature.get(index))
        if not isinstance(signature, str):
            return BatchDecline("NESTED_SIGNATURE_NOT_SUPPORTED", "signature")
        if signature != "constexpr" and (index, ) not in constants and index not in constants and name not in constants:
            result.append(index)
    return result


def _reduces_blocks(grid, axis, factor):
    """Whether batching this axis shrinks the flattened block count at all."""
    batched = list(grid)
    batched[axis] = (grid[axis] + factor - 1) // factor
    return batched[0] * batched[1] * batched[2] < grid[0] * grid[1] * grid[2]


def _batched_blocks(grid, axis, factor):
    batched = list(grid)
    batched[axis] = (grid[axis] + factor - 1) // factor
    return batched[0] * batched[1] * batched[2]


def _core_floor():
    """Minimum merged task count for the supported AIV-only candidates.

    Batching validates final execution mode as AIV, so its scheduling floor
    must use available vector cores rather than cube cores. Device limits
    remain owned and validated by the device interface. An explicit override
    is honored only after that validation; zero disables the floor.
    """
    import os
    from triton.runtime import driver
    available_aiv = driver.active.utils.get_aivector_core_num()
    override = os.getenv("TA_BLOCK_BATCHING_CORE_FLOOR")
    return int(override) if override is not None else available_aiv


def _capture_state():
    import sys
    torch = sys.modules.get("torch")
    query = getattr(getattr(torch, "npu", None), "is_current_stream_capturing", None)
    return query() if query is not None else None


class BatchKernel:
    """CompiledKernel-compatible facade selecting a complete variant per call."""

    def __init__(self, baseline, candidate, request, report, positions, *, publish_manifest=True):
        import hashlib
        import threading
        from collections import namedtuple
        self.baseline, self.candidate = baseline, candidate
        self._vv_mode = report.get("cv_mode") == "disabled"
        if self._vv_mode:
            from copy import deepcopy
            report = deepcopy(report)
        self.request, self.block_batching = request, report
        self.positions = positions
        from .block_batching_report import validate_vv_loop_contract
        self._preserve_grid = (validate_vv_loop_contract(candidate.metadata._asdict())
                               if candidate is not None else False)
        from .block_batching_contract import ExecutionContract
        self._contract = (ExecutionContract.from_validated_candidate(
            candidate, request, report, positions, self._preserve_grid) if candidate is not None else None)
        # Legacy performance diagnostics inspect this evaluator. Dispatch and
        # ownership stay in the common contract; this is an alias, not a copy.
        self._guard = self._contract.guard if self._contract is not None and not self._vv_mode else None
        self._primary = candidate if candidate is not None else baseline
        self._lock = threading.RLock()
        self._ready = False
        self._local = threading.local()
        self.hash = candidate.hash if candidate is not None else hashlib.sha256(
            f"{baseline.hash}:block_batching:{request.factor}:{request.axis}:{IMPLEMENTATION_VERSION}".encode(
            )).hexdigest()
        fields = dict(self._primary.metadata._asdict(), block_batching=report)
        self.metadata = namedtuple("BatchKernelMetadata", sorted(fields))(**fields)
        if publish_manifest:
            self.publish_manifest()

    def publish_manifest(self):
        import json
        from triton.runtime.cache import get_cache_manager
        manifest = {
            "version": 1, "baseline_hash": self.baseline.hash, "candidate_hash":
            self.candidate.hash if self.candidate is not None else None, "request":
            {"factor": self.request.factor, "axis": self.request.axis}, "report": self.block_batching
        }
        get_cache_manager(self.hash).put(json.dumps(manifest, sort_keys=True), "block_batching.bundle.json")

    def __getattr__(self, name):
        return getattr(self._primary, name)

    @property
    def last_launch(self):
        result = getattr(self._local, "launch", {"runtime_status": "NOT_LAUNCHED"})
        if self._vv_mode:
            from copy import deepcopy
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
                    self._primary = self.baseline
                    from collections import namedtuple
                    fields = dict(self.baseline.metadata._asdict(), block_batching=self.block_batching)
                    self.metadata = namedtuple("BatchKernelMetadata", sorted(fields))(**fields)
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
            from .block_batching_launch import candidate_launch_refusal
            live = contract._legacy_report
            request = contract._legacy_request if live is not None else contract
            report = live if live is not None else {'selected_axis': contract.axis}
            reason = candidate_launch_refusal(request, report, grid, contract.preserve_grid, capture,
                                              cv=_is_cv_report(live) if live is not None else False,
                                              reduces_blocks=_reduces_blocks, batched_blocks=_batched_blocks,
                                              core_floor=_core_floor,
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


def _freeze_source(backend, src, baseline_options, backend_revision):
    """Generate frontend IR once, then parse the same immutable snapshot per variant."""
    import hashlib
    from triton._C.libtriton import ir
    from triton.compiler.compiler import ASTSource
    from triton.runtime.cache import get_cache_manager
    if not isinstance(src, ASTSource) or hasattr(src, "block_batching_source_hash"):
        return src
    options = backend.parse_options(baseline_options)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = src.make_ir(backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                         context)
    text = str(module)
    fingerprint = hashlib.sha256((src.hash() + "\n" + backend_revision + "\n" + text).encode()).hexdigest()
    from triton.runtime.cache import FileCacheManager
    cache = get_cache_manager(fingerprint)
    if type(cache) is not FileCacheManager:
        from .block_batching_worker import BatchDecline
        return BatchDecline("CUSTOM_CACHE_NOT_SUPPORTED", "freeze")
    path = cache.put(text, "block_batching.original.ttir", binary=False)

    from .block_batching_source import FrozenSource
    return FrozenSource(src, path, fingerprint, hashlib.sha256(text.encode()).hexdigest())


def rejected_bundle(baseline, request, decline, positions=(), retry=None):
    reason = decline['reason']
    report = dict(report_version=1, compile_status='REJECTED', owner='none', requested_factor=request.factor,
                  requested_axis=request.axis, applied_factor=1, reason=reason)
    if decline.get('analysis') is not None:
        report['candidate_analysis'] = decline['analysis']
    if 'diagnostic' in decline:
        report.update(backend_diagnostic=decline['diagnostic'], stage=decline['stage'])
    if retry:
        report['layout'] = dict(requested_policy='auto', selected_policy='baseline', compile_fallback=retry)
    return BatchKernel(baseline, None, request, report, positions)


def _freeze_refusal(normalized):
    from triton import knobs
    if knobs.compilation.override or getattr(normalized, 'ir_override', None):
        return 'IR_OVERRIDE_NOT_SUPPORTED'
    if knobs.cache.manager_class is not None or knobs.cache.remote_manager_class is not None:
        return 'CUSTOM_CACHE_NOT_SUPPORTED'
    return None


def _candidate_refusal(backend, src, normalized):
    from .utils import backend_policy
    if not str(backend.target.arch).startswith(('Ascend950', 'Ascend910_95')):
        return 'TARGET_NOT_VALIDATED'
    if normalized.is_pure_simt or normalized.compile_mode == 'simt_only':
        return 'SIMT_NOT_SUPPORTED'
    if backend_policy not in (None, 'torch_npu'):
        return 'FRAMEWORK_NOT_VALIDATED'
    if not hasattr(src, 'fn'):
        return 'AST_SOURCE_REQUIRED'
    return None


def compile_bundle(backend, src, options, compile_variant):
    from .block_batching_worker import BatchDecline, CandidateCompiler, _baseline_environment
    request = BatchRequest(options.get('block_batching_factor', 1), options.get('block_batching_axis', -1))
    if request.factor == 1:
        return compile_variant(dict(options), src)
    normalized = backend.parse_options(dict(options))
    baseline_options = dict(options, block_batching_factor=1, block_batching_axis=-1)
    baseline_options.pop('block_batching_grid', None)
    refusal = _freeze_refusal(normalized)
    frozen = src if refusal else _freeze_source(backend, src, baseline_options,
                                                normalized.block_batching_backend_revision)
    if isinstance(frozen, BatchDecline):
        refusal, frozen = frozen.reason, src
    baseline = compile_variant(baseline_options, frozen)
    refusal = refusal or _candidate_refusal(backend, src, normalized)
    if refusal:
        return rejected_bundle(baseline, request, dict(reason=refusal))
    positions = _runtime_positions(src)
    if isinstance(positions, BatchDecline):
        return rejected_bundle(baseline, request, positions.payload())
    env_vars = _baseline_environment(frozen, backend, baseline_options, baseline)
    if env_vars is None:
        return rejected_bundle(baseline, request, dict(reason='BASELINE_ENVIRONMENT_NOT_PROVEN'), positions)
    return CandidateCompiler(backend, frozen, env_vars).compile(baseline, options, request, positions)
