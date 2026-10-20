"""Host regression tests for constexpr analysis and reuse contracts.

Run with a dev build: python -m pytest -q third_party/ascend/unittest/test_constexpr_reuse.py
Tensor metadata, the native binder, loaded handles and the compiler worker use test doubles.
"""
import inspect
import io
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from threading import Event
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl
from triton import knobs
from triton.backends.ascend.reuse import analysis, bridge, config, context, protocol, service, transport, worker
from triton.backends.ascend.reuse.binding import ArgumentBinder
from triton.backends.ascend.reuse.bridge import ReuseHandler, requires_original
from triton.backends.ascend.reuse.dispatch import resolve_dispatch
from triton.backends.ascend.reuse.dsl.source import source_stamp
from triton.backends.ascend.reuse.guards import adapt_tile
from triton.backends.ascend.reuse.identity import family_key
from triton.backends.ascend.reuse.model import BuildSpec, JitTailRequest, ParameterClass, PreparedLaunch
from triton.backends.ascend.reuse.registry import VariantRegistry
from triton.backends.ascend.reuse.transform import FrozenJITFunction, ReuseASTSource


@dataclass(frozen=True)
class HostOptions:
    compile_mode: str = "simd"
    ir_override: object = None

    def hash(self):
        return "host-options"


@pytest.fixture(scope="module", autouse=True)
def assign_npu():
    """Override the parent conftest's device fixture for these host-only tests."""


@triton.jit
def scale(x, SCALE: tl.constexpr):
    return x * SCALE


@triton.jit
def scalar_store(out, N: tl.constexpr, SCALE: tl.constexpr, MODE: tl.constexpr):
    extent: tl.constexpr = N * 2
    i = tl.program_id(0)
    if i < extent:
        value = scale(N, SCALE=SCALE)
        if MODE:
            value = scale(value, SCALE)
        tl.store(out + i, value)


@triton.jit
def partial_unknown(out, N: tl.constexpr, BINS: tl.constexpr):
    histogram = tl.histogram(tl.arange(0, 32), BINS)
    tl.store(out, tl.sum(histogram, 0) + N)


@triton.jit
def block_dot(a, b, out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr, ALLOW: tl.constexpr):
    ap = tl.make_block_ptr(a, (M, K), (K, 1), (0, 0), (BM, BK), (1, 0))
    bp = tl.make_block_ptr(b, (K, N), (N, 1), (0, 0), (BK, BN), (1, 0))
    x = tl.load(ap, boundary_check=(0, 1), padding_option="zero")
    y = tl.load(bp, boundary_check=(0, 1), padding_option="zero")
    result = tl.dot(x, y, allow_tf32=ALLOW)
    cp = tl.make_block_ptr(out, (M, N), (N, 1), (0, 0), (BM, BN), (1, 0))
    tl.store(cp, result, boundary_check=(0, 1))


@triton.jit
def copy_kernel(src, dst, N: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    value = tl.load(src + index, mask=index < N, other=0)
    tl.store(dst + index, value, mask=index < N)


@triton.jit
def grid_stride_copy(src, dst, n, work, TILE: tl.constexpr):
    for block in range(tl.program_id(0), work, tl.num_programs(0)):
        index = block * TILE + tl.arange(0, TILE)
        value = tl.load(src + index, mask=index < n, other=0)
        tl.store(dst + index, value, mask=index < n)


class Pointer:
    dtype = "float32"

    def data_ptr(self):
        pytest.fail("analysis must not inspect tensor addresses")


def analyze(fn, *arguments, allow_tile_reuse=True):
    return analysis.analyze(fn, dict(zip(fn.arg_names, arguments)), fn.cache_key, allow_tile_reuse=allow_tile_reuse)


def test_dynamic_transform_preserves_static_control_and_original():
    original = (scalar_store.src, scalar_store.signature, scalar_store.params, scalar_store.device_caches, scale.src)
    profile = analyze(scalar_store, Pointer(), 7, 3, True)
    assert [d.classification for d in profile.decisions
            ] == [ParameterClass.RuntimeEligible, ParameterClass.RuntimeEligible, ParameterClass.StaticRequired]
    clone = FrozenJITFunction.from_plan(profile.plan)
    assert clone.constexprs == [3]
    for i in (1, 2):
        assert clone.signature.parameters[clone.arg_names[i]].annotation is inspect.Parameter.empty
        assert clone.params[i].do_not_specialize and clone.params[i].do_not_specialize_on_alignment
    assert "extent: " not in clone.src
    helpers = [value for name, value in clone.get_capture_scope().items() if name.startswith("_reuse_helper_")]
    assert len(helpers) == 2 and all(not helper.constexprs for helper in helpers)
    assert clone.device_caches is not scalar_store.device_caches
    assert original == (scalar_store.src, scalar_store.signature, scalar_store.params, scalar_store.device_caches,
                        scale.src)
    assert all(scalar_store.params[i].is_constexpr for i in (1, 2, 3))
    assert scale.params[1].is_constexpr


def test_unknown_operation_only_blocks_dependent_parameters():
    profile = analyze(partial_unknown, Pointer(), 7, 32)
    assert profile.plan.dynamic == (1, )
    assert profile.decisions[1].classification is ParameterClass.Unknown
    assert "OPERATION_SUMMARY_MISSING" in repr(profile.decisions[1].reasons)


def test_dot_and_block_pointer_shapes_remain_static():
    profile = analyze(block_dot, Pointer(), Pointer(), Pointer(), 35, 47, 65, 32, 32, 32, False)
    assert profile.plan.dynamic == (3, 4, 5)
    assert all(d.classification is ParameterClass.StaticRequired for d in profile.decisions[3:])
    assert profile.recipe is None


def test_analysis_cache_relaxes_only_proven_values(monkeypatch):
    handler = ReuseHandler()
    handler.registry = VariantRegistry()

    def lookup(fn, args, **kwargs):
        with handler.lock:
            return handler._analysis_entry(fn, dict(zip(fn.arg_names, args)), fn.cache_key, "npu", "options", **kwargs)

    args = (Pointer(), Pointer(), 67, 64)
    first = lookup(copy_kernel, args, allow_tile_reuse=True)
    profile, owner = first
    assert profile.plan.dynamic == (2, ) and profile.recipe.tiles == (3, )
    assert profile.decisions[1].classification is ParameterClass.ScheduleReusable
    assert owner.params[3].is_constexpr
    with monkeypatch.context() as patch:
        patch.setattr(analysis, "analyze", lambda *a, **kw: pytest.fail("redundant analysis"))
        patch.setattr(FrozenJITFunction, "from_plan", lambda *a: pytest.fail("redundant JIT clone"))
        assert lookup(copy_kernel, (*args[:2], 129, 32), allow_tile_reuse=True) is first
    dynamic_only = lookup(copy_kernel, args, allow_tile_reuse=False)
    assert dynamic_only is not first
    assert dynamic_only[0].plan.dynamic == (2, ) and dynamic_only[0].recipe is None
    assert dynamic_only[0].decisions[1].classification is ParameterClass.Unknown
    controlled = lookup(scalar_store, (Pointer(), 7, 3, True), allow_tile_reuse=True)
    assert lookup(scalar_store, (Pointer(), 19, 5, True), allow_tile_reuse=True) is controlled
    assert lookup(scalar_store, (Pointer(), 19, 5, False), allow_tile_reuse=True) is not controlled
    assert lookup(scalar_store, (Pointer(), 19.0, 5, True), allow_tile_reuse=True) is not controlled


def test_frozen_request_round_trip_and_live_source_invalidation(monkeypatch):
    profile = analyze(scalar_store, Pointer(), 7, 3, True)
    owner = FrozenJITFunction.from_plan(profile.plan)
    source = ReuseASTSource(owner, {"out": "*fp32", "N": "i32", "SCALE": "i32", "MODE": "constexpr"}, {(3, ): True})
    stamp = source_stamp(scalar_store)
    spec = BuildSpec(owner, source, "memory", "variant", ("family", ), SimpleNamespace(compile_mode="simd"), (),
                     stamp.weak_guard())
    request = protocol.build_request(spec, {}, 0)
    restored = protocol.rebuild_source(request)
    assert restored.hash() == source.hash()
    assert restored.signature == source.signature and restored.constants == source.constants
    assert restored.fn.cache_key == owner.cache_key
    assert restored.fn.device_caches is not owner.device_caches
    frozen = protocol.freeze_source(spec, request)
    assert frozen.is_current()
    monkeypatch.setattr(scale, "_src", scale.src.replace("x * SCALE", "x + SCALE"))
    assert not stamp.matches() and not frozen.is_current()


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_ASCEND_ENABLE_DYNAMIC_REUSE", "2")
    config.mode.cache_clear()
    monkeypatch.setattr(knobs, "propagate_env", False)
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path))
    for name in ("manager_class", "remote_manager_class"):
        monkeypatch.setattr(knobs.cache, name, None)
    for name in ("jit_cache_hook", "jit_post_compile_hook", "add_stages_inspection_hook"):
        monkeypatch.setattr(knobs.runtime, name, None)
    for name in ("override", "always_compile"):
        monkeypatch.setattr(knobs.compilation, name, False)
    yield
    config.mode.cache_clear()


@pytest.mark.parametrize("value,expected", [(None, 0), ("1", 1), ("2", 2), ("invalid", 0)])
def test_startup_modes_gate_tile_reuse_and_background(runtime, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("TRITON_ASCEND_ENABLE_DYNAMIC_REUSE")
    else:
        monkeypatch.setenv("TRITON_ASCEND_ENABLE_DYNAMIC_REUSE", value)
    target, options = SimpleNamespace(backend="npu"), SimpleNamespace(compile_mode="simd")
    assert config.mode() == expected and config.enabled() == (expected != 0)
    assert config.supports_tile_reuse(target, options) == (expected == 2)
    assert not config.supports_tile_reuse(target, SimpleNamespace(compile_mode="simt_only"))
    monkeypatch.setenv("TRITON_ASCEND_ENABLE_DYNAMIC_REUSE", "0" if expected else "2")
    assert config.mode() == expected  # A launch must not reread startup policy.
    if expected != 2:
        instance = service.PreparationService(VariantRegistry(), {}, service.Capabilities(compile=True))
        try:
            assert not instance.try_submit(SimpleNamespace(variant_key="unused"), (os.getpid(), 0))
            assert instance.compiler_executor is None and instance.worker is None
        finally:
            instance.close()


@pytest.fixture
def tensors(monkeypatch):
    # Metadata only: production guards run unchanged; no device allocation.
    from triton.runtime.cache import triton_key

    # Cache the native installation fingerprint before replacing tensor metadata.
    # Its package scan must import optional runtime modules with the real torch.
    triton_key()

    class Tensor:
        dtype = "float32"
        device = SimpleNamespace(type="npu", index=0)

        def __init__(self, address, count=67):
            self.address, self.count = address, count

        def data_ptr(self):
            return self.address

        def element_size(self):
            return 4

        def numel(self):
            return self.count

        def is_contiguous(self):
            return True

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Tensor=Tensor))
    return Tensor


def test_grid_stride_reuse_updates_and_validates_external_block_count(tensors):
    args = (tensors(4096), tensors(8192), 67, 3, 32)
    recipe = analyze(grid_stride_copy, *args).recipe
    assert recipe is not None and recipe.bound == 3
    actual = adapt_tile(recipe, args, {(4, ): 64}, (2, ), 0, compile_mode="simd")
    assert actual[:3] == args[:3] and actual[3:] == (2, 64)
    visited = [
        block * actual[4] + lane
        for pid in range(2)
        for block in range(pid, actual[3], 2)
        for lane in range(actual[4])
        if block * actual[4] + lane < actual[2]
    ]
    assert sorted(visited) == list(range(67))
    assert adapt_tile(recipe, (*args[:3], 2, 32), {(4, ): 64}, (2, ), 0, compile_mode="simd") is None


@pytest.fixture
def dispatch_case(runtime, tensors):
    args = (tensors(4096), tensors(8192), 67, 64)
    profile = analyze(copy_kernel, *args)
    owner = FrozenJITFunction.from_plan(profile.plan)
    signature = {"src": "*fp32", "dst": "*fp32", "N": "i32", "TILE": "constexpr"}

    def binder(*args, **kwargs):
        return dict(zip(owner.arg_names, args)), [(ty, value if ty == "constexpr" else None)
                                                  for ty, value in zip(signature.values(), args)], kwargs

    owner.device_caches[0] = ({}, {}, None, None, binder)
    options = SimpleNamespace(compile_mode="simd")

    def make_spec(tile):
        source = ReuseASTSource(owner, signature.copy(), {(3, ): tile})
        family = family_key(profile, source, options, "npu", {}, options_hash="options")
        return BuildSpec(owner, source, str(tile), source.hash(), family, options, ())

    def kernel(spec, **overrides):
        result = SimpleNamespace(src=spec.source, hash=spec.variant_key, module=1, function=2, _run=lambda: None,
                                 _initialization_complete=True)
        result.__dict__.update(overrides)
        return result

    return SimpleNamespace(args=args, profile=profile, registry=VariantRegistry(), context=(os.getpid(), 0),
                           desired=make_spec(64), old=make_spec(32), make_spec=make_spec, kernel=kernel)


def dispatch(case, **kwargs):
    return resolve_dispatch(case.registry, case.desired, case.profile, case.args, kwargs.pop("grid", (2, )),
                            case.context, None, **kwargs)


def test_dispatch_cold_family_native_cache_and_exact_priority(dispatch_case):
    case = dispatch_case
    assert isinstance(dispatch(case), JitTailRequest)
    # A rejected candidate must not hide the next compatible tile.
    bad = case.make_spec(0)
    case.registry.register_ready_if_eligible(case.kernel(bad), bad, case.context)
    old = case.kernel(case.old)
    case.registry.register_ready_if_eligible(old, case.old, case.context)
    grid_calls = []
    frame = context.InvocationFrame()

    def grid(meta):
        grid_calls.append(meta["TILE"])
        return (2, )

    result = dispatch(case, grid=grid, resolve_grid=lambda: frame.resolve_grid(grid, {"TILE": 64}))
    assert isinstance(result, PreparedLaunch) and result.executable.kernel is old
    assert result.arguments[3] == 32 and result.grid == (3, )
    assert grid_calls == [64]
    visited = [pid * 32 + lane for pid in range(result.grid[0]) for lane in range(32) if pid * 32 + lane < 67]
    assert visited == list(range(67))  # Independent coverage oracle, including the tail.
    assert isinstance(dispatch(case, target_cached=lambda *args: True), JitTailRequest)
    exact = case.kernel(case.desired)
    case.registry.register_ready_if_eligible(exact, case.desired, case.context)
    result = dispatch(case, target_cached=lambda *args: pytest.fail("exact READY bypassed"))
    assert result.executable.kernel is exact and result.arguments == case.args
    assert case.registry.lookup_exact_ready(case.desired.variant_key, (case.context[0], 1)) is None


@pytest.mark.parametrize("hazard", ["overlap", "out_of_bounds", "overflow", "abi_mismatch"])
def test_unsafe_tile_candidate_returns_to_jit(dispatch_case, hazard):
    case = dispatch_case
    if hazard == "overlap":
        case.args[1].address = case.args[0].address + 4
    elif hazard == "out_of_bounds":
        case.args[1].count = 66
    elif hazard == "overflow":
        case.args = (*case.args[:2], 2**31, 64)
    else:
        case.old.source.signature["N"] = "i64"
    assert case.registry.register_ready_if_eligible(case.kernel(case.old), case.old, case.context) is not None
    assert isinstance(dispatch(case), JitTailRequest)


def test_registry_publishes_only_successful_initialization(dispatch_case):
    case = dispatch_case
    kernel = case.kernel(case.desired, _initialization_complete=False)
    assert case.registry.register_ready_if_eligible(kernel, case.desired, case.context) is None
    error = RuntimeError("load callback failed")
    calls = []

    def fail():
        calls.append("load")
        raise error

    kernel._init_handles = fail
    with pytest.raises(RuntimeError) as caught:
        case.registry.initialize(kernel, case.desired, case.context)
    assert caught.value is error and calls == ["load"]
    assert not case.registry.exact and not case.registry.families
    ready = case.kernel(case.desired, _initialization_complete=False)

    def initialize():
        calls.append("ready")
        ready._initialization_complete = True

    ready._init_handles = initialize
    assert case.registry.initialize(ready, case.desired, case.context) is ready
    assert case.registry.initialize(kernel, case.desired, case.context) is ready
    assert calls == ["load", "ready"]


@pytest.fixture
def background(dispatch_case, monkeypatch, tmp_path):
    case = dispatch_case
    started, release = Event(), Event()
    calls = []

    class Worker:

        def __init__(self, **kwargs):
            pass

        def compile(self, request):
            calls.append(request["variant_key"])
            started.set()
            assert release.wait(5), "test did not release compiler"
            return request["variant_key"]

        def close(self):
            pass

    monkeypatch.setattr(service, "CompileWorker", Worker)
    instance = service.PreparationService(case.registry, {"device": 0, "cache": str(tmp_path)},
                                          service.Capabilities(compile=True))
    try:
        yield SimpleNamespace(service=instance, worker=Worker, case=case, started=started, release=release, calls=calls)
    finally:
        release.set()
        instance.close()
        if instance.compiler_executor is not None:
            instance.compiler_executor.shutdown(wait=True, cancel_futures=True)


def drain(instance):
    instance.compiler_executor.submit(lambda: None).result(timeout=5)


def test_background_deduplicates_and_only_hands_off_to_native_jit(background, monkeypatch):
    bg = background
    case, instance = bg.case, bg.service
    # Observational hooks remain foreground-only and do not prohibit reuse.
    for owner, name in ((knobs.compilation, "listener"), (knobs.runtime, "kernel_load_start_hook")):
        monkeypatch.setattr(owner, name, lambda *a, **kw: pytest.fail("foreground hook ran in background"))
    assert not requires_original(copy_kernel, False)
    assert instance.try_submit(case.desired, case.context)
    assert bg.started.wait(5)
    assert not instance.try_submit(replace(case.desired, family_key=("another-proof", )), case.context)
    bg.release.set()
    drain(instance)
    assert bg.calls == [case.desired.variant_key]
    assert not instance.tasks and not case.registry.exact and not case.registry.families
    assert instance.has_compiled(case.desired, case.context)
    assert isinstance(dispatch(case, target_cached=instance.has_compiled), JitTailRequest)
    monkeypatch.setattr(case.desired.source.fn, "_src", case.desired.source.fn.src + "\n# changed\n")
    assert not instance.has_compiled(case.desired, case.context)
    assert not instance.completed


def test_background_failure_is_contained_and_backs_off(background, monkeypatch):
    bg = background

    def fail(self, request):
        bg.calls.append(request["variant_key"])
        raise ValueError("compile failed")

    monkeypatch.setattr(bg.worker, "compile", fail)
    assert bg.service.try_submit(bg.case.desired, bg.case.context)
    drain(bg.service)
    assert not bg.service.tasks and not bg.service.completed and not bg.case.registry.exact
    assert bg.service.failures[bg.case.desired.variant_key].classification == "ValueError"
    assert not bg.service.try_submit(bg.case.desired, bg.case.context)
    assert bg.calls == [bg.case.desired.variant_key]


class FalseyHook:

    def __bool__(self):
        return False

    def __call__(self, *args, **kwargs):
        pytest.fail("inactive callback invoked")


@pytest.mark.parametrize("name,value,blocked", [
    ("jit_cache_hook", lambda *a: None, True),
    ("jit_cache_hook", FalseyHook(), False),
    ("add_stages_inspection_hook", FalseyHook(), True),
])
def test_frontend_and_background_share_hook_policy(background, monkeypatch, name, value, blocked):
    bg = background
    monkeypatch.setattr(knobs.runtime, name, value)
    assert bool(requires_original(copy_kernel, False)) == blocked
    assert bg.service.try_submit(bg.case.desired, bg.case.context) == (not blocked)
    if blocked:
        assert bg.service.compiler_executor is None and not bg.calls
    else:
        bg.release.set()
        drain(bg.service)


@pytest.fixture
def frontend(runtime, tensors, monkeypatch):
    """Run the real frontend; only the native binder/compiler/launcher are doubles."""
    handler = ReuseHandler()
    target = SimpleNamespace(backend="npu", arch="Ascend950", warp_size=32)
    backend = SimpleNamespace(target=target, parse_options=lambda kw: HostOptions(), hash=lambda: "host-backend")
    calls, compiled = [], []

    def create_binder(owner):
        bind = ArgumentBinder(owner.signature)

        def specialize(*args, **kwargs):
            bound, options = bind(*args, **kwargs)
            signature = [("constexpr", value) if p.is_constexpr else
                         ("*fp32" if hasattr(value, "data_ptr") else "i32", None)
                         for p, value in zip(owner.params, bound.values())]
            return bound, signature, options

        return {}, {}, target, backend, specialize

    monkeypatch.setattr(triton.JITFunction, "create_binder", create_binder)
    monkeypatch.setattr(service, "capabilities", lambda: service.Capabilities(compile=True))
    monkeypatch.setattr(protocol, "environment", lambda device, target: {"device": device, "cache": knobs.cache.dir})

    class Compiler:

        def __init__(self, **kwargs):
            pass

        def compile(self, request):
            assert calls[-1][0] == "prepared"  # Foreground launch must finish first.
            compiled.append(request["variant_key"])
            return request["variant_key"]

        def close(self):
            pass

    monkeypatch.setattr(service, "CompileWorker", Compiler)

    def tail(args, kwargs, grid, warmup, device, stream, **state):
        if "owner" not in state:
            calls.append(("original", state))
            return "original"
        frame = state["frame"]
        frame.resolve_grid(grid, state["bound"])
        if "prepared" in state:
            kernel = state["prepared"]
            kind = "prepared"
        else:
            spec = next(value for value in handler.cache.values()
                        if isinstance(value, BuildSpec) and value.memory_key == state["memory_key"])
            kernel = SimpleNamespace(src=spec.source, hash=spec.variant_key, module=1, function=2, _run=lambda: None,
                                     _initialization_complete=False)
            kernel._init_handles = lambda: setattr(kernel, "_initialization_complete", True)
            kernel = state["initialize"](kernel)
            state["owner"].device_caches[device][0][state["memory_key"]] = kernel
            kind = "native"
        frame.before_launch()
        calls.append((kind, state))
        return kernel

    def run(tile=32, *, jit_tail=tail, **kwargs):
        args = (tensors(4096), tensors(8192), 67, tile)
        return handler.run(copy_kernel, args, {}, kwargs.pop("grid", (triton.cdiv(67, tile), )),
                           kwargs.pop("warmup", False), kwargs.pop("device", 0), 0, backend, jit_tail)

    try:
        yield SimpleNamespace(handler=handler, backend=backend, run=run, tail=tail, calls=calls, compiled=compiled)
    finally:
        if handler.preparation is not None:
            executor = handler.preparation.compiler_executor
            handler.preparation.close()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)


def test_frontend_cold_reuse_background_and_exact_handoff(frontend, monkeypatch, capsys):
    f = frontend
    monkeypatch.setattr(knobs.runtime, "debug", True)
    first = f.run()
    owner = f.calls[-1][1]["owner"]
    assert owner is not copy_kernel and first.src.constants == {(3, ): 32}
    assert first.src.signature["N"] == "i32"
    grid_calls = []
    assert f.run(64, grid=lambda meta: grid_calls.append(meta["TILE"]) or (2, )) is first
    assert grid_calls == [64] and f.calls[-1][1]["frame"].grid == (3, )
    drain(f.handler.preparation)
    assert len(f.compiled) == 1
    target = f.run(64)
    assert target is not first and target.src.constants == {(3, ): 64}
    assert f.run(64) is target
    assert [kind for kind, _ in f.calls] == ["native", "prepared", "native", "prepared"]
    assert f.handler.registry.snapshot()["analysis_miss"] == 1
    monkeypatch.setattr(bridge, "_handler", f.handler)
    diagnostic = bridge.diagnostics()
    assert diagnostic["profiles"][0]["dynamic_slots"] == (2, )
    assert diagnostic["counts"]["native_init"] == 2
    assert "schedule-reused=[TILE: 64 -> 32]" in capsys.readouterr().err


@pytest.mark.parametrize("phase", [None, "launch", "load", "listener"])
def test_frontend_fallback_does_not_replay_observed_effects(frontend, phase):
    from triton.runtime.errors import OutOfResources

    f = frontend
    error = OutOfResources(2, 1, "test resource")
    attempts = []

    def failing_tail(*args, **state):
        attempts.append("private" if "owner" in state else "original")
        if "owner" not in state:
            return "fallback"
        frame = state["frame"]
        if phase == "launch":
            frame.before_launch()
        elif phase == "load":
            frame.load_hook_started = True
        elif phase == "listener":
            frame.before_compilation_listener()
        raise error

    if phase is not None:
        with pytest.raises(OutOfResources) as caught:
            f.run(jit_tail=failing_tail)
        assert caught.value is error and attempts == ["private"]
        assert f.handler.registry.snapshot().get("fallback", 0) == 0
    else:
        for _ in range(4):
            assert f.run(jit_tail=failing_tail) == "fallback"
        assert attempts == ["private", "original"] * 3 + ["original"]
        assert f.handler.registry.snapshot()["circuit_open"] == 1


def test_frontend_exact_scope_and_argument_errors_precede_reuse(frontend):
    f = frontend
    with pytest.raises(TypeError):
        f.handler.run(copy_kernel, (Pointer(), ), {}, (1, ), False, 0, 0, f.backend,
                      lambda *a, **kw: pytest.fail("invalid arguments reached native JIT"))
    assert f.handler.registry is None and f.handler.preparation is None
    with pytest.raises(RuntimeError, match="scope exit"):
        with context.exact_request_scope():
            assert f.run() == "original"
            raise RuntimeError("scope exit")
    assert not context.requires_exact_request()
    assert f.run(warmup=True) == "original"
    f.run()
    assert f.run(device=1) == "original"
    frame = context.InvocationFrame()
    frame.before_launch()
    with pytest.raises(RuntimeError, match="after launch"):
        frame.select_grid((1, ))


@triton.jit
def tile_indices(start, TILE: tl.constexpr, n, ENABLE: tl.constexpr = True):
    if ENABLE:
        index = start + tl.arange(0, TILE)
        return index, n > index
    return start + tl.arange(0, TILE + 1), True


@triton.jit
def local_reduce(src, dst, n, TILE: tl.constexpr, MAXIMUM: tl.constexpr, COUNTED: tl.constexpr):
    base = tl.program_id(0) * n
    if MAXIMUM:
        state = tl.full((TILE, ), -float("inf"), dtype=tl.float32)
    else:
        state = tl.zeros((TILE, ), dtype=tl.float32)
    step: tl.constexpr = 1 if COUNTED else TILE
    stop = tl.cdiv(n, TILE) if COUNTED else n
    for block in range(0, stop, step):
        start = block * TILE if COUNTED else block
        index, mask = tile_indices(start, TILE, n)
        if MAXIMUM:
            value = tl.load(src + base + index, mask=mask, other=-float("inf"))
            state = tl.maximum(state, value)
        else:
            value = tl.load(src + base + index, mask=mask, other=0.0)
            state += value * value
    if MAXIMUM:
        total = tl.max(state, 0)
    else:
        total = tl.sqrt(tl.sum(state, 0))
    tl.store(dst + tl.program_id(0), total, mask=tl.program_id(0) < tl.num_programs(0))


@pytest.mark.parametrize("maximum,counted", [(False, False), (True, True)])
def test_local_reduction_helpers_keep_neutral_padding_and_static_controls(tensors, maximum, counted):
    args = (tensors(4096, 201), tensors(8192, 3), 67, 32, maximum, counted)
    profile = analyze(local_reduce, *args)
    recipe = profile.recipe
    assert recipe is not None and recipe.tiles == (3, )
    assert {4, 5} <= set(recipe.static_parameters)
    assert adapt_tile(recipe, args, {(3, ): 16}, (3, ), 0, compile_mode="simd") == (*args[:3], 16, *args[4:])
    assert adapt_tile(recipe, args, {(3, ): 16, (4, ): not maximum}, (3, ), 0, compile_mode="simd") is None
    # A changed reduction identity must not silently reuse the successful proof.
    damaged = local_reduce.src.replace('other=0.0', 'other=1.0').replace('other=-float("inf")', 'other=0.0')
    original = local_reduce.src
    try:
        local_reduce._src = damaged
        assert analyze(local_reduce, *args).recipe is None
    finally:
        local_reduce._src = original


@triton.jit
def matrix_walk(src, dst, rows, cols, BM: tl.constexpr, BN: tl.constexpr, REDUCE: tl.constexpr):
    for block in range(tl.program_id(0), tl.cdiv(rows, BM), tl.num_programs(0)):
        row = block * BM + tl.arange(0, BM)
        if REDUCE:
            state = tl.zeros((BM, ), dtype=tl.float32)
        for start in range(0, cols, BN):
            column = start + tl.arange(0, BN)
            mask = (row[:, None] < rows) & (column[None, :] < cols)
            offset = row[:, None] * cols + column[None, :]
            value = tl.load(src + offset, mask=mask, other=0.0)
            if REDUCE:
                state += tl.sum(value, 1)
            else:
                tl.store(dst + offset, tl.exp(value), mask=mask)
        if REDUCE:
            tl.store(dst + row, state, mask=row < rows)


@pytest.mark.parametrize("reduce", [False, True])
def test_nested_axes_preserve_full_matrix_or_fixed_row_reduction(tensors, reduce):
    args = (tensors(4096, 335), tensors(8192, 335), 5, 67, 2, 32, reduce)
    recipe = analyze(matrix_walk, *args).recipe
    assert recipe is not None
    assert recipe.tiles == ((4, ) if reduce else (4, 5))
    constants = {(4, ): 1, (5, ): 32 if reduce else 16}
    actual = adapt_tile(recipe, args, constants, (2, ), 0, compile_mode="simd")
    assert actual == (*args[:4], 1, constants[(5, )], reduce)
    assert adapt_tile(recipe, args, constants, (2, ), 0, compile_mode="simt_only") is None
    args[1].count = 4 if reduce else 334
    assert adapt_tile(recipe, args, constants, (2, ), 0, compile_mode="simd") is None


@triton.jit
def partitioned_copy(src, dst, rows, cols, STRIDE: tl.constexpr, TILE: tl.constexpr):
    begin = tl.program_id(0) * STRIDE
    end = tl.minimum(begin + tl.cdiv(rows, tl.num_programs(0)), rows)
    for row in range(begin, end):
        for start in range(0, cols, TILE):
            column = start + tl.arange(0, TILE)
            mask = column < cols
            value = tl.load(src + row * cols + column, mask=mask, other=0.0)
            tl.store(dst + row * cols + column, value, mask=mask)


def test_preserved_outer_partitions_reject_overlap_and_excessive_proof_work(tensors):
    args = (tensors(4096, 335), tensors(8192, 335), 5, 67, 2, 32)
    recipe = analyze(partitioned_copy, *args).recipe
    assert recipe is not None and recipe.tiles == (5, )
    assert adapt_tile(recipe, args, {(5, ): 16}, (3, ), 0, compile_mode="simd") == (*args[:5], 16)
    assert adapt_tile(recipe, (*args[:4], 1, 32), {(5, ): 16}, (3, ), 0, compile_mode="simd") is None
    assert adapt_tile(recipe, args, {(5, ): 16}, (4097, ), 0, compile_mode="simd") is None


@pytest.mark.parametrize("failure,error_type", [("identity", transport.WorkerProtocolError),
                                                ("compile", transport.WorkerCompileError), ("eof", BrokenPipeError),
                                                ("timeout", TimeoutError)])
def test_worker_pipe_protocol_failure_and_owned_shutdown(monkeypatch, tmp_path, failure, error_type):
    # A real owned subprocess exercises nonblocking pipes and shutdown. The native
    # compiler boundary is represented by this deterministic protocol peer.
    child = '''
import json, os, sys, time
with os.fdopen(int(sys.argv[1]), "w", buffering=1) as replies:
    for line in sys.stdin:
        message = json.loads(line)
        action = message["request"]["action"]
        print("compiler diagnostic", flush=True)
        if action == "eof":
            break
        if action == "timeout":
            time.sleep(10)
        reply = dict(transport=2, request_id=message["request_id"], stage="compile", variant_key="result")
        if action == "identity":
            reply["request_id"] = "wrong"
        if action == "compile":
            reply.update(error="ValueError", message="compiler failed")
        replies.write(json.dumps(reply) + "\\n")
'''
    popen = transport.subprocess.Popen

    def launch(command, **kwargs):
        return popen([sys.executable, "-u", "-c", child, command[-1]], **kwargs)

    monkeypatch.setattr(transport.subprocess, "Popen", launch)
    compiler = transport.CompileWorker(cache=tmp_path)
    try:
        assert compiler.compile({"action": "ok"}) == "result"
        assert b"compiler diagnostic" in compiler.log
        if failure == "timeout":
            monkeypatch.setattr(transport, "_REQUEST_TIMEOUT", 0.1)
        with pytest.raises(error_type):
            compiler.compile({"action": failure})
    finally:
        compiler.close()
    assert compiler.process.poll() is not None and compiler.read_fd is None
    compiler.close()
    with pytest.raises(BrokenPipeError, match="closed"):
        compiler.compile({"action": "ok"})


def test_worker_compile_validates_frozen_context_and_replies(dispatch_case, monkeypatch, tmp_path):
    from triton.runtime import driver

    case = dispatch_case
    target = SimpleNamespace(backend="npu", arch="Ascend950", warp_size=32)
    utils = SimpleNamespace(get_device_core=lambda: (24, 48), get_aicore_num=lambda: 24,
                            get_aivector_core_num=lambda: 48)
    monkeypatch.setattr(driver, "_active", SimpleNamespace(utils=utils, get_current_target=lambda: target))
    devices, compilations = [], []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(set_device=devices.append)))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(worker.compile_request, "cache_root", str(tmp_path), raising=False)
    options = HostOptions()
    backend = SimpleNamespace(parse_options=lambda values: options, hash=lambda: "host-backend")
    # compile_request imports these names from the public compiler module.
    import triton.compiler
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton.backends.ascend.reuse.identity import variant_key

    monkeypatch.setattr(triton.compiler, "make_backend", lambda selected: backend)
    key = variant_key(case.desired.source, backend, options, get_cache_invalidating_env_vars())
    spec = replace(case.desired, variant_key=key, options=options)
    request = protocol.build_request(spec, protocol.environment(0, target), 0)

    def compile(source, **kwargs):
        compilations.append(source.hash())
        return SimpleNamespace(hash=key)

    monkeypatch.setattr(triton.compiler, "compile", compile)
    assert worker.compile_request(request) == key
    assert devices == [0] and compilations == [spec.source.hash()]
    bad = dict(request, variant_key="wrong")
    with pytest.raises(ValueError, match="VariantKey differs"):
        worker.compile_request(bad)
    assert len(compilations) == 1
    changed_env = dict(request, environment=dict(request["environment"], core_limit=[1, 1]))
    with pytest.raises(ValueError, match="environment mismatch"):
        worker.compile_request(changed_env)
    changed_cache = dict(request, environment=dict(request["environment"], cache=str(tmp_path / "other")))
    with pytest.raises(ValueError, match="cache root differs"):
        worker.compile_request(changed_cache)
    output = io.StringIO()
    messages = [
        dict(transport=2, request_id="success", request=request),
        dict(transport=2, request_id="failure", request=bad),
        dict(transport=99, request_id="invalid", request=request)
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(json.dumps(m) for m in messages)))
    monkeypatch.setattr(sys, "argv", ["worker", "123"])
    monkeypatch.setattr(os, "fdopen", lambda *args, **kwargs: nullcontext(output))
    worker.main()
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert replies[0]["variant_key"] == key
    assert [r.get("error") for r in replies] == [None, "ValueError", "ValueError"]
    assert [r["request_id"] for r in replies] == ["success", "failure", "invalid"]
