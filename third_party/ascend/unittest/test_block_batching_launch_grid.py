"""Host regressions for callable-grid hints through the real JIT frontend."""
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl
from triton.backends.ascend.compiler import AscendBackend
from triton.backends.ascend.reuse.binding import ArgumentBinder
from triton.backends.ascend.reuse.context import InvocationFrame
from triton.runtime import jit as jit_module


@pytest.fixture(scope="module", autouse=True)
def assign_npu():
    """These tests do not initialize a device."""


@pytest.fixture(params=[False, True], ids=["native", "reuse-continuation"])
def frontend(monkeypatch, request):

    @triton.jit
    def kernel(out, rows, N: tl.constexpr, BLOCK_N: tl.constexpr = 4096):
        row = tl.program_id(1)
        col = tl.program_id(0) * BLOCK_N
        tl.store(out + row * N + col, 1)

    events, compiled, launches, callback_calls = [], [], [], []
    kernel.pre_run_hooks.append(lambda *args, **kwargs: events.append("pre_run"))
    bind = ArgumentBinder(kernel.signature)

    def binder(*args, **kwargs):
        bound, options = bind(*args, **kwargs)
        signature = [("constexpr", v) if p.is_constexpr else ("i32", None)
                     for p, v in zip(kernel.params, bound.values())]
        return bound, signature, options

    def pack(backend, kwargs, bound, specialization, options):
        return dict(options), {}, {}, {}

    def compile_kernel(key, signature, device, constexprs, options, attrs, warmup):
        events.append("compile")
        result = SimpleNamespace(options=options, function=0, packed_metadata=(), launch_metadata=lambda *a: None,
                                 run=lambda *a: launches.append(a[:3]))
        compiled.append(result)
        kernel.device_caches[0][0][key] = result
        return result

    class Handler:

        def run(self, fn, args, kwargs, grid, warmup, device, stream, backend, tail):
            events.append(("handler", kwargs.get("block_batching_grid"), callable(grid)))
            return tail(args, kwargs, grid, warmup, device, stream, frame=InvocationFrame())

    from triton.backends.ascend.reuse import bridge, config
    monkeypatch.setattr(config, "enabled", lambda: request.param)
    monkeypatch.setattr(bridge, "get_handler", lambda: Handler())
    backend = SimpleNamespace(get_jit_reuse_handler=AscendBackend.get_jit_reuse_handler)
    kernel.device_caches[0] = ({}, {}, None, backend, binder)
    monkeypatch.setattr(kernel, "_pack_args", pack)
    monkeypatch.setattr(kernel, "_do_compile", compile_kernel)
    monkeypatch.setattr(
        jit_module, "driver",
        SimpleNamespace(active=SimpleNamespace(get_current_device=lambda: 0, get_current_stream=lambda d: 0)))

    def grid(meta):
        callback_calls.append(dict(meta))
        events.append("grid")
        return (triton.cdiv(meta["N"], meta["BLOCK_N"]), meta["rows"])

    def run(rows=16, *, launch_grid=grid, warmup=False, **options):
        options.setdefault("block_batching_factor", 2)
        options.setdefault("block_batching_axis", -1)
        return kernel.run(0, rows, N=4096, grid=launch_grid, warmup=warmup, **options)

    return SimpleNamespace(run=run, grid=grid, compiled=compiled, launches=launches, calls=callback_calls,
                           events=events, reuse=request.param)


def test_callable_hint_precedes_cache_and_reuse(frontend):
    f = frontend
    first = f.run()
    assert first.options.get("block_batching_grid") == (1, 16, 1)
    assert f.events.index("pre_run") < f.events.index("grid") < f.events.index("compile")
    assert len(f.calls) == 1 and f.launches == [(1, 16, 1)]
    again = f.run()
    assert again is first and len(f.compiled) == 1 and len(f.calls) == 2
    changed = f.run(32)
    assert changed is not first and changed.options["block_batching_grid"] == (1, 32, 1)
    assert f.run() is first and len(f.compiled) == 2 and len(f.calls) == 4
    if f.reuse:
        handlers = [e for e in f.events if isinstance(e, tuple)]
        assert all(e[1] is not None and e[2] is False for e in handlers)


def test_callable_closure_changes_axis_without_argument_changes(frontend):
    f = frontend
    dimensions = [1, 16]
    calls = []

    def grid(meta):
        calls.append(1)
        return tuple(dimensions)

    first = f.run(launch_grid=grid)
    dimensions[:] = [16, 1]
    second = f.run(launch_grid=grid)
    assert second is not first
    assert second.options["block_batching_grid"] == (16, 1, 1)
    dimensions[:] = [1, 16]
    assert f.run(launch_grid=grid) is first
    assert len(calls) == 3 and len(f.compiled) == 2


def test_static_and_callable_share_cache_identity(frontend):
    f = frontend
    first = f.run(launch_grid=(1, 16))
    assert f.run() is first
    assert len(f.compiled) == 1 and len(f.calls) == 1


@pytest.mark.parametrize("options", [
    {"block_batching_factor": 1},
    {"block_batching_axis": 0},
    {"block_batching_grid": (1, 16, 1)},
])
def test_explicit_and_unmerged_callable_keep_late_resolution(frontend, options):
    f = frontend
    result = f.run(**options)
    assert f.events.index("compile") < f.events.index("grid")
    assert len(f.calls) == 1
    assert result.options.get("block_batching_grid") == options.get("block_batching_grid")


def test_warmup_does_not_evaluate_callable(frontend):
    f = frontend
    result = f.run(warmup=True)
    assert not f.calls and not f.launches
    assert "block_batching_grid" not in result.options
    f.run()
    assert len(f.calls) == 1 and f.launches == [(1, 16, 1)]


def test_callable_error_propagates_once_without_compilation(frontend):
    f = frontend
    calls = []
    error = RuntimeError("grid callback failed")

    def fail(meta):
        calls.append(meta)
        raise error

    with pytest.raises(RuntimeError) as exc:
        f.run(launch_grid=fail)
    assert exc.value is error and len(calls) == 1
    assert not f.compiled and not f.launches


def test_zero_grid_is_not_encoded_as_positive_hint(frontend):
    f = frontend
    result = f.run(0)
    assert "block_batching_grid" not in result.options
    assert len(f.calls) == 1 and f.launches == [(1, 0, 1)]


@pytest.mark.parametrize("launch_grid", [(1, 16), [1, 16]])
def test_unmerged_static_grid_keeps_original_options(frontend, launch_grid):
    f = frontend
    result = f.run(launch_grid=launch_grid, block_batching_factor=1)
    assert "block_batching_grid" not in result.options
    assert not f.calls and f.launches == [(1, 16, 1)]
    assert f.events.index("pre_run") < f.events.index("compile")


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_explicit_axis_grid_changes_reuse_one_compilation(frontend, axis):
    f = frontend
    calls = []

    def callable_grid(extent):

        def grid(meta):
            calls.append(extent)
            return (extent, )

        return grid

    first = f.run(launch_grid=(128, ), block_batching_axis=axis)
    for grid in ((256, ), [128], callable_grid(256), callable_grid(128)):
        assert f.run(launch_grid=grid, block_batching_axis=axis) is first
    assert len(f.compiled) == 1
    assert "block_batching_grid" not in first.options
    assert f.launches == [(128, 1, 1), (256, 1, 1), (128, 1, 1), (256, 1, 1), (128, 1, 1)]
    assert calls == [256, 128]


def test_explicit_axis_warmup_reuses_compilation_for_other_grid(frontend):
    f = frontend
    first = f.run(launch_grid=(128, ), block_batching_axis=0, warmup=True)
    assert not f.launches
    assert f.run(launch_grid=(256, ), block_batching_axis=0) is first
    assert len(f.compiled) == 1 and f.launches == [(256, 1, 1)]
