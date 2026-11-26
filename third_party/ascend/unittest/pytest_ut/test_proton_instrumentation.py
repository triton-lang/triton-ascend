"""Exercise the same instrumented vector add on software-buffer and hardware-sampled NPUs."""
import json
import math
from collections import Counter
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("triton._C.libproton")
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.backends.ascend.compiler import NPUOptions


def _is_a5():
    arch = triton.runtime.driver.active.get_current_target().arch
    return isinstance(arch, str) and arch.startswith(("Ascend910_95", "Ascend950"))


@triton.jit
def scoped_add(x, y, z, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with pl.scope("kernel"):
        with pl.scope("load"):
            left = tl.load(x + offsets, offsets < n, other=0)
            right = tl.load(y + offsets, offsets < n, other=0)
        with pl.scope("compute"):
            result = left + right
        with pl.scope("store"):
            tl.store(z + offsets, result, offsets < n)


@pytest.fixture(autouse=True)
def instrumentation_semantic(monkeypatch):
    monkeypatch.setattr(pl, "_SEMANTICS", pl._SEMANTICS.copy())
    pl.enable_semantic("triton")
    monkeypatch.delenv("PROTON_INSTRUMENTATION_WARMUP_LAUNCHES", raising=False)


@pytest.mark.parametrize("n", [256, 4096])
def test_instrumentation(tmp_path, n):
    x = torch.randn(n, device="npu")
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    grid = (16, )
    block = triton.cdiv(n, grid[0])
    # Keep one group per logical program on software buffers. Older compilers do not batch.
    options = {"block_batching_factor": 1} if hasattr(NPUOptions, "block_batching_factor") else {}

    def launch():
        kernel = scoped_add[grid](x, y, out, n, BLOCK=block, **options)
        torch.npu.synchronize()
        torch.testing.assert_close(out, x + y)
        return kernel

    before = launch()
    assert (before.metadata.profile_scratch_size, before.metadata.profile_scratch_align) == (0, 1)
    assert "proton.record" not in before.asm["ttir"]
    base = tmp_path / "instrumentation"
    mixed = _is_a5()
    if mixed:
        plain_out = torch.empty_like(x[:256])
    session = proton.start(str(base), backend="instrumentation", data="trace")
    try:
        if mixed:
            plain_add[(1, )](x, y, plain_out, BLOCK=256)
        for _ in range(2):
            compiled = launch()
        if mixed:
            plain_add[(1, )](plain_out, y, plain_out, BLOCK=256)
            torch.npu.synchronize()
    finally:
        proton.finalize(session, output_format="chrome_trace")
    if mixed:
        torch.testing.assert_close(plain_out, x[:256] + y[:256] + y[:256])
    after = launch()
    assert (after.metadata.profile_scratch_size, after.metadata.profile_scratch_align) == (0, 1)
    assert after.asm["ttir"] == before.asm["ttir"]

    trace = json.loads(base.with_suffix(".chrome_trace").read_text())
    events = [e for e in (trace if isinstance(trace, list) else trace["traceEvents"]) if e["ph"] in ("X", "i")]
    counts = Counter(e["name"] for e in events)
    assert counts.keys() == {"kernel", "load", "compute", "store"}
    assert len(set(counts.values())) == 1
    assert all(math.isfinite(e["ts"]) for e in events)

    # Hardware traces cover sampled channels; only software buffers promise one group per program.
    if getattr(compiled.metadata, "proton_collection", None) != "perf_monitor":
        assert compiled.metadata.profile_scratch_size > 0 and compiled.metadata.profile_scratch_align == 64
        assert compiled.metadata.profile_num_units == 1
        assert counts["kernel"] == 2 * grid[0]
        assert all(e["ph"] == "X" and math.isfinite(e["dur"]) and e["dur"] >= 0 for e in events)
        return
    assert (compiled.metadata.profile_scratch_size, compiled.metadata.profile_scratch_align) == (0, 1)
    for e in events:
        args = e["args"]
        assert args["coverage"] == "sampled_hardware_channels"
        assert args["all_logical_blocks"] is False and args["channel_id_is_core_id"] is False
        freq = args["Frequency (MHz)"]
        cycles = args["end_cycle"] - args["start_cycle"]
        assert cycles > 0 and math.isfinite(freq) and freq >= 0
        if freq == 0:
            assert e["ph"] == "i" and e["ts"] == 0 and "dur" not in e
        else:
            assert e["ph"] == "X" and e["dur"] == pytest.approx(cycles / freq)
    for parent in (e for e in events if e["name"] == "kernel"):
        children = [
            e["name"] for e in events if e["pid"] == parent["pid"] and e["name"] != "kernel" and e["args"]
            ["start_cycle"] >= parent["args"]["start_cycle"] and e["args"]["end_cycle"] <= parent["args"]["end_cycle"]
        ]
        assert sorted(children) == ["compute", "load", "store"]


@triton.jit
def plain_add(x, y, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(out + offsets, tl.load(x + offsets) + tl.load(y + offsets))


@triton.jit
def restart_add(x, y, out, BLOCK: tl.constexpr):
    with pl.scope("iks_add"):
        plain_add(x, y, out, BLOCK)


@pytest.mark.parametrize(("data", "suffix"), [("tree", ".hatchet"), ("trace", ".chrome_trace")])
def test_kernel_without_records(tmp_path, data, suffix):
    x = torch.arange(256, device="npu", dtype=torch.float32)
    out = torch.full_like(x, float("nan"))
    base = tmp_path / "empty"
    session = proton.start(str(base), backend="instrumentation", data=data)
    try:
        compiled = plain_add[(1, )](x, x, out, BLOCK=256)
        torch.npu.synchronize()
    finally:
        proton.finalize(session)
    torch.testing.assert_close(out, x + x)
    if _is_a5():
        assert compiled.metadata.proton_collection == "none"
        assert not hasattr(compiled.metadata, "proton_metadata")
    path = base.with_suffix(suffix)
    if data == "trace":
        assert path.read_bytes() == b""
    else:
        root = json.loads(path.read_text())[0]
        assert root["frame"]["name"] == "ROOT" and root["children"] == []


def _scopes(node, name):
    if node["frame"]["name"] == name:
        yield node
    for child in node["children"]:
        yield from _scopes(child, name)


def test_cached_kernel_hook_routing(tmp_path, monkeypatch):
    if not _is_a5():
        pytest.skip("Requires the A5 performance monitor hook")
    from triton.profiler import ascend_perf_monitor

    collector = Mock(active=False)
    collector.stop.return_value = [(12, 0, 0, 100, 117, 0)]
    monkeypatch.setattr(ascend_perf_monitor, "PerfMonitorCollector", lambda: collector)
    x = torch.arange(256, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)
    handles = []
    for i in range(2):
        base = tmp_path / f"cached_{i}"
        session = proton.start(str(base), backend="instrumentation")
        try:
            handles.append(restart_add[(1, )](x, x, out, BLOCK=256, block_batching_factor=1).function)
            torch.npu.synchronize()
        finally:
            proton.finalize(session)
        kernel, = json.loads(base.with_suffix(".hatchet").read_text())[0]["children"]
        scope, = kernel["children"]
        assert scope["frame"]["name"] == "iks_add"
        assert scope["metrics"]["cycles"] == scope["metrics"]["normalized_cycles"] == 17
    assert handles[0] == handles[1]
    assert collector.start.call_count == collector.stop.call_count == 2
    torch.testing.assert_close(out, x + x)


def test_restart_with_loaded_kernel(tmp_path):
    x = torch.arange(256, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)
    handles = []
    for label, launches in (("warm", 1), ("single", 1), ("aggregate", 3)):
        out.fill_(float("nan"))
        base = tmp_path / label
        session = proton.start(str(base), backend="instrumentation", data="tree")
        try:
            for _ in range(launches):
                compiled = restart_add[(1, )](x, x, out, BLOCK=256)
                handles.append(compiled.function)
            torch.npu.synchronize()
        finally:
            proton.finalize(session)
        torch.testing.assert_close(out, x + x)
        root = json.loads(base.with_suffix(".hatchet").read_text())[0]
        scopes = list(_scopes(root, "iks_add"))
        assert scopes, f"{label}: no iks_add scope in {root}"
        for scope in scopes:
            for key in ("cycles", "normalized_cycles"):
                value = scope["metrics"][key]
                assert math.isfinite(value) and value > 0, (label, key, value)
    assert handles[0] and len(set(handles)) == 1


def test_partial_clock_trace(tmp_path, monkeypatch):
    if not _is_a5():
        pytest.skip("Requires the A5 performance monitor hook")
    from triton.profiler import ascend_perf_monitor as pm

    def info(device, module, kind, out):
        if module == 4:
            return 8
        out._obj.value = 1650
        return 0

    driver = object.__new__(pm._Driver)
    driver.info = info
    with pytest.warns(RuntimeWarning, match="module 4"):
        clocks = driver.clocks(0)
    assert clocks == [0, 1650]
    collector = Mock(active=False)
    collector.stop.return_value = [(11, 0, 0, 100, 133, clocks[0]), (12, 0, 0, 400, 433, clocks[1])]
    monkeypatch.setattr(pm, "PerfMonitorCollector", lambda: collector)
    x = torch.arange(256, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)
    base = tmp_path / "partial_clock"
    session = proton.start(str(base), backend="instrumentation", data="trace")
    try:
        restart_add[(1, )](x, x, out, BLOCK=256, block_batching_factor=1)
        torch.npu.synchronize()
    finally:
        proton.finalize(session)
    events = json.loads(base.with_suffix(".chrome_trace").read_text())["traceEvents"]
    markers = {e["args"]["channel_id"]: e for e in events if e.get("name") == "iks_add"}
    unknown, known = markers[11], markers[12]
    assert unknown["args"]["Frequency (MHz)"] == 0 and unknown["ph"] == "i" and "dur" not in unknown
    assert unknown["args"]["end_cycle"] - unknown["args"]["start_cycle"] == 33
    assert known["ph"] == "X" and known["dur"] == pytest.approx(33 / 1650)
