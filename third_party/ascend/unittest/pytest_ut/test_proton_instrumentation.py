"""Exercise the same instrumented vector add on software-buffer and hardware-sampled NPUs."""
import json
import math
from collections import Counter

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("triton._C.libproton")
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.backends.ascend.compiler import NPUOptions


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


@pytest.mark.parametrize("n", [256, 4096])
def test_instrumentation(tmp_path, monkeypatch, n):
    monkeypatch.setattr(pl, "_SEMANTICS", pl._SEMANTICS.copy())
    pl.enable_semantic("triton")
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
    session = proton.start(str(base), backend="instrumentation", data="trace")
    try:
        for _ in range(2):
            compiled = launch()
    finally:
        proton.finalize(session, output_format="chrome_trace")
    after = launch()
    assert (after.metadata.profile_scratch_size, after.metadata.profile_scratch_align) == (0, 1)
    assert after.asm["ttir"] == before.asm["ttir"]

    trace = json.loads(base.with_suffix(".chrome_trace").read_text())
    events = [e for e in (trace if isinstance(trace, list) else trace["traceEvents"]) if e["ph"] == "X"]
    counts = Counter(e["name"] for e in events)
    assert counts.keys() == {"kernel", "load", "compute", "store"}
    assert len(set(counts.values())) == 1
    assert all(math.isfinite(e["ts"]) and math.isfinite(e["dur"]) and e["dur"] >= 0 for e in events)

    # Hardware traces cover sampled channels; only software buffers promise one group per program.
    if getattr(compiled.metadata, "proton_collection", None) != "perf_monitor":
        assert compiled.metadata.profile_scratch_size > 0 and compiled.metadata.profile_scratch_align == 64
        assert compiled.metadata.profile_num_units == 1
        assert counts["kernel"] == 2 * grid[0]
        return
    assert (compiled.metadata.profile_scratch_size, compiled.metadata.profile_scratch_align) == (0, 1)
    for e in events:
        args = e["args"]
        assert args["coverage"] == "sampled_hardware_channels"
        assert args["all_logical_blocks"] is False and args["channel_id_is_core_id"] is False
        freq = args["Frequency (MHz)"]
        assert math.isfinite(freq) and freq > 0 and e["dur"] > 0
        assert e["dur"] == pytest.approx((args["end_cycle"] - args["start_cycle"]) / freq)
    for parent in (e for e in events if e["name"] == "kernel"):
        children = [
            e["name"]
            for e in events
            if e["pid"] == parent["pid"] and e["name"] != "kernel" and e["ts"] >= parent["ts"] - 1e-9 and e["ts"] +
            e["dur"] <= parent["ts"] + parent["dur"] + 1e-9
        ]
        assert sorted(children) == ["compute", "load", "store"]
