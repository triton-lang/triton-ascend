import json
import math

import pytest
import torch

pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("Ascend NPU is required", allow_module_level=True)

import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl

if triton.runtime.driver.active.get_current_target().arch != "Ascend910_9382":
    pytest.skip("This regression is validated on A3 only", allow_module_level=True)

pl.enable_semantic("triton")


@triton.jit
def add_kernel(x, y, z, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with pl.scope("kernel"):
        with pl.scope("load"):
            left = tl.load(x + offsets, offsets < n, other=0)
            right = tl.load(y + offsets, offsets < n, other=0)
        with pl.scope("compute"):
            result = left + right
        with pl.scope("store"):
            tl.store(z + offsets, result, offsets < n)


@pytest.mark.parametrize("enabled", [False, True])
def test_ascend_instrumentation(enabled, tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    output = tmp_path / "profile"
    x = torch.randn(4096, device="npu")
    y = torch.randn_like(x)
    z = torch.empty_like(x)

    session = None
    try:
        if enabled:
            session = proton.start(str(output), data="trace", context="shadow", backend="instrumentation")
        for _ in range(2):
            compiled = add_kernel[(16, )](x, y, z, 4096, BLOCK=256)
            torch.npu.synchronize()
        torch.testing.assert_close(z, x + y)
        scratch_size = getattr(compiled.metadata, "profile_scratch_size", 0)
        if enabled:
            assert scratch_size > 0
            assert compiled.metadata.profile_num_units == 1
        else:
            assert scratch_size == 0
            assert "proton.record" not in compiled.asm["ttir"]
    finally:
        if session is not None:
            proton.finalize(output_format="chrome_trace")

    trace = output.with_suffix(".chrome_trace")
    if not enabled:
        assert not trace.exists()
        return

    data = json.loads(trace.read_text())
    events = data if isinstance(data, list) else data["traceEvents"]
    complete = [event for event in events if event.get("ph") == "X"]
    assert len(complete) == 128
    assert {"kernel", "load", "compute", "store"} <= {event["name"] for event in complete}
    for event in complete:
        assert math.isfinite(float(event["ts"]))
        assert math.isfinite(float(event["dur"]))
        assert float(event["dur"]) >= 0
