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

arch = triton.runtime.driver.active.get_current_target().arch
if not isinstance(arch, str) or not arch.startswith(("Ascend910_95", "Ascend950")):
    pytest.skip("A5 hardware instrumentation is required", allow_module_level=True)

pl.enable_semantic("triton")


@triton.jit
def scoped_add(x, y, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with pl.scope("kernel"):
        with pl.scope("load"):
            left = tl.load(x + offsets, offsets < n, other=0)
            right = tl.load(y + offsets, offsets < n, other=0)
        with pl.scope("compute"):
            result = left + right
        with pl.scope("store"):
            tl.store(out + offsets, result, offsets < n)


def test_ascend_instrumentation_a5(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    output = tmp_path / "a5_instrumentation"
    x = torch.randn(4096, device="npu")
    y = torch.randn_like(x)
    out = torch.empty_like(x)

    scoped_add[(16, )](x, y, out, 4096, BLOCK=256)
    torch.npu.synchronize()

    session = proton.start(str(output), backend="instrumentation", data="trace")
    try:
        compiled = scoped_add[(16, )](x, y, out, 4096, BLOCK=256)
        torch.npu.synchronize()
        assert compiled.metadata.proton_collection == "mspti"
        assert compiled.metadata.profile_scratch_size == 0
    finally:
        proton.finalize(session, output_format="chrome_trace")

    torch.testing.assert_close(out, x + y)
    trace = json.loads(output.with_suffix(".chrome_trace").read_text())
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    assert events, "A5 instrumentation produced an empty trace"
    assert {event["name"] for event in events} == {"kernel", "load", "compute", "store"}

    # Hardware sampling does not guarantee one record per Triton program/block.
    for event in events:
        assert math.isfinite(event["ts"])
        assert math.isfinite(event["dur"]) and event["dur"] > 0
        args = event["args"]
        assert args["coverage"] == "sampled_hardware_channels"
        assert args["all_logical_blocks"] is False
        assert args["channel_id_is_core_id"] is False
        frequency = args["Frequency (MHz)"]
        assert math.isfinite(frequency) and frequency > 0
        assert event["dur"] == pytest.approx((args["end_cycle"] - args["start_cycle"]) / frequency)

    for pid in {event["pid"] for event in events}:
        local = [event for event in events if event["pid"] == pid]
        parents = [event for event in local if event["name"] == "kernel"]
        assert parents
        for name in ("load", "compute", "store"):
            children = [event for event in local if event["name"] == name]
            assert len(children) == len(parents)
            for parent in parents:
                contained = [
                    child for child in children if child["ts"] >= parent["ts"] - 1e-9 and child["ts"] +
                    child["dur"] <= parent["ts"] + parent["dur"] + 1e-9
                ]
                assert len(contained) == 1, (pid, name, parent, children)
