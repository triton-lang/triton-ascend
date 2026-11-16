"""Integration coverage for MSPTI and instrumentation in an Ascend-only build."""
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


def launch_metadata(grid, metadata, args):
    return {"name": "named_add", "flops32": args["n"], "bytes": args["n"] * 12}


@triton.jit(launch_metadata=launch_metadata)
def metadata_add(x, y, z, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(z + offsets,
             tl.load(x + offsets, offsets < n, other=0) + tl.load(y + offsets, offsets < n, other=0), offsets < n)


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


def walk(node):
    yield node
    for child in node.get("children", []):
        yield from walk(child)


@pytest.mark.parametrize("hook", [None, "triton"])
@pytest.mark.parametrize("limit_cores", [False, True])
def test_mspti_metadata_and_activation(tmp_path, monkeypatch, hook, limit_cores):
    n = 1024
    x = torch.arange(n, device="npu", dtype=torch.float32)
    y = torch.ones_like(x)
    z = torch.empty_like(x)

    def launch():
        metadata_add[(4, )](x, y, z, n, BLOCK=256)
        torch.npu.synchronize()

    launch()
    utils = triton.runtime.driver.active.utils
    hardware_aic, hardware_aiv = utils.get_device_core()
    divisor = math.gcd(hardware_aic, hardware_aiv)
    if limit_cores:
        monkeypatch.setenv("NPU_DEVICE_LIMIT", f"{hardware_aic // divisor},{hardware_aiv // divisor}")
    else:
        monkeypatch.delenv("NPU_DEVICE_LIMIT", raising=False)
    expected_aiv = hardware_aiv // divisor if limit_cores else hardware_aiv
    base = tmp_path / "mspti"
    # Also exercises default NPU -> MSPTI backend selection.
    session = proton.start(str(base), hook=hook)
    try:
        with proton.scope("before"):
            launch()
        proton.deactivate(session)
        launch()
        proton.activate(session)
        with proton.scope("after"):
            launch()
            launch()
    finally:
        proton.finalize(session)
    torch.testing.assert_close(z, x + y)
    profile = json.loads(base.with_suffix(".hatchet").read_text())
    scopes = {child["frame"]["name"]: child for child in profile[0]["children"]}
    assert scopes.keys() == {"before", "after"}
    for scope, count in (("before", 1), ("after", 2)):
        kernels = [node for node in walk(scopes[scope]) if node.get("metrics", {}).get("count", 0)]
        assert len(kernels) == 1
        kernel = kernels[0]
        metrics = kernel["metrics"]
        assert metrics["count"] == count
        assert metrics["time (ns)"] > 0
        assert metrics["device_type"] == "NPU"
        assert str(metrics["device_id"]) in profile[1]["NPU"]
        device_info = profile[1]["NPU"][str(metrics["device_id"])]
        assert device_info["num_sms"] == expected_aiv
        assert device_info["arch"] == utils.get_arch()
        logical_device = int(metrics["device_id"])
        assert device_info["clock_rate"] == utils.get_vector_clock_rate(logical_device)
        assert device_info["memory_clock_rate"] == utils.get_hbm_clock_rate(logical_device)
        if hook == "triton":
            assert kernel["frame"]["name"] == "named_add"
            assert metrics["flops32"] == n * count
            assert metrics["bytes"] == n * 12 * count
        else:
            assert kernel["frame"]["name"] == "metadata_add"
            assert "flops32" not in metrics


def test_mspti_unavailable_device_clocks(tmp_path, monkeypatch):
    n = 1024
    x = torch.arange(n, device="npu", dtype=torch.float32)
    y = torch.ones_like(x)
    z = torch.empty_like(x)
    metadata_add[(4, )](x, y, z, n, BLOCK=256)
    torch.npu.synchronize()
    utils = triton.runtime.driver.active.utils
    native = utils._load_mod()
    monkeypatch.setattr(utils, "get_vector_clock_rate", lambda device: native.get_vector_clock_rate(999999))
    monkeypatch.setattr(utils, "get_hbm_clock_rate", lambda device: native.get_hbm_clock_rate(999999))
    base = tmp_path / "unavailable_clocks"
    session = proton.start(str(base), backend="mspti", hook="triton")
    try:
        metadata_add[(4, )](x, y, z, n, BLOCK=256)
        torch.npu.synchronize()
    finally:
        with pytest.warns(RuntimeWarning, match="clock rate is unavailable"):
            proton.finalize(session)
    profile = json.loads(base.with_suffix(".hatchet").read_text())
    kernels = [node for node in walk(profile[0]) if node.get("metrics", {}).get("count", 0)]
    assert len(kernels) == 1
    assert kernels[0]["metrics"]["count"] == 1
    assert kernels[0]["metrics"]["time (ns)"] > 0
    info = profile[1]["NPU"][str(kernels[0]["metrics"]["device_id"])]
    assert info["clock_rate"] == info["memory_clock_rate"] == 0


@pytest.mark.parametrize("n", [256, 4096])
def test_instrumentation_off_on_off(tmp_path, n):
    pl.enable_semantic("triton")
    x = torch.randn(n, device="npu")
    y = torch.randn_like(x)
    z = torch.empty_like(x)
    grid = (triton.cdiv(n, 256), )

    def launch():
        kernel = scoped_add[grid](x, y, z, n, BLOCK=256)
        torch.npu.synchronize()
        torch.testing.assert_close(z, x + y)
        return kernel

    before = launch()
    assert before.metadata.profile_scratch_size == 0
    assert "proton.record" not in before.asm["ttir"]
    base = tmp_path / "instrumentation"
    session = proton.start(str(base), backend="instrumentation", data="trace")
    try:
        for _ in range(2):
            compiled = launch()
        assert compiled.metadata.profile_scratch_size > 0
        assert compiled.metadata.profile_num_units == 1
    finally:
        proton.finalize(session, output_format="chrome_trace")
    after = launch()
    assert after.metadata.profile_scratch_size == 0
    assert after.asm["ttir"] == before.asm["ttir"]
    trace = json.loads(base.with_suffix(".chrome_trace").read_text())
    events = trace if isinstance(trace, list) else trace["traceEvents"]
    complete = [event for event in events if event.get("ph") == "X"]
    assert len(complete) == 2 * grid[0] * 4
    assert {event["name"] for event in complete} == {"kernel", "load", "compute", "store"}
    for event in complete:
        assert math.isfinite(float(event["ts"]))
        assert math.isfinite(float(event["dur"]))
        assert float(event["dur"]) >= 0
