# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
import torch
import torch_npu  # noqa: F401
import triton
from block_batching_reduction_cases import REDUCTION_KINDS, reduction_ir, reduction_reference
from test_block_batching_elementwise_e2e import OpSource


@pytest.mark.parametrize("kind",
                         [kind for kind in REDUCTION_KINDS if kind not in ("maxnum_capture", "argmax", "argmax_index")])
@pytest.mark.parametrize("factor", [3, 6])
def test_reduction_and_scan_lanes_have_independent_results(tmp_path, kind, factor):
    kernel = triton.compile(
        OpSource(reduction_ir(kind), tmp_path / "reduction-device.ttir"), options={
            "block_batching_factor": factor, "block_batching_axis": 0, "debug": True, "enable_graph_optimize": False
        })
    assert kernel.candidate is not None, kernel.block_batching
    operation = "tt.scan" if "scan" in kind else "tt.reduce"
    assert kernel.block_batching["operations"].get(operation, 0)
    for blocks in (factor, factor + 1):
        n = blocks * 4 - 1
        x_cpu = (torch.arange(blocks).float()[:, None] * 16 + torch.tensor([1., 7., 7., 0.])).reshape(-1)[:n]
        x = x_cpu.to("npu")
        baseline, actual = torch.empty_like(x), torch.empty_like(x)
        kernel.baseline[(blocks, 1, 1)](x, baseline, n)
        kernel[(blocks, 1, 1)](x, actual, n)
        torch.npu.synchronize()
        assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.last_launch
        expected = reduction_reference(kind, x_cpu, blocks)
        torch.testing.assert_close(baseline.cpu(), expected, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("kind", ["max"])
def test_nan_and_first_index_ties_have_declared_semantics(tmp_path, kind):
    kernel = triton.compile(
        OpSource(reduction_ir(kind), tmp_path / "nan-tie.ttir"),
        options={"block_batching_factor": 3, "block_batching_axis": 0, "debug": True, "enable_graph_optimize": False})
    assert kernel.candidate is not None, kernel.block_batching
    x_cpu = torch.tensor([[float("nan"), 7., 7., float("nan")], [7., float("nan"), 7., 0.], [float("nan")] * 4,
                          [7., 7., 0., 0.]]).reshape(-1)
    x = x_cpu.to("npu")
    baseline, actual = torch.empty_like(x), torch.empty_like(x)
    kernel.baseline[(4, 1, 1)](x, baseline, x.numel())
    kernel[(4, 1, 1)](x, actual, x.numel())
    torch.npu.synchronize()
    assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.last_launch
    expected = reduction_reference(kind, x_cpu, 4)
    torch.testing.assert_close(baseline.cpu(), expected, equal_nan=True)
    torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)


@pytest.mark.parametrize("kind", ["argmax", "argmax_index"])
def test_nan_sensitive_argmax_uses_the_complete_standard_baseline(tmp_path, kind):
    kernel = triton.compile(
        OpSource(reduction_ir(kind), tmp_path / "legacy-argmax.ttir"),
        options={"block_batching_factor": 3, "block_batching_axis": 0, "enable_graph_optimize": False})
    assert kernel.candidate is None, kernel.block_batching
    for special in (False, True):
        x_cpu = torch.tensor([[1., 7., 7., 0.], [7., 1., 7., 0.], [1., 2., 3., 4.], [7., 7., 0., 0.]])
        if special:
            x_cpu[0, 0] = float("nan")
            x_cpu[2, :] = float("nan")
        x_cpu = x_cpu.reshape(-1)
        x = x_cpu.to("npu")
        baseline, actual = torch.empty_like(x), torch.empty_like(x)
        kernel.baseline[(4, 1, 1)](x, baseline, x.numel())
        kernel[(4, 1, 1)](x, actual, x.numel())
        torch.npu.synchronize()
        assert kernel.last_launch["runtime_status"] == "BASELINE_SELECTED", kernel.last_launch
        assert kernel.last_launch["reason"] == "UNSUPPORTED_NAN_REDUCTION_COMBINER"
        torch.testing.assert_close(actual.cpu(), baseline.cpu(), equal_nan=True)
        # The user explicitly leaves the original backend's NaN/tie issue out
        # of scope. Check CPU semantics on ordinary data; on NaNs check only
        # that fallback faithfully executes the unchanged standard baseline.
        if not special:
            torch.testing.assert_close(actual.cpu(), reduction_reference(kind, x_cpu, 4))


def test_number_max_standard_special_values(tmp_path, record_property):
    import json
    kernel = triton.compile(
        OpSource(reduction_ir("maxnum"), tmp_path / "standard-maxnum.ttir"),
        options={"block_batching_factor": 3, "block_batching_axis": 0, "enable_graph_optimize": False})
    x_cpu = torch.tensor([[float("nan"), 7., 7., float("nan")], [7., float("nan"), 7., 0.], [float("nan")] * 4,
                          [7., 7., 0., 0.]]).reshape(-1)
    x = x_cpu.to("npu")
    baseline, actual = torch.empty_like(x), torch.empty_like(x)
    kernel.baseline[(4, 1, 1)](x, baseline, x.numel())
    kernel[(4, 1, 1)](x, actual, x.numel())
    torch.npu.synchronize()
    assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.last_launch
    baseline_cpu, actual_cpu = baseline.cpu(), actual.cpu()
    record_property("baseline_rows", json.dumps(baseline_cpu.reshape(4, 4)[:, 0].tolist()))
    record_property("candidate_rows", json.dumps(actual_cpu.reshape(4, 4)[:, 0].tolist()))
    expected = reduction_reference("maxnum", x_cpu, 4)
    supported = ~torch.isnan(expected)
    torch.testing.assert_close(baseline_cpu[supported], expected[supported])
    torch.testing.assert_close(actual_cpu[supported], expected[supported])
    # This test promises equivalence to the restored standard backend, not
    # a fix for its all-NaN seed behavior. Keep that reference failure below.
    torch.testing.assert_close(actual_cpu, baseline_cpu, equal_nan=True)


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="Known standard-backend maxnum all-NaN seed behavior; no global correctness compensation in block batching")
def test_number_max_all_nan_reference(tmp_path, record_property):
    import json
    baseline = triton.compile(OpSource(reduction_ir("maxnum"), tmp_path / "standard-maxnum-all-nan.ttir"),
                              options={"block_batching_factor": 1, "enable_graph_optimize": False})
    x = torch.full((16, ), float("nan"), device="npu")
    out = torch.empty_like(x)
    baseline[(4, 1, 1)](x, out, x.numel())
    torch.npu.synchronize()
    host = out.cpu()
    record_property("standard_baseline_rows", json.dumps(host.reshape(4, 4)[:, 0].tolist()))
    assert torch.isnan(host).all().item()
