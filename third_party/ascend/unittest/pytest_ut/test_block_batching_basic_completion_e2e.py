# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
import torch
import torch_npu  # noqa: F401
import triton
from block_batching_op_cases import BASIC_COMPLETION_OPS, case_ir, reference
from test_block_batching_elementwise_e2e import OpSource


@pytest.mark.parametrize("op", BASIC_COMPLETION_OPS)
@pytest.mark.parametrize("factor", [3, 6])
def test_remaining_basic_input_ops_have_observable_device_results(tmp_path, op, factor):
    source = OpSource(case_ir(op), tmp_path / "device.ttir")
    kernel = triton.compile(
        source, options={
            "block_batching_factor": factor, "block_batching_axis": 0, "debug": True, "enable_graph_optimize": False
        })
    assert kernel.candidate is not None, kernel.block_batching
    assert kernel.block_batching["lane_width"] == factor
    assert kernel.block_batching["operations"].get(op, 0), "op disappeared before batching"
    for blocks in (factor, factor + 1):
        n = blocks * 4 - 1
        x_cpu = torch.linspace(.125, 3.125, n, dtype=torch.float32)
        x = x_cpu.to("npu")
        result, baseline = torch.empty_like(x), torch.empty_like(x)
        kernel.baseline[(blocks, 1, 1)](x, baseline, n)
        kernel[(blocks, 1, 1)](x, result, n)
        torch.npu.synchronize()
        assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.last_launch
        assert kernel.last_launch["applied_factor"] == factor
        expected = reference(op, x_cpu, blocks)
        torch.testing.assert_close(baseline.cpu(), expected, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(result.cpu(), expected, rtol=2e-4, atol=2e-5)
