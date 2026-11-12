# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
import torch
import torch_npu  # noqa: F401
import triton
from test_block_batching_memory_e2e import MemorySource, block_batching_memory
from triton.compiler import ASTSource
from block_batching_control_flow_cases import CONTROL_KINDS, control_ir, control_reference


class ControlSource(MemorySource):

    def __init__(self, kind):
        ASTSource.__init__(self, block_batching_memory, {"x": "*fp32", "y": "*fp32", "n": "i32", "k": "i32"})
        self.text = control_ir(kind).replace("@block_batching_control", "@block_batching_memory")


@pytest.mark.parametrize("kind", CONTROL_KINDS)
@pytest.mark.parametrize("factor", [3, 6])
def test_uniform_regions_execute_original_branch_and_iteration_sequence(kind, factor):
    kernel = triton.compile(
        ControlSource(kind), options={
            "block_batching_factor": factor, "block_batching_axis": 0, "enable_graph_optimize": False, "debug": True
        })
    assert kernel.candidate is not None, kernel.block_batching
    for blocks in (factor, factor + 1):
        cases = (1, ) if kind == "if_untaken" else (0, 1, 4, -1)
        for k in cases:
            x_cpu = torch.arange(blocks * 8, dtype=torch.float32) / 4 + (1 if k > 0 else -1)
            x = x_cpu.to("npu")
            baseline, result = (torch.full((blocks, ), -37., device="npu") for _ in range(2))
            kernel.baseline[(blocks, 1, 1)](x, baseline, blocks, k)
            kernel[(blocks, 1, 1)](x, result, blocks, k)
            torch.npu.synchronize()
            assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.last_launch
            expected = control_reference(kind, x_cpu, blocks, k)
            torch.testing.assert_close(baseline.cpu(), expected)
            torch.testing.assert_close(result.cpu(), expected)
