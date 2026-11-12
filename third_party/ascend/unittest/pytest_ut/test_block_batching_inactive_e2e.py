# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
import torch
import torch_npu  # noqa: F401
import triton
from test_block_batching_memory_e2e import MemorySource, block_batching_memory
from triton.compiler import ASTSource
from block_batching_inactive_cases import KINDS, inactive_ir


class InactiveSource(MemorySource):

    def __init__(self, kind):
        ASTSource.__init__(self, block_batching_memory, {'x': '*fp32', 'y': '*fp32', 'n': 'i32', 'k': 'i32'})
        self.text = inactive_ir(kind)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('factor', [3, 6])
def test_fp_dead_lanes_and_region_lift_reuse(kind, factor):
    kernel = triton.compile(
        InactiveSource(kind), options={
            'block_batching_factor': factor, 'block_batching_axis': 0, 'enable_graph_optimize': False, 'debug': True
        })
    assert kernel.candidate is not None, kernel.block_batching
    for blocks in (factor, factor + 1):
        n = blocks * 4 - 1
        cpu = torch.linspace(.25, 2., n)
        x = cpu.to('npu')
        p = torch.arange(n) // 4
        for k in (-1, 1) if kind == 'sibling_lifts' else (1, ):
            baseline, actual = (torch.empty(n, device='npu') for _ in range(2))
            kernel.baseline[(blocks, 1, 1)](x, baseline, n, k)
            kernel[(blocks, 1, 1)](x, actual, n, k)
            torch.npu.synchronize()
            assert kernel.last_launch['runtime_status'] == 'APPLIED', kernel.last_launch
            if kind == 'div_nan_tail': expected = torch.ones_like(cpu)
            elif kind == 'log_to_int': expected = torch.log((blocks - p) * cpu).to(torch.int32).float()
            elif kind == 'exp_to_int': expected = torch.exp((p - blocks + 1) * 1000 * cpu).to(torch.int32).float()
            else: expected = cpu + (2 if k > 0 else 0)
            torch.testing.assert_close(baseline.cpu(), expected, rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)
