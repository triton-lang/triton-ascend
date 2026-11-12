# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import hashlib
import pytest
import torch
import torch_npu  # noqa: F401
import triton
from triton._C.libtriton import ir
from triton.compiler import ASTSource
from triton.runtime.cache import get_cache_manager
from block_batching_memory_cases import KINDS, memory_ir, memory_signature


@triton.jit
def block_batching_memory(x, y, n, k):
    pass


class MemorySource(ASTSource):

    def __init__(self, kind):
        super().__init__(block_batching_memory, memory_signature(kind))
        self.text = memory_ir(kind)

    def hash(self):
        return hashlib.sha256((super().hash() + self.text).encode()).hexdigest()

    def make_ir(self, target, options, codegen_fns, module_map, context):
        path = get_cache_manager(self.hash()).put(self.text, "memory.fixture.ttir", binary=False)
        result = ir.parse_mlir_module(path, context)
        result.context = context
        return result


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("factor", [3, 6])
def test_memory_models_are_applied_without_source_or_abi_changes(kind, factor):
    source = MemorySource(kind)
    kernel = triton.compile(
        source, options={
            "block_batching_factor": factor, "block_batching_axis": 0, "enable_graph_optimize": False, "debug": True
        })
    assert kernel.candidate is not None, kernel.block_batching
    for blocks in (factor, factor + 1):
        if kind == "nd_reused_range":
            n, k = blocks * 4 - 1, 3
            x_cpu = torch.arange(n * k, dtype=torch.float32) / 16
            expected, grid = x_cpu + 1, (blocks, 1, 1)
        elif kind == "masked_scalar_like":
            n, k = blocks * 4, 0
            x_cpu = torch.zeros(1, dtype=torch.float32)
            expected, grid = torch.full((n, ), 7.0), (blocks, 1, 1)
        elif kind == "ptr_bitcast":
            n, k = blocks, 0
            x_cpu = torch.arange(n, dtype=torch.float32) * .25 - n * .125
            expected, grid = x_cpu.view(torch.int32), (blocks, 1, 1)
        elif kind == "integer64":
            n, k = blocks, 0
            x_cpu = torch.arange(n, dtype=torch.int64) * (1 << 35) - (1 << 40)
            expected, grid = x_cpu, (blocks, 1, 1)
        elif kind == "ptr_roundtrip":
            n, k = blocks, 0
            x_cpu = torch.arange(n, dtype=torch.float32) / 4
            expected, grid = x_cpu + 1, (blocks, 1, 1)
        else:
            n, k = blocks * 4 - 1, -3
            x_cpu = torch.arange(n, dtype=torch.float32) / 4
            integer = k % (1 << 32) if kind == "index_castui" else k
            expected, grid = x_cpu + torch.tensor(integer + 1, dtype=torch.float32), (blocks, 1, 1)
        x = x_cpu.to("npu")
        result = torch.empty_like(expected, device="npu")
        baseline = torch.empty_like(result)
        kernel.baseline[grid](x, baseline, n, k)
        kernel[grid](x, result, n, k)
        torch.npu.synchronize()
        assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.last_launch
        assert kernel.block_batching["guard"]["version"] == 3
        if expected.dtype in (torch.int32, torch.int64):
            assert torch.equal(result.cpu(), expected)
            assert torch.equal(baseline.cpu(), expected)
        else:
            torch.testing.assert_close(baseline.cpu(), expected, rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(result.cpu(), expected, rtol=2e-4, atol=2e-5)
