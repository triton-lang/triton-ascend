# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Standard-backend/device acceptance for each newly admitted elementwise op."""
import hashlib

import pytest
import torch
import torch_npu  # noqa: F401 -- register torch.npu before device fixtures run
import triton
from triton._C.libtriton import ir
from triton.compiler import ASTSource

from block_batching_op_cases import ELEMENTWISE_OPS, case_ir, reference


@triton.jit
def block_batching_elementwise(x, y, n):
    # The test AST source below supplies observable, verifier-checked TTIR.
    pass


class OpSource(ASTSource):

    def __init__(self, text, path):
        super().__init__(block_batching_elementwise, {"x": "*fp32", "y": "*fp32", "n": "i32"})
        self.text, self.path = text, path

    def hash(self):
        return hashlib.sha256((super().hash() + self.text).encode()).hexdigest()

    def make_ir(self, target, options, codegen_fns, module_map, context):
        from triton.runtime.cache import get_cache_manager
        self.path.write_text(self.text)
        # Stable source locations let both task-queue runs reuse the same
        # kernel cache instead of recompiling for each pytest temporary path.
        path = get_cache_manager(self.hash()).put(self.text, "elementwise.fixture.ttir", binary=False)
        module = ir.parse_mlir_module(path, context)
        module.context = context
        return module


@pytest.mark.parametrize("op", ELEMENTWISE_OPS)
@pytest.mark.parametrize("factor", [3, 6])
def test_elementwise_is_applied_on_complete_and_tail_groups(tmp_path, op, factor):
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
