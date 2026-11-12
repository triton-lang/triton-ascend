# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import hashlib
import pytest
import torch
import torch_npu  # noqa: F401
import triton
from triton._C.libtriton import ir
from triton.compiler import ASTSource
from triton.runtime.cache import get_cache_manager
from block_batching_byte_extern_cases import case_ir


@triton.jit
def block_batching_byte_extern(x, y, c, out, n):
    pass


class Source(ASTSource):

    def __init__(self, kind):
        super().__init__(block_batching_byte_extern,
                         dict(x="*fp32", y="*fp32", c="*i8" if kind == "byte" else "*i1", out="*fp32", n="i32"))
        self.text = case_ir(kind)

    def hash(self):
        return hashlib.sha256((super().hash() + self.text).encode()).hexdigest()

    def make_ir(self, target, options, codegen_fns, module_map, context):
        path = get_cache_manager(self.hash()).put(self.text, "byte-extern.fixture.ttir", binary=False)
        result = ir.parse_mlir_module(path, context)
        result.context = context
        return result


@pytest.mark.parametrize("kind", ["bool", "byte", "pow", "tanh"])
@pytest.mark.parametrize("factor", [3, 6])
def test_new_subsets_execute_complete_and_tail_groups(kind, factor):
    kernel = triton.compile(
        Source(kind), options=dict(block_batching_factor=factor, block_batching_axis=0, enable_graph_optimize=False,
                                   debug=True))
    assert kernel.candidate is not None, kernel.block_batching
    for blocks in (factor, factor + 1):
        n = blocks * 8 - 1
        seeds = torch.tensor([-float("inf"), -2., -0., 0., .5, 2., float("inf"), float("nan")])
        x_cpu = seeds.repeat(triton.cdiv(n, 8))[:n]
        y_cpu = torch.tensor([2., 3., 1., 2., .5, -1., 0., 2.]).repeat(triton.cdiv(n, 8))[:n]
        c_cpu = torch.tensor([-128, -1, 0, 1, 127], dtype=torch.int8).repeat(triton.cdiv(n, 5))[:n]
        if kind != "byte": c_cpu = c_cpu != 0
        expected = (torch.where(c_cpu != 0, x_cpu, y_cpu)
                    if kind in ("byte", "bool") else torch.pow(x_cpu, y_cpu) if kind == "pow" else torch.tanh(x_cpu))
        x, y, c = (v.to("npu") for v in (x_cpu, y_cpu, c_cpu))
        storage = torch.full((n + 32, ), -123., device="npu")
        out = storage[16:16 + n]
        baseline = torch.full_like(out, float("nan"))
        grid = (blocks, 1, 1)
        kernel.baseline[grid](x, y, c, baseline, n)
        kernel[grid](x, y, c, out, n)
        torch.npu.synchronize()
        assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.last_launch
        torch.testing.assert_close(out.cpu(), baseline.cpu(), rtol=2e-4, atol=2e-5, equal_nan=True)
        torch.testing.assert_close(baseline.cpu(), expected, rtol=2e-4, atol=2e-5, equal_nan=True)
        zeros = torch.isfinite(expected) & (expected == 0)
        assert torch.equal(torch.signbit(out.cpu()[zeros]), torch.signbit(baseline.cpu()[zeros]))
        host = storage.cpu()
        assert (host[:16] == -123).all() and (host[16 + n:] == -123).all()


import triton.language as tl


@triton.jit
def byte_copy_kernel(x, out, n, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    mask = offsets < n
    value = tl.load(x + offsets, mask, other=0)
    tl.store(out + offsets, value, mask)


@pytest.mark.parametrize("factor", [3, 6])
def test_byte_stores_preserve_exact_values_and_tail_sentinels(factor):
    n = (factor + 1) * 128 - 3
    x_cpu = torch.arange(n, dtype=torch.int32).to(torch.int8)
    x = x_cpu.to("npu")
    storage = torch.full((n + 32, ), -123, dtype=torch.int8, device="npu")
    out = storage[16:16 + n]
    kernel = byte_copy_kernel[(triton.cdiv(n, 128), )](x, out, n, 128, block_batching_factor=factor,
                                                       block_batching_axis=0)
    torch.npu.synchronize()
    assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.last_launch
    assert torch.equal(out.cpu(), x_cpu)
    host = storage.cpu()
    assert (host[:16] == -123).all() and (host[16 + n:] == -123).all()
