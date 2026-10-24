# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _gather_scatter(src, indices, gathered, scattered, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    index = tl.load(indices + lane).to(tl.int64)
    # Exercise signed deltas in the migrated i64 arithmetic subset. Narrow
    # tensor arithmetic remains a local boundary in the first consumer.
    ptr = (src + 64) + (index - 64)
    value = tl.load(ptr, lane < N, other=-7)
    tl.store(gathered + lane, value)
    tl.store((scattered + 64) + (index - 64), value, lane < N)


@triton.jit
def _indirect_atomic(indices, output, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    index = tl.load(indices + lane).to(tl.int64)
    tl.atomic_add((output + 8) + (index - 8), lane + 1, lane < N)


@pytest.mark.parametrize("n", [53, 64])
def test_public_pointer_gather_scatter(n):
    block = 64
    indices_cpu = (torch.arange(block, dtype=torch.int32) * 37) % 128
    src_cpu = torch.arange(128, dtype=torch.int32) * 13 - 21
    indices = indices_cpu.npu()
    src = src_cpu.npu()
    gathered = torch.empty(block, dtype=torch.int32, device="npu")
    scattered = torch.full((128, ), -999, dtype=torch.int32, device="npu")
    _gather_scatter[(1, )](src, indices, gathered, scattered, n, block)
    torch.npu.synchronize()

    expected_gather = torch.full((block, ), -7, dtype=torch.int32)
    expected_gather[:n] = src_cpu[indices_cpu[:n].long()]
    expected_scatter = torch.full((128, ), -999, dtype=torch.int32)
    expected_scatter[indices_cpu[:n].long()] = expected_gather[:n]
    assert torch.equal(gathered.cpu(), expected_gather)
    assert torch.equal(scattered.cpu(), expected_scatter)


@pytest.mark.parametrize("n", [53, 64])
def test_public_pointer_indirect_atomic(n):
    block = 64
    indices_cpu = (torch.arange(block, dtype=torch.int32) * 7) % 17
    indices = indices_cpu.npu()
    output = torch.zeros(32, dtype=torch.int32, device="npu")
    _indirect_atomic[(1, )](indices, output, n, block)
    torch.npu.synchronize()

    expected = torch.zeros(32, dtype=torch.int32)
    expected.index_add_(0, indices_cpu[:n].long(), torch.arange(1, n + 1, dtype=torch.int32))
    assert torch.equal(output.cpu(), expected)
