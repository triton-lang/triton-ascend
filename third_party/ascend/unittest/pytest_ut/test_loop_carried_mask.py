# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _loop_carried_mask_kernel(src, dst, n, steps, BLOCK: tl.constexpr, MASK_USE: tl.constexpr,
                              CARRY_INDEX: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    offsets = lanes
    src_ptrs = src + lanes
    dst_ptrs = dst + lanes
    for iteration in range(steps):
        if CARRY_INDEX:
            mask = offsets < n
        else:
            # Equivalent scalar-index control; both variants advance pointers.
            mask = iteration * BLOCK + lanes < n
        if MASK_USE == "load":
            values = tl.load(src_ptrs, mask=mask, other=-1)
            tl.store(dst_ptrs, values)
        elif MASK_USE == "store":
            values = tl.load(src_ptrs)
            tl.store(dst_ptrs, values, mask=mask)
        else:
            values = tl.load(src_ptrs)
            tl.store(dst_ptrs, tl.where(mask, values, -1))
        offsets += BLOCK
        src_ptrs += BLOCK
        dst_ptrs += BLOCK


@pytest.mark.parametrize("mask_use", ["load", "store", "select"])
@pytest.mark.parametrize("carry_index", [False, True], ids=["scalar_index", "tensor_index"])
def test_loop_carried_mask(mask_use, carry_index):
    """A full tile, a 3-lane tail and an empty tile need different masks."""
    block, n, steps = 32, 35, 3
    # Back the entire traversal, including masked-off lanes, with real storage.
    # Incorrect masks then cause deterministic value errors, not allocator OOB.
    source_cpu = torch.arange(1, steps * block + 1, dtype=torch.int32)
    output_cpu = torch.full_like(source_cpu, -2)
    source, output = source_cpu.npu(), output_cpu.npu()

    _loop_carried_mask_kernel[(1, )](
        source,
        output,
        n,
        steps,
        BLOCK=block,
        MASK_USE=mask_use,
        CARRY_INDEX=carry_index,
        enable_auto_bind_sub_block=False,
    )

    expected = source_cpu.clone()
    expected[n:] = -2 if mask_use == "store" else -1
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@triton.jit
def _masked_access(src, dst, mask, MASK_USE: tl.constexpr):
    if MASK_USE == "load":
        tl.store(dst, tl.load(src, mask=mask, other=-1))
    elif MASK_USE == "store":
        tl.store(dst, tl.load(src), mask=mask)
    elif MASK_USE == "atomic_add":
        tl.atomic_add(dst, tl.load(src), mask=mask)
    else:
        tl.store(dst, tl.where(mask, tl.load(src), -1))


@triton.jit
def _changing_predicate_kernel(src, dst, steps, BLOCK: tl.constexpr, MASK_USE: tl.constexpr, WHILE_LOOP: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < 3
    if WHILE_LOOP:
        iteration = 0
        while iteration < steps:
            offsets = iteration * BLOCK + lanes
            _masked_access(src + offsets, dst + offsets, mask, MASK_USE)
            mask = ~mask
            iteration += 1
    else:
        for iteration in range(steps):
            offsets = iteration * BLOCK + lanes
            _masked_access(src + offsets, dst + offsets, mask, MASK_USE)
            mask = ~mask


@pytest.mark.parametrize("mask_use", ["load", "store", "select", "atomic_add"])
@pytest.mark.parametrize("while_loop", [False, True], ids=["for", "while"])
def test_changing_loop_predicate(mask_use, while_loop):
    """The predicate itself changes; scalarizing pointer offsets cannot fix it."""
    block, steps = 32, 3
    source_cpu = torch.arange(1, steps * block + 1, dtype=torch.int32)
    source = source_cpu.npu()
    output = torch.full_like(source, -2)
    _changing_predicate_kernel[(1, )](
        source,
        output,
        steps,
        BLOCK=block,
        MASK_USE=mask_use,
        WHILE_LOOP=while_loop,
        enable_auto_bind_sub_block=False,
        compile_mode="simd",
    )
    active = torch.arange(block) < 3
    mask = torch.stack([active, ~active, active]).flatten()
    if mask_use == "atomic_add":
        expected = torch.where(mask, source_cpu - 2, -2)
    else:
        other = -2 if mask_use == "store" else -1
        expected = torch.where(mask, source_cpu, other)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
