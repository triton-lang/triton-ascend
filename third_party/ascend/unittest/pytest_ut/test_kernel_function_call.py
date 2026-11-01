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
import test_common


@triton.jit
def load_effective_token(
    committed_ptr,
    committed_stride,
    pending_ptr,
    request_start,
    request_idx,
    committed_len,
    pos,
):
    if pos < committed_len:
        return tl.load(committed_ptr + request_idx * committed_stride + pos)
    pending_pos = request_start + pos - committed_len + 1
    return tl.load(pending_ptr + pending_pos)


@triton.jit
def reproduce_kernel(
    committed_ptr,
    pending_ptr,
    lengths_ptr,
    request_mapping_ptr,
    output_ptr,
    committed_stride,
):
    token_idx = tl.program_id(0).to(tl.int64)
    request_idx = tl.load(request_mapping_ptr + token_idx)
    committed_len = tl.load(lengths_ptr + request_idx)
    effective_len = committed_len + 1
    total = tl.zeros((), dtype=tl.int32)
    for pos in tl.range(committed_len - 1, effective_len):
        token = load_effective_token(
            committed_ptr,
            committed_stride,
            pending_ptr,
            token_idx,
            request_idx,
            committed_len,
            pos,
        )
        total += token
    tl.store(output_ptr + token_idx, total)


@triton.jit
def helper_top(x_ptr, idx):
    v = tl.load(x_ptr + idx)
    if v > 0:
        return v + 1
    return v + 2


@triton.jit
def helper_add_const(x_ptr, idx, c: tl.constexpr):
    return tl.load(x_ptr + idx) + c


@triton.jit
def helper_loop_sum_early(x_ptr, idx, N: tl.constexpr):
    s = tl.zeros((), dtype=tl.int32)
    for j in tl.range(0, N):
        s += helper_add_const(x_ptr, idx * N + j, 1)
    if s >= 0:
        return s
    return 0


@triton.jit
def helper_early_select(x_ptr, idx):
    v = tl.load(x_ptr + idx)
    if v > 0:
        return v + 1
    return v + 2


@triton.jit
def helper_add_const_inline(x_ptr, idx, c: tl.constexpr):
    return tl.load(x_ptr + idx) + c


@triton.jit
def kernel_top_call(x_ptr, out_ptr):
    v = helper_top(x_ptr, 0)
    tl.store(out_ptr, v)


@triton.jit
def kernel_nested_loop_call(x_ptr, out_ptr, N: tl.constexpr):
    total = tl.zeros((), dtype=tl.int32)
    for i in tl.range(0, N):
        total += helper_loop_sum_early(x_ptr, i, N)
    tl.store(out_ptr, total)


@triton.jit
def kernel_if_call(x_ptr, out_ptr, mode_ptr):
    mode = tl.load(mode_ptr)
    v = tl.zeros((), dtype=tl.int32)
    if mode > 0:
        if mode > 1:
            v = helper_early_select(x_ptr, 0)
        else:
            v = helper_early_select(x_ptr, 1)
    else:
        v = helper_early_select(x_ptr, 2)
    tl.store(out_ptr, v)


@triton.jit
def kernel_loop_call_inline(x_ptr, out_ptr, N: tl.constexpr):
    total = tl.zeros((), dtype=tl.int32)
    for i in tl.range(0, N):
        total += helper_add_const_inline(x_ptr, i, 1)
    tl.store(out_ptr, total)


def _early_select_ref(v):
    return int(v) + 1 if int(v) > 0 else int(v) + 2


def _effective_token_reference(committed, pending, lengths, mapping):
    totals = []
    for t, r in enumerate(mapping.tolist()):
        committed_len = int(lengths[r].item())
        total = 0
        for pos in range(committed_len - 1, committed_len + 1):
            if pos < committed_len:
                total += int(committed[r, pos].item())
            else:
                pending_pos = t + pos - committed_len + 1
                total += int(pending[pending_pos].item())
        totals.append(total)
    return torch.tensor(totals, dtype=torch.int32)


def _run_effective_token_case(num_requests, max_committed_len, committed_stride):
    device = "npu"
    torch.manual_seed(0)
    committed = torch.randint(-100, 100, (num_requests, max_committed_len), dtype=torch.int32, device=device)
    pending_len = num_requests + max_committed_len
    pending = torch.randint(-100, 100, (pending_len, ), dtype=torch.int32, device=device)
    lengths = torch.randint(1, max_committed_len + 1, (num_requests, ), dtype=torch.int32, device=device)
    mapping = torch.randperm(num_requests, dtype=torch.int32, device=device)
    output = torch.empty(num_requests, dtype=torch.int32, device=device)
    reproduce_kernel[(num_requests, )](
        committed,
        pending,
        lengths,
        mapping,
        output,
        committed.stride(0),
    )
    torch.npu.synchronize()
    expected = _effective_token_reference(committed.cpu(), pending.cpu(), lengths.cpu(), mapping.cpu())
    test_common.validate_cmp("int32", output.cpu(), expected)


@pytest.mark.parametrize("N", [32])
def test_jit_call_at_top_level(N):
    x = test_common.generate_tensor(shape=(N, ), dtype="int32")
    out = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_top_call[(1, )](x.npu(), out)
    expected = torch.tensor([_early_select_ref(x[0])], dtype=torch.int32)
    test_common.validate_cmp("int32", out.cpu(), expected)


@pytest.mark.parametrize(
    "num_requests, max_committed_len, committed_stride",
    [
        (3, 5, 4),
    ],
)
def test_jit_call_in_for_loop_effective_token(num_requests, max_committed_len, committed_stride):
    _run_effective_token_case(num_requests, max_committed_len, committed_stride)


@pytest.mark.parametrize("N", [4, 16, 33])
def test_jit_call_nested_loop(N):
    x = torch.randint(0, 100, (N * N, ), dtype=torch.int32)
    out = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_nested_loop_call[(1, )](x.npu(), out, N)
    expected = torch.tensor([int(x.sum()) + N * N], dtype=torch.int32)
    test_common.validate_cmp("int32", out.cpu(), expected)


@pytest.mark.parametrize("mode, x_idx", [(1, 1)])
def test_jit_call_in_nested_if(mode, x_idx):
    x = test_common.generate_tensor(shape=(4, ), dtype="int32")
    mode_tensor = torch.tensor([mode], dtype=torch.int32, device="npu")
    out = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_if_call[(1, )](x.npu(), out, mode_tensor)
    expected = torch.tensor([_early_select_ref(x[x_idx])], dtype=torch.int32)
    test_common.validate_cmp("int32", out.cpu(), expected)


@pytest.mark.parametrize("N", [32])
def test_jit_call_in_for_loop_inlined(N):
    x = test_common.generate_tensor(shape=(N, ), dtype="int32")
    out = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_loop_call_inline[(1, )](x.npu(), out, N)
    expected = torch.tensor([int(x.sum()) + N], dtype=torch.int32)
    test_common.validate_cmp("int32", out.cpu(), expected)


@triton.jit
def helper_early_return_select(x_ptr, y_ptr, offset, pick_y, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    if pick_y:
        return tl.load(y_ptr + offset + idx)
    return tl.load(x_ptr + offset + idx) + 1


@triton.jit
def kernel_early_return_select(x_ptr, y_ptr, mode_ptr, out_ptr, BLOCK: tl.constexpr):
    pick_y = tl.load(mode_ptr)
    block = helper_early_return_select(x_ptr, y_ptr, 0, pick_y, BLOCK)
    tl.store(out_ptr + tl.arange(0, BLOCK), block)


@triton.jit
def helper_early_return_three(x_ptr, offset, mode, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offset + idx)
    if mode == 0:
        return v
    if mode == 1:
        return v + 1
    return v - 1


@triton.jit
def kernel_early_return_three(x_ptr, mode_ptr, out_ptr, BLOCK: tl.constexpr):
    mode = tl.load(mode_ptr)
    block = helper_early_return_three(x_ptr, 0, mode, BLOCK)
    tl.store(out_ptr + tl.arange(0, BLOCK), block)


@triton.jit
def kernel_early_return_in_loop(x_ptr, y_ptr, mode_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pick_y = tl.load(mode_ptr)
    for i in tl.range(0, N):
        block = helper_early_return_select(x_ptr, y_ptr, i * BLOCK, pick_y, BLOCK)
        tl.store(out_ptr + i * BLOCK + tl.arange(0, BLOCK), block)


@pytest.mark.parametrize("BLOCK, pick_y", [(32, 1)])
def test_early_return_tensor_select(BLOCK, pick_y):
    x = test_common.generate_tensor(shape=(BLOCK, ), dtype="int32")
    y = test_common.generate_tensor(shape=(BLOCK, ), dtype="int32")
    out = torch.empty((BLOCK, ), dtype=torch.int32, device="npu")
    mode_tensor = torch.tensor([pick_y], dtype=torch.int32, device="npu")
    kernel_early_return_select[(1, )](x.npu(), y.npu(), mode_tensor, out, BLOCK)
    expected = y if pick_y else x + 1
    test_common.validate_cmp("int32", out.cpu(), expected)


@pytest.mark.parametrize("BLOCK, mode", [(127, 2)])
def test_early_return_tensor_three(BLOCK, mode):
    x = test_common.generate_tensor(shape=(BLOCK, ), dtype="int32")
    out = torch.empty((BLOCK, ), dtype=torch.int32, device="npu")
    mode_tensor = torch.tensor([mode], dtype=torch.int32, device="npu")
    kernel_early_return_three[(1, )](x.npu(), mode_tensor, out, BLOCK)
    expected = {0: x, 1: x + 1, 2: x - 1}[mode]
    test_common.validate_cmp("int32", out.cpu(), expected)


@pytest.mark.parametrize("N, BLOCK, pick_y", [(3, 32, 1)])
def test_early_return_tensor_in_loop(N, BLOCK, pick_y):
    size = N * BLOCK
    x = test_common.generate_tensor(shape=(size, ), dtype="int32")
    y = test_common.generate_tensor(shape=(size, ), dtype="int32")
    out = torch.empty((size, ), dtype=torch.int32, device="npu")
    mode_tensor = torch.tensor([pick_y], dtype=torch.int32, device="npu")
    kernel_early_return_in_loop[(1, )](x.npu(), y.npu(), mode_tensor, out, N, BLOCK)
    expected = y if pick_y else x + 1
    test_common.validate_cmp("int32", out.cpu(), expected)


@triton.jit(noinline=True)
def helper_early_return_tensor_arg(x_ptr, offset, cond, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offset + idx)
    if cond == 0:
        return v + 1
    if cond == 1:
        return v - 1
    if cond == 2:
        return v * 2
    return v + 100


@triton.jit
def kernel_early_return_tensor_arg(x_ptr, cond_ptr, out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    cond = tl.load(cond_ptr)
    r = helper_early_return_tensor_arg(x_ptr, 0, cond, BLOCK)
    tl.store(out_ptr + idx, r)


@pytest.mark.parametrize("BLOCK, cond", [(32, 2), (127, 3)])
def test_early_return_tensor_arg(BLOCK, cond):
    # Multi-block callee with a tensor argument and a tensor result, covering
    # every return point of the four-way early return.
    x = test_common.generate_tensor(shape=(BLOCK, ), dtype="int32")
    out = torch.empty((BLOCK, ), dtype=torch.int32, device="npu")
    cond_tensor = torch.tensor([cond], dtype=torch.int32, device="npu")
    kernel_early_return_tensor_arg[(1, )](x.npu(), cond_tensor, out, BLOCK)
    if cond == 0:
        expected = x + 1
    elif cond == 1:
        expected = x - 1
    elif cond == 2:
        expected = 2 * x
    else:
        expected = x + 100
    test_common.validate_cmp("int32", out.cpu(), expected)


@triton.jit
def helper_multi_return_scalar(x_ptr, idx):
    v = tl.load(x_ptr + idx)
    return v + 1, v - 1


@triton.jit
def kernel_multi_return_scalar(x_ptr, a_ptr, b_ptr):
    a, b = helper_multi_return_scalar(x_ptr, 0)
    tl.store(a_ptr, a)
    tl.store(b_ptr, b)


@triton.jit
def helper_multi_return_block_and_sum(x_ptr, offset, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    block = tl.load(x_ptr + offset + idx)
    return block, tl.sum(block)


@triton.jit
def kernel_multi_return_block_and_sum(x_ptr, out_ptr, sum_ptr, BLOCK: tl.constexpr):
    block, s = helper_multi_return_block_and_sum(x_ptr, 0, BLOCK)
    tl.store(out_ptr + tl.arange(0, BLOCK), block)
    tl.store(sum_ptr, s)


@triton.jit
def helper_multi_return_two_blocks(x_ptr, offset, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offset + idx)
    return v + 1, v - 1


@triton.jit
def kernel_multi_return_two_blocks(x_ptr, out_a_ptr, out_b_ptr, BLOCK: tl.constexpr):
    a, b = helper_multi_return_two_blocks(x_ptr, 0, BLOCK)
    tl.store(out_a_ptr + tl.arange(0, BLOCK), a)
    tl.store(out_b_ptr + tl.arange(0, BLOCK), b)


@pytest.mark.parametrize("N", [32])
def test_multi_return_scalar(N):
    x = test_common.generate_tensor(shape=(N, ), dtype="int32")
    a = torch.empty((1, ), dtype=torch.int32, device="npu")
    b = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_multi_return_scalar[(1, )](x.npu(), a, b)
    test_common.validate_cmp("int32", a.cpu(), torch.tensor([int(x[0]) + 1], dtype=torch.int32))
    test_common.validate_cmp("int32", b.cpu(), torch.tensor([int(x[0]) - 1], dtype=torch.int32))


@pytest.mark.parametrize("BLOCK", [32])
def test_multi_return_block_and_sum(BLOCK):
    x = test_common.generate_tensor(shape=(BLOCK, ), dtype="int32")
    out = torch.empty((BLOCK, ), dtype=torch.int32, device="npu")
    s = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_multi_return_block_and_sum[(1, )](x.npu(), out, s, BLOCK)
    test_common.validate_cmp("int32", out.cpu(), x)
    test_common.validate_cmp("int32", s.cpu(), torch.tensor([int(x.sum())], dtype=torch.int32))


@pytest.mark.parametrize("BLOCK", [32])
def test_multi_return_two_blocks(BLOCK):
    x = test_common.generate_tensor(shape=(BLOCK, ), dtype="int32")
    out_a = torch.empty((BLOCK, ), dtype=torch.int32, device="npu")
    out_b = torch.empty((BLOCK, ), dtype=torch.int32, device="npu")
    kernel_multi_return_two_blocks[(1, )](x.npu(), out_a, out_b, BLOCK)
    test_common.validate_cmp("int32", out_a.cpu(), x + 1)
    test_common.validate_cmp("int32", out_b.cpu(), x - 1)


@triton.jit(noinline=True)
def helper_noinline_add_const(x_ptr, idx, c: tl.constexpr):
    return tl.load(x_ptr + idx) + c


@triton.jit(noinline=True)
def helper_noinline_multi_return_scalar(x_ptr, idx):
    v = tl.load(x_ptr + idx)
    return v + 1, v - 1


@triton.jit
def kernel_noinline_call(x_ptr, out_ptr, c: tl.constexpr):
    v = helper_noinline_add_const(x_ptr, 0, c)
    tl.store(out_ptr, v)


@triton.jit
def kernel_noinline_multi_return(x_ptr, a_ptr, b_ptr):
    a, b = helper_noinline_multi_return_scalar(x_ptr, 0)
    tl.store(a_ptr, a)
    tl.store(b_ptr, b)


@pytest.mark.parametrize("N", [32])
def test_noinline_call(N):
    x = test_common.generate_tensor(shape=(N, ), dtype="int32")
    out = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_noinline_call[(1, )](x.npu(), out, 3)
    expected = torch.tensor([int(x[0]) + 3], dtype=torch.int32)
    test_common.validate_cmp("int32", out.cpu(), expected)


@pytest.mark.parametrize("N", [32])
def test_noinline_multi_return(N):
    x = test_common.generate_tensor(shape=(N, ), dtype="int32")
    a = torch.empty((1, ), dtype=torch.int32, device="npu")
    b = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_noinline_multi_return[(1, )](x.npu(), a, b)
    test_common.validate_cmp("int32", a.cpu(), torch.tensor([int(x[0]) + 1], dtype=torch.int32))
    test_common.validate_cmp("int32", b.cpu(), torch.tensor([int(x[0]) - 1], dtype=torch.int32))


@triton.jit(noinline=True)
def helper_noinline_block(x_ptr, offset, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    return tl.load(x_ptr + offset + idx)


@triton.jit
def kernel_noinline_block(x_ptr, out_ptr, BLOCK: tl.constexpr):
    block = helper_noinline_block(x_ptr, 0, BLOCK)
    tl.store(out_ptr + tl.arange(0, BLOCK), block)


@triton.jit(noinline=True)
def helper_noinline_block_and_sum(x_ptr, offset, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    block = tl.load(x_ptr + offset + idx)
    return block, tl.sum(block)


@triton.jit
def kernel_noinline_block_and_sum(x_ptr, out_ptr, sum_ptr, BLOCK: tl.constexpr):
    block, s = helper_noinline_block_and_sum(x_ptr, 0, BLOCK)
    tl.store(out_ptr + tl.arange(0, BLOCK), block)
    tl.store(sum_ptr, s)


@pytest.mark.parametrize("BLOCK", [32])
def test_noinline_block(BLOCK):
    x = test_common.generate_tensor(shape=(BLOCK, ), dtype="int32")
    out = torch.empty((BLOCK, ), dtype=torch.int32, device="npu")
    kernel_noinline_block[(1, )](x.npu(), out, BLOCK)
    test_common.validate_cmp("int32", out.cpu(), x)


@pytest.mark.parametrize("BLOCK", [32])
def test_noinline_multi_return_block_and_sum(BLOCK):
    x = test_common.generate_tensor(shape=(BLOCK, ), dtype="int32")
    out = torch.empty((BLOCK, ), dtype=torch.int32, device="npu")
    s = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_noinline_block_and_sum[(1, )](x.npu(), out, s, BLOCK)
    test_common.validate_cmp("int32", out.cpu(), x)
    test_common.validate_cmp("int32", s.cpu(), torch.tensor([int(x.sum())], dtype=torch.int32))


@triton.jit
def kernel_noinline_multiple_callees(x1_ptr, x2_ptr, out_ptr, sum_ptr, BLOCK: tl.constexpr):
    block = helper_noinline_block(x1_ptr, 0, BLOCK)
    s = helper_noinline_add_const(x2_ptr, 0, 2)
    a, b = helper_noinline_multi_return_scalar(x2_ptr, 0)
    block2, total = helper_noinline_block_and_sum(x2_ptr, 0, BLOCK)
    tl.store(out_ptr + tl.arange(0, BLOCK), block + block2 + a)
    tl.store(sum_ptr, s + b + total)


@pytest.mark.parametrize("BLOCK", [32])
def test_noinline_multiple_callees(BLOCK):
    x1 = test_common.generate_tensor(shape=(BLOCK, ), dtype="int32")
    x2 = test_common.generate_tensor(shape=(BLOCK, ), dtype="int32")
    out = torch.empty((BLOCK, ), dtype=torch.int32, device="npu")
    s = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_noinline_multiple_callees[(1, )](x1.npu(), x2.npu(), out, s, BLOCK)
    expected_out = x1 + x2 + int(x2[0]) + 1
    expected_s = torch.tensor([2 * int(x2[0]) + 1 + int(x2.sum())], dtype=torch.int32)
    test_common.validate_cmp("int32", out.cpu(), expected_out)
    test_common.validate_cmp("int32", s.cpu(), expected_s)


@triton.jit(noinline=True)
def helper_noinline_nested_leaf(x_ptr, idx, c: tl.constexpr, N: tl.constexpr):
    # Noinline leaf: a loop with a three-way conditional accumulation.
    total = tl.zeros((), dtype=tl.int32)
    for i in tl.range(0, N):
        v = tl.load(x_ptr + idx + i)
        if v > 100:
            total += v + c
        elif v > 50:
            total += v - c
        else:
            total += v * 2
    return total


@triton.jit(noinline=True)
def helper_noinline_nested_middle(x_ptr, idx, c: tl.constexpr, N: tl.constexpr):
    # A noinline callee that itself calls another noinline callee. The early
    # return makes this one multi-block, so it survives as a real call, while
    # the leaf (single-block) is inlined into it.
    s = helper_noinline_nested_leaf(x_ptr, idx, c, N)
    if s > 1000:
        return s + 10
    return s - 10


@triton.jit
def kernel_noinline_nested(x_ptr, out_ptr, N: tl.constexpr):
    v = helper_noinline_nested_middle(x_ptr, 0, 3, N)
    tl.store(out_ptr, v)


def _noinline_nested_ref(x, c, N):
    total = 0
    for i in range(N):
        v = int(x[i])
        if v > 100:
            total += v + c
        elif v > 50:
            total += v - c
        else:
            total += v * 2
    return total + 10 if total > 1000 else total - 10


@pytest.mark.parametrize("N", [32])
def test_noinline_nested(N):
    x = test_common.generate_tensor(shape=(N, ), dtype="int32")
    out = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_noinline_nested[(1, )](x.npu(), out, N)
    expected = torch.tensor([_noinline_nested_ref(x, 3, N)], dtype=torch.int32)
    test_common.validate_cmp("int32", out.cpu(), expected)


@triton.jit(noinline=True)
def helper_noinline_block_2d(x_ptr, offset, BM: tl.constexpr, BN: tl.constexpr):
    row = tl.arange(0, BM)[:, None]
    col = tl.arange(0, BN)[None, :]
    idx = row * BN + col
    block = tl.load(x_ptr + offset + idx)
    return block, tl.sum(block, axis=1)


@triton.jit
def kernel_noinline_block_2d(x_ptr, out_ptr, sum_ptr, BM: tl.constexpr, BN: tl.constexpr):
    row = tl.arange(0, BM)[:, None]
    col = tl.arange(0, BN)[None, :]
    block, row_sum = helper_noinline_block_2d(x_ptr, 0, BM, BN)
    tl.store(out_ptr + row * BN + col, block)
    tl.store(sum_ptr + tl.arange(0, BM), row_sum)


@pytest.mark.parametrize("BM, BN", [(8, 128)])
def test_noinline_block_2d(BM, BN):
    # noinline callee returning a 2D tile plus its per-row sums.
    size = BM * BN
    x = test_common.generate_tensor(shape=(size, ), dtype="int32")
    out = torch.empty((size, ), dtype=torch.int32, device="npu")
    s = torch.empty((BM, ), dtype=torch.int32, device="npu")
    kernel_noinline_block_2d[(1, )](x.npu(), out, s, BM, BN)
    x2d = x.view(BM, BN)
    test_common.validate_cmp("int32", out.cpu(), x2d.reshape(-1))
    test_common.validate_cmp("int32", s.cpu(), x2d.sum(dim=1).to(torch.int32))


@triton.jit(noinline=True)
def helper_noinline_block_3d(x_ptr, offset, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    r = tl.arange(0, BM)[:, None, None]
    c = tl.arange(0, BN)[None, :, None]
    k = tl.arange(0, BK)[None, None, :]
    idx = (r * BN + c) * BK + k
    block = tl.load(x_ptr + offset + idx)
    return block, tl.sum(block)


@triton.jit
def kernel_noinline_block_3d(x_ptr, out_ptr, sum_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    r = tl.arange(0, BM)[:, None, None]
    c = tl.arange(0, BN)[None, :, None]
    k = tl.arange(0, BK)[None, None, :]
    block, s = helper_noinline_block_3d(x_ptr, 0, BM, BN, BK)
    tl.store(out_ptr + (r * BN + c) * BK + k, block)
    tl.store(sum_ptr, s)


@pytest.mark.parametrize("BM, BN, BK", [(2, 64, 20)])
def test_noinline_block_3d(BM, BN, BK):
    # noinline callee returning a 3D tile plus its global sum.
    size = BM * BN * BK
    x = test_common.generate_tensor(shape=(size, ), dtype="int32")
    out = torch.empty((size, ), dtype=torch.int32, device="npu")
    s = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_noinline_block_3d[(1, )](x.npu(), out, s, BM, BN, BK)
    x3d = x.view(BM, BN, BK)
    test_common.validate_cmp("int32", out.cpu(), x3d.reshape(-1))
    test_common.validate_cmp("int32", s.cpu(), torch.tensor([int(x.sum())], dtype=torch.int32))
