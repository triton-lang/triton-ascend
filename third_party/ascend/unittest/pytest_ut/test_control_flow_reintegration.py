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
import triton
import triton.language as tl
import triton.language.extra.cann as al

torch_npu = pytest.importorskip("torch_npu")

DEVICE = "npu"
DTYPE = torch.float32

BLOCK = 32
N_LOOPS = 4
N_OUTER = 2
N_INNER = 2


def _require_npu():
    if not torch.npu.is_available():
        pytest.skip("NPU is unavailable")


def _assert_close(actual, expected, atol=1e-4, rtol=1e-4):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@triton.jit
def kernel_cf_rein_s068_for_outer_scope(
    in_ptr,
    out_ptr,
    n_loops,
    block: tl.constexpr,
    off: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    ptr = in_ptr + offsets
    acc = tl.zeros((block, ), dtype=tl.float32)

    for _ in range(n_loops):
        with al.scope(core_mode="vector"):
            ptr = ptr + off
            acc += tl.load(ptr)

    tl.store(out_ptr + offsets, acc)


def test_cf_rein_s068_for_outer_scope():
    """An outer loop containing a vector scope accumulates shifted loads."""
    _require_npu()

    x = torch.randn((BLOCK * 4, ), dtype=DTYPE, device=DEVICE)
    out = torch.zeros((BLOCK, ), dtype=DTYPE, device=DEVICE)

    kernel_cf_rein_s068_for_outer_scope[(1, )](
        x,
        out,
        N_LOOPS,
        BLOCK,
        1,
    )

    expected = sum(x[offset:offset + BLOCK] for offset in range(1, N_LOOPS + 1))
    _assert_close(out, expected)


@triton.jit
def kernel_cf_rein_s051_multi_cf_chain_ptr(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    ptr = in_ptr + offsets

    if flag != 0:
        for _ in range(n_loops):
            ptr = ptr + off
    else:
        for _ in range(n_loops):
            ptr = ptr - off

    acc = tl.zeros((block, ), dtype=tl.float32)
    for _ in range(n_loops):
        acc += tl.load(ptr)

    tl.store(out_ptr + offsets, acc)


def test_cf_rein_s051_multi_cf_chain_ptr():
    """A pointer produced by one control-flow region is consumed by another."""
    _require_npu()

    torch.manual_seed(0)

    # Leave room on both sides so that both branch directions are valid.
    x = torch.randn((BLOCK * 8, ), dtype=DTYPE, device=DEVICE)
    in_view = x[2 * BLOCK:]
    out = torch.zeros((BLOCK, ), dtype=DTYPE, device=DEVICE)

    for flag in (1, 0):
        out.zero_()

        kernel_cf_rein_s051_multi_cf_chain_ptr[(1, )](
            in_view,
            out,
            flag,
            N_LOOPS,
            BLOCK,
            1,
        )

        signed_shift = N_LOOPS if flag != 0 else -N_LOOPS
        expected = N_LOOPS * x[2 * BLOCK + signed_shift:2 * BLOCK + signed_shift + BLOCK]
        _assert_close(out, expected)


@triton.jit
def kernel_cf_rein_s025_l2_while_while_load(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    block: tl.constexpr,
    off_then: tl.constexpr,
    off_else: tl.constexpr,
    off_step: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    ptr = in_ptr + offsets
    acc = tl.zeros((block, ), dtype=tl.float32)

    i = 0
    while i < n_outer:
        j = 0
        while j < n_inner:
            ptr = ptr + off_step
            acc += tl.load(ptr)
            j = j + 1
        i = i + 1

    tl.store(out_ptr + offsets, acc.to(out_ptr.dtype.element_ty))


def test_cf_rein_s025_l2_while_while_load():
    """Nested while loops repeatedly advance a pointer and load from it."""
    _require_npu()

    x = torch.randn((BLOCK * 8, ), dtype=DTYPE, device=DEVICE)
    out = torch.zeros((BLOCK, ), dtype=DTYPE, device=DEVICE)

    kernel_cf_rein_s025_l2_while_while_load[(1, )](
        x,
        out,
        1,
        N_OUTER,
        N_INNER,
        BLOCK,
        1,
        2,
        1,
    )

    num_loads = N_OUTER * N_INNER
    expected = sum(x[offset:offset + BLOCK] for offset in range(1, num_loads + 1))
    _assert_close(out, expected)


@triton.jit
def kernel_cf_rein_s056_scope_cube_cube(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)

    with al.scope(core_mode="cube"):
        a = tl.load(a_ptr + offsets_m[:, None] * k + offsets_k[None, :])
        b = tl.load(b_ptr + offsets_k[:, None] * n + offsets_n[None, :])
        accumulator = tl.dot(a, b)

    tl.store(
        c_ptr + offsets_m[:, None] * n + offsets_n[None, :],
        accumulator,
    )


def test_cf_rein_s056_scope_cube_cube():
    """A cube scope performs a complete matrix multiplication tile."""
    _require_npu()

    m = 16
    n = 16
    k = 16
    block_m = 16
    block_n = 16
    block_k = 16

    a = torch.randn((m, k), dtype=torch.float16, device=DEVICE)
    b = torch.randn((k, n), dtype=torch.float16, device=DEVICE)
    out = torch.zeros((m, n), dtype=torch.float16, device=DEVICE)

    kernel_cf_rein_s056_scope_cube_cube[(1, 1)](
        a,
        b,
        out,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
    )

    expected = torch.matmul(a, b)
    _assert_close(out, expected, atol=1e-2, rtol=1e-2)


@triton.jit
def kernel_cf_rein_s087_diff_block_ptr_merge(
    in_a_ptr,
    in_b_ptr,
    out_ptr,
    flag,
    n_elements,
    block: tl.constexpr,
):
    if flag != 0:
        block_ptr = tl.make_block_ptr(
            base=in_a_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(0, ),
            block_shape=(block, ),
            order=(0, ),
        )
    else:
        block_ptr = tl.make_block_ptr(
            base=in_b_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(0, ),
            block_shape=(block, ),
            order=(0, ),
        )

    value = tl.load(block_ptr)

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(block, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    tl.store(out_block_ptr, value)


def test_cf_rein_s087_diff_block_ptr_merge():
    """Different block-pointer bases selected by an if produce correct values."""
    _require_npu()

    torch.manual_seed(0)

    a = torch.randn((BLOCK, ), dtype=DTYPE, device=DEVICE)
    b = torch.randn((BLOCK, ), dtype=DTYPE, device=DEVICE)

    for flag, expected in ((1, a), (0, b)):
        out = torch.empty_like(a)

        kernel_cf_rein_s087_diff_block_ptr_merge[(1, )](
            a,
            b,
            out,
            flag,
            BLOCK,
            BLOCK,
        )

        _assert_close(out, expected)
