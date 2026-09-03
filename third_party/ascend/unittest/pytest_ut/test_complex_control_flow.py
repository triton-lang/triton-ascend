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

# ============================================================
# 复杂控制流指针分析 — 测试用例
# 【AI生成标识】
#   生成工具: Cursor Agent
#   用例数量: 主门禁 24 + 扩展 36 + atomic/IV/different-base = 62 TC
#   生成日期: 2026-05-28
#   设计文档: TEST_DESIGN_complex_control_flow.md
#   Spec: spec_complex_control_flow.yaml
#   适用平台: Ascend NPU
# ============================================================

import pytest
import torch
import torch_npu
import triton
import triton.language as tl

DEVICE = "npu"
DTYPES_E2E = [torch.float16, torch.float32]
BLOCK = 32
N_LOOPS = 4
OFF0 = 1
OFF1 = 2
ADV_IF = 1
ADV_ELSE = 2
N_OUTER = 4
N_INNER = 4
OFF2 = 3
ADV_IEL_IF = 1
ADV_IEL_ELIF = 2
ADV_IEL_ELSE = 3
STRIDE0 = 1
STRIDE1 = 1


def _require_npu():
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU not available")


def _assert_close(actual, expected, dtype):
    actual = actual.cpu()
    expected = expected.cpu()
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32,
                 torch.uint64):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    else:
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def compare_results(actual, expected, dtype):
    _assert_close(actual, expected, dtype)


def _n_elements_for_loops(block, n_loops, off0, off1):
    return block + n_loops * max(off0, off1) + 8


def _branch_delta(flag, off0, off1):
    return off0 if flag != 0 else off1


def _ref_src_tile(flat, block):
    idx = torch.arange(block, device=flat.device, dtype=torch.long)
    return flat[idx.clamp(0, flat.numel() - 1)].clone()


def _ref_st_scatter_carry(flat, block, step, n_iters):
    """Simulate in-loop store + final load on a pristine buffer (flat is a clone)."""
    src = _ref_src_tile(flat, block)
    pos = 0
    for _ in range(n_iters):
        pos += step
        end = min(pos + block, flat.numel())
        flat[pos:end] = src[:end - pos]
    end = min(pos + block, flat.numel())
    out = flat[pos:end]
    if out.numel() < block:
        out = torch.cat([
            out,
            torch.zeros(block - out.numel(), dtype=out.dtype, device=out.device),
        ])
    return out


def _ref_atomic_add_carry(flat, block, step, n_iters):
    """Simulate loop-carried atomic additions and return the final tile."""
    buffer = flat.clone()
    src = _ref_src_tile(buffer, block)
    pos = 0
    for _ in range(n_iters):
        pos += step
        buffer[pos:pos + block] += src
    return buffer[pos:pos + block]


def _ref_fp32_chunk_add(acc, flat, pos, block):
    end = min(pos + block, flat.numel())
    chunk = flat[pos:end]
    if chunk.numel() < block:
        pad = torch.zeros(block - chunk.numel(), dtype=chunk.dtype, device=chunk.device)
        chunk = torch.cat([chunk, pad])
    return acc + chunk.float()


def _ref_adv_ld_accumulate(x, block, step_fn, n_iters):
    flat = x.flatten()
    pos = 0
    acc = torch.zeros(block, dtype=torch.float32, device=x.device)
    for _ in range(n_iters):
        pos += step_fn()
        acc = _ref_fp32_chunk_add(acc, flat, pos, block)
    return acc.to(x.dtype)


# ---------------------------------------------------------------------------
# D5=2 addptr + load kernels (TC 001–006)
# ---------------------------------------------------------------------------


@triton.jit
def kernel_ap_ld_in_ie(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n_loops):
        if flag != 0:
            a_ptr = a_ptr + off0
            val = tl.load(a_ptr)
        else:
            a_ptr = a_ptr + off1
            val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_ld_out_ie(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n_loops):
        if flag != 0:
            a_ptr = a_ptr + off0
        else:
            a_ptr = a_ptr + off1
        val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_ld_in_for(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    if flag != 0:
        for _ in range(n_loops):
            a_ptr = a_ptr + off0
            val = tl.load(a_ptr)
    else:
        for _ in range(n_loops):
            a_ptr = a_ptr + off1
            val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_ld_out_for(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    if flag != 0:
        for _ in range(n_loops):
            a_ptr = a_ptr + off0
    else:
        for _ in range(n_loops):
            a_ptr = a_ptr + off1
    val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_ld_in_wh(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    if flag != 0:
        i = 0
        while i < n_loops:
            a_ptr = a_ptr + off0
            val = tl.load(a_ptr)
            i = i + 1
    else:
        i = 0
        while i < n_loops:
            a_ptr = a_ptr + off1
            val = tl.load(a_ptr)
            i = i + 1
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_ld_out_wh(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    if flag != 0:
        i = 0
        while i < n_loops:
            a_ptr = a_ptr + off0
            i = i + 1
    else:
        i = 0
        while i < n_loops:
            a_ptr = a_ptr + off1
            i = i + 1
    val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


def reference_ap_ld_loop(x, flag, n_loops, off0, off1, block, outer_for):
    flat = x.flatten()
    idx = torch.arange(block, device=x.device, dtype=torch.long)
    val = flat[idx.clamp(0, flat.numel() - 1)]
    if outer_for:
        for _ in range(n_loops):
            idx = idx + _branch_delta(flag, off0, off1)
            idx = idx.clamp(0, flat.numel() - 1)
            val = flat[idx]
    else:
        delta = _branch_delta(flag, off0, off1)
        for _ in range(n_loops):
            idx = idx + delta
            idx = idx.clamp(0, flat.numel() - 1)
            val = flat[idx]
    return val


def _run_ap_ld(kernel, x, flag, n_loops, off0, off1, block):
    out = torch.empty(block, dtype=x.dtype, device=x.device)
    kernel[(1, )](x, out, flag, n_loops, block, off0, off1)
    return out


# ---------------------------------------------------------------------------
# D5=2 addptr + store kernels (TC 007–012)
# ---------------------------------------------------------------------------


@triton.jit
def kernel_ap_st_in_ie(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    src = tl.load(in_ptr + offs)
    for _ in range(n_loops):
        if flag != 0:
            a_ptr = a_ptr + off0
            tl.store(a_ptr, src)
        else:
            a_ptr = a_ptr + off1
            tl.store(a_ptr, src)
    tl.store(out_ptr + offs, tl.load(a_ptr))


@triton.jit
def kernel_ap_st_out_ie(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    src = tl.load(in_ptr + offs)
    for _ in range(n_loops):
        if flag != 0:
            a_ptr = a_ptr + off0
        else:
            a_ptr = a_ptr + off1
        tl.store(a_ptr, src)
    tl.store(out_ptr + offs, tl.load(a_ptr))


@triton.jit
def kernel_ap_st_in_for(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    src = tl.load(in_ptr + offs)
    if flag != 0:
        for _ in range(n_loops):
            a_ptr = a_ptr + off0
            tl.store(a_ptr, src)
    else:
        for _ in range(n_loops):
            a_ptr = a_ptr + off1
            tl.store(a_ptr, src)
    tl.store(out_ptr + offs, tl.load(a_ptr))


@triton.jit
def kernel_ap_st_out_for(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    src = tl.load(in_ptr + offs)
    if flag != 0:
        for _ in range(n_loops):
            a_ptr = a_ptr + off0
            tl.store(a_ptr, src)
    else:
        for _ in range(n_loops):
            a_ptr = a_ptr + off1
            tl.store(a_ptr, src)
    tl.store(out_ptr + offs, tl.load(a_ptr))


@triton.jit
def kernel_ap_st_in_wh(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    src = tl.load(in_ptr + offs)
    if flag != 0:
        i = 0
        while i < n_loops:
            a_ptr = a_ptr + off0
            tl.store(a_ptr, src)
            i = i + 1
    else:
        i = 0
        while i < n_loops:
            a_ptr = a_ptr + off1
            tl.store(a_ptr, src)
            i = i + 1
    tl.store(out_ptr + offs, tl.load(a_ptr))


@triton.jit
def kernel_ap_st_out_wh(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    src = tl.load(in_ptr + offs)
    if flag != 0:
        i = 0
        while i < n_loops:
            a_ptr = a_ptr + off0
            i = i + 1
    else:
        i = 0
        while i < n_loops:
            a_ptr = a_ptr + off1
            i = i + 1
    tl.store(a_ptr, src)
    tl.store(out_ptr + offs, tl.load(a_ptr))


def reference_ap_st_ie(x, flag, n_loops, off0, off1, block):
    flat = x.flatten().clone()
    return _ref_st_scatter_carry(
        flat,
        block,
        _branch_delta(flag, off0, off1),
        n_loops,
    )


# ---------------------------------------------------------------------------
# D5=2 advance + load kernels (TC 013–018)
# ---------------------------------------------------------------------------


@triton.jit
def kernel_adv_ld_in_ie(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for _ in range(n_loops):
        if flag != 0:
            bp = tl.advance(bp, (adv_if, ))
            acc += tl.load(bp)
        else:
            bp = tl.advance(bp, (adv_else, ))
            acc += tl.load(bp)
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_ld_out_ie(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for _ in range(n_loops):
        if flag != 0:
            bp = tl.advance(bp, (adv_if, ))
        else:
            bp = tl.advance(bp, (adv_else, ))
        acc += tl.load(bp)
    tl.store(out_ptr + tl.arange(0, block), acc)


def reference_adv_ld_ie(x, flag, n_loops, adv_if, adv_else, block):
    adv = adv_if if flag != 0 else adv_else
    return _ref_adv_ld_accumulate(x, block, lambda: adv, n_loops)


def _run_adv_ld_ie(kernel, x, flag, n_loops, adv_if, adv_else, block):
    n_elements = _n_elements_for_loops(block, n_loops, adv_if, adv_else)
    out = torch.empty(block, dtype=x.dtype, device=x.device)
    kernel[(1, )](x, out, flag, n_loops, n_elements, block, adv_if, adv_else)
    return out


@triton.jit
def kernel_adv_ld_in_for(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    if flag != 0:
        for _ in range(n_loops):
            bp = tl.advance(bp, (adv_if, ))
            acc += tl.load(bp)
    else:
        for _ in range(n_loops):
            bp = tl.advance(bp, (adv_else, ))
            acc += tl.load(bp)
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_ld_out_for(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    if flag != 0:
        for _ in range(n_loops):
            bp = tl.advance(bp, (adv_if, ))
            acc += tl.load(bp)
    else:
        for _ in range(n_loops):
            bp = tl.advance(bp, (adv_else, ))
            acc += tl.load(bp)
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_ld_in_wh(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    if flag != 0:
        i = 0
        while i < n_loops:
            bp = tl.advance(bp, (adv_if, ))
            acc += tl.load(bp)
            i = i + 1
    else:
        i = 0
        while i < n_loops:
            bp = tl.advance(bp, (adv_else, ))
            acc += tl.load(bp)
            i = i + 1
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_ld_out_wh(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    if flag != 0:
        i = 0
        while i < n_loops:
            bp = tl.advance(bp, (adv_if, ))
            acc += tl.load(bp)
            i = i + 1
    else:
        i = 0
        while i < n_loops:
            bp = tl.advance(bp, (adv_else, ))
            acc += tl.load(bp)
            i = i + 1
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_st_in_ie(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    src = tl.load(in_ptr + tl.arange(0, block))
    for _ in range(n_loops):
        if flag != 0:
            bp = tl.advance(bp, (adv_if, ))
            tl.store(bp, src)
        else:
            bp = tl.advance(bp, (adv_else, ))
            tl.store(bp, src)
    tl.store(out_ptr + tl.arange(0, block), tl.load(bp))


@triton.jit
def kernel_adv_st_out_ie(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    src = tl.load(in_ptr + tl.arange(0, block))
    for _ in range(n_loops):
        if flag != 0:
            bp = tl.advance(bp, (adv_if, ))
        else:
            bp = tl.advance(bp, (adv_else, ))
        tl.store(bp, src)
    tl.store(out_ptr + tl.arange(0, block), tl.load(bp))


@triton.jit
def kernel_adv_st_in_for(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    src = tl.load(in_ptr + tl.arange(0, block))
    if flag != 0:
        for _ in range(n_loops):
            bp = tl.advance(bp, (adv_if, ))
            tl.store(bp, src)
    else:
        for _ in range(n_loops):
            bp = tl.advance(bp, (adv_else, ))
            tl.store(bp, src)
    tl.store(out_ptr + tl.arange(0, block), tl.load(bp))


@triton.jit
def kernel_adv_st_out_for(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    src = tl.load(in_ptr + tl.arange(0, block))
    if flag != 0:
        for _ in range(n_loops):
            bp = tl.advance(bp, (adv_if, ))
            tl.store(bp, src)
    else:
        for _ in range(n_loops):
            bp = tl.advance(bp, (adv_else, ))
            tl.store(bp, src)
    tl.store(out_ptr + tl.arange(0, block), tl.load(bp))


@triton.jit
def kernel_adv_st_in_wh(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    src = tl.load(in_ptr + tl.arange(0, block))
    if flag != 0:
        i = 0
        while i < n_loops:
            bp = tl.advance(bp, (adv_if, ))
            tl.store(bp, src)
            i = i + 1
    else:
        i = 0
        while i < n_loops:
            bp = tl.advance(bp, (adv_else, ))
            tl.store(bp, src)
            i = i + 1
    tl.store(out_ptr + tl.arange(0, block), tl.load(bp))


@triton.jit
def kernel_adv_st_out_wh(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    src = tl.load(in_ptr + tl.arange(0, block))
    if flag != 0:
        i = 0
        while i < n_loops:
            bp = tl.advance(bp, (adv_if, ))
            i = i + 1
    else:
        i = 0
        while i < n_loops:
            bp = tl.advance(bp, (adv_else, ))
            i = i + 1
    tl.store(bp, src)
    tl.store(out_ptr + tl.arange(0, block), tl.load(bp))


def reference_adv_st_ie(x, flag, n_loops, adv_if, adv_else, block):
    flat = x.flatten().clone()
    adv = adv_if if flag != 0 else adv_else
    return _ref_st_scatter_carry(flat, block, adv, n_loops)


def _run_adv_st_ie(kernel, x, flag, n_loops, adv_if, adv_else, block):
    n_elements = _n_elements_for_loops(block, n_loops, adv_if, adv_else)
    out = torch.empty(block, dtype=x.dtype, device=x.device)
    kernel[(1, )](x, out, flag, n_loops, n_elements, block, adv_if, adv_else)
    return out


def reference_adv_ld_branch(x, flag, n_loops, adv_if, adv_else, block):
    adv = adv_if if flag != 0 else adv_else
    return _ref_adv_ld_accumulate(x, block, lambda: adv, n_loops)


def _run_adv_ld_branch(kernel, x, flag, n_loops, adv_if, adv_else, block):
    n_elements = _n_elements_for_loops(block, n_loops, adv_if, adv_else)
    out = torch.empty(block, dtype=x.dtype, device=x.device)
    kernel[(1, )](x, out, flag, n_loops, n_elements, block, adv_if, adv_else)
    return out


# ---------------------------------------------------------------------------
# Extension: if-elif-else (TC 025–032)
# ---------------------------------------------------------------------------


@triton.jit
def kernel_ap_ld_in_iee(
    in_ptr,
    out_ptr,
    branch_id,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
    off2: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n_loops):
        if branch_id == 0:
            a_ptr = a_ptr + off0
            val = tl.load(a_ptr)
        elif branch_id == 1:
            a_ptr = a_ptr + off1
            val = tl.load(a_ptr)
        else:
            a_ptr = a_ptr + off2
            val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_ld_out_iee(
    in_ptr,
    out_ptr,
    branch_id,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
    off2: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n_loops):
        if branch_id == 0:
            a_ptr = a_ptr + off0
        elif branch_id == 1:
            a_ptr = a_ptr + off1
        else:
            a_ptr = a_ptr + off2
        val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_st_in_iee(
    in_ptr,
    out_ptr,
    branch_id,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
    off2: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    src = tl.load(in_ptr + offs)
    for _ in range(n_loops):
        if branch_id == 0:
            a_ptr = a_ptr + off0
            tl.store(a_ptr, src)
        elif branch_id == 1:
            a_ptr = a_ptr + off1
            tl.store(a_ptr, src)
        else:
            a_ptr = a_ptr + off2
            tl.store(a_ptr, src)
    tl.store(out_ptr + offs, tl.load(a_ptr))


@triton.jit
def kernel_ap_st_out_iee(
    in_ptr,
    out_ptr,
    branch_id,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
    off2: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    src = tl.load(in_ptr + offs)
    for _ in range(n_loops):
        if branch_id == 0:
            a_ptr = a_ptr + off0
        elif branch_id == 1:
            a_ptr = a_ptr + off1
        else:
            a_ptr = a_ptr + off2
        tl.store(a_ptr, src)
    tl.store(out_ptr + offs, tl.load(a_ptr))


@triton.jit
def kernel_adv_ld_in_iee(
    in_ptr,
    out_ptr,
    branch_id,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv0: tl.constexpr,
    adv1: tl.constexpr,
    adv2: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for _ in range(n_loops):
        if branch_id == 0:
            bp = tl.advance(bp, (adv0, ))
            acc += tl.load(bp)
        elif branch_id == 1:
            bp = tl.advance(bp, (adv1, ))
            acc += tl.load(bp)
        else:
            bp = tl.advance(bp, (adv2, ))
            acc += tl.load(bp)
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_ld_out_iee(
    in_ptr,
    out_ptr,
    branch_id,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv0: tl.constexpr,
    adv1: tl.constexpr,
    adv2: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for _ in range(n_loops):
        if branch_id == 0:
            bp = tl.advance(bp, (adv0, ))
        elif branch_id == 1:
            bp = tl.advance(bp, (adv1, ))
        else:
            bp = tl.advance(bp, (adv2, ))
        acc += tl.load(bp)
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_st_in_iee(
    in_ptr,
    out_ptr,
    branch_id,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv0: tl.constexpr,
    adv1: tl.constexpr,
    adv2: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    src = tl.load(in_ptr + tl.arange(0, block))
    for _ in range(n_loops):
        if branch_id == 0:
            bp = tl.advance(bp, (adv0, ))
            tl.store(bp, src)
        elif branch_id == 1:
            bp = tl.advance(bp, (adv1, ))
            tl.store(bp, src)
        else:
            bp = tl.advance(bp, (adv2, ))
            tl.store(bp, src)
    tl.store(out_ptr + tl.arange(0, block), tl.load(bp))


@triton.jit
def kernel_adv_st_out_iee(
    in_ptr,
    out_ptr,
    branch_id,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv0: tl.constexpr,
    adv1: tl.constexpr,
    adv2: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    src = tl.load(in_ptr + tl.arange(0, block))
    for _ in range(n_loops):
        if branch_id == 0:
            bp = tl.advance(bp, (adv0, ))
        elif branch_id == 1:
            bp = tl.advance(bp, (adv1, ))
        else:
            bp = tl.advance(bp, (adv2, ))
        tl.store(bp, src)
    tl.store(out_ptr + tl.arange(0, block), tl.load(bp))


def reference_adv_st_iee(x, branch_id, n_loops, block):
    flat = x.flatten().clone()
    return _ref_st_scatter_carry(flat, block, _iel_adv(branch_id), n_loops)


def _iel_delta(branch_id):
    return {0: OFF0, 1: OFF1, 2: OFF2}[branch_id]


def _iel_adv(branch_id):
    return {0: ADV_IEL_IF, 1: ADV_IEL_ELIF, 2: ADV_IEL_ELSE}[branch_id]


def reference_ap_ld_iee(x, branch_id, n_loops, block):
    flat = x.flatten()
    idx = torch.arange(block, device=x.device, dtype=torch.long)
    delta = _iel_delta(branch_id)
    for _ in range(n_loops):
        idx = (idx + delta).clamp(0, flat.numel() - 1)
    return flat[idx]


def reference_ap_st_iee(x, branch_id, n_loops, block):
    flat = x.flatten().clone()
    return _ref_st_scatter_carry(flat, block, _iel_delta(branch_id), n_loops)


def reference_adv_ld_iee(x, branch_id, n_loops, block):
    adv = _iel_adv(branch_id)
    return _ref_adv_ld_accumulate(x, block, lambda: adv, n_loops)


# ---------------------------------------------------------------------------
# Extension: D5=3/4 load/store (TC 049–084)
# ---------------------------------------------------------------------------


@triton.jit
def kernel_ap_ld_in_ie_d3(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n_outer):
        for _ in range(n_inner):
            if flag != 0:
                a_ptr = a_ptr + off0
                val = tl.load(a_ptr)
            else:
                a_ptr = a_ptr + off1
                val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_ld_out_ie_d3(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n_outer):
        for _ in range(n_inner):
            if flag != 0:
                a_ptr = a_ptr + off0
            else:
                a_ptr = a_ptr + off1
            val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_ld_in_for_d3(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n_outer):
        if flag != 0:
            for _ in range(n_inner):
                a_ptr = a_ptr + off0
                val = tl.load(a_ptr)
        else:
            for _ in range(n_inner):
                a_ptr = a_ptr + off1
                val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_ld_out_for_d3(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n_outer):
        if flag != 0:
            for _ in range(n_inner):
                a_ptr = a_ptr + off0
        else:
            for _ in range(n_inner):
                a_ptr = a_ptr + off1
        val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_adv_ld_in_wh_ie_d3(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for _ in range(n_outer):
        i = 0
        while i < n_inner:
            if flag != 0:
                bp = tl.advance(bp, (adv_if, ))
                acc += tl.load(bp)
            else:
                bp = tl.advance(bp, (adv_else, ))
                acc += tl.load(bp)
            i = i + 1
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_ld_out_wh_ie_d3(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for _ in range(n_outer):
        i = 0
        while i < n_inner:
            if flag != 0:
                bp = tl.advance(bp, (adv_if, ))
            else:
                bp = tl.advance(bp, (adv_else, ))
            acc += tl.load(bp)
            i = i + 1
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_ld_in_for_wh_ie_d3(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    i = 0
    while i < n_outer:
        for _ in range(n_inner):
            if flag != 0:
                bp = tl.advance(bp, (adv_if, ))
                acc += tl.load(bp)
            else:
                bp = tl.advance(bp, (adv_else, ))
                acc += tl.load(bp)
        i = i + 1
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_ld_out_for_wh_ie_d3(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    i = 0
    while i < n_outer:
        for _ in range(n_inner):
            if flag != 0:
                bp = tl.advance(bp, (adv_if, ))
            else:
                bp = tl.advance(bp, (adv_else, ))
            acc += tl.load(bp)
        i = i + 1
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_ap_st_in_ie_for_d3(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    src = tl.load(in_ptr + offs)
    for _ in range(n_outer):
        if flag != 0:
            for _ in range(n_inner):
                a_ptr = a_ptr + off0
                tl.store(a_ptr, src)
        else:
            for _ in range(n_inner):
                a_ptr = a_ptr + off1
                tl.store(a_ptr, src)
    tl.store(out_ptr + offs, tl.load(a_ptr))


@triton.jit
def kernel_ap_st_out_ie_for_d3(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    src = tl.load(in_ptr + offs)
    for _ in range(n_outer):
        if flag != 0:
            for _ in range(n_inner):
                a_ptr = a_ptr + off0
                tl.store(a_ptr, src)
        else:
            for _ in range(n_inner):
                a_ptr = a_ptr + off1
                tl.store(a_ptr, src)
    tl.store(out_ptr + offs, tl.load(a_ptr))


@triton.jit
def kernel_ap_st_in_iee_for_d3(
    in_ptr,
    out_ptr,
    branch_id,
    n_outer,
    n_inner,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
    off2: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    src = tl.load(in_ptr + offs)
    for _ in range(n_outer):
        if branch_id == 0:
            for _ in range(n_inner):
                a_ptr = a_ptr + off0
                tl.store(a_ptr, src)
        elif branch_id == 1:
            for _ in range(n_inner):
                a_ptr = a_ptr + off1
                tl.store(a_ptr, src)
        else:
            for _ in range(n_inner):
                a_ptr = a_ptr + off2
                tl.store(a_ptr, src)
    tl.store(out_ptr + offs, tl.load(a_ptr))


@triton.jit
def kernel_ap_st_out_iee_for_d3(
    in_ptr,
    out_ptr,
    branch_id,
    n_outer,
    n_inner,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
    off2: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    src = tl.load(in_ptr + offs)
    for _ in range(n_outer):
        if branch_id == 0:
            for _ in range(n_inner):
                a_ptr = a_ptr + off0
                tl.store(a_ptr, src)
        elif branch_id == 1:
            for _ in range(n_inner):
                a_ptr = a_ptr + off1
                tl.store(a_ptr, src)
        else:
            for _ in range(n_inner):
                a_ptr = a_ptr + off2
                tl.store(a_ptr, src)
    tl.store(out_ptr + offs, tl.load(a_ptr))


@triton.jit
def kernel_adv_st_in_wh_for_iee_d3(
    in_ptr,
    out_ptr,
    branch_id,
    n_outer,
    n_inner,
    n_elements,
    block: tl.constexpr,
    adv0: tl.constexpr,
    adv1: tl.constexpr,
    adv2: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    src = tl.load(in_ptr + tl.arange(0, block))
    i = 0
    while i < n_outer:
        for _ in range(n_inner):
            if branch_id == 0:
                bp = tl.advance(bp, (adv0, ))
                tl.store(bp, src)
            elif branch_id == 1:
                bp = tl.advance(bp, (adv1, ))
                tl.store(bp, src)
            else:
                bp = tl.advance(bp, (adv2, ))
                tl.store(bp, src)
        i = i + 1
    tl.store(out_ptr + tl.arange(0, block), tl.load(bp))


@triton.jit
def kernel_adv_st_out_wh_for_iee_d3(
    in_ptr,
    out_ptr,
    branch_id,
    n_outer,
    n_inner,
    n_elements,
    block: tl.constexpr,
    adv0: tl.constexpr,
    adv1: tl.constexpr,
    adv2: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    src = tl.load(in_ptr + tl.arange(0, block))
    i = 0
    while i < n_outer:
        for _ in range(n_inner):
            if branch_id == 0:
                bp = tl.advance(bp, (adv0, ))
            elif branch_id == 1:
                bp = tl.advance(bp, (adv1, ))
            else:
                bp = tl.advance(bp, (adv2, ))
            tl.store(bp, src)
        i = i + 1
    tl.store(out_ptr + tl.arange(0, block), tl.load(bp))


@triton.jit
def kernel_adv_st_in_for_wh_iee_d3(
    in_ptr,
    out_ptr,
    branch_id,
    n_outer,
    n_inner,
    n_elements,
    block: tl.constexpr,
    adv0: tl.constexpr,
    adv1: tl.constexpr,
    adv2: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    src = tl.load(in_ptr + tl.arange(0, block))
    for _ in range(n_outer):
        i = 0
        while i < n_inner:
            if branch_id == 0:
                bp = tl.advance(bp, (adv0, ))
                tl.store(bp, src)
            elif branch_id == 1:
                bp = tl.advance(bp, (adv1, ))
                tl.store(bp, src)
            else:
                bp = tl.advance(bp, (adv2, ))
                tl.store(bp, src)
            i = i + 1
    tl.store(out_ptr + tl.arange(0, block), tl.load(bp))


@triton.jit
def kernel_adv_st_out_for_wh_iee_d3(
    in_ptr,
    out_ptr,
    branch_id,
    n_outer,
    n_inner,
    n_elements,
    block: tl.constexpr,
    adv0: tl.constexpr,
    adv1: tl.constexpr,
    adv2: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    src = tl.load(in_ptr + tl.arange(0, block))
    for _ in range(n_outer):
        i = 0
        while i < n_inner:
            if branch_id == 0:
                bp = tl.advance(bp, (adv0, ))
            elif branch_id == 1:
                bp = tl.advance(bp, (adv1, ))
            else:
                bp = tl.advance(bp, (adv2, ))
            tl.store(bp, src)
            i = i + 1
    tl.store(out_ptr + tl.arange(0, block), tl.load(bp))


@triton.jit
def kernel_ap_ld_in_iee_d4(
    in_ptr,
    out_ptr,
    branch_id,
    n_outer,
    n_inner,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
    off2: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n_outer):
        for _ in range(n_inner):
            if branch_id == 0:
                a_ptr = a_ptr + off0
                val = tl.load(a_ptr)
            elif branch_id == 1:
                a_ptr = a_ptr + off1
                val = tl.load(a_ptr)
            else:
                a_ptr = a_ptr + off2
                val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_ld_out_iee_d4(
    in_ptr,
    out_ptr,
    branch_id,
    n_outer,
    n_inner,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
    off2: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n_outer):
        for _ in range(n_inner):
            if branch_id == 0:
                a_ptr = a_ptr + off0
            elif branch_id == 1:
                a_ptr = a_ptr + off1
            else:
                a_ptr = a_ptr + off2
            val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_ld_in_for_iee_d4(
    in_ptr,
    out_ptr,
    branch_id,
    n_outer,
    n_inner,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
    off2: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n_outer):
        if branch_id == 0:
            for _ in range(n_inner):
                a_ptr = a_ptr + off0
                val = tl.load(a_ptr)
        elif branch_id == 1:
            for _ in range(n_inner):
                a_ptr = a_ptr + off1
                val = tl.load(a_ptr)
        else:
            for _ in range(n_inner):
                a_ptr = a_ptr + off2
                val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_ld_out_for_iee_d4(
    in_ptr,
    out_ptr,
    branch_id,
    n_outer,
    n_inner,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
    off2: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n_outer):
        if branch_id == 0:
            for _ in range(n_inner):
                a_ptr = a_ptr + off0
        elif branch_id == 1:
            for _ in range(n_inner):
                a_ptr = a_ptr + off1
        else:
            for _ in range(n_inner):
                a_ptr = a_ptr + off2
        val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_adv_ld_in_wh_ie_d4(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    i = 0
    while i < n_outer:
        for _ in range(n_inner):
            if flag != 0:
                bp = tl.advance(bp, (adv_if, ))
                acc += tl.load(bp)
            else:
                bp = tl.advance(bp, (adv_else, ))
                acc += tl.load(bp)
        i = i + 1
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_ld_out_wh_ie_d4(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    i = 0
    while i < n_outer:
        for _ in range(n_inner):
            if flag != 0:
                bp = tl.advance(bp, (adv_if, ))
            else:
                bp = tl.advance(bp, (adv_else, ))
            acc += tl.load(bp)
        i = i + 1
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_ld_in_for_wh_ie_d4(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for _ in range(n_outer):
        i = 0
        while i < n_inner:
            if flag != 0:
                bp = tl.advance(bp, (adv_if, ))
                acc += tl.load(bp)
            else:
                bp = tl.advance(bp, (adv_else, ))
                acc += tl.load(bp)
            i = i + 1
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_ld_out_for_wh_ie_d4(
    in_ptr,
    out_ptr,
    flag,
    n_outer,
    n_inner,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for _ in range(n_outer):
        i = 0
        while i < n_inner:
            if flag != 0:
                bp = tl.advance(bp, (adv_if, ))
            else:
                bp = tl.advance(bp, (adv_else, ))
            acc += tl.load(bp)
            i = i + 1
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_ap_ld_in_for3(
    in_ptr,
    out_ptr,
    n0,
    n1,
    n2,
    block: tl.constexpr,
    off: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n0):
        for _ in range(n1):
            for _ in range(n2):
                a_ptr = a_ptr + off
                val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_ap_ld_out_for3(
    in_ptr,
    out_ptr,
    n0,
    n1,
    n2,
    block: tl.constexpr,
    off: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    val = tl.load(a_ptr)
    for _ in range(n0):
        for _ in range(n1):
            for _ in range(n2):
                a_ptr = a_ptr + off
                val = tl.load(a_ptr)
    tl.store(out_ptr + offs, val)


@triton.jit
def kernel_adv_ld_in_for3(
    in_ptr,
    out_ptr,
    n0,
    n1,
    n2,
    n_elements,
    block: tl.constexpr,
    adv: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for _ in range(n0):
        for _ in range(n1):
            for _ in range(n2):
                bp = tl.advance(bp, (adv, ))
                acc += tl.load(bp)
    tl.store(out_ptr + tl.arange(0, block), acc)


@triton.jit
def kernel_adv_ld_out_for3(
    in_ptr,
    out_ptr,
    n0,
    n1,
    n2,
    n_elements,
    block: tl.constexpr,
    adv: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for _ in range(n0):
        for _ in range(n1):
            for _ in range(n2):
                bp = tl.advance(bp, (adv, ))
                acc += tl.load(bp)
    tl.store(out_ptr + tl.arange(0, block), acc)


def reference_ap_ld_nested(x, flag, n_outer, n_inner, off0, off1, block):
    flat = x.flatten()
    idx = torch.arange(block, device=x.device, dtype=torch.long)
    delta = _branch_delta(flag, off0, off1)
    for _ in range(n_outer):
        for _ in range(n_inner):
            idx = (idx + delta).clamp(0, flat.numel() - 1)
    return flat[idx]


def reference_ap_ld_outer_for(x, flag, n_outer, n_inner, off0, off1, block):
    flat = x.flatten()
    idx = torch.arange(block, device=x.device, dtype=torch.long)
    for _ in range(n_outer):
        delta = _branch_delta(flag, off0, off1)
        for _ in range(n_inner):
            idx = (idx + delta).clamp(0, flat.numel() - 1)
    return flat[idx]


def reference_ap_ld_for3(x, n0, n1, n2, off, block):
    flat = x.flatten()
    idx = torch.arange(block, device=x.device, dtype=torch.long)
    for _ in range(n0):
        for _ in range(n1):
            for _ in range(n2):
                idx = (idx + off).clamp(0, flat.numel() - 1)
    return flat[idx]


def reference_adv_ld_wh_ie_d3(x, flag, n_outer, n_inner, adv_if, adv_else, block):
    adv = adv_if if flag != 0 else adv_else
    return _ref_adv_ld_accumulate(
        x,
        block,
        lambda: adv,
        n_outer * n_inner,
    )


def reference_adv_ld_for_wh_ie_d3(x, flag, n_outer, n_inner, adv_if, adv_else, block):
    adv = adv_if if flag != 0 else adv_else
    flat = x.flatten()
    pos = 0
    acc = torch.zeros(block, dtype=torch.float32, device=x.device)
    i = 0
    while i < n_outer:
        for _ in range(n_inner):
            pos += adv
            acc = _ref_fp32_chunk_add(acc, flat, pos, block)
        i += 1
    return acc.to(x.dtype)


def reference_ap_st_outer_for(x, flag, n_outer, n_inner, off0, off1, block):
    flat = x.flatten().clone()
    idx = torch.arange(block, device=x.device, dtype=torch.long)
    src = _ref_src_tile(flat, block)
    delta = _branch_delta(flag, off0, off1)
    for _ in range(n_outer):
        for _ in range(n_inner):
            idx = (idx + delta).clamp(0, flat.numel() - 1)
            flat[idx] = src
    return flat[idx]


def reference_ap_st_iee_for_d3(x, branch_id, n_outer, n_inner, block):
    flat = x.flatten().clone()
    return _ref_st_scatter_carry(
        flat,
        block,
        _iel_delta(branch_id),
        n_outer * n_inner,
    )


def reference_adv_st_wh_for_iee_d3(x, branch_id, n_outer, n_inner, block):
    flat = x.flatten().clone()
    idx = torch.arange(block, device=x.device, dtype=torch.long)
    src = _ref_src_tile(flat, block)
    adv = _iel_adv(branch_id)
    i = 0
    while i < n_outer:
        for _ in range(n_inner):
            idx = (idx + adv).clamp(0, flat.numel() - 1)
            flat[idx] = src
        i += 1
    return flat[idx]


def reference_adv_st_for_wh_iee_d3(x, branch_id, n_outer, n_inner, block):
    flat = x.flatten().clone()
    return _ref_st_scatter_carry(
        flat,
        block,
        _iel_adv(branch_id),
        n_outer * n_inner,
    )


def reference_ap_ld_iee_nested_d4(x, branch_id, n_outer, n_inner, block):
    flat = x.flatten()
    idx = torch.arange(block, device=x.device, dtype=torch.long)
    delta = _iel_delta(branch_id)
    for _ in range(n_outer):
        for _ in range(n_inner):
            idx = (idx + delta).clamp(0, flat.numel() - 1)
    return flat[idx]


def reference_ap_ld_iee_outer_for_d4(x, branch_id, n_outer, n_inner, block):
    flat = x.flatten()
    idx = torch.arange(block, device=x.device, dtype=torch.long)
    delta = _iel_delta(branch_id)
    for _ in range(n_outer):
        for _ in range(n_inner):
            idx = (idx + delta).clamp(0, flat.numel() - 1)
    return flat[idx]


def reference_adv_ld_for3(x, n0, n1, n2, adv, block):
    return _ref_adv_ld_accumulate(x, block, lambda: adv, n0 * n1 * n2)


# ---------------------------------------------------------------------------
# atomic 代表 (TC 085–086), IV 仿射 (087–088), different-base (089)
# ---------------------------------------------------------------------------


@triton.jit
def kernel_adv_at_in_ie(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    n_elements,
    block: tl.constexpr,
    adv_if: tl.constexpr,
    adv_else: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    src = tl.load(in_ptr + tl.arange(0, block))
    for _ in range(n_loops):
        if flag != 0:
            bp = tl.advance(bp, (adv_if, ))
            tl.atomic_add(bp, src)
        else:
            bp = tl.advance(bp, (adv_else, ))
            tl.atomic_add(bp, src)
    tl.store(out_ptr + tl.arange(0, block), tl.load(bp))


@triton.jit
def kernel_ap_at_in_ie(
    in_ptr,
    out_ptr,
    flag,
    n_loops,
    block: tl.constexpr,
    off0: tl.constexpr,
    off1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    src = tl.load(in_ptr + offs)
    for _ in range(n_loops):
        if flag != 0:
            a_ptr = a_ptr + off0
            tl.atomic_add(a_ptr, src)
        else:
            a_ptr = a_ptr + off1
            tl.atomic_add(a_ptr, src)
    tl.store(out_ptr + offs, tl.load(a_ptr))


@triton.jit
def kernel_ap_ld_in_for2_affine(
    in_ptr,
    out_ptr,
    n_outer,
    n_inner,
    block: tl.constexpr,
    stride0: tl.constexpr,
    stride1: tl.constexpr,
):
    offs = tl.arange(0, block)
    a_ptr = in_ptr + offs
    acc = tl.zeros([block], tl.float32)
    for i in range(n_outer):
        for j in range(n_inner):
            a_ptr = a_ptr + i * stride0
            a_ptr = a_ptr + j * stride1
            acc += tl.load(a_ptr)
    tl.store(out_ptr + offs, acc)


@triton.jit
def kernel_adv_ld_in_for2_affine(
    in_ptr,
    out_ptr,
    m_size,
    n_size,
    n_outer,
    n_inner,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(m_size, n_size),
        strides=(n_size, 1),
        offsets=(0, 0),
        block_shape=(block_m, block_n),
        order=(1, 0),
    )
    acc = tl.zeros([block_m, block_n], tl.float32)
    for i in range(n_outer):
        for j in range(n_inner):
            bp = tl.advance(bp, (i, j))
            acc += tl.load(bp)
    out_bp = tl.make_block_ptr(
        base=out_ptr,
        shape=(block_m, block_n),
        strides=(block_n, 1),
        offsets=(0, 0),
        block_shape=(block_m, block_n),
        order=(1, 0),
    )
    tl.store(out_bp, acc.to(out_bp.type.element_ty))


@triton.jit
def kernel_cf_diff_base(
    in_a_ptr,
    in_b_ptr,
    out_ptr,
    flag,
    block: tl.constexpr,
):
    offs = tl.arange(0, block)
    val = tl.load(in_a_ptr + offs)
    if flag != 0:
        val = tl.load(in_a_ptr + offs)
    else:
        val = tl.load(in_b_ptr + offs)
    tl.store(out_ptr + offs, val)


def _total_affine_span(n_outer, n_inner, stride0, stride1):
    total = 0
    for i in range(n_outer):
        for j in range(n_inner):
            total += i * stride0 + j * stride1
    return total


def reference_ap_ld_in_for2_affine(x, n_outer, n_inner, block, stride0, stride1):
    flat = x.flatten()
    pos = 0
    acc = torch.zeros(block, dtype=torch.float32, device=x.device)
    for i in range(n_outer):
        for j in range(n_inner):
            pos += i * stride0 + j * stride1
            acc = acc + flat[pos:pos + block].float()
    return acc.to(x.dtype)


def reference_adv_ld_in_for2_affine(x, n_outer, n_inner, block_m, block_n):
    row, col = 0, 0
    acc = torch.zeros((block_m, block_n), dtype=torch.float32, device=x.device)
    for i in range(n_outer):
        for j in range(n_inner):
            row += i
            col += j
            acc = acc + x[row:row + block_m, col:col + block_n].float()
    return acc.to(x.dtype)


def _affine_2d_shape(n_outer, n_inner, block_m, block_n):
    row_span = sum(i for i in range(n_outer) for _ in range(n_inner))
    col_span = sum(j for _ in range(n_outer) for j in range(n_inner))
    return block_m + row_span, block_n + col_span


# ---------------------------------------------------------------------------
# Batch-1 load 主门禁 (001–006, 013–018)
# ---------------------------------------------------------------------------

_AP_LD_CASES = [
    ("001", kernel_ap_ld_in_ie, True),
    ("002", kernel_ap_ld_out_ie, True),
    ("003", kernel_ap_ld_in_for, False),
    ("004", kernel_ap_ld_out_for, False),
    ("005", kernel_ap_ld_in_wh, False),
    ("006", kernel_ap_ld_out_wh, False),
]

for _tc_id, _kernel, _outer_for in _AP_LD_CASES:

    def _make_ap_ld_test(tcid, kernel, outer_for):

        @pytest.mark.parametrize("dtype", DTYPES_E2E)
        @pytest.mark.parametrize("flag", [1, 0])
        def _test(dtype, flag):
            _require_npu()
            n_elements = _n_elements_for_loops(BLOCK, N_LOOPS, OFF0, OFF1)
            x = torch.randn(n_elements, dtype=dtype, device=DEVICE)
            actual = _run_ap_ld(kernel, x, flag, N_LOOPS, OFF0, OFF1, BLOCK)
            expected = reference_ap_ld_loop(
                x,
                flag,
                N_LOOPS,
                OFF0,
                OFF1,
                BLOCK,
                outer_for,
            )
            compare_results(actual, expected, dtype)

        _test.__name__ = f"test_spec_ccf_{tcid}_ap_ld"
        _test.__doc__ = f"TC_CCF_{tcid}: addptr + load (D5=2 主门禁)"
        return _test

    globals()[f"test_spec_ccf_{_tc_id}_ap_ld"] = _make_ap_ld_test(
        _tc_id,
        _kernel,
        _outer_for,
    )


@pytest.mark.parametrize("dtype", DTYPES_E2E)
@pytest.mark.parametrize("flag", [1, 0])
def test_spec_ccf_013_adv_ld_in_ie(dtype, flag):
    """TC_CCF_013: advance + load + 内 + if-else + D5=2"""
    _require_npu()
    x = torch.randn(
        _n_elements_for_loops(BLOCK, N_LOOPS, ADV_IF, ADV_ELSE),
        dtype=dtype,
        device=DEVICE,
    )
    actual = _run_adv_ld_ie(kernel_adv_ld_in_ie, x, flag, N_LOOPS, ADV_IF, ADV_ELSE, BLOCK)
    expected = reference_adv_ld_ie(x, flag, N_LOOPS, ADV_IF, ADV_ELSE, BLOCK)
    compare_results(actual, expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES_E2E)
@pytest.mark.parametrize("flag", [1, 0])
def test_spec_ccf_014_adv_ld_out_ie(dtype, flag):
    """TC_CCF_014: advance + load + 外 + if-else + D5=2"""
    _require_npu()
    x = torch.randn(
        _n_elements_for_loops(BLOCK, N_LOOPS, ADV_IF, ADV_ELSE),
        dtype=dtype,
        device=DEVICE,
    )
    actual = _run_adv_ld_ie(
        kernel_adv_ld_out_ie,
        x,
        flag,
        N_LOOPS,
        ADV_IF,
        ADV_ELSE,
        BLOCK,
    )
    expected = reference_adv_ld_ie(x, flag, N_LOOPS, ADV_IF, ADV_ELSE, BLOCK)
    compare_results(actual, expected, dtype)


# ---------------------------------------------------------------------------
# Batch-1b store 全组合 (007–012)
# ---------------------------------------------------------------------------

_AP_ST_CASES = [
    ("007", kernel_ap_st_in_ie),
    ("008", kernel_ap_st_out_ie),
    ("009", kernel_ap_st_in_for),
    ("010", kernel_ap_st_out_for),
    ("011", kernel_ap_st_in_wh),
    ("012", kernel_ap_st_out_wh),
]

for _tc_id, _kernel in _AP_ST_CASES:

    def _make_ap_st_test(tcid, kernel):

        @pytest.mark.parametrize("dtype", DTYPES_E2E)
        @pytest.mark.parametrize("flag", [1, 0])
        def _test(dtype, flag):
            _require_npu()
            n_elements = _n_elements_for_loops(BLOCK, N_LOOPS, OFF0, OFF1)
            x = torch.randn(n_elements, dtype=dtype, device=DEVICE)
            out = torch.empty(BLOCK, dtype=dtype, device=DEVICE)
            expected = reference_ap_st_ie(x, flag, N_LOOPS, OFF0, OFF1, BLOCK)
            kernel[(1, )](x, out, flag, N_LOOPS, BLOCK, OFF0, OFF1)
            compare_results(out, expected, dtype)

        _test.__name__ = f"test_spec_ccf_{tcid}_ap_st"
        _test.__doc__ = f"TC_CCF_{tcid}: addptr + store (D5=2 补充)"
        return _test

    globals()[f"test_spec_ccf_{_tc_id}_ap_st"] = _make_ap_st_test(
        _tc_id,
        _kernel,
    )

# ---------------------------------------------------------------------------
# Batch-1b advance store 全组合 (019–024)
# ---------------------------------------------------------------------------

_ADV_ST_CASES = [
    ("019", kernel_adv_st_in_ie),
    ("020", kernel_adv_st_out_ie),
    ("021", kernel_adv_st_in_for),
    ("022", kernel_adv_st_out_for),
    ("023", kernel_adv_st_in_wh),
    ("024", kernel_adv_st_out_wh),
]

for _tc_id, _kernel in _ADV_ST_CASES:

    def _make_adv_st_test(tcid, kernel):

        @pytest.mark.parametrize("dtype", DTYPES_E2E)
        @pytest.mark.parametrize("flag", [1, 0])
        def _test(dtype, flag):
            _require_npu()
            x = torch.randn(
                _n_elements_for_loops(BLOCK, N_LOOPS, ADV_IF, ADV_ELSE),
                dtype=dtype,
                device=DEVICE,
            )
            expected = reference_adv_st_ie(
                x,
                flag,
                N_LOOPS,
                ADV_IF,
                ADV_ELSE,
                BLOCK,
            )
            actual = _run_adv_st_ie(
                kernel,
                x,
                flag,
                N_LOOPS,
                ADV_IF,
                ADV_ELSE,
                BLOCK,
            )
            compare_results(actual, expected, dtype)

        _test.__name__ = f"test_spec_ccf_{tcid}_adv_st"
        _test.__doc__ = f"TC_CCF_{tcid}: advance + store (D5=2 补充)"
        return _test

    globals()[f"test_spec_ccf_{_tc_id}_adv_st"] = _make_adv_st_test(
        _tc_id,
        _kernel,
    )

# ---------------------------------------------------------------------------
# Batch-1 advance load 015–018
# ---------------------------------------------------------------------------

_ADV_LD_BRANCH_CASES = [
    ("015", kernel_adv_ld_in_for),
    ("016", kernel_adv_ld_out_for),
    ("017", kernel_adv_ld_in_wh),
    ("018", kernel_adv_ld_out_wh),
]

for _tc_id, _kernel in _ADV_LD_BRANCH_CASES:

    def _make_adv_ld_branch_test(tcid, kernel):

        @pytest.mark.parametrize("dtype", DTYPES_E2E)
        @pytest.mark.parametrize("flag", [1, 0])
        def _test(dtype, flag):
            _require_npu()
            x = torch.randn(
                _n_elements_for_loops(BLOCK, N_LOOPS, ADV_IF, ADV_ELSE),
                dtype=dtype,
                device=DEVICE,
            )
            actual = _run_adv_ld_branch(
                kernel,
                x,
                flag,
                N_LOOPS,
                ADV_IF,
                ADV_ELSE,
                BLOCK,
            )
            expected = reference_adv_ld_branch(
                x,
                flag,
                N_LOOPS,
                ADV_IF,
                ADV_ELSE,
                BLOCK,
            )
            compare_results(actual, expected, dtype)

        _test.__name__ = f"test_spec_ccf_{tcid}_adv_ld"
        _test.__doc__ = f"TC_CCF_{tcid}: advance + load (D5=2 主门禁)"
        return _test

    globals()[f"test_spec_ccf_{_tc_id}_adv_ld"] = _make_adv_ld_branch_test(
        _tc_id,
        _kernel,
    )

# ---------------------------------------------------------------------------
# Batch-1c if-elif-else 扩展 (025–032)
# ---------------------------------------------------------------------------

_IEL_CASES = [
    ("025", "ap_ld_in", kernel_ap_ld_in_iee, "ld"),
    ("026", "ap_ld_out", kernel_ap_ld_out_iee, "ld"),
    ("027", "ap_st_in", kernel_ap_st_in_iee, "st"),
    ("028", "ap_st_out", kernel_ap_st_out_iee, "st"),
    ("029", "adv_ld_in", kernel_adv_ld_in_iee, "adv"),
    ("030", "adv_ld_out", kernel_adv_ld_out_iee, "adv"),
    ("031", "adv_st_in", kernel_adv_st_in_iee, "adv_st"),
    ("032", "adv_st_out", kernel_adv_st_out_iee, "adv_st"),
]

for _tc_id, _suffix, _kernel, _kind in _IEL_CASES:

    def _make_iel_test(tcid, suffix, kernel, kind):

        @pytest.mark.parametrize("dtype", DTYPES_E2E)
        @pytest.mark.parametrize("branch_id", [0, 1, 2])
        def _test(dtype, branch_id):
            _require_npu()
            n_elements = _n_elements_for_loops(BLOCK, N_LOOPS, OFF0, OFF2)
            x = torch.randn(n_elements, dtype=dtype, device=DEVICE)
            out = torch.empty(BLOCK, dtype=dtype, device=DEVICE)
            if kind == "adv":
                expected = reference_adv_ld_iee(x, branch_id, N_LOOPS, BLOCK)
            elif kind == "adv_st":
                expected = reference_adv_st_iee(x, branch_id, N_LOOPS, BLOCK)
            elif kind == "st":
                expected = reference_ap_st_iee(x, branch_id, N_LOOPS, BLOCK)
            else:
                expected = reference_ap_ld_iee(x, branch_id, N_LOOPS, BLOCK)
            if kind in ("adv", "adv_st"):
                kernel[(1, )](
                    x,
                    out,
                    branch_id,
                    N_LOOPS,
                    n_elements,
                    BLOCK,
                    ADV_IEL_IF,
                    ADV_IEL_ELIF,
                    ADV_IEL_ELSE,
                )
            else:
                kernel[(1, )](
                    x,
                    out,
                    branch_id,
                    N_LOOPS,
                    BLOCK,
                    OFF0,
                    OFF1,
                    OFF2,
                )
            compare_results(out, expected, dtype)

        _test.__name__ = f"test_spec_ccf_{tcid}_{suffix}_iee"
        _test.__doc__ = f"TC_CCF_{tcid}: if-elif-else 扩展"
        return _test

    globals()[f"test_spec_ccf_{_tc_id}_{_suffix}_iee"] = _make_iel_test(
        _tc_id,
        _suffix,
        _kernel,
        _kind,
    )

# ---------------------------------------------------------------------------
# Batch-2 D5=3 load 扩展 (049–056, 081–084)
# ---------------------------------------------------------------------------

_EXT_LOAD_D3 = [
    ("049", kernel_ap_ld_in_ie_d3, reference_ap_ld_nested),
    ("050", kernel_ap_ld_out_ie_d3, reference_ap_ld_nested),
    ("051", kernel_ap_ld_in_for_d3, reference_ap_ld_outer_for),
    ("052", kernel_ap_ld_out_for_d3, reference_ap_ld_outer_for),
    ("053", kernel_adv_ld_in_wh_ie_d3, reference_adv_ld_wh_ie_d3),
    ("054", kernel_adv_ld_out_wh_ie_d3, reference_adv_ld_wh_ie_d3),
    ("055", kernel_adv_ld_in_for_wh_ie_d3, reference_adv_ld_for_wh_ie_d3),
    ("056", kernel_adv_ld_out_for_wh_ie_d3, reference_adv_ld_for_wh_ie_d3),
    ("081", kernel_ap_ld_in_for3, reference_ap_ld_for3),
    ("082", kernel_ap_ld_out_for3, reference_ap_ld_for3),
    ("083", kernel_adv_ld_in_for3, reference_adv_ld_for3),
    ("084", kernel_adv_ld_out_for3, reference_adv_ld_for3),
]

for _tc_id, _kernel, _ref_fn in _EXT_LOAD_D3:

    def _make_ext_load_d3_test(tcid, kernel, ref_fn):

        @pytest.mark.parametrize("dtype", DTYPES_E2E)
        @pytest.mark.parametrize("flag", [1, 0])
        def _test(dtype, flag):
            _require_npu()
            if tcid in ("081", "082"):
                n0, n1, n2, off = 2, 2, 2, 1
                n_elements = BLOCK + n0 * n1 * n2 * off + 4
                x = torch.randn(n_elements, dtype=dtype, device=DEVICE)
                out = torch.empty(BLOCK, dtype=dtype, device=DEVICE)
                kernel[(1, )](x, out, n0, n1, n2, BLOCK, off)
                expected = ref_fn(x, n0, n1, n2, off, BLOCK)
            elif tcid in ("083", "084"):
                n0, n1, n2, adv = 2, 2, 2, 1
                n_elements = _n_elements_for_loops(BLOCK, n0 * n1 * n2, adv, adv)
                x = torch.randn(n_elements, dtype=dtype, device=DEVICE)
                out = torch.empty(BLOCK, dtype=dtype, device=DEVICE)
                kernel[(1, )](x, out, n0, n1, n2, n_elements, BLOCK, adv)
                expected = ref_fn(x, n0, n1, n2, adv, BLOCK)
            elif tcid.startswith("05") and int(tcid) >= 53:
                n_elements = _n_elements_for_loops(
                    BLOCK,
                    N_OUTER * N_INNER,
                    ADV_IF,
                    ADV_ELSE,
                )
                x = torch.randn(n_elements, dtype=dtype, device=DEVICE)
                out = torch.empty(BLOCK, dtype=dtype, device=DEVICE)
                kernel[(1, )](
                    x,
                    out,
                    flag,
                    N_OUTER,
                    N_INNER,
                    n_elements,
                    BLOCK,
                    ADV_IF,
                    ADV_ELSE,
                )
                expected = ref_fn(x, flag, N_OUTER, N_INNER, ADV_IF, ADV_ELSE, BLOCK)
            else:
                n_elements = _n_elements_for_loops(
                    BLOCK,
                    N_OUTER * N_INNER,
                    OFF0,
                    OFF1,
                )
                x = torch.randn(n_elements, dtype=dtype, device=DEVICE)
                out = torch.empty(BLOCK, dtype=dtype, device=DEVICE)
                if tcid in ("051", "052"):
                    kernel[(1, )](x, out, flag, N_OUTER, N_INNER, BLOCK, OFF0, OFF1)
                    expected = ref_fn(x, flag, N_OUTER, N_INNER, OFF0, OFF1, BLOCK)
                else:
                    kernel[(1, )](x, out, flag, N_OUTER, N_INNER, BLOCK, OFF0, OFF1)
                    expected = ref_fn(x, flag, N_OUTER, N_INNER, OFF0, OFF1, BLOCK)
            compare_results(out, expected, dtype)

        _test.__name__ = f"test_spec_ccf_{tcid}_ext_load_d3"
        _test.__doc__ = f"TC_CCF_{tcid}: D5=3 load 扩展"
        return _test

    globals()[f"test_spec_ccf_{_tc_id}_ext_load_d3"] = _make_ext_load_d3_test(
        _tc_id,
        _kernel,
        _ref_fn,
    )

# ---------------------------------------------------------------------------
# Batch-3 D5=3 store 扩展 (057–064)
# ---------------------------------------------------------------------------

_EXT_STORE_D3 = [
    ("057", kernel_ap_st_in_ie_for_d3, "ie", reference_ap_st_outer_for),
    ("058", kernel_ap_st_out_ie_for_d3, "ie", reference_ap_st_outer_for),
    ("059", kernel_ap_st_in_iee_for_d3, "iee", reference_ap_st_iee_for_d3),
    ("060", kernel_ap_st_out_iee_for_d3, "iee", reference_ap_st_iee_for_d3),
    ("061", kernel_adv_st_in_wh_for_iee_d3, "adv_wh", reference_adv_st_wh_for_iee_d3),
    ("062", kernel_adv_st_out_wh_for_iee_d3, "adv_wh", reference_adv_st_wh_for_iee_d3),
    ("063", kernel_adv_st_in_for_wh_iee_d3, "adv_fw", reference_adv_st_for_wh_iee_d3),
    ("064", kernel_adv_st_out_for_wh_iee_d3, "adv_fw", reference_adv_st_for_wh_iee_d3),
]

for _tc_id, _kernel, _store_kind, _ref_fn in _EXT_STORE_D3:

    def _make_ext_store_d3_test(tcid, kernel, store_kind, ref_fn):

        @pytest.mark.parametrize("dtype", DTYPES_E2E)
        def _test(dtype):
            _require_npu()
            n_elements = _n_elements_for_loops(
                BLOCK,
                N_OUTER * N_INNER,
                OFF0,
                OFF2,
            )
            x = torch.randn(n_elements, dtype=dtype, device=DEVICE)
            out = torch.empty(BLOCK, dtype=dtype, device=DEVICE)
            if store_kind == "ie":
                for flag in (1, 0):
                    out.zero_()
                    x_copy = x.clone()
                    kernel[(1, )](x_copy, out, flag, N_OUTER, N_INNER, BLOCK, OFF0, OFF1)
                    expected = ref_fn(x, flag, N_OUTER, N_INNER, OFF0, OFF1, BLOCK)
                    compare_results(out, expected, dtype)
            elif store_kind == "iee":
                for branch_id in (0, 1, 2):
                    out.zero_()
                    x_copy = x.clone()
                    kernel[(1, )](
                        x_copy,
                        out,
                        branch_id,
                        N_OUTER,
                        N_INNER,
                        BLOCK,
                        OFF0,
                        OFF1,
                        OFF2,
                    )
                    expected = ref_fn(x, branch_id, N_OUTER, N_INNER, BLOCK)
                    compare_results(out, expected, dtype)
            else:
                for branch_id in (0, 1, 2):
                    out.zero_()
                    x_copy = x.clone()
                    kernel[(1, )](
                        x_copy,
                        out,
                        branch_id,
                        N_OUTER,
                        N_INNER,
                        n_elements,
                        BLOCK,
                        ADV_IEL_IF,
                        ADV_IEL_ELIF,
                        ADV_IEL_ELSE,
                    )
                    expected = ref_fn(x, branch_id, N_OUTER, N_INNER, BLOCK)
                    compare_results(out, expected, dtype)

        _test.__name__ = f"test_spec_ccf_{tcid}_ext_store_d3"
        _test.__doc__ = f"TC_CCF_{tcid}: D5=3 store 扩展"
        return _test

    globals()[f"test_spec_ccf_{_tc_id}_ext_store_d3"] = _make_ext_store_d3_test(
        _tc_id,
        _kernel,
        _store_kind,
        _ref_fn,
    )

# ---------------------------------------------------------------------------
# Batch-4 D5=4 load 扩展 (065–072)
# ---------------------------------------------------------------------------

_EXT_LOAD_D4 = [
    ("065", kernel_ap_ld_in_iee_d4, reference_ap_ld_iee_nested_d4),
    ("066", kernel_ap_ld_out_iee_d4, reference_ap_ld_iee_nested_d4),
    ("067", kernel_ap_ld_in_for_iee_d4, reference_ap_ld_iee_outer_for_d4),
    ("068", kernel_ap_ld_out_for_iee_d4, reference_ap_ld_iee_outer_for_d4),
    ("069", kernel_adv_ld_in_wh_ie_d4, reference_adv_ld_wh_ie_d3),
    ("070", kernel_adv_ld_out_wh_ie_d4, reference_adv_ld_wh_ie_d3),
    ("071", kernel_adv_ld_in_for_wh_ie_d4, reference_adv_ld_wh_ie_d3),
    ("072", kernel_adv_ld_out_for_wh_ie_d4, reference_adv_ld_wh_ie_d3),
]

for _tc_id, _kernel, _ref_fn in _EXT_LOAD_D4:

    def _make_ext_load_d4_test(tcid, kernel, ref_fn):

        @pytest.mark.parametrize("dtype", DTYPES_E2E)
        def _test(dtype):
            _require_npu()
            n_elements = _n_elements_for_loops(
                BLOCK,
                N_OUTER * N_INNER,
                OFF0,
                OFF2,
            )
            x = torch.randn(n_elements, dtype=dtype, device=DEVICE)
            out = torch.empty(BLOCK, dtype=dtype, device=DEVICE)
            if int(tcid) < 69:
                for branch_id in (0, 1, 2):
                    kernel[(1, )](
                        x,
                        out,
                        branch_id,
                        N_OUTER,
                        N_INNER,
                        BLOCK,
                        OFF0,
                        OFF1,
                        OFF2,
                    )
                    expected = ref_fn(x, branch_id, N_OUTER, N_INNER, BLOCK)
                    compare_results(out, expected, dtype)
            else:
                for flag in (1, 0):
                    kernel[(1, )](
                        x,
                        out,
                        flag,
                        N_OUTER,
                        N_INNER,
                        n_elements,
                        BLOCK,
                        ADV_IF,
                        ADV_ELSE,
                    )
                    expected = ref_fn(
                        x,
                        flag,
                        N_OUTER,
                        N_INNER,
                        ADV_IF,
                        ADV_ELSE,
                        BLOCK,
                    )
                    compare_results(out, expected, dtype)

        _test.__name__ = f"test_spec_ccf_{tcid}_ext_load_d4"
        _test.__doc__ = f"TC_CCF_{tcid}: D5=4 load 扩展"
        return _test

    globals()[f"test_spec_ccf_{_tc_id}_ext_load_d4"] = _make_ext_load_d4_test(
        _tc_id,
        _kernel,
        _ref_fn,
    )

# ---------------------------------------------------------------------------
# Batch-5 atomic / Batch-2b IV 仿射 / Batch-6 different-base
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES_E2E)
@pytest.mark.parametrize("flag", [1, 0])
def test_spec_ccf_085_ap_at_in_ie(dtype, flag):
    """TC_CCF_085: addptr + atomic numerical result."""
    _require_npu()
    n_elements = _n_elements_for_loops(BLOCK, N_LOOPS, OFF0, OFF1)
    x = torch.randn(n_elements, dtype=dtype, device=DEVICE)
    out = torch.empty(BLOCK, dtype=dtype, device=DEVICE)
    step = _branch_delta(flag, OFF0, OFF1)
    expected = _ref_atomic_add_carry(x, BLOCK, step, N_LOOPS)
    kernel_ap_at_in_ie[(1, )](x, out, flag, N_LOOPS, BLOCK, OFF0, OFF1)
    compare_results(out, expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES_E2E)
@pytest.mark.parametrize("flag", [1, 0])
def test_spec_ccf_086_adv_at_in_ie(dtype, flag):
    """TC_CCF_086: advance + atomic numerical result."""
    _require_npu()
    n_elements = _n_elements_for_loops(BLOCK, N_LOOPS, ADV_IF, ADV_ELSE)
    x = torch.randn(n_elements, dtype=dtype, device=DEVICE)
    out = torch.empty(BLOCK, dtype=dtype, device=DEVICE)
    step = _branch_delta(flag, ADV_IF, ADV_ELSE)
    expected = _ref_atomic_add_carry(x, BLOCK, step, N_LOOPS)
    kernel_adv_at_in_ie[(1, )](
        x,
        out,
        flag,
        N_LOOPS,
        n_elements,
        BLOCK,
        ADV_IF,
        ADV_ELSE,
    )
    compare_results(out, expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES_E2E)
def test_spec_ccf_087_ap_ld_in_for2_affine(dtype):
    """TC_CCF_087: IV 仿射 addptr (7.5.2)"""
    _require_npu()
    block = 16
    n_outer, n_inner = N_OUTER, N_INNER
    n_elements = block + _total_affine_span(n_outer, n_inner, STRIDE0, STRIDE1)
    x = torch.randn(n_elements, dtype=dtype, device=DEVICE)
    out = torch.empty(block, dtype=dtype, device=DEVICE)
    kernel_ap_ld_in_for2_affine[(1, )](
        x,
        out,
        n_outer,
        n_inner,
        block,
        STRIDE0,
        STRIDE1,
    )
    expected = reference_ap_ld_in_for2_affine(
        x,
        n_outer,
        n_inner,
        block,
        STRIDE0,
        STRIDE1,
    )
    compare_results(out, expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES_E2E)
def test_spec_ccf_088_adv_ld_in_for2_affine(dtype):
    """TC_CCF_088: IV 仿射 advance (7.4.2)"""
    _require_npu()
    block_m, block_n = 2, 8
    m_size, n_size = _affine_2d_shape(N_OUTER, N_INNER, block_m, block_n)
    x = torch.randn((m_size, n_size), dtype=dtype, device=DEVICE)
    out = torch.empty((block_m, block_n), dtype=dtype, device=DEVICE)
    kernel_adv_ld_in_for2_affine[(1, )](
        x,
        out,
        m_size,
        n_size,
        N_OUTER,
        N_INNER,
        block_m,
        block_n,
    )
    expected = reference_adv_ld_in_for2_affine(x, N_OUTER, N_INNER, block_m, block_n)
    compare_results(out, expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES_E2E)
@pytest.mark.parametrize("flag", [1, 0])
def test_spec_ccf_089_cf_diff_base(dtype, flag):
    """TC_CCF_089: control flow selects between different pointer bases."""
    _require_npu()
    a = torch.randn(BLOCK, dtype=dtype, device=DEVICE)
    b = torch.randn(BLOCK, dtype=dtype, device=DEVICE)
    out = torch.empty(BLOCK, dtype=dtype, device=DEVICE)
    kernel_cf_diff_base[(1, )](a, b, out, flag, BLOCK)
    expected = a if flag != 0 else b
    compare_results(out, expected, dtype)
