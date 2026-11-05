# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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
"""
Tensor pointer demotion - pytest version.

Covers the pre-memory-stage demotion of int-derived pointer tensors to i64
address tensors (see tensor_pointer_demotion_implementation.md in the repo
root): tensor-form tt.int_to_ptr / tt.ptr_to_int, pointer select, pointer
arithmetic, structure ops, bitcast, control flow, view-chain address
materialization, and the upstream community case test_where.

The reference values are computed on the host from the *same* address values:
a pointer tensor is data-wise nothing but the i64 addresses it carries, so
every value-level operation on it reduces to plain int64 arithmetic.
"""

import numpy as np
import torch
import torch_npu
from numpy.random import RandomState

import triton
import triton.language as tl
from triton._internal_testing import numpy_random, to_triton

BLOCK = 1024


def _addresses(base, count=BLOCK, stride=16):
    """Host-known int64 address values derived from a real device pointer."""
    return torch.from_numpy(base + np.arange(count, dtype=np.int64) * stride).npu()


def _i64_offsets(count=BLOCK):
    return torch.from_numpy(np.arange(count, dtype=np.int64) % 7).npu()


def _cond_i8(count=BLOCK):
    rs = RandomState(17)
    return torch.from_numpy(numpy_random(count, "int8", rs) % 2).npu()


def _perm_addresses(base, count=BLOCK, seed=3):
    """Distinct element indices (permutation) -> addresses inside a buffer."""
    perm = np.random.RandomState(seed).permutation(count)
    return perm, torch.from_numpy(base + perm.astype(np.int64) * 4).npu()


# ---------------------------------------------------------------------------
# 1. Community case: test_where with TEST_POINTERS (tensor int_to_ptr path)
# ---------------------------------------------------------------------------
@triton.jit
def where_pointers_kernel(cond_ptr, a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    decide = tl.load(cond_ptr + offsets, mask=mask)
    a = tl.load(a_ptr + offsets, mask=mask).to(tl.pi32_t)
    b = tl.load(b_ptr + offsets, mask=mask).to(tl.pi32_t)
    output = tl.where(decide, a, b)
    tl.store(output_ptr + offsets, output, mask=mask)


def test_where_pointer_tensors():
    # Mirrors upstream test_core.py::test_where[1-*int32], first launch.
    size = 1000
    rs = RandomState(17)
    cond = numpy_random(size, "bool", rs)
    x = numpy_random(size, "int64", rs=rs)
    y = numpy_random(size, "int64", rs=rs)
    z = np.where(cond, x, y)

    cond_tri = to_triton(cond, device="npu")
    x_tri = to_triton(x, device="npu", dst_type="int64")
    y_tri = to_triton(y, device="npu", dst_type="int64")
    z_tri = to_triton(np.empty(size, dtype=z.dtype), device="npu", dst_type="int64")

    grid = (triton.cdiv(size, BLOCK), )
    where_pointers_kernel[grid](cond_tri, x_tri, y_tri, z_tri, size, BLOCK_SIZE=BLOCK)
    assert (z == z_tri.cpu().numpy()).all()


# ---------------------------------------------------------------------------
# 2. Community case: test_where with TEST_SCALAR_POINTERS (scalar select path)
# ---------------------------------------------------------------------------
@triton.jit
def where_scalar_pointers_kernel(cond_ptr, a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    ptr = tl.where(tl.load(cond_ptr), a_ptr, b_ptr)
    output = tl.load(ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, output, mask=mask)


def test_where_scalar_pointers():
    # Mirrors upstream test_core.py::test_where[1-*int32], second launch.
    size = 1000
    rs = RandomState(17)
    cond = numpy_random(size, "bool", rs)
    x = numpy_random(size, "int64", rs=rs)
    y = numpy_random(size, "int64", rs=rs)
    z = np.where(cond[0], x, y)

    cond_tri = to_triton(cond, device="npu")
    x_tri = to_triton(x, device="npu", dst_type="int64")
    y_tri = to_triton(y, device="npu", dst_type="int64")
    z_tri = to_triton(np.empty(size, dtype=z.dtype), device="npu", dst_type="int64")

    grid = (triton.cdiv(size, BLOCK), )
    where_scalar_pointers_kernel[grid](cond_tri, x_tri, y_tri, z_tri, size, BLOCK_SIZE=BLOCK)
    assert (z == z_tri.cpu().numpy()).all()


# ---------------------------------------------------------------------------
# 3. Pointer arithmetic on int-derived tensors: scalar / tensor / negative /
#    i64 offsets (R4).
# ---------------------------------------------------------------------------
@triton.jit
def addptr_kernel(addr_ptr, i64_offs_ptr, data_ptr, out1_ptr, out2_ptr, out3_ptr, out4_ptr, out5_ptr,
                  base_addr: tl.int64, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    base = tl.load(addr_ptr + offs).to(tl.pi32_t)
    p1 = base + 8  # scalar i32 offset (broadcast)
    p2 = base + offs  # tensor i32 offset
    p3 = base - 4  # negative offset
    i64o = tl.load(i64_offs_ptr + offs)
    p4 = base + i64o  # i64 offset
    # End-to-end stage: the demoted addresses become element offsets and read
    # real data through a view chain, which then takes part in computation.
    idx = ((p1.to(tl.int64) - base_addr) // 4).to(tl.int32)
    tl.store(out1_ptr + offs, p1.to(tl.int64))
    tl.store(out2_ptr + offs, p2.to(tl.int64))
    tl.store(out3_ptr + offs, p3.to(tl.int64))
    tl.store(out4_ptr + offs, p4.to(tl.int64))
    tl.store(out5_ptr + offs, tl.load(data_ptr + idx) * 3)


def test_addptr_on_int_derived_pointers():
    data = (torch.arange(BLOCK + 16, dtype=torch.int32) + 100).npu()
    base = data.data_ptr()
    perm, addrs = _perm_addresses(base)

    i64o = _i64_offsets()
    outs = [torch.empty(BLOCK, dtype=torch.int64).npu() for _ in range(4)]
    out5 = torch.empty(BLOCK, dtype=torch.int32).npu()
    addptr_kernel[(1, )](addrs, i64o, data, *outs, out5, base, BLOCK_SIZE=BLOCK)

    offs = np.arange(BLOCK, dtype=np.int64)
    expected_addr = [
        addrs.cpu().numpy() + 8 * 4,
        addrs.cpu().numpy() + offs * 4,
        addrs.cpu().numpy() - 4 * 4,
        addrs.cpu().numpy() + i64o.cpu().numpy() * 4,
    ]
    for out, exp in zip(outs, expected_addr):
        assert (out.cpu().numpy() == exp).all()
    # idx = (addr + 8*4 - base) // 4 = perm + 8, inside the oversized buffer.
    expected_data = (data.cpu().numpy()[perm + 8] * 3).astype(np.int32)
    assert (out5.cpu().numpy() == expected_data).all()


# ---------------------------------------------------------------------------
# 4. Select mixing an int-derived side and a view-derived side (R3 + R2b).
# ---------------------------------------------------------------------------
@triton.jit
def mixed_select_kernel(cond_ptr, addr_ptr, data_ptr, out_ptr, out_val_ptr, base_addr: tl.int64,
                        BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    p_int = tl.load(addr_ptr + offs).to(tl.pi32_t)  # int-derived chain
    p_view = data_ptr + offs  # view-derived addptr chain
    p = tl.where(tl.load(cond_ptr + offs) > 0, p_int, p_view)
    tl.store(out_ptr + offs, p.to(tl.int64))
    # End-to-end stage: read real data through the demoted address selection.
    idx = ((p.to(tl.int64) - base_addr) // 4).to(tl.int32)
    tl.store(out_val_ptr + offs, tl.load(data_ptr + idx) + 1)


def test_select_mixed_int_and_view_derived():
    data = (torch.arange(BLOCK, dtype=torch.int32) + 7).npu()
    base = data.data_ptr()
    perm, addrs = _perm_addresses(base)

    cond = _cond_i8()
    out = torch.empty(BLOCK, dtype=torch.int64).npu()
    out_val = torch.empty(BLOCK, dtype=torch.int32).npu()
    mixed_select_kernel[(1, )](cond, addrs, data, out, out_val, base, BLOCK_SIZE=BLOCK)

    offs = np.arange(BLOCK, dtype=np.int64)
    view_addrs = base + offs * 4
    expected = np.where(cond.cpu().numpy() > 0, addrs.cpu().numpy(), view_addrs)
    assert (out.cpu().numpy() == expected).all()
    # int side idx = perm; view side idx = offs.
    expected_data = (np.where(cond.cpu().numpy() > 0,
                              data.cpu().numpy()[perm],
                              data.cpu().numpy()) + 1).astype(np.int32)
    assert (out_val.cpu().numpy() == expected_data).all()


# ---------------------------------------------------------------------------
# 5. ptr_to_int of a view-derived chain materializes base + offs * sizeof (R2b).
# ---------------------------------------------------------------------------
@triton.jit
def ptr_to_int_view_kernel(base_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    addrs = (base_ptr + offs).to(tl.int64)
    tl.store(out_ptr + offs, addrs)


def test_ptr_to_int_of_view_chain():
    dummy = torch.empty(BLOCK, dtype=torch.int64).npu()  # i64 pointee: 8 bytes
    base = dummy.data_ptr()
    out = torch.empty(BLOCK, dtype=torch.int64).npu()
    ptr_to_int_view_kernel[(1, )](dummy, out, BLOCK_SIZE=BLOCK)

    expected = base + np.arange(BLOCK, dtype=np.int64) * 8
    assert (out.cpu().numpy() == expected).all()


# ---------------------------------------------------------------------------
# 6. Null pointer sentinel built from an int64 zero tensor (S3).
# ---------------------------------------------------------------------------
@triton.jit
def nullptr_kernel(cond_ptr, addr_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    valid = tl.load(cond_ptr + offs) > 0
    p = tl.load(addr_ptr + offs).to(tl.pi32_t)
    nullp = tl.cast(tl.zeros((BLOCK_SIZE, ), tl.int64), tl.pointer_type(tl.int32))
    p = tl.where(valid, p, nullp)
    tl.store(out_ptr + offs, p.to(tl.int64))


def test_null_pointer_sentinel():
    dummy = torch.empty(16, dtype=torch.int32).npu()
    addrs = _addresses(dummy.data_ptr())
    cond = _cond_i8()
    out = torch.empty(BLOCK, dtype=torch.int64).npu()
    nullptr_kernel[(1, )](cond, addrs, out, BLOCK_SIZE=BLOCK)

    expected = np.where(cond.cpu().numpy() > 0, addrs.cpu().numpy(), 0)
    assert (out.cpu().numpy() == expected).all()


# ---------------------------------------------------------------------------
# 7. Structure ops on pointer tensors: reshape / trans / broadcast_to (C5).
# ---------------------------------------------------------------------------
@triton.jit
def structure_kernel(addr_ptr, data_ptr, out_r_ptr, out_t_ptr, out_b_ptr, out_rv_ptr, base_addr: tl.int64,
                     BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    p = tl.load(addr_ptr + offs).to(tl.pi32_t)  # (BLOCK,)
    r = tl.reshape(p, (32, 32)).to(tl.int64)
    t = tl.trans(tl.reshape(p, (32, 32))).to(tl.int64)
    b = tl.broadcast_to(tl.expand_dims(p, 0), 4, BLOCK_SIZE).to(tl.int64)
    idx2d = tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
    tl.store(out_r_ptr + idx2d, r)
    tl.store(out_t_ptr + idx2d, t)
    idx2d_b = tl.arange(0, 4)[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    tl.store(out_b_ptr + idx2d_b, b)
    # End-to-end stage: reshaped addresses become 2D offsets and read real data.
    idx_r = ((r - base_addr) // 4).to(tl.int32)
    tl.store(out_rv_ptr + idx2d, tl.load(data_ptr + idx_r) + 1)


def test_pointer_tensor_structure_ops():
    data = (torch.arange(BLOCK, dtype=torch.int32) + 11).npu()
    base = data.data_ptr()
    perm, addrs = _perm_addresses(base)
    out_r = torch.empty((32, 32), dtype=torch.int64).npu()
    out_t = torch.empty((32, 32), dtype=torch.int64).npu()
    out_b = torch.empty((4, BLOCK), dtype=torch.int64).npu()
    out_rv = torch.empty((32, 32), dtype=torch.int32).npu()
    structure_kernel[(1, )](addrs, data, out_r, out_t, out_b, out_rv, base, BLOCK_SIZE=BLOCK)

    a = addrs.cpu().numpy().reshape(32, 32)
    assert (out_r.cpu().numpy() == a).all()
    assert (out_t.cpu().numpy() == a.T).all()
    assert (out_b.cpu().numpy() == np.broadcast_to(addrs.cpu().numpy(), (4, BLOCK))).all()
    expected_data = (data.cpu().numpy()[perm].reshape(32, 32) + 1).astype(np.int32)
    assert (out_rv.cpu().numpy() == expected_data).all()


# ---------------------------------------------------------------------------
# 8. join / split on pointer tensors (C5, exercises the multi-result rebuild).
# ---------------------------------------------------------------------------
@triton.jit
def join_split_kernel(addr_ptr, out_j_ptr, out_lo_ptr, out_hi_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    p = tl.load(addr_ptr + offs).to(tl.pi32_t)  # (BLOCK,)
    j = tl.join(p, p)  # (BLOCK, 2)
    lo, hi = tl.split(j)  # two (BLOCK,) halves
    idx2d = tl.arange(0, BLOCK_SIZE)[:, None] * 2 + tl.arange(0, 2)[None, :]
    tl.store(out_j_ptr + idx2d, j.to(tl.int64))
    tl.store(out_lo_ptr + offs, lo.to(tl.int64))
    tl.store(out_hi_ptr + offs, hi.to(tl.int64))


def test_pointer_tensor_join_split():
    dummy = torch.empty(16, dtype=torch.int32).npu()
    addrs = _addresses(dummy.data_ptr())
    out_j = torch.empty((BLOCK, 2), dtype=torch.int64).npu()
    out_lo = torch.empty(BLOCK, dtype=torch.int64).npu()
    out_hi = torch.empty(BLOCK, dtype=torch.int64).npu()
    join_split_kernel[(1, )](addrs, out_j, out_lo, out_hi, BLOCK_SIZE=BLOCK)

    a = addrs.cpu().numpy()
    assert (out_j.cpu().numpy() == np.stack([a, a], axis=1)).all()
    assert (out_lo.cpu().numpy() == a).all()
    assert (out_hi.cpu().numpy() == a).all()


# ---------------------------------------------------------------------------
# 9. Pointer type intercast: pi32 <-> pi64 bitcast keeps the addresses (C6).
# ---------------------------------------------------------------------------
@triton.jit
def bitcast_kernel(addr_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    p32 = tl.load(addr_ptr + offs).to(tl.pi32_t)
    tl.store(out_ptr + offs, p32.to(tl.int64))


def test_pointer_bitcast():
    dummy = torch.empty(16, dtype=torch.int32).npu()
    addrs = _addresses(dummy.data_ptr())
    out = torch.empty(BLOCK, dtype=torch.int64).npu()
    bitcast_kernel[(1, )](addrs, out, BLOCK_SIZE=BLOCK)

    assert (out.cpu().numpy() == addrs.cpu().numpy()).all()


# ---------------------------------------------------------------------------
# 10. Pointer tensors across control flow: scf.if and scf.for (C2).
# ---------------------------------------------------------------------------
@triton.jit
def cf_if_kernel(cond_ptr, addr_ptr, data_ptr, out_ptr, out_val_ptr, base_addr: tl.int64, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    p = tl.load(addr_ptr + offs).to(tl.pi32_t)
    if tl.load(cond_ptr) > 0:
        p = p + 2
    tl.store(out_ptr + offs, p.to(tl.int64))
    # End-to-end stage: the pointer tensor carried through scf.if reads data.
    idx = ((p.to(tl.int64) - base_addr) // 4).to(tl.int32)
    tl.store(out_val_ptr + offs, tl.load(data_ptr + idx) - 1)


@triton.jit
def cf_for_kernel(addr_ptr, data_ptr, out_ptr, out_val_ptr, base_addr: tl.int64, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    p = tl.load(addr_ptr + offs).to(tl.pi32_t)
    for _ in range(n):
        p = p + 1
    tl.store(out_ptr + offs, p.to(tl.int64))
    # End-to-end stage: the scf.for iter_arg addresses read data.
    idx = ((p.to(tl.int64) - base_addr) // 4).to(tl.int32)
    tl.store(out_val_ptr + offs, tl.load(data_ptr + idx) - 1)


def test_pointer_tensor_control_flow():
    data = (torch.arange(BLOCK + 16, dtype=torch.int32) + 3).npu()
    base = data.data_ptr()
    perm, addrs = _perm_addresses(base)

    cond = torch.tensor([1], dtype=torch.int8).npu()
    out_if = torch.empty(BLOCK, dtype=torch.int64).npu()
    out_if_val = torch.empty(BLOCK, dtype=torch.int32).npu()
    cf_if_kernel[(1, )](cond, addrs, data, out_if, out_if_val, base, BLOCK_SIZE=BLOCK)
    assert (out_if.cpu().numpy() == addrs.cpu().numpy() + 2 * 4).all()
    assert (out_if_val.cpu().numpy() == (data.cpu().numpy()[perm + 2] - 1).astype(np.int32)).all()

    cond_neg = torch.tensor([0], dtype=torch.int8).npu()
    out_if_neg = torch.empty(BLOCK, dtype=torch.int64).npu()
    cf_if_kernel[(1, )](cond_neg, addrs, data, out_if_neg, out_if_val, base, BLOCK_SIZE=BLOCK)
    assert (out_if_neg.cpu().numpy() == addrs.cpu().numpy()).all()
    assert (out_if_val.cpu().numpy() == (data.cpu().numpy()[perm] - 1).astype(np.int32)).all()

    n = 5
    out_for = torch.empty(BLOCK, dtype=torch.int64).npu()
    out_for_val = torch.empty(BLOCK, dtype=torch.int32).npu()
    cf_for_kernel[(1, )](addrs, data, out_for, out_for_val, base, n, BLOCK_SIZE=BLOCK)
    assert (out_for.cpu().numpy() == addrs.cpu().numpy() + n * 4).all()
    assert (out_for_val.cpu().numpy() == (data.cpu().numpy()[perm + n] - 1).astype(np.int32)).all()


# ---------------------------------------------------------------------------
# 11. Address comparison via explicit ptr_to_int (C3).
# ---------------------------------------------------------------------------
@triton.jit
def compare_kernel(addr1_ptr, addr2_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    p1 = tl.load(addr1_ptr + offs).to(tl.pi32_t) + 1
    p2 = tl.load(addr2_ptr + offs).to(tl.pi32_t)
    eq = p1.to(tl.int64) == p2.to(tl.int64)
    tl.store(out_ptr + offs, eq.to(tl.int8))


def test_pointer_address_compare():
    dummy = torch.empty(16, dtype=torch.int32).npu()
    base = dummy.data_ptr()
    addrs1 = _addresses(base)
    addrs2 = _addresses(base, stride=16)
    addrs2 = torch.from_numpy(addrs1.cpu().numpy() + 4).npu()  # p1 == p2 exactly
    out = torch.empty(BLOCK, dtype=torch.int8).npu()
    compare_kernel[(1, )](addrs1, addrs2, out, BLOCK_SIZE=BLOCK)

    assert (out.cpu().numpy() == 1).all()


# ---------------------------------------------------------------------------
# 12. End-to-end gather emulation: pointer select feeds real data loads.
# ---------------------------------------------------------------------------
@triton.jit
def gather_select_kernel(cond_ptr, addr1_ptr, addr2_ptr, data_ptr, out_ptr, base_addr: tl.int64,
                         BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    p1 = tl.load(addr1_ptr + offs).to(tl.pi32_t)
    p2 = tl.load(addr2_ptr + offs).to(tl.pi32_t)
    sel = tl.where(tl.load(cond_ptr + offs) > 0, p1, p2)  # 指针张量 select
    idx = ((sel.to(tl.int64) - base_addr) // 4).to(tl.int32)  # 地址 → 元素偏移
    vals = tl.load(data_ptr + idx)  # 视图链读真实数据
    tl.store(out_ptr + offs, vals * 2 + 1)  # 参与运算写回


def test_gather_emulation_through_view_chain():
    data = (torch.arange(BLOCK, dtype=torch.int32) + 5).npu()
    base = data.data_ptr()
    perm1, addrs1 = _perm_addresses(base, seed=3)
    perm2, addrs2 = _perm_addresses(base, seed=7)
    cond = _cond_i8()
    out = torch.empty(BLOCK, dtype=torch.int32).npu()
    gather_select_kernel[(1, )](cond, addrs1, addrs2, data, out, base, BLOCK_SIZE=BLOCK)

    c = cond.cpu().numpy() > 0
    expected = (np.where(c, data.cpu().numpy()[perm1], data.cpu().numpy()[perm2]) * 2 + 1).astype(np.int32)
    assert (out.cpu().numpy() == expected).all()


# ---------------------------------------------------------------------------
# 13. End-to-end scatter emulation: processed addresses write real data.
# ---------------------------------------------------------------------------
@triton.jit
def scatter_kernel(addr_ptr, data_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    base_addr = tl.cast(data_ptr, tl.int64)  # 标量 ptr_to_int(既有机制)
    p = tl.load(addr_ptr + offs).to(tl.pi32_t) + 2  # 整型派生 + addptr
    idx = ((p.to(tl.int64) - base_addr) // 4).to(tl.int32)
    tl.store(data_ptr + idx, offs + 1000)  # scatter 写回(视图链)
    tl.store(out_ptr + offs, idx.to(tl.int64))  # 记录写的位置


def test_scatter_emulation_through_view_chain():
    data = torch.full((BLOCK, ), -1, dtype=torch.int32).npu()
    base = data.data_ptr()
    size = BLOCK - 4
    perm, addrs = _perm_addresses(base, count=size, seed=5)  # idx = perm + 2 < BLOCK
    out = torch.empty(size, dtype=torch.int64).npu()
    scatter_kernel[(1, )](addrs, data, out, BLOCK_SIZE=size)

    idx = perm + 2
    expected_data = np.full(BLOCK, -1, dtype=np.int32)
    expected_data[idx] = np.arange(size, dtype=np.int32) + 1000
    assert (data.cpu().numpy() == expected_data).all()
    assert (out.cpu().numpy() == idx).all()
