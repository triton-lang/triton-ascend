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

import torch
import torch_npu
import pytest
import triton
import triton.language as tl


# CTA scoped indirect bitwise atomic lowers to a BiShengIR template that tries
# to aggregate updates in UB before writing the final value back to GM. The
# template allocates a UB temp buffer with the same element count as the offsets
# tensor. Therefore offsets in [0, offsets.numel()) can use the fast UB path,
# while sparse offsets outside that range must locally fall back to the normal
# GM atomic path to avoid UB overflow.
#
# This file covers all supported CTA bitwise ops, integer dtypes, and routing
# modes:
# - ops: or / and / xor
# - dtypes: int32 / uint32 / int64 / uint64
# - "none": every offset is in range, so every lane uses the CTA UB path.
# - "partial": some lanes are in range and some lanes fall back to GM atomic.
# - "all": every offset is out of range, so the CTA wrapper degenerates to the
#   old GM atomic behavior.
#
# The offsets are unique in all modes. That keeps the returned old value
# deterministic, making it safe to check both the final output and the atomic
# return tensor.
@triton.jit
def indirect_atomic_bitwise_cta_1d(
    offset_ptr,
    value_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    OP: tl.constexpr,
):
    i0 = tl.arange(0, D0)
    linear = i0
    offsets = tl.load(offset_ptr + linear)
    values = tl.load(value_ptr + linear)
    if OP == "or":
        old = tl.atomic_or(out_ptr + offsets, values, scope="cta")
    elif OP == "and":
        old = tl.atomic_and(out_ptr + offsets, values, scope="cta")
    else:
        old = tl.atomic_xor(out_ptr + offsets, values, scope="cta")
    tl.store(old_ptr + linear, old)


@triton.jit
def indirect_atomic_bitwise_cta_2d(
    offset_ptr,
    value_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    OP: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None]
    i1 = tl.arange(0, D1)[None, :]
    linear = i0 * D1 + i1
    offsets = tl.load(offset_ptr + linear)
    values = tl.load(value_ptr + linear)
    if OP == "or":
        old = tl.atomic_or(out_ptr + offsets, values, scope="cta")
    elif OP == "and":
        old = tl.atomic_and(out_ptr + offsets, values, scope="cta")
    else:
        old = tl.atomic_xor(out_ptr + offsets, values, scope="cta")
    tl.store(old_ptr + linear, old)


@triton.jit
def indirect_atomic_bitwise_cta_3d(
    offset_ptr,
    value_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    OP: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None, None]
    i1 = tl.arange(0, D1)[None, :, None]
    i2 = tl.arange(0, D2)[None, None, :]
    linear = (i0 * D1 + i1) * D2 + i2
    offsets = tl.load(offset_ptr + linear)
    values = tl.load(value_ptr + linear)
    if OP == "or":
        old = tl.atomic_or(out_ptr + offsets, values, scope="cta")
    elif OP == "and":
        old = tl.atomic_and(out_ptr + offsets, values, scope="cta")
    else:
        old = tl.atomic_xor(out_ptr + offsets, values, scope="cta")
    tl.store(old_ptr + linear, old)


@triton.jit
def indirect_atomic_bitwise_cta_4d(
    offset_ptr,
    value_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    OP: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None, None, None]
    i1 = tl.arange(0, D1)[None, :, None, None]
    i2 = tl.arange(0, D2)[None, None, :, None]
    i3 = tl.arange(0, D3)[None, None, None, :]
    linear = ((i0 * D1 + i1) * D2 + i2) * D3 + i3
    offsets = tl.load(offset_ptr + linear)
    values = tl.load(value_ptr + linear)
    if OP == "or":
        old = tl.atomic_or(out_ptr + offsets, values, scope="cta")
    elif OP == "and":
        old = tl.atomic_and(out_ptr + offsets, values, scope="cta")
    else:
        old = tl.atomic_xor(out_ptr + offsets, values, scope="cta")
    tl.store(old_ptr + linear, old)


@triton.jit
def indirect_atomic_bitwise_cta_5d(
    offset_ptr,
    value_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
    OP: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None, None, None, None]
    i1 = tl.arange(0, D1)[None, :, None, None, None]
    i2 = tl.arange(0, D2)[None, None, :, None, None]
    i3 = tl.arange(0, D3)[None, None, None, :, None]
    i4 = tl.arange(0, D4)[None, None, None, None, :]
    linear = (((i0 * D1 + i1) * D2 + i2) * D3 + i3) * D4 + i4
    offsets = tl.load(offset_ptr + linear)
    values = tl.load(value_ptr + linear)
    if OP == "or":
        old = tl.atomic_or(out_ptr + offsets, values, scope="cta")
    elif OP == "and":
        old = tl.atomic_and(out_ptr + offsets, values, scope="cta")
    else:
        old = tl.atomic_xor(out_ptr + offsets, values, scope="cta")
    tl.store(old_ptr + linear, old)


RANK_SHAPES = {
    1: (8,),
    2: (2, 4),
    3: (2, 2, 4),
    4: (2, 2, 2, 4),
    5: (2, 2, 2, 2, 2),
}

KERNELS = {
    1: indirect_atomic_bitwise_cta_1d,
    2: indirect_atomic_bitwise_cta_2d,
    3: indirect_atomic_bitwise_cta_3d,
    4: indirect_atomic_bitwise_cta_4d,
    5: indirect_atomic_bitwise_cta_5d,
}

DTYPES = [
    ("int32", torch.int32),
    ("uint32", torch.uint32),
    ("int64", torch.int64),
    ("uint64", torch.uint64),
]


def _build_offsets(shape, fallback_mode):
    numel = 1
    for dim in shape:
        numel *= dim

    offsets = torch.arange(numel, dtype=torch.int64)
    if fallback_mode == "none":
        pass
    elif fallback_mode == "partial":
        offsets[1::3] += numel
    elif fallback_mode == "all":
        offsets += numel
    else:
        raise AssertionError(f"Unknown fallback mode: {fallback_mode}")

    return offsets.reshape(shape), numel * 2


def _build_output(output_size, torch_dtype):
    if torch_dtype in (torch.uint32, torch.uint64):
        output = torch.arange(output_size, dtype=torch.int64).to(torch_dtype)
    else:
        output = torch.arange(output_size, dtype=torch_dtype)
    return output * 256


def _build_values(shape, torch_dtype):
    numel = 1
    for dim in shape:
        numel *= dim
    return torch.tensor(
        [1 << (idx % 16) for idx in range(numel)], dtype=torch_dtype
    ).reshape(shape)


def _simulate_atomic_bitwise(op, base_output, offsets, values):
    expected_output = base_output.clone()
    expected_old = torch.zeros_like(values)
    flat_offsets = offsets.reshape(-1).to(torch.int64)
    flat_values = values.reshape(-1)
    flat_old = expected_old.reshape(-1)

    for idx, offset in enumerate(flat_offsets.tolist()):
        flat_old[idx] = expected_output[offset]
        if op == "or":
            expected_output[offset] = expected_output[offset] | flat_values[idx]
        elif op == "and":
            expected_output[offset] = expected_output[offset] & flat_values[idx]
        elif op == "xor":
            expected_output[offset] = expected_output[offset] ^ flat_values[idx]
        else:
            raise AssertionError(f"Unknown atomic op: {op}")

    return expected_output, expected_old


def _launch(rank, op, offsets, values, output, old, shape):
    kwargs = {f"D{dim}": size for dim, size in enumerate(shape)}
    KERNELS[rank][(1,)](offsets, values, output, old, OP=op, **kwargs)


@pytest.mark.parametrize("dtype_name, torch_dtype", DTYPES)
@pytest.mark.parametrize("op", ["or", "and", "xor"])
@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("fallback_mode", ["none", "partial", "all"])
def test_indirect_atomic_bitwise_cta_fallback_modes(
    dtype_name, torch_dtype, op, rank, fallback_mode
):
    shape = RANK_SHAPES[rank]
    offsets, output_size = _build_offsets(shape, fallback_mode)
    values = _build_values(shape, torch_dtype)
    output = _build_output(output_size, torch_dtype)
    old = torch.zeros(shape, dtype=torch_dtype)

    expected_output, expected_old = _simulate_atomic_bitwise(
        op, output, offsets, values
    )

    offsets_npu = offsets.npu()
    values_npu = values.npu()
    output_npu = output.npu()
    old_npu = old.npu()

    _launch(rank, op, offsets_npu, values_npu, output_npu, old_npu, shape)

    assert torch.equal(output_npu.cpu(), expected_output), (
        f"dtype={dtype_name}, op={op}, rank={rank}, "
        f"fallback_mode={fallback_mode}, output mismatch"
    )
    assert torch.equal(old_npu.cpu(), expected_old), (
        f"dtype={dtype_name}, op={op}, rank={rank}, "
        f"fallback_mode={fallback_mode}, old mismatch"
    )
