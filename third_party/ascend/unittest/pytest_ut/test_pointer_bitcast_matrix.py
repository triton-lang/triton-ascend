# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import struct

import pytest
import torch
import torch_npu
import triton
import triton.language as tl


def _sync_npu():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _pack_i16(value):
    return struct.pack("<h", int(value))


def _pack_i32(value):
    return struct.pack("<i", int(value))


def _pack_f32(value):
    return struct.pack("<f", float(value))


def _float_to_bf16_bits(value):
    f32_bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return f32_bits >> 16


def _pack_bf16(value):
    bits = _float_to_bf16_bits(value)
    return bytes((bits & 0xFF, (bits >> 8) & 0xFF))


def _write_packed_values(buf, byte_offsets, values, pack_fn):
    for byte_offset, value in zip(byte_offsets, values):
        payload = pack_fn(value)
        start = int(byte_offset)
        for byte_idx, byte_value in enumerate(payload):
            buf[start + byte_idx] = byte_value


def _expected_bytes(values, pack_fn):
    out = []
    for value in values:
        out.extend(pack_fn(value))
    return torch.tensor(out, dtype=torch.uint8)


def _float_bits_to_i32(value):
    return struct.unpack("<i", struct.pack("<f", float(value)))[0]


@triton.jit
def _load_widen_from_u8(src, out, n_elements, BASE: tl.constexpr, BYTE_WIDTH: tl.constexpr, TARGET_DTYPE: tl.constexpr,
                        BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < n_elements
    byte_offsets = BASE + lanes * BYTE_WIDTH
    target_ptrs = (src + byte_offsets).to(tl.pointer_type(TARGET_DTYPE, 1), bitcast=True)
    values = tl.load(target_ptrs, mask=mask, other=0)
    tl.store(out + lanes, values)


@triton.jit
def _store_widen_to_u8(src, dst, n_elements, BASE: tl.constexpr, BYTE_WIDTH: tl.constexpr, TARGET_DTYPE: tl.constexpr,
                       BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < n_elements
    values = tl.load(src + lanes, mask=mask, other=0)
    byte_offsets = BASE + lanes * BYTE_WIDTH
    target_ptrs = (dst + byte_offsets).to(tl.pointer_type(TARGET_DTYPE, 1), bitcast=True)
    tl.store(target_ptrs, values, mask=mask)


@triton.jit
def _atomic_add_widen_to_u8(updates_ptr, dst, n_elements, BASE: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < n_elements
    updates = tl.load(updates_ptr + lanes, mask=mask, other=0)
    byte_offsets = BASE + lanes * 4
    target_ptrs = (dst + byte_offsets).to(tl.pointer_type(tl.int32, 1), bitcast=True)
    tl.atomic_add(target_ptrs, updates, mask=mask)


@triton.jit
def _atomic_cas_widen_to_u8(compare_ptr, updates_ptr, dst, old_ptr, BASE: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    compare = tl.load(compare_ptr + lanes)
    updates = tl.load(updates_ptr + lanes)
    byte_offsets = BASE + lanes * 4
    target_ptrs = (dst + byte_offsets).to(tl.pointer_type(tl.int32, 1), bitcast=True)
    old = tl.atomic_cas(target_ptrs, compare, updates)
    tl.store(old_ptr + lanes, old)


@triton.jit
def _same_width_i32_pointer_bitcast(src, out, word_base: tl.constexpr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    src_words = src + word_base
    f32_ptr = src_words.to(tl.pointer_type(tl.float32), bitcast=True)
    values = tl.load(f32_ptr + lanes, mask=lanes < n_elements, other=0.0)
    tl.store(out + lanes, values)


@triton.jit
def _value_bitcast_i32_to_f32(src, out, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    raw = tl.load(src + lanes, mask=lanes < n_elements, other=0)
    values = raw.to(tl.float32, bitcast=True)
    tl.store(out + lanes, values)


@triton.jit
def _narrow_i32_to_u8_load(src, out, word_base: tl.constexpr, n_bytes: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    src_words = src + word_base
    byte_ptr = src_words.to(tl.pointer_type(tl.uint8), bitcast=True)
    values = tl.load(byte_ptr + lanes, mask=lanes < n_bytes, other=0)
    tl.store(out + lanes, values)


@triton.jit
def _narrow_i32_tensor_ptrs_to_u8_load(src, word_offsets_ptr, out, N: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    word_offsets = tl.load(word_offsets_ptr + lanes, mask=mask, other=0)
    word_ptrs = src + word_offsets
    byte_ptrs = word_ptrs.to(tl.pointer_type(tl.uint8), bitcast=True)
    values = tl.load(byte_ptrs, mask=mask, other=0)
    tl.store(out + lanes, values, mask=mask)


@triton.jit
def _legacy_i1_pointer_to_u8_load(src, out, N: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    byte_ptr = src.to(tl.pointer_type(tl.uint8), bitcast=True)
    values = tl.load(byte_ptr + lanes, mask=mask, other=0)
    tl.store(out + lanes, values, mask=mask)


@triton.jit
def _plain_u8_copy(src, out, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    values = tl.load(src + lanes, mask=lanes < n_elements, other=0)
    tl.store(out + lanes, values)


@triton.jit
def _static_pre_and_post_bitcast_i32_load(src, out):
    ptr = (src + 4).to(tl.pointer_type(tl.int32, 1), bitcast=True)
    value = tl.load(ptr + 2)
    tl.store(out, value)


@pytest.mark.parametrize(
    "name,byte_width,target_dtype,torch_dtype,values,pack_fn",
    [
        ("i16", 2, tl.int16, torch.int16, [11, 22, 33, 44, 55], _pack_i16),
        ("i32", 4, tl.int32, torch.int32, [0x01020304, 0x05060708, 0x11121314, 0x21222324], _pack_i32),
        ("f32", 4, tl.float32, torch.float32, [1.0, 2.0, 4.0, 8.0, 16.0], _pack_f32),
        ("bf16", 2, tl.bfloat16, torch.bfloat16, [1.0, 2.0, 3.0, 4.0, 5.0], _pack_bf16),
    ],
)
def test_widen_load_from_uint8_dtype_matrix(name, byte_width, target_dtype, torch_dtype, values, pack_fn):
    block = 8
    base = 32
    host = torch.zeros(base + block * byte_width, dtype=torch.uint8)
    byte_offsets = [base + idx * byte_width for idx in range(len(values))]
    _write_packed_values(host, byte_offsets, values, pack_fn)

    src = host.npu()
    out = torch.empty((block, ), device="npu", dtype=torch_dtype)

    _load_widen_from_u8[(1, )](src, out, len(values), BASE=base, BYTE_WIDTH=byte_width, TARGET_DTYPE=target_dtype,
                               BLOCK=block)
    _sync_npu()

    expected = torch.zeros((block, ), dtype=torch_dtype)
    expected[:len(values)] = torch.tensor(values, dtype=torch_dtype)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name,byte_width,target_dtype,torch_dtype,values,pack_fn",
    [
        ("i16", 2, tl.int16, torch.int16, [101, 202, 303, 404], _pack_i16),
        ("i32", 4, tl.int32, torch.int32, [0x01010101, 0x02020202, 0x03030303], _pack_i32),
        ("f32", 4, tl.float32, torch.float32, [1.0, 2.0, 4.0], _pack_f32),
        ("bf16", 2, tl.bfloat16, torch.bfloat16, [1.0, 2.0, 3.0, 4.0], _pack_bf16),
    ],
)
def test_widen_store_to_uint8_dtype_matrix(name, byte_width, target_dtype, torch_dtype, values, pack_fn):
    block = 8
    base = 16
    src = torch.tensor(values, dtype=torch_dtype).npu()
    dst = torch.zeros(base + block * byte_width, device="npu", dtype=torch.uint8)

    _store_widen_to_u8[(1, )](src, dst, len(values), BASE=base, BYTE_WIDTH=byte_width, TARGET_DTYPE=target_dtype,
                              BLOCK=block)
    _sync_npu()

    actual = dst.cpu()[base:base + len(values) * byte_width]
    expected = _expected_bytes(values, pack_fn)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_widen_atomic_add_scalarizes_value_and_mask():
    block = 8
    base = 32
    updates_host = torch.tensor([3, 7, 11, 13, 17], dtype=torch.int32)
    initial = torch.full((base + block * 4 + 8, ), 0xA5, dtype=torch.uint8)
    byte_offsets = [base + lane * 4 for lane in range(block)]
    _write_packed_values(initial, byte_offsets, [0] * block, _pack_i32)

    updates = updates_host.npu()
    dst = initial.npu()
    _atomic_add_widen_to_u8[(1, )](updates, dst, updates_host.numel(), BASE=base, BLOCK=block)
    _sync_npu()

    expected = initial.clone()
    _write_packed_values(expected, byte_offsets[:updates_host.numel()], updates_host, _pack_i32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


def test_widen_atomic_cas_scalarizes_compare_value_and_result():
    block = 4
    base = 32
    initial_values = [10, 20, 30, 40]
    compare_values = [10, 99, 30, 77]
    update_values = [101, 202, 303, 404]
    expected_values = [101, 20, 303, 40]
    byte_offsets = [base + lane * 4 for lane in range(block)]
    initial = torch.full((base + block * 4 + 8, ), 0x5A, dtype=torch.uint8)
    _write_packed_values(initial, byte_offsets, initial_values, _pack_i32)

    compare = torch.tensor(compare_values, dtype=torch.int32).npu()
    updates = torch.tensor(update_values, dtype=torch.int32).npu()
    dst = initial.npu()
    old = torch.empty((block, ), dtype=torch.int32, device="npu")
    _atomic_cas_widen_to_u8[(1, )](compare, updates, dst, old, BASE=base, BLOCK=block)
    _sync_npu()

    expected = initial.clone()
    _write_packed_values(expected, byte_offsets, expected_values, _pack_i32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(old.cpu(), torch.tensor(initial_values, dtype=torch.int32), rtol=0, atol=0)


def test_same_width_pointer_bitcast_i32_to_f32_is_unchanged():
    block = 8
    word_base = 1
    values = [1.0, 2.0, 4.0, 8.0]
    raw = [0] + [_float_bits_to_i32(value) for value in values] + [0] * 4
    src = torch.tensor(raw, dtype=torch.int32).npu()
    out = torch.empty((block, ), device="npu", dtype=torch.float32)

    _same_width_i32_pointer_bitcast[(1, )](src, out, word_base=word_base, n_elements=len(values), BLOCK=block)
    _sync_npu()

    expected = torch.zeros((block, ), dtype=torch.float32)
    expected[:len(values)] = torch.tensor(values, dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_value_bitcast_i32_to_f32_is_unchanged():
    block = 8
    values = [1.0, 2.0, 4.0, 8.0]
    raw = [_float_bits_to_i32(value) for value in values] + [0] * 4
    src = torch.tensor(raw, dtype=torch.int32).npu()
    out = torch.empty((block, ), device="npu", dtype=torch.float32)

    _value_bitcast_i32_to_f32[(1, )](src, out, n_elements=len(values), BLOCK=block)
    _sync_npu()

    expected = torch.zeros((block, ), dtype=torch.float32)
    expected[:len(values)] = torch.tensor(values, dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_narrow_i32_to_uint8_load():
    block = 16
    word_base = 1
    words = [0, 0x01020304, 0x11121314, 0x21222324, 0x31323334]
    src = torch.tensor(words, dtype=torch.int32).npu()
    out = torch.empty((block, ), device="npu", dtype=torch.uint8)

    _narrow_i32_to_u8_load[(1, )](src, out, word_base=word_base, n_bytes=12, BLOCK=block)
    _sync_npu()

    expected = _expected_bytes(words[1:4], _pack_i32)
    padded = torch.zeros((block, ), dtype=torch.uint8)
    padded[:expected.numel()] = expected
    torch.testing.assert_close(out.cpu(), padded, rtol=0, atol=0)


def test_narrow_i32_tensor_pointers_to_uint8_load():
    words = [
        0,
        0x01020304,
        0,
        0x11121314,
        0x21222324,
        0,
        0x31323334,
    ]
    word_offsets = [1, 3, 4, 6]
    src = torch.tensor(words, dtype=torch.int32).npu()
    offsets = torch.tensor(word_offsets, dtype=torch.int64).npu()
    out = torch.empty((len(word_offsets), ), device="npu", dtype=torch.uint8)

    _narrow_i32_tensor_ptrs_to_u8_load[(1, )](src, offsets, out, N=len(word_offsets), BLOCK=4)
    _sync_npu()

    expected = torch.tensor([0x04, 0x14, 0x24, 0x34], dtype=torch.uint8)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_legacy_i1_pointer_to_uint8_load_is_unchanged():
    values = [False, True, True, False, True, False, False, True]
    src = torch.tensor(values, dtype=torch.bool).npu()
    out = torch.empty((len(values), ), device="npu", dtype=torch.uint8)

    _legacy_i1_pointer_to_u8_load[(1, )](src, out, N=len(values), BLOCK=8)
    _sync_npu()

    expected = torch.tensor(values, dtype=torch.uint8)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_plain_uint8_load_store_is_unchanged():
    block = 32
    n_elements = 19
    src_host = torch.arange(block, dtype=torch.uint8)
    src = src_host.npu()
    out = torch.empty((block, ), device="npu", dtype=torch.uint8)

    _plain_u8_copy[(1, )](src, out, n_elements=n_elements, BLOCK=block)
    _sync_npu()

    expected = torch.zeros((block, ), dtype=torch.uint8)
    expected[:n_elements] = src_host[:n_elements]
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_static_offsets_before_and_after_bitcast_preserve_address():
    value = 0x10203040
    host = torch.zeros((32, ), dtype=torch.uint8)
    _write_packed_values(host, [12], [value], _pack_i32)
    src = host.npu()
    out = torch.empty((), device="npu", dtype=torch.int32)

    _static_pre_and_post_bitcast_i32_load[(1, )](src, out)
    _sync_npu()

    assert int(out.cpu()) == value
