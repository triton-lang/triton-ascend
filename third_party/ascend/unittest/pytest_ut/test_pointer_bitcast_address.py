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

import torch
import torch_npu
import triton
import triton.language as tl


def _sync_npu():
    torch.npu.synchronize()


def _write_u32_le(buf, byte_offsets, values):
    for byte_offset, value in zip(byte_offsets, values):
        payload = struct.pack("<I", int(value))
        for byte_idx, byte_value in enumerate(payload):
            buf[int(byte_offset) + byte_idx] = byte_value


def _float_to_bf16_bits(value):
    f32_bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return f32_bits >> 16


def _write_bf16_le(buf, byte_offset, values):
    for idx, value in enumerate(values):
        bits = _float_to_bf16_bits(value)
        buf[int(byte_offset) + idx * 2] = bits & 0xFF
        buf[int(byte_offset) + idx * 2 + 1] = (bits >> 8) & 0xFF


def _ttadapter(compiled):
    assert "ttadapter" in compiled.asm, compiled.asm.keys()
    return compiled.asm["ttadapter"]


def _require_no_runtime_check(compiled):
    ttadapter = _ttadapter(compiled)
    assert "arith.remsi" not in ttadapter, ttadapter
    assert "triton_assert" not in ttadapter, ttadapter


def _require_no_pointer_bitcast_assert(compiled):
    ttadapter = _ttadapter(compiled)
    assert "triton_assert" not in ttadapter, ttadapter


@triton.jit
def _runtime_paged_scale_load(kv_ptr, block_tables_ptr, out_ptr, n_positions, stride_kvblk, stride_kvpos, stride_kvbyte,
                              DIM: tl.constexpr, CACHE_BLOCK_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    kv_global_pos = tl.arange(0, BLOCK)
    valid = kv_global_pos < n_positions
    logical_block = kv_global_pos // CACHE_BLOCK_SIZE
    intra_block_pos = kv_global_pos % CACHE_BLOCK_SIZE
    physical_block = tl.load(block_tables_ptr + logical_block, mask=valid, other=0)
    kv_base = (physical_block * stride_kvblk + intra_block_pos * stride_kvpos)
    scale_addr = kv_base + DIM * stride_kvbyte
    scale_ptr = (kv_ptr + scale_addr).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    scale_u32 = tl.load(scale_ptr, mask=valid, other=0)
    tl.store(out_ptr + kv_global_pos, scale_u32, mask=valid)


@triton.jit
def _runtime_scalar_bf16_load(src, byte_offset_ptr, out, N: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    byte_offset = tl.load(byte_offset_ptr)
    bf16_ptr = (src + byte_offset).to(tl.pointer_type(tl.bfloat16), bitcast=True)
    values = tl.load(bf16_ptr + lanes, mask=lanes < N, other=0.0)
    tl.store(out + lanes, values, mask=lanes < N)


@triton.jit
def _runtime_tensor_u32_load(src, byte_offsets_ptr, out, N: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    byte_offsets = tl.load(byte_offsets_ptr + lanes, mask=mask, other=0)
    u32_ptrs = (src + byte_offsets).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    values = tl.load(u32_ptrs, mask=mask, other=0)
    tl.store(out + lanes, values, mask=mask)


@triton.jit
def _runtime_tensor_u32_store(src, dst, byte_offsets_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    byte_offsets = tl.load(byte_offsets_ptr + lanes, mask=mask, other=0)
    values = tl.load(src + lanes, mask=mask, other=0)
    u32_ptrs = (dst + byte_offsets).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    tl.store(u32_ptrs, values, mask=mask)


@triton.jit
def _runtime_2d_tensor_u32_load(src, byte_offsets_ptr, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    linear = rows * BLOCK_N + cols
    byte_offsets = tl.load(byte_offsets_ptr + linear)
    u32_ptrs = (src + byte_offsets).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    values = tl.load(u32_ptrs)
    tl.store(out + linear, values)


@triton.jit
def _runtime_2d_tensor_u32_store(src, dst, byte_offsets_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    linear = rows * BLOCK_N + cols
    byte_offsets = tl.load(byte_offsets_ptr + linear)
    values = tl.load(src + linear)
    u32_ptrs = (dst + byte_offsets).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    tl.store(u32_ptrs, values)


@triton.jit
def _runtime_2d_block_ptr_bf16_load(src, byte_offset_ptr, out, ROWS: tl.constexpr, COLS: tl.constexpr,
                                    ROW_STRIDE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    byte_offset = tl.load(byte_offset_ptr)
    bf16_base = (src + byte_offset).to(tl.pointer_type(tl.bfloat16), bitcast=True)
    block_ptr = tl.make_block_ptr(
        base=bf16_base,
        shape=(ROWS, COLS),
        strides=(ROW_STRIDE, 1),
        offsets=(1, 2),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    values = tl.load(block_ptr)
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    tl.store(out + rows * BLOCK_N + cols, values)


@triton.jit
def _runtime_2d_block_ptr_bf16_store(src, dst, byte_offset_ptr, row_stride, ROWS: tl.constexpr, COLS: tl.constexpr,
                                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    values = tl.load(src + rows * BLOCK_N + cols)
    byte_offset = tl.load(byte_offset_ptr)
    bf16_base = (dst + byte_offset).to(tl.pointer_type(tl.bfloat16), bitcast=True)
    block_ptr = tl.make_block_ptr(
        base=bf16_base,
        shape=(ROWS, COLS),
        strides=(row_stride, 1),
        offsets=(1, 2),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(block_ptr, values)


@triton.jit
def _runtime_pointer_then_value_bitcast(src, byte_offsets_ptr, out, N: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    byte_offsets = tl.load(byte_offsets_ptr + lanes, mask=mask, other=0)
    u32_ptrs = (src + byte_offsets).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    scale_u32 = tl.load(u32_ptrs, mask=mask, other=0)
    scale_f32 = scale_u32.to(tl.float32, bitcast=True)
    tl.store(out + lanes, scale_f32, mask=mask)


@triton.jit
def _loaded_offset_division_without_pointer_bitcast(src, offsets_ptr, out, N: tl.constexpr, DIVISOR: tl.constexpr,
                                                    BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    loaded_offsets = tl.load(offsets_ptr + lanes, mask=mask, other=0)
    element_offsets = loaded_offsets // DIVISOR
    values = tl.load(src + element_offsets, mask=mask, other=0)
    tl.store(out + lanes, values, mask=mask)


@triton.jit
def _runtime_scalar_bf16_store(src, dst, byte_offset_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    byte_offset = tl.load(byte_offset_ptr)
    values = tl.load(src + lanes, mask=mask, other=0.0)
    bf16_ptr = (dst + byte_offset).to(tl.pointer_type(tl.bfloat16), bitcast=True)
    tl.store(bf16_ptr + lanes, values, mask=mask)


@triton.jit
def _static_byte_offset_u32_load(src, out, N: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    byte_offsets = lanes * 4
    u32_ptrs = (src + byte_offsets).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    values = tl.load(u32_ptrs, mask=mask, other=0)
    tl.store(out + lanes, values, mask=mask)


@triton.jit
def _runtime_multi_addptr_bf16_load(src, params_ptr, out, N: tl.constexpr, BLOCK: tl.constexpr):
    block_idx = tl.load(params_ptr + 0)
    token_pos = tl.load(params_ptr + 1)
    block_stride = tl.load(params_ptr + 2)
    token_bytes = tl.load(params_ptr + 3)
    bf16_offset = tl.load(params_ptr + 4)
    block_base = src + block_idx * block_stride
    token_base = block_base + token_pos * token_bytes
    bf16_ptr = (token_base + bf16_offset).to(tl.pointer_type(tl.bfloat16), bitcast=True)
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    values = tl.load(bf16_ptr + lanes, mask=mask, other=0.0)
    tl.store(out + lanes, values, mask=mask)


@triton.jit
def _runtime_scalar_u32_load(src, byte_offset_ptr, out):
    byte_offset = tl.load(byte_offset_ptr)
    u32_ptr = (src + byte_offset).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    value = tl.load(u32_ptr)
    tl.store(out, value)


@triton.jit
def _runtime_tensor_pre_post_u32_load(src, element_offsets, out, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    pre_cast_byte_offsets = lanes * 8
    post_cast_element_offsets = tl.load(element_offsets + lanes)
    ptrs = (src + pre_cast_byte_offsets).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    values = tl.load(ptrs + post_cast_element_offsets)
    tl.store(out + lanes, values)


@triton.jit
def _runtime_scalar_pre_post_u32_load(src, offsets, out):
    pre_cast_byte_offset = tl.load(offsets + 0)
    post_cast_element_offset = tl.load(offsets + 1)
    ptr = (src + pre_cast_byte_offset).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    tl.store(out, tl.load(ptr + post_cast_element_offset))


@triton.jit
def _runtime_multiple_bitcast_boundaries_load(src, offsets, out):
    pre_bytes = tl.load(offsets + 0)
    u16_elements = tl.load(offsets + 1)
    u8_bytes = tl.load(offsets + 2)
    u32_elements = tl.load(offsets + 3)
    p16 = (src + pre_bytes).to(tl.pointer_type(tl.uint16, 1), bitcast=True)
    p8 = (p16 + u16_elements).to(tl.pointer_type(tl.uint8, 1), bitcast=True)
    p32 = (p8 + u8_bytes).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    tl.store(out, tl.load(p32 + u32_elements))


@triton.jit
def _runtime_negative_post_bitcast_u32_load(src, element_offset, out):
    p32 = src.to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    tl.store(out, tl.load(p32 + tl.load(element_offset)))


@triton.jit
def _runtime_tensor_pre_post_u32_store(values, dst, element_offsets, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    pre_cast_byte_offsets = lanes * 8
    post_cast_element_offsets = tl.load(element_offsets + lanes)
    ptrs = (dst + pre_cast_byte_offsets).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    tl.store(ptrs + post_cast_element_offsets, tl.load(values + lanes))


@triton.jit
def _runtime_tensor_pre_post_atomic_add(values, dst, element_offsets, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    pre_cast_byte_offsets = lanes * 8
    post_cast_element_offsets = tl.load(element_offsets + lanes)
    ptrs = (dst + pre_cast_byte_offsets).to(tl.pointer_type(tl.int32, 1), bitcast=True)
    tl.atomic_add(ptrs + post_cast_element_offsets, tl.load(values + lanes))


def test_runtime_paged_mqa_scale_preserves_address():
    cache_block_size = 4
    dim = 8
    record_bytes = dim + 4
    stride_kvpos = record_bytes
    stride_kvblk = cache_block_size * record_bytes
    stride_kvbyte = 1
    n_positions = 8
    block = 8
    block_table = [2, 0]
    values = [100 + idx for idx in range(n_positions)]

    host = torch.zeros(3 * stride_kvblk, dtype=torch.uint8)
    byte_offsets = []
    for logical_pos in range(n_positions):
        logical_block = logical_pos // cache_block_size
        intra_block_pos = logical_pos % cache_block_size
        physical_block = block_table[logical_block]
        byte_offsets.append(physical_block * stride_kvblk + intra_block_pos * stride_kvpos + dim)
    _write_u32_le(host, byte_offsets, values)

    kv = host.npu()
    block_tables = torch.tensor(block_table, dtype=torch.int32).npu()
    out = torch.empty((block, ), device="npu", dtype=torch.int32)
    args = (kv, block_tables, out, n_positions, stride_kvblk, stride_kvpos, stride_kvbyte)
    meta = {"DIM": dim, "CACHE_BLOCK_SIZE": cache_block_size, "BLOCK": block}

    compiled = _runtime_paged_scale_load.warmup(*args, **meta, grid=(1, ))
    _require_no_pointer_bitcast_assert(compiled)
    _runtime_paged_scale_load[(1, )](*args, **meta)
    _sync_npu()

    torch.testing.assert_close(out.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0)


def test_runtime_scalar_bf16_load_preserves_address():
    byte_offset = 18
    values = [1.0, 2.0, 4.0, 8.0]
    host = torch.zeros(64, dtype=torch.uint8)
    _write_bf16_le(host, byte_offset, values)
    src = host.npu()
    offset = torch.tensor([byte_offset], dtype=torch.int64).npu()
    out = torch.empty((len(values), ), device="npu", dtype=torch.bfloat16)

    compiled = _runtime_scalar_bf16_load.warmup(src, offset, out, N=len(values), BLOCK=8, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_scalar_bf16_load[(1, )](src, offset, out, N=len(values), BLOCK=8)
    _sync_npu()

    expected = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_runtime_tensor_offsets_u32_load_preserves_address():
    byte_offsets = [4, 20, 36, 52]
    values = [0x01020304, 0x11121314, 0x21222324, 0x31323334]
    host = torch.zeros(80, dtype=torch.uint8)
    _write_u32_le(host, byte_offsets, values)
    src = host.npu()
    offsets = torch.tensor(byte_offsets, dtype=torch.int64).npu()
    out = torch.empty((len(values), ), device="npu", dtype=torch.int32)

    compiled = _runtime_tensor_u32_load.warmup(src, offsets, out, N=len(values), BLOCK=4, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_tensor_u32_load[(1, )](src, offsets, out, N=len(values), BLOCK=4)
    _sync_npu()

    torch.testing.assert_close(out.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0)


def test_runtime_tensor_offsets_u32_store_preserves_address():
    byte_offsets = [8, 24, 40, 56]
    values = [101, 202, 303, 404]
    src = torch.tensor(values, dtype=torch.int32).npu()
    dst = torch.zeros(80, device="npu", dtype=torch.uint8)
    offsets = torch.tensor(byte_offsets, dtype=torch.int64).npu()

    compiled = _runtime_tensor_u32_store.warmup(src, dst, offsets, N=len(values), BLOCK=4, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_tensor_u32_store[(1, )](src, dst, offsets, N=len(values), BLOCK=4)
    _sync_npu()

    expected = torch.zeros(80, dtype=torch.uint8)
    _write_u32_le(expected, byte_offsets, values)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


def test_runtime_2d_tensor_offsets_u32_load_preserves_address():
    block_m = 2
    block_n = 4
    byte_offsets = [0, 12, 24, 36, 48, 60, 72, 84]
    values = [11, 22, 33, 44, 55, 66, 77, 88]
    host = torch.zeros(96, dtype=torch.uint8)
    _write_u32_le(host, byte_offsets, values)
    src = host.npu()
    offsets = torch.tensor(byte_offsets, dtype=torch.int64).npu()
    out = torch.empty((block_m, block_n), device="npu", dtype=torch.int32)

    compiled = _runtime_2d_tensor_u32_load.warmup(src, offsets, out, BLOCK_M=block_m, BLOCK_N=block_n, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_2d_tensor_u32_load[(1, )](src, offsets, out, BLOCK_M=block_m, BLOCK_N=block_n)
    _sync_npu()

    expected = torch.tensor(values, dtype=torch.int32).reshape(block_m, block_n)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_runtime_2d_tensor_offsets_u32_store_preserves_address():
    block_m = 2
    block_n = 4
    byte_offsets = [4, 16, 28, 40, 52, 64, 76, 88]
    values = [111, 222, 333, 444, 555, 666, 777, 888]
    src = torch.tensor(values, dtype=torch.int32).reshape(block_m, block_n).npu()
    dst = torch.zeros(96, device="npu", dtype=torch.uint8)
    offsets = torch.tensor(byte_offsets, dtype=torch.int64).npu()

    compiled = _runtime_2d_tensor_u32_store.warmup(src, dst, offsets, BLOCK_M=block_m, BLOCK_N=block_n, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_2d_tensor_u32_store[(1, )](src, dst, offsets, BLOCK_M=block_m, BLOCK_N=block_n)
    _sync_npu()

    expected = torch.zeros(96, dtype=torch.uint8)
    _write_u32_le(expected, byte_offsets, values)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


def test_runtime_2d_block_ptr_bf16_load_preserves_address():
    block_m = 4
    block_n = 8
    row_offset = 1
    col_offset = 2
    rows = block_m + row_offset
    cols = block_n + col_offset
    row_stride = 12
    byte_offset = 8
    storage_bytes = 144
    values = [float(idx + 1) for idx in range(block_m * block_n)]
    host = torch.zeros(storage_bytes, dtype=torch.uint8)
    for row in range(block_m):
        start = row * block_n
        _write_bf16_le(
            host,
            byte_offset + ((row + row_offset) * row_stride + col_offset) * 2,
            values[start:start + block_n],
        )
    src = host.npu()
    offset = torch.tensor([byte_offset], dtype=torch.int64).npu()
    out = torch.empty((block_m, block_n), device="npu", dtype=torch.bfloat16)
    meta = {
        "ROWS": rows,
        "COLS": cols,
        "ROW_STRIDE": row_stride,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
    }

    compiled = _runtime_2d_block_ptr_bf16_load.warmup(src, offset, out, **meta, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_2d_block_ptr_bf16_load[(1, )](src, offset, out, **meta)
    _sync_npu()

    expected = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16).reshape(block_m, block_n)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_runtime_2d_block_ptr_bf16_store_preserves_address():
    block_m = 4
    block_n = 8
    row_offset = 1
    col_offset = 2
    rows = block_m + row_offset
    cols = block_n + col_offset
    row_stride = 12
    byte_offset = 8
    storage_bytes = 144
    values = [float(idx + 1) for idx in range(block_m * block_n)]
    src = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16).reshape(block_m, block_n).npu()
    dst = torch.zeros(storage_bytes, device="npu", dtype=torch.uint8)
    offset = torch.tensor([byte_offset], dtype=torch.int64).npu()
    meta = {
        "ROWS": rows,
        "COLS": cols,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
    }

    compiled = _runtime_2d_block_ptr_bf16_store.warmup(src, dst, offset, row_stride, **meta, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_2d_block_ptr_bf16_store[(1, )](src, dst, offset, row_stride, **meta)
    _sync_npu()

    expected = torch.zeros(storage_bytes, dtype=torch.uint8)
    for row in range(block_m):
        start = row * block_n
        _write_bf16_le(
            expected,
            byte_offset + ((row + row_offset) * row_stride + col_offset) * 2,
            values[start:start + block_n],
        )
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


def test_runtime_pointer_bitcast_followed_by_value_bitcast():
    byte_offsets = [4, 12, 20, 28]
    values = [1.0, -2.0, 0.5, 8.0]
    bit_patterns = [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]
    host = torch.zeros(40, dtype=torch.uint8)
    _write_u32_le(host, byte_offsets, bit_patterns)
    src = host.npu()
    offsets = torch.tensor(byte_offsets, dtype=torch.int64).npu()
    out = torch.empty((len(values), ), device="npu", dtype=torch.float32)

    compiled = _runtime_pointer_then_value_bitcast.warmup(src, offsets, out, N=len(values), BLOCK=4, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_pointer_then_value_bitcast[(1, )](src, offsets, out, N=len(values), BLOCK=4)
    _sync_npu()

    torch.testing.assert_close(out.cpu(), torch.tensor(values, dtype=torch.float32), rtol=0, atol=0)


def test_loaded_tensor_offset_division_without_pointer_bitcast_is_unchanged():
    loaded_offsets = [1, 2, 3, 4]
    source = torch.arange(16, dtype=torch.int32).npu()
    offsets = torch.tensor(loaded_offsets, dtype=torch.int64).npu()
    out = torch.empty((len(loaded_offsets), ), device="npu", dtype=torch.int32)

    compiled = _loaded_offset_division_without_pointer_bitcast.warmup(source, offsets, out, N=len(loaded_offsets),
                                                                      DIVISOR=2, BLOCK=4, grid=(1, ))
    _require_no_pointer_bitcast_assert(compiled)
    _loaded_offset_division_without_pointer_bitcast[(1, )](source, offsets, out, N=len(loaded_offsets), DIVISOR=2,
                                                           BLOCK=4)
    _sync_npu()

    expected = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_runtime_scalar_bf16_store_preserves_address():
    byte_offset = 22
    values = [3.0, 6.0, 9.0, 12.0]
    src = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16).npu()
    dst = torch.zeros(64, device="npu", dtype=torch.uint8)
    offset = torch.tensor([byte_offset], dtype=torch.int64).npu()

    compiled = _runtime_scalar_bf16_store.warmup(src, dst, offset, N=len(values), BLOCK=8, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_scalar_bf16_store[(1, )](src, dst, offset, N=len(values), BLOCK=8)
    _sync_npu()

    expected = torch.zeros(64, dtype=torch.uint8)
    _write_bf16_le(expected, byte_offset, values)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


def test_static_byte_offsets_preserve_address_without_runtime_check():
    block = 4
    values = [11, 22, 33, 44]
    host = torch.zeros(block * 4, dtype=torch.uint8)
    _write_u32_le(host, [idx * 4 for idx in range(block)], values)
    src = host.npu()
    out = torch.empty((block, ), device="npu", dtype=torch.int32)

    compiled = _static_byte_offset_u32_load.warmup(src, out, N=block, BLOCK=block, grid=(1, ))
    _require_no_runtime_check(compiled)
    _static_byte_offset_u32_load[(1, )](src, out, N=block, BLOCK=block)
    _sync_npu()

    torch.testing.assert_close(out.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0)


def test_runtime_multi_addptr_bf16_load_preserves_address():
    params_host = [1, 1, 64, 16, 8]
    byte_offset = 64 + 16 + 8
    values = [2.0, 4.0, 6.0, 8.0]
    host = torch.zeros(128, dtype=torch.uint8)
    _write_bf16_le(host, byte_offset, values)
    src = host.npu()
    params = torch.tensor(params_host, dtype=torch.int64).npu()
    out = torch.empty((len(values), ), device="npu", dtype=torch.bfloat16)

    compiled = _runtime_multi_addptr_bf16_load.warmup(src, params, out, N=len(values), BLOCK=8, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_multi_addptr_bf16_load[(1, )](src, params, out, N=len(values), BLOCK=8)
    _sync_npu()

    expected = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_runtime_negative_scalar_offset_u32_load_preserves_address():
    host = torch.zeros(64, dtype=torch.uint8)
    value = 0x10203040
    _write_u32_le(host, [12], [value])
    storage = host.npu()
    src = storage[16:]
    offset = torch.tensor([-4], dtype=torch.int64).npu()
    out = torch.empty((), device="npu", dtype=torch.int32)

    compiled = _runtime_scalar_u32_load.warmup(src, offset, out, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_scalar_u32_load[(1, )](src, offset, out)
    _sync_npu()

    assert int(out.cpu()) == value


def test_tensor_offsets_before_and_after_bitcast_preserve_address():
    block = 4
    post_offsets = [0, 1, 0, 1]
    byte_offsets = [lane * 8 + post_offsets[lane] * 4 for lane in range(block)]
    values = [101, 202, 303, 404]
    host = torch.zeros(40, dtype=torch.uint8)
    _write_u32_le(host, byte_offsets, values)
    src = host.npu()
    offsets = torch.tensor(post_offsets, dtype=torch.int64).npu()
    out = torch.empty((block, ), device="npu", dtype=torch.int32)

    compiled = _runtime_tensor_pre_post_u32_load.warmup(src, offsets, out, BLOCK=block, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_tensor_pre_post_u32_load[(1, )](src, offsets, out, BLOCK=block)
    _sync_npu()

    torch.testing.assert_close(out.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0)


def test_scalar_offsets_before_and_after_bitcast_preserve_address():
    pre_bytes = 4
    post_elements = 3
    expected_byte_address = pre_bytes + post_elements * 4
    value = 0x10203040
    host = torch.zeros(40, dtype=torch.uint8)
    _write_u32_le(host, [expected_byte_address], [value])
    src = host.npu()
    offsets = torch.tensor([pre_bytes, post_elements], dtype=torch.int64).npu()
    out = torch.empty((), device="npu", dtype=torch.int32)

    compiled = _runtime_scalar_pre_post_u32_load.warmup(src, offsets, out, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_scalar_pre_post_u32_load[(1, )](src, offsets, out)
    _sync_npu()

    assert int(out.cpu()) == value


def test_multiple_bitcast_boundaries_preserve_address():
    offsets_host = [3, 2, 1, 2]
    expected_byte_address = 3 + 2 * 2 + 1 + 2 * 4
    value = 0x11223344
    host = torch.zeros(40, dtype=torch.uint8)
    _write_u32_le(host, [expected_byte_address], [value])
    src = host.npu()
    offsets = torch.tensor(offsets_host, dtype=torch.int64).npu()
    out = torch.empty((), device="npu", dtype=torch.int32)

    compiled = _runtime_multiple_bitcast_boundaries_load.warmup(src, offsets, out, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_multiple_bitcast_boundaries_load[(1, )](src, offsets, out)
    _sync_npu()

    assert int(out.cpu()) == value


def test_negative_post_bitcast_offset_preserves_address():
    value = 0x21324354
    host = torch.zeros(32, dtype=torch.uint8)
    _write_u32_le(host, [12], [value])
    storage = host.npu()
    src = storage[16:]
    offset = torch.tensor([-1], dtype=torch.int64).npu()
    out = torch.empty((), device="npu", dtype=torch.int32)

    compiled = _runtime_negative_post_bitcast_u32_load.warmup(src, offset, out, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_negative_post_bitcast_u32_load[(1, )](src, offset, out)
    _sync_npu()

    assert int(out.cpu()) == value


def test_tensor_store_offsets_after_bitcast_preserve_address():
    block = 4
    post_offsets = [1, 0, 1, 0]
    byte_offsets = [lane * 8 + post_offsets[lane] * 4 for lane in range(block)]
    values_host = [11, 22, 33, 44]
    values = torch.tensor(values_host, dtype=torch.int32).npu()
    dst = torch.zeros(40, device="npu", dtype=torch.uint8)
    offsets = torch.tensor(post_offsets, dtype=torch.int64).npu()

    compiled = _runtime_tensor_pre_post_u32_store.warmup(values, dst, offsets, BLOCK=block, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_tensor_pre_post_u32_store[(1, )](values, dst, offsets, BLOCK=block)
    _sync_npu()

    expected = torch.zeros(40, dtype=torch.uint8)
    _write_u32_le(expected, byte_offsets, values_host)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


def test_tensor_atomic_offsets_after_bitcast_preserve_address():
    block = 4
    post_offsets = [0, 1, 0, 1]
    element_indices = [lane * 2 + post_offsets[lane] for lane in range(block)]
    initial = torch.arange(12, dtype=torch.int32)
    increments_host = [10, 20, 30, 40]
    increments = torch.tensor(increments_host, dtype=torch.int32).npu()
    dst = initial.clone().npu()
    dst_bytes = dst.view(torch.uint8)
    offsets = torch.tensor(post_offsets, dtype=torch.int64).npu()

    compiled = _runtime_tensor_pre_post_atomic_add.warmup(increments, dst_bytes, offsets, BLOCK=block, grid=(1, ))
    _require_no_runtime_check(compiled)
    _runtime_tensor_pre_post_atomic_add[(1, )](increments, dst_bytes, offsets, BLOCK=block)
    _sync_npu()

    expected = initial.clone()
    for index, increment in zip(element_indices, increments_host):
        expected[index] += increment
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
