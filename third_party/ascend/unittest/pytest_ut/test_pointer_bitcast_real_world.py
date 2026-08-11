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
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _write_u32_le(buf, byte_offsets, values):
    for byte_offset, value in zip(byte_offsets, values):
        value = int(value)
        for byte_idx in range(4):
            buf[int(byte_offset) + byte_idx] = (value >> (8 * byte_idx)) & 0xFF


def _float_to_bf16_bits(value):
    f32_bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return f32_bits >> 16


def _write_bf16_le(buf, byte_offset, values):
    for idx, value in enumerate(values):
        bits = _float_to_bf16_bits(value)
        buf[int(byte_offset) + idx * 2] = bits & 0xFF
        buf[int(byte_offset) + idx * 2 + 1] = (bits >> 8) & 0xFF


@triton.jit
def _load_u32_from_u8_tensor_offsets(src, out, n_elements, BASE: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < n_elements
    byte_offsets = BASE + lanes * 4
    u32_ptrs = (src + byte_offsets).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    values = tl.load(u32_ptrs, mask=mask, other=0).to(tl.int32)
    tl.store(out + lanes, values)


@triton.jit
def _load_bf16_from_u8_multi_addptr(src, out, block_idx, token_pos, block_stride, N: tl.constexpr,
                                    TOKEN_BYTES: tl.constexpr, BF16_OFFSET: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    hinted_stride = tl.multiple_of(block_stride, 16)
    block_base = src + block_idx * hinted_stride
    token_fp8_ptr = block_base + token_pos * TOKEN_BYTES
    bf16_ptr = (token_fp8_ptr + BF16_OFFSET).to(tl.pointer_type(tl.bfloat16))
    values = tl.load(bf16_ptr + lanes, mask=lanes < N, other=0.0)
    tl.store(out + lanes, values, mask=lanes < N)


@triton.jit
def _load_bf16_from_u8_inside_loop(src, out, count, NOPE_DIM: tl.constexpr, TOKEN_BYTES: tl.constexpr,
                                   ROPE_DIM: tl.constexpr, BLOCK: tl.constexpr, NUM_WORKERS: tl.constexpr):
    worker = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    mask = lanes < ROPE_DIM
    for token_idx in range(worker, count, NUM_WORKERS):
        token_data = src + token_idx * TOKEN_BYTES
        bf16_ptr = (token_data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        values = tl.load(bf16_ptr + lanes, mask=mask, other=0.0)
        tl.store(out + token_idx * ROPE_DIM + lanes, values, mask=mask)


def test_load_uint32_scale_words_from_uint8_tensor_offsets():
    block = 32
    n_words = 17
    base = 16
    values = [0x01020304 + idx * 0x00010101 for idx in range(n_words)]
    host = torch.zeros(base + block * 4, dtype=torch.uint8, device="cpu")
    _write_u32_le(host, [base + idx * 4 for idx in range(n_words)], values)

    src = host.npu()
    out = torch.empty((block, ), device="npu", dtype=torch.int32)

    _load_u32_from_u8_tensor_offsets[(1, )](src, out, n_words, BASE=base, BLOCK=block)
    _sync_npu()

    expected = torch.zeros((block, ), dtype=torch.int32, device="cpu")
    expected[:n_words] = torch.tensor(values, dtype=torch.int32, device="cpu")
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_load_bf16_rope_region_from_uint8_multi_addptr():
    block_idx = 1
    token_pos = 2
    block_stride = 2048
    token_bytes = 576
    bf16_offset = 448
    n_values = 16
    base = block_idx * block_stride + token_pos * token_bytes + bf16_offset
    values = [float(idx + 1) for idx in range(n_values)]

    host = torch.zeros(base + n_values * 2 + 64, dtype=torch.uint8, device="cpu")
    _write_bf16_le(host, base, values)

    src = host.npu()
    out = torch.empty((n_values, ), device="npu", dtype=torch.bfloat16)

    _load_bf16_from_u8_multi_addptr[(1, )](
        src,
        out,
        block_idx,
        token_pos,
        block_stride,
        N=n_values,
        TOKEN_BYTES=token_bytes,
        BF16_OFFSET=bf16_offset,
        BLOCK=32,
    )
    _sync_npu()

    expected = torch.tensor(values, dtype=torch.float32, device="cpu").to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_load_bf16_rope_region_from_uint8_inside_dynamic_loop():
    token_count = 5
    nope_dim = 448
    rope_dim = 8
    token_bytes = 576
    num_workers = 2
    host = torch.zeros(token_count * token_bytes, dtype=torch.uint8, device="cpu")
    expected_values = []

    for token_idx in range(token_count):
        values = [float(token_idx * rope_dim + idx + 1) for idx in range(rope_dim)]
        expected_values.extend(values)
        _write_bf16_le(host, token_idx * token_bytes + nope_dim, values)

    src = host.npu()
    out = torch.empty((token_count * rope_dim, ), device="npu", dtype=torch.bfloat16)

    _load_bf16_from_u8_inside_loop[(num_workers, )](
        src,
        out,
        token_count,
        NOPE_DIM=nope_dim,
        TOKEN_BYTES=token_bytes,
        ROPE_DIM=rope_dim,
        BLOCK=16,
        NUM_WORKERS=num_workers,
    )
    _sync_npu()

    expected = torch.tensor(expected_values, dtype=torch.float32, device="cpu").to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
