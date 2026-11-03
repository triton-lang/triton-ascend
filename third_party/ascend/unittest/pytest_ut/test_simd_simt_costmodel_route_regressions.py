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

import json

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend.utils import is_compile_on_910_95

simd_simt_910_95_only = pytest.mark.xfail(
    not is_compile_on_910_95(),
    reason="SIMD/SIMT cost model only supports 910_95",
    run=False,
)


def _assert_effective_route(path, expected):
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["effective_decision_kind"] == expected


def _route_options(report_path, logical_programs=0):
    options = {
        "num_warps": 32,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
        "enable_auto_blockify": True,
    }
    if logical_programs:
        options["logical_program_count_hint"] = logical_programs
    return options


@triton.jit
def index_put_atomic_kernel(
    indices_ptr,
    enabled_ptr,
    values_ptr,
    add1_ptr,
    add2_ptr,
    output_ptr,
    z0_numel,
    y1_numel,
    x2_numel,
    Z0BLOCK: tl.constexpr,
    Z0BLOCK_SUB: tl.constexpr,
    Y1BLOCK_SUB: tl.constexpr,
):
    x2_numel = 128
    X2BLOCK_SUB: tl.constexpr = 128
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0 = tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    base_y1 = tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (y1_numel + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2 = tl.arange(0, X2BLOCK_SUB)
    output_rows = tl.full([Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB], 98166, tl.int32)
    negative_one = tl.full([1, 1, 1], -1, tl.int64)

    for loop_z0 in range(loops_z0):
        z0 = z0_offset + loop_z0 * Z0BLOCK_SUB + base_z0[:, None, None]
        z0_mask = z0 < min(Z0BLOCK + z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = loop_y1 * Y1BLOCK_SUB + base_y1[None, :, None]
            y1_mask = y1 < y1_numel
            x2 = base_x2[None, None, :]
            x2_mask = x2 < x2_numel
            index = tl.load(indices_ptr + y1 + 38 * z0, mask=y1_mask & z0_mask)
            enabled = tl.load(enabled_ptr + y1 + 38 * z0, mask=y1_mask & z0_mask)
            value = tl.load(values_ptr + 1064 + x2 + 2280 * z0, mask=x2_mask & z0_mask)
            add1 = tl.load(add1_ptr + x2 + 128 * z0, mask=x2_mask & z0_mask)
            add2 = tl.load(add2_ptr + x2 + 128 * z0, mask=x2_mask & z0_mask)

            index = tl.where(index < 0, index + output_rows, index)
            update = tl.where(enabled != 0, value + add1 + add2, 0.0)
            update = tl.where(index == negative_one, 0.0, update)
            tl.atomic_add(output_ptr + x2 + 128 * index, update, mask=x2_mask & y1_mask & z0_mask)


@triton.jit
def dacs_segsum_kernel(
    da_ptr,
    da_cs_ptr,
    da_cs_rev_ptr,
    segsum_ptr,
    stride_da_batch,
    stride_da_head,
    stride_da_seq,
    stride_da_cs_batch,
    stride_da_cs_head,
    stride_da_cs_seq,
    stride_da_cs_rev_batch,
    stride_da_cs_rev_head,
    stride_da_cs_rev_seq,
    stride_segsum_batch,
    stride_segsum_head,
    stride_segsum_chunk,
    stride_segsum_row,
    stride_segsum_col,
    SEQLEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    chunk_start = pid_chunk * CHUNK_SIZE
    offsets = tl.arange(0, CHUNK_SIZE)
    sequence_offsets = chunk_start + offsets
    mask = sequence_offsets < SEQLEN

    da_base = pid_batch * stride_da_batch + pid_head * stride_da_head
    da_chunk = tl.load(da_ptr + da_base + sequence_offsets * stride_da_seq, mask=mask, other=0.0)

    da_cs = tl.minimum(tl.cumsum(da_chunk, axis=0), 0.0)

    da_cs_rev = tl.cumsum(da_chunk, axis=0, reverse=True)
    row = tl.arange(0, CHUNK_SIZE)[:, None]
    column = tl.arange(0, CHUNK_SIZE)[None, :]
    shift_mask = row == column - 1
    da_cs_rev = tl.sum(tl.where(shift_mask, da_cs_rev, 0.0), axis=1)
    da_cs_rev = tl.minimum(da_cs_rev, 0.0)

    da_cs_base = pid_batch * stride_da_cs_batch + pid_head * stride_da_cs_head
    da_cs_rev_base = pid_batch * stride_da_cs_rev_batch + pid_head * stride_da_cs_rev_head
    tl.store(da_cs_ptr + da_cs_base + sequence_offsets * stride_da_cs_seq, da_cs, mask=mask)
    tl.store(da_cs_rev_ptr + da_cs_rev_base + sequence_offsets * stride_da_cs_rev_seq, da_cs_rev, mask=mask)

    broadcasted_indices = tl.zeros_like(offsets)
    segsum = tl.load(da_ptr + da_base + sequence_offsets[:, None] * stride_da_seq + broadcasted_indices[None, :])
    segsum = tl.where(row > column, segsum, 0.0)
    segsum = tl.minimum(tl.cumsum(segsum, axis=0), 0.0)

    segsum_base = (pid_batch * stride_segsum_batch + pid_head * stride_segsum_head + pid_chunk * stride_segsum_chunk)
    tl.store(segsum_ptr + segsum_base + row * stride_segsum_row + column * stride_segsum_col, segsum)


@simd_simt_910_95_only
def test_index_put_atomic_selects_all_simt_only(tmp_path):
    z0_numel = 4096
    y1_numel = 38
    x2_numel = 128
    indices = torch.arange(z0_numel, dtype=torch.int64, device="npu")[:, None].expand(-1, y1_numel).contiguous()
    enabled = torch.zeros((z0_numel, y1_numel), dtype=torch.bool, device="npu")
    enabled[:, 0] = True
    values = torch.zeros((z0_numel, 2280), dtype=torch.float32, device="npu")
    add1 = torch.ones((z0_numel, 1, x2_numel), dtype=torch.float32, device="npu")
    add2 = torch.zeros_like(add1)
    output = torch.zeros((98166, x2_numel), dtype=torch.float32, device="npu")
    report_path = tmp_path / "index_put_atomic_route.json"

    index_put_atomic_kernel[(128, 1, 1)](
        indices,
        enabled,
        values,
        add1,
        add2,
        output,
        z0_numel,
        y1_numel,
        x2_numel,
        Z0BLOCK=32,
        Z0BLOCK_SUB=8,
        Y1BLOCK_SUB=2,
        multibuffer=False,
        num_ctas=1,
        num_stages=2,
        superblock_factor=2,
        has_auto_blockify_blacklist_op=False,
        **_route_options(report_path),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(output[:z0_numel], torch.ones_like(output[:z0_numel]))
    assert torch.count_nonzero(output[z0_numel:]).item() == 0
    _assert_effective_route(report_path, "all_simt_only")
