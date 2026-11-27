# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _packed_mask_load(Packed, Score, Output, Q: tl.constexpr, K: tl.constexpr, ROW_BYTES: tl.constexpr,
                      BLOCK_N: tl.constexpr, USE_BLOCK_PTR: tl.constexpr, ROW_START: tl.constexpr,
                      COL_START: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    if USE_BLOCK_PTR:
        block = tl.make_block_ptr(Packed, (Q, tl.cdiv(K, 8)), (ROW_BYTES, 1), (ROW_START, COL_START // 8),
                                  (32, BLOCK_N // 8), (1, 0))
        keep = tl.load(block, boundary_check=(0, 1), padding_option="zero", compile_hint="packed_mask_lsb")
    else:
        byte_cols = COL_START // 8 + tl.arange(0, BLOCK_N // 8)[None, :]
        ptrs = Packed + (ROW_START + rows) * ROW_BYTES + byte_cols
        valid = (ROW_START + rows < Q) & (byte_cols < tl.cdiv(K, 8))
        keep = tl.load(ptrs, mask=valid, other=0, compile_hint="packed_mask_lsb")
    keep = keep & (COL_START + cols < K)
    # Store the complete logical tile so invalid rows and last-byte bits are
    # observable in the reference check, rather than hidden by a store mask.
    offsets = rows * BLOCK_N + cols
    score = tl.load(Score + offsets)
    penalty = tl.where(keep, 0.0, -float("inf"))
    penalty = penalty.to(score.dtype)
    tl.store(Output + offsets, score + penalty)


@pytest.mark.parametrize("block_n", [64, 128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("use_block_ptr", [False, True])
@pytest.mark.parametrize("shape", ["aligned", "short", "tail", "strided_offset"])
def test_packed_mask_load_hint(block_n, dtype, use_block_ptr, shape):
    properties = torch.npu.get_device_properties(0)
    if "950" not in properties.name and "910_95" not in properties.name:
        pytest.skip("packed mask hint currently targets Ascend950")
    q = 32 if shape == "aligned" else 19
    k = block_n if shape == "aligned" else (32 if shape == "short" else block_n - 13)
    row_start = 3 if shape == "strided_offset" else 0
    col_start = 8 if shape == "strided_offset" else 0
    k += col_start
    row_bytes = triton.cdiv(k, 8)
    if shape == "strided_offset":
        row_bytes += 3
    dense = torch.rand((q, row_bytes * 8), device="npu") > 0.5
    # Poison unused bits and row padding: neither may enter logical consumers.
    dense[:, k:] = True
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device="npu", dtype=torch.int32)
    groups = dense.reshape(q, row_bytes, 8).to(torch.int32)
    weighted = groups * weights
    packed = weighted.sum(-1).to(torch.uint8)
    score = torch.randn((32, block_n), device="npu", dtype=dtype)
    output = torch.empty_like(score)
    _packed_mask_load[(1, )](packed, score, output, q, k, row_bytes, block_n, use_block_ptr, row_start, col_start,
                             num_warps=4, compile_mode="simd", enable_dynamic_cv_pipeline=True)
    expected_keep = torch.zeros((32, block_n), device="npu", dtype=torch.bool)
    expected_keep[:q - row_start, :k - col_start] = dense[row_start:, col_start:k]
    expected = score + torch.where(expected_keep, 0.0, -float("inf")).to(dtype)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
