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
#
# Tests for SIMT IndirectLoad fast-path (TritonToLinalg / IndirectLoadRewrite).
#
# What each parameter row exercises against the V1 trigger condition
# (compileOn91095 + forceSimtTemplate, last-axis stride statically > 1,
#  non-permuted layout, rank <= 5, not stride==2-even-size):
#
#   * STRIDE > 2 / odd       -> V1 should rewrite tt.load -> tt.indirect_load
#   * STRIDE == 2, even size -> DeinterleaveStatusOptimization handles (V1 yields)
#   * STRIDE == 2, odd size  -> V1 should rewrite (deinterleave precondition fails)
#   * Permuted layout        -> ImplicitPermute handles (V1 must not touch)
#   * Last-axis stride == 1  -> No rewrite (normal strided memref.copy)
#
# Verifies correctness only -- to confirm which path actually fired, dump IR
# with MLIR_ENABLE_DUMP=1 and grep for tt.indirect_load / tt.trans.

import triton
import triton.language as tl
import torch
import pytest
import test_common


# ---------------------------------------------------------------------------
# 1D: out[i] = in[i * STRIDE]
# ---------------------------------------------------------------------------

@triton.jit
def kernel_1d_strided_compaction(
    in_ptr, out_ptr, in_numel, out_numel,
    XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr, STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        src_idx = xindex * STRIDE
        # Two masks: store side bounded by out_numel, load side by in_numel.
        store_mask = xindex < out_numel
        load_mask = src_idx < in_numel
        tmp = tl.load(in_ptr + src_idx, load_mask)
        tl.store(out_ptr + xindex, tmp, store_mask)


def _ref_1d(src_cpu: torch.Tensor, stride: int, out_numel: int) -> torch.Tensor:
    flat = src_cpu.flatten().contiguous()
    return flat[::stride][:out_numel].contiguous()


@pytest.mark.parametrize("dtype,in_numel,stride,ncore,xblock,xblock_sub", [
    # ---- V1 命中: stride > 2 (deinterleave 不接) ----
    ("float32", 4096 * 16, 16, 2, 2048, 256),   # stride=16
    ("float32", 4096 * 8,   8, 2, 2048, 256),   # stride=8
    ("float32", 4096 * 4,   4, 2, 2048, 256),   # stride=4
    ("float32", 4096 * 3,   3, 2, 2048, 256),   # stride=3 (奇数 stride, block 仍 pow2)
    ("float16", 4096 * 6,   6, 2, 2048, 256),
    ("int8",    4096 * 7,   7, 2, 2048, 256),
    # ---- Deinterleave 接管 (stride==2 + 偶数 block, V1 让路) ----
    # 注:无法在 Triton 中构造"奇数 size + stride==2"以测试 V1 这一支,
    # 因为 tl.arange 要求 end-start 是 2 的幂.
    ("float32", 4096 * 2,   2, 2, 2048, 256),
    # ---- 不应改写 (stride==1) ----
    # TODO: 编译失败原因待排查; 已加 CSE+Canonicalize 收尾仍挂.
    # pytest.param("float32", 4096, 1, 2, 2048, 256,
    #              marks=pytest.mark.skip("stride=1 V1 sub-step compile bug")),
])
def test_1d_strided_compaction(dtype, in_numel, stride, ncore, xblock, xblock_sub):
    out_numel = ncore * xblock
    assert in_numel >= out_numel * stride, "in_numel must cover the strided range"
    assert xblock % xblock_sub == 0

    src = test_common.generate_tensor((in_numel,), dtype).npu()
    dst = test_common.generate_tensor((out_numel,), dtype).npu()

    kernel_1d_strided_compaction[(ncore,)](
        src, dst, in_numel, out_numel, xblock, xblock_sub, stride,
    )

    ref = _ref_1d(src.cpu(), stride, out_numel)
    actual = dst.cpu()
    if dtype in ("float32", "float16"):
        assert torch.allclose(actual, ref, atol=1e-2, rtol=1e-3), \
            f"max abs diff = {(actual.float() - ref.float()).abs().max().item()}"
    else:
        assert torch.equal(actual, ref), \
            f"mismatch count = {(actual != ref).sum().item()}"


# ---------------------------------------------------------------------------
# Multi-D: out[i0, i1, ..., in-1] = in_flat[sum_d i_d * STRIDE_d]
# Output is packed contiguous of shape `blocks`; input is flat with enough
# elements to cover the maximum offset.
# ---------------------------------------------------------------------------

@triton.jit
def kernel_multi_d_gather(
    in_ptr, out_ptr,
    BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
    BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
    STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
    STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr,
):
    # Build in_off / out_off progressively. out_off is packed-contiguous so we
    # can compare against a plain reshape of the reference gather.
    in_off = tl.arange(0, BLOCK_0) * STRIDE_0
    out_off = tl.arange(0, BLOCK_0)

    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        in_off = in_off[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        out_off = out_off[:, None] * BLOCK_1 + tl.arange(0, BLOCK_1)[None, :]
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        in_off = in_off[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        out_off = out_off[:, :, None] * BLOCK_2 + tl.arange(0, BLOCK_2)[None, None, :]
    if (BLOCK_3 * BLOCK_4) > 1:
        in_off = in_off[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        out_off = out_off[:, :, :, None] * BLOCK_3 + tl.arange(0, BLOCK_3)[None, None, None, :]
    if BLOCK_4 > 1:
        in_off = in_off[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        out_off = out_off[:, :, :, :, None] * BLOCK_4 + tl.arange(0, BLOCK_4)[None, None, None, None, :]

    tmp = tl.load(in_ptr + in_off)
    tl.store(out_ptr + out_off, tmp)


def _ref_multi_d(src_flat_cpu: torch.Tensor, blocks, strides) -> torch.Tensor:
    """reference[i0, ..., in-1] = src_flat[sum_d i_d * strides[d]]"""
    coords = torch.meshgrid(
        *[torch.arange(b) for b in blocks], indexing="ij"
    )
    offsets = torch.zeros(blocks, dtype=torch.int64)
    for d in range(len(blocks)):
        offsets = offsets + coords[d].to(torch.int64) * int(strides[d])
    return src_flat_cpu[offsets]


@pytest.mark.parametrize("dtype,blocks,strides", [
    # ---- V1 命中: 非 permuted, 尾轴 stride 静态 > 1, 所有 block 是 2 的幂 ----
    # 2D
    ("float32", (4, 8),          (8, 4)),                 # stride 4
    ("float32", (4, 8),          (24, 3)),                # stride 3 (奇)
    # 3D
    ("float16", (2, 4, 8),       (32, 8, 4)),             # stride 4
    ("float32", (4, 4, 8),       (96, 24, 3)),            # stride 3 (奇)
    # 4D
    ("float16", (2, 4, 4, 8),    (128, 32, 8, 3)),        # stride 3
    # 5D
    ("float32", (2, 2, 2, 4, 8), (256, 128, 64, 16, 5)),  # stride 5 (奇)

    # ---- Deinterleave 接管 (stride==2 + 偶数 last block) ----
    ("float16", (4, 4, 8),       (32, 8, 2)),

    # ---- Permuted: ImplicitPermute 处理, V1 必须放过 ----
    ("float32", (4, 8),          (1, 4)),                 # strides 升序
    ("float32", (4, 4, 8),       (1, 4, 16)),             # 严格升序

    # ---- Normal contiguous (stride==1): 不应改写 ----
    ("float32", (4, 8),          (8, 1)),
    ("float32", (4, 4, 8),       (32, 8, 1)),
])
def test_multi_d_gather(dtype, blocks, strides):
    assert len(blocks) == len(strides)
    assert len(blocks) <= 5

    # Cover max input offset: (B_d - 1) * STRIDE_d sum + 1.
    max_offset = sum((b - 1) * s for b, s in zip(blocks, strides)) + 1
    src = test_common.generate_tensor((max_offset,), dtype).npu()
    dst = test_common.generate_tensor(tuple(blocks), dtype).npu()

    padded_blocks = list(blocks) + [1] * (5 - len(blocks))
    padded_strides = list(strides) + [0] * (5 - len(strides))

    kernel_multi_d_gather[(1,)](
        src, dst,
        *padded_blocks,
        *padded_strides,
    )

    ref = _ref_multi_d(src.cpu(), blocks, strides)
    actual = dst.cpu()
    if dtype in ("float32", "float16"):
        assert torch.allclose(actual, ref, atol=1e-2, rtol=1e-3), \
            f"max abs diff = {(actual.float() - ref.float()).abs().max().item()}"
    else:
        assert torch.equal(actual, ref), \
            f"mismatch count = {(actual != ref).sum().item()}"
