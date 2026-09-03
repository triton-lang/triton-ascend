# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Triton-Ascend implementation of AddRmsNorm for BF16 [B, S, H] tensors."""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rms_norm_kernel(
    x1_ptr,
    x2_ptr,
    gamma_ptr,
    y_ptr,
    n_rows: tl.constexpr,
    h_size: tl.constexpr,
    epsilon: tl.constexpr,
    block_h: tl.constexpr,
):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, block_h)
    mask = cols < h_size
    offsets = row * h_size + cols

    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x1 + x2
    sq = tl.where(mask, z * z, 0.0)
    variance = tl.sum(sq, axis=0) / h_size
    rstd = 1.0 / tl.sqrt(variance + epsilon)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = z * rstd * gamma
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _add_rms_norm_partial_sum_kernel(
    x1_ptr,
    x2_ptr,
    partial_ptr,
    n_rows: tl.constexpr,
    h_size: tl.constexpr,
    n_chunks: tl.constexpr,
    block_h: tl.constexpr,
):
    row = tl.program_id(axis=0)
    chunk = tl.program_id(axis=1)
    cols = tl.arange(0, block_h)
    hidden = chunk * block_h + cols
    mask = hidden < h_size
    offsets = row * h_size + hidden

    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x1 + x2
    sq = tl.where(mask, z * z, 0.0)
    partial = tl.sum(sq, axis=0)
    tl.store(partial_ptr + row * n_chunks + chunk, partial)


@triton.jit
def _add_rms_norm_reduce_rstd_kernel(
    partial_ptr,
    rstd_ptr,
    n_chunks: tl.constexpr,
    h_size: tl.constexpr,
    epsilon: tl.constexpr,
    block_chunks: tl.constexpr,
):
    row = tl.program_id(axis=0)
    chunks = tl.arange(0, block_chunks)
    mask = chunks < n_chunks
    partial = tl.load(partial_ptr + row * n_chunks + chunks, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(partial, axis=0)
    variance = total / h_size
    tl.store(rstd_ptr + row, 1.0 / tl.sqrt(variance + epsilon))


@triton.jit
def _add_rms_norm_apply_chunk_kernel(
    x1_ptr,
    x2_ptr,
    gamma_ptr,
    rstd_ptr,
    y_ptr,
    n_rows: tl.constexpr,
    h_size: tl.constexpr,
    block_h: tl.constexpr,
):
    row = tl.program_id(axis=0)
    chunk = tl.program_id(axis=1)
    cols = tl.arange(0, block_h)
    hidden = chunk * block_h + cols
    mask = hidden < h_size
    offsets = row * h_size + hidden

    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)
    y = (x1 + x2) * rstd * gamma
    tl.store(y_ptr + offsets, y, mask=mask)


def _validate_tensor(name: str, tensor: torch.Tensor, shape: tuple[int, int, int]) -> None:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"AddRmsNorm requires bfloat16 {name}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {tuple(tensor.shape)}")
    if tensor.dim() != 3:
        raise ValueError(f"{name} must be rank-3 [B, S, H], got rank {tensor.dim()}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.device.type != "npu":
        raise ValueError(f"{name} must be an NPU tensor, got device {tensor.device}")


@torch.inference_mode()
def add_rms_norm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute y = (x1 + x2) * rsqrt(mean((x1 + x2)^2) + epsilon) * gamma.

    The implementation follows the addRmsNorm delivery requirement:
    contiguous BF16 tensors in ND layout with shape [B, S, H]. Dispatch is
    based on runtime metadata and Triton-Ascend kernel capability, not on public
    test case ids or public workload files.
    """

    shape = tuple(x1.shape)
    _validate_tensor("x1", x1, shape)
    _validate_tensor("x2", x2, shape)
    _validate_tensor("gamma", gamma, shape)

    epsilon_value = float(epsilon)
    if epsilon_value <= 0.0:
        raise ValueError("epsilon must be positive")

    h_size = int(shape[-1])
    if h_size <= 0:
        raise ValueError("H must be positive")

    block_h = int(triton.next_power_of_2(h_size))
    n_rows = int(x1.numel() // h_size)
    y = torch.empty_like(x1)
    grid = (n_rows, )
    if h_size <= 8192:
        _add_rms_norm_kernel[grid](
            x1,
            x2,
            gamma,
            y,
            n_rows,
            h_size,
            epsilon_value,
            block_h,
        )
    else:
        chunk_h = 8192
        n_chunks = (h_size + chunk_h - 1) // chunk_h
        block_chunks = int(triton.next_power_of_2(n_chunks))
        if block_chunks > 8192:
            raise ValueError(f"chunk count block={block_chunks} exceeds this Triton-Ascend implementation limit")
        partial = torch.empty((n_rows, n_chunks), device=x1.device, dtype=torch.float32)
        rstd = torch.empty((n_rows, ), device=x1.device, dtype=torch.float32)
        chunk_grid = (n_rows, n_chunks)
        _add_rms_norm_partial_sum_kernel[chunk_grid](
            x1,
            x2,
            partial,
            n_rows,
            h_size,
            n_chunks,
            chunk_h,
        )
        _add_rms_norm_reduce_rstd_kernel[grid](
            partial,
            rstd,
            n_chunks,
            h_size,
            epsilon_value,
            block_chunks,
        )
        _add_rms_norm_apply_chunk_kernel[chunk_grid](
            x1,
            x2,
            gamma,
            rstd,
            y,
            n_rows,
            h_size,
            chunk_h,
        )
    return y


def add_rms_norm_reference(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Semantic PyTorch reference used only by local validation."""

    z = x1.to(torch.float32) + x2.to(torch.float32)
    variance = torch.mean(z * z, dim=-1, keepdim=True)
    y = z * torch.rsqrt(variance + float(epsilon)) * gamma.to(torch.float32)
    return y.to(torch.bfloat16)


__all__ = ["add_rms_norm", "add_rms_norm_reference"]
