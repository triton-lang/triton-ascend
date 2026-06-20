"""Triton-Ascend implementation of DynamicQuant for BF16 [B, S, H] tensors.

The measured entrypoint is :func:`dynamic_quant`. It implements the custom
DynamicQuant task contract directly in Triton-Ascend kernels and does not call
PyTorch, torch_npu, CANN/vendor DynamicQuant, CPU code, or benchmark golden
logic for the scored computation.
"""

from __future__ import annotations

import os
from typing import Tuple

os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

_ROW_STRIDE_MIN_ROWS = 48
_ROW_STRIDE_GRID = 32


@triton.jit
def _dynamic_quant_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_H: tl.constexpr,
    q_abs: tl.constexpr,
    q_min: tl.constexpr,
    q_max: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    x = tl.load(x_ptr + row * hidden + offs)
    al.multibuffer(x, 2)
    abs_x = tl.abs(x)
    abs_max = tl.max(abs_x, axis=0).to(tl.float32)
    safe_max = tl.where(abs_max == 0.0, 1.0e-12, abs_max)
    scale = safe_max / q_abs
    inv_scale = q_abs / safe_max
    tl.store(scale_ptr + row, scale)
    q = x * inv_scale
    tl.store(out_ptr + row * hidden + offs, q.to(tl.int8))


@triton.jit
def _dynamic_quant_kernel_row_stride(
    x_ptr,
    out_ptr,
    scale_ptr,
    rows,
    hidden: tl.constexpr,
    BLOCK_H: tl.constexpr,
    q_abs: tl.constexpr,
    q_min: tl.constexpr,
    q_max: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_H)
    for row in tl.range(row_start, rows, row_step, num_stages=1):
        x = tl.load(x_ptr + row * hidden + offs)
        al.multibuffer(x, 2)
        abs_x = tl.abs(x)
        abs_max = tl.max(abs_x, axis=0).to(tl.float32)
        safe_max = tl.where(abs_max == 0.0, 1.0e-12, abs_max)
        scale = safe_max / q_abs
        inv_scale = q_abs / safe_max
        tl.store(scale_ptr + row, scale)
        q = x * inv_scale
        tl.store(out_ptr + row * hidden + offs, q.to(tl.int8))


@triton.jit
def _dynamic_quant_loop_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    hidden: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    q_abs: tl.constexpr,
    q_min: tl.constexpr,
    q_max: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    abs_max = tl.full((), 0.0, dtype=tl.float32)
    for chunk in range(0, NUM_CHUNKS):
        h = chunk * BLOCK_H + offs
        mask = h < hidden
        x = tl.load(x_ptr + row * hidden + h, mask=mask, other=0.0).to(tl.float32)
        chunk_abs = tl.max(tl.where(mask, tl.abs(x), 0.0), axis=0)
        abs_max = tl.maximum(abs_max, chunk_abs)
    safe_max = tl.where(abs_max == 0.0, 1.0e-12, abs_max)
    scale = safe_max / q_abs
    inv_scale = q_abs / safe_max
    tl.store(scale_ptr + row, scale)
    for chunk in range(0, NUM_CHUNKS):
        h = chunk * BLOCK_H + offs
        mask = h < hidden
        x = tl.load(x_ptr + row * hidden + h, mask=mask, other=0.0).to(tl.float32)
        q = x * inv_scale
        tl.store(out_ptr + row * hidden + h, q.to(tl.int8), mask=mask)


@triton.jit
def _dynamic_quant_loop_nomask_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    hidden: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    q_abs: tl.constexpr,
    q_min: tl.constexpr,
    q_max: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    abs_max = tl.full((), 0.0, dtype=tl.float32)
    for chunk in range(0, NUM_CHUNKS):
        h = chunk * BLOCK_H + offs
        x = tl.load(x_ptr + row * hidden + h)
        al.multibuffer(x, 2)
        chunk_abs = tl.max(tl.abs(x), axis=0).to(tl.float32)
        abs_max = tl.maximum(abs_max, chunk_abs)
    safe_max = tl.where(abs_max == 0.0, 1.0e-12, abs_max)
    scale = safe_max / q_abs
    inv_scale = q_abs / safe_max
    tl.store(scale_ptr + row, scale)
    for chunk in range(0, NUM_CHUNKS):
        h = chunk * BLOCK_H + offs
        x = tl.load(x_ptr + row * hidden + h)
        al.multibuffer(x, 2)
        q = x * inv_scale
        tl.store(out_ptr + row * hidden + h, q.to(tl.int8))


def _dst_type_params(dst_type: str) -> tuple[float, float, float]:
    value = str(dst_type).lower()
    if value in {"int8", "torch.int8"}:
        return 127.0, -128.0, 127.0
    if value in {"int4", "torch.int4"}:
        return 7.0, -8.0, 7.0
    raise ValueError("dynamic_quant Triton path supports dst_type in {int8, int4}")


def dynamic_quant(x: torch.Tensor, dst_type: str = "int8") -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 tokens along the last dimension.

    Args:
        x: contiguous NPU BF16 tensor with shape ``[B, S, H]``.
        dst_type: ``"int8"`` or ``"int4"``. Logical INT4 values are returned as
            unpacked signed values in int8 storage, matching the CANN-Bench
            task contract.

    Returns:
        ``(output, scale)`` where ``output`` has the same shape as ``x`` and
        int8 storage, and ``scale`` is float32 with shape ``[B, S]``.
    """

    q_abs, q_min, q_max = _dst_type_params(dst_type)
    if x.dim() != 3:
        raise ValueError("dynamic_quant expects rank-3 [B, S, H]")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"dynamic_quant expects bfloat16 input, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("dynamic_quant expects contiguous input")
    if x.device.type != "npu":
        raise ValueError(f"dynamic_quant expects an NPU tensor, got device {x.device}")

    bsz, seq, hidden = [int(v) for v in x.shape]
    if bsz <= 0 or seq <= 0 or hidden <= 0:
        raise ValueError("dynamic_quant expects positive B, S, and H dimensions")
    rows = bsz * seq
    output = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    scale = torch.empty((bsz, seq), device=x.device, dtype=torch.float32)

    if hidden <= 8192:
        block_h = hidden
        if rows == 1 or rows == 8 or rows > _ROW_STRIDE_MIN_ROWS:
            _dynamic_quant_kernel_row_stride[(_ROW_STRIDE_GRID, )](
                x,
                output,
                scale,
                rows,
                hidden,
                BLOCK_H=block_h,
                q_abs=q_abs,
                q_min=q_min,
                q_max=q_max,
            )
            return output, scale
        _dynamic_quant_kernel[(rows, )](
            x,
            output,
            scale,
            rows,
            hidden,
            BLOCK_H=block_h,
            q_abs=q_abs,
            q_min=q_min,
            q_max=q_max,
        )
        return output, scale

    chunk_h = 4096 if hidden >= 8192 else 2048
    num_chunks = (hidden + chunk_h - 1) // chunk_h
    if num_chunks >= 65536:
        raise ValueError("dynamic_quant hidden dimension exceeds Triton-Ascend block limit")
    if hidden % chunk_h == 0:
        _dynamic_quant_loop_nomask_kernel[(rows, )](
            x,
            output,
            scale,
            hidden,
            BLOCK_H=chunk_h,
            NUM_CHUNKS=num_chunks,
            q_abs=q_abs,
            q_min=q_min,
            q_max=q_max,
        )
        return output, scale
    _dynamic_quant_loop_kernel[(rows, )](
        x,
        output,
        scale,
        hidden,
        BLOCK_H=chunk_h,
        NUM_CHUNKS=num_chunks,
        q_abs=q_abs,
        q_min=q_min,
        q_max=q_max,
    )
    return output, scale
