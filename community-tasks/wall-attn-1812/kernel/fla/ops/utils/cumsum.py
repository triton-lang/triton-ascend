# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Vendored from upstream fla/ops/utils/cumsum.py, reduced to the
# chunk_global_cumsum family required by fla.ops.wall_attn.
# Ascend adaptations:
#   - @fla.ops.backends.dispatch removed (GPU backend routing, NPU-irrelevant);
#   - autotune spaces pruned (num_warps {2,4} x num_stages {1,2}) to bound the
#     NPU compile/autotune sweep — semantic keys and kernel bodies unchanged.

import torch
import triton
import triton.language as tl

from fla.utils import autotune_cache_kwargs, input_guard


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [32, 64, 128, 256]
        for num_warps in [2, 4]
        for num_stages in [1, 2]
    ],
    key=['B', 'H', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos

    b_z = tl.zeros([], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_s = s + bos * H + i_h + o_t * H
        p_o = o + bos * H + i_h + o_t * H
        b_s = tl.load(p_s, mask=m_t, other=0.0).to(tl.float32)
        if REVERSE:
            b_o = tl.cumsum(b_s, axis=0, reverse=True)
        else:
            b_o = tl.cumsum(b_s, axis=0)
        b_ss = tl.sum(b_s, 0)
        b_o += b_z
        if i_c >= 0:
            b_z += b_ss
        if HAS_SCALE:
            b_o *= scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t)


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [16, 32, 64, 128]
        for num_warps in [2, 4]
        for num_stages in [1, 2]
    ],
    key=['B', 'H', 'S', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_s, i_nh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos

    b_z = tl.zeros([BS], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        o_t = i_t * BT + tl.arange(0, BT)
        o_s = i_s * BS + tl.arange(0, BS)
        m_s = (o_t[:, None] < T) & (o_s[None, :] < S)
        p_s = s + (bos * H + i_h) * S + o_t[:, None] * (H * S) + o_s[None, :]
        p_o = o + (bos * H + i_h) * S + o_t[:, None] * (H * S) + o_s[None, :]
        # [BT, BS]
        b_s = tl.load(p_s, mask=m_s, other=0.0).to(tl.float32)
        if REVERSE:
            b_c = b_z[None, :] + tl.cumsum(b_s, axis=0, reverse=True)
        else:
            b_c = b_z[None, :] + tl.cumsum(b_s, axis=0)
        if HAS_SCALE:
            b_c *= scale
        tl.store(p_o, b_c.to(p_o.dtype.element_ty), mask=m_s)
        b_z += tl.sum(b_s, 0)


@input_guard
def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    output_dtype: torch.dtype | None = torch.float,
    **kwargs,
) -> torch.Tensor:
    if 'head_first' in kwargs:
        raise DeprecationWarning("head_first has been removed. Inputs must be in `[B, T, H, ...]` format.", )
    B, T, H = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    grid = (N * H, )
    chunk_global_cumsum_scalar_kernel[grid](
        s=s,
        o=z,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        REVERSE=reverse,
    )
    return z


@input_guard
def chunk_global_cumsum_vector(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    output_dtype: torch.dtype | None = torch.float,
    **kwargs,
) -> torch.Tensor:
    if 'head_first' in kwargs:
        raise DeprecationWarning("head_first has been removed. Inputs must be in `[B, T, H, ...]` format.", )
    B, T, H, S = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BS = min(32, triton.next_power_of_2(S))

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    grid = (triton.cdiv(S, BS), N * H)
    chunk_global_cumsum_vector_kernel[grid](
        s=s,
        o=z,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        S=S,
        BS=BS,
        REVERSE=reverse,
    )
    return z


@input_guard
def chunk_global_cumsum(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    output_dtype: torch.dtype | None = torch.float,
    **kwargs,
) -> torch.Tensor:
    if 'head_first' in kwargs:
        raise DeprecationWarning("head_first has been removed. Inputs must be in `[B, T, H, ...]` format.", )
    if cu_seqlens is not None:
        assert s.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            output_dtype=output_dtype,
        )
    elif len(s.shape) == 4:
        return chunk_global_cumsum_vector(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            output_dtype=output_dtype,
        )
    else:
        raise ValueError(f"Unsupported input shape {s.shape}, "
                         f"which should be [B, T, H] or [B, T, H, D]", )
