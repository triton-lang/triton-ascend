"""Example 1 optimized variant: aggregate adjacent heads in one program.

Extracted from ``example1/merge_split_ub_aggr.py``.  The fixed
``BLOCK_H=32`` and ``BLOCK_D=head_dim`` choices are intentionally preserved so
this file represents the uploaded after-version exactly.  The missing
``NEGATIVE_INFINITY`` dependency is restored locally.
"""

import torch
import triton
import triton.language as tl

NEGATIVE_INFINITY = tl.constexpr(float("-inf"))


@triton.jit
def _merge_split_states_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    out_ptr,
    lse_ptr,
    stride_po_split,
    stride_po_token,
    stride_po_head,
    stride_pl_split,
    stride_pl_token,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    num_splits,
    num_heads,
    head_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Combine split states for several consecutive heads of one token."""
    token = tl.program_id(0)
    head_block = tl.program_id(1)

    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    splits = tl.arange(0, BLOCK_S)
    dims = tl.arange(0, BLOCK_D)
    head_ok = heads < num_heads
    split_ok = splits < num_splits
    dim_ok = dims < head_dim

    partial_lse = tl.load(
        partial_lse_ptr + splits[:, None] * stride_pl_split + token * stride_pl_token + heads[None, :],
        mask=split_ok[:, None] & head_ok[None, :],
        other=NEGATIVE_INFINITY,
    )
    peak = tl.max(partial_lse, axis=0)
    empty = peak == NEGATIVE_INFINITY
    shift = tl.where(empty, 0.0, peak)
    weights = tl.where(
        split_ok[:, None] & head_ok[None, :],
        tl.exp(partial_lse - shift[None, :]),
        0.0,
    )
    total = tl.sum(weights, axis=0)

    states = tl.load(
        partial_out_ptr + splits[:, None, None] * stride_po_split + token * stride_po_token +
        heads[None, :, None] * stride_po_head + dims[None, None, :],
        mask=(split_ok[:, None, None]
              & head_ok[None, :, None]
              & dim_ok[None, None, :]),
        other=0.0,
    ).to(tl.float32)
    merged = tl.sum(states * weights[:, :, None], axis=0) / tl.maximum(total[:, None], 1.0e-20)

    tl.store(
        out_ptr + token * stride_out_token + heads[:, None] * stride_out_head + dims[None, :],
        merged.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None] & dim_ok[None, :],
    )
    tl.store(
        lse_ptr + token * stride_lse_token + heads,
        tl.where(empty, NEGATIVE_INFINITY, shift + tl.log(total)),
        mask=head_ok,
    )


def merge_split_states(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce split states with 32 adjacent heads aggregated per program."""
    num_splits, tokens, num_heads, head_dim = partial_out.shape
    if partial_lse.shape != (num_splits, tokens, num_heads):
        raise ValueError("partial_lse shape mismatch: expected "
                         f"{(num_splits, tokens, num_heads)}, got {tuple(partial_lse.shape)}")

    block_heads = 32
    out = torch.empty(
        (tokens, num_heads, head_dim),
        device=partial_out.device,
        dtype=partial_out.dtype,
    )
    lse = torch.empty((tokens, num_heads), device=partial_out.device, dtype=torch.float32)
    _merge_split_states_kernel[(tokens, triton.cdiv(num_heads, block_heads))](
        partial_out,
        partial_lse,
        out,
        lse,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        num_splits,
        num_heads,
        head_dim=head_dim,
        BLOCK_H=block_heads,
        BLOCK_S=triton.next_power_of_2(num_splits),
        BLOCK_D=head_dim,
    )
    return out, lse
