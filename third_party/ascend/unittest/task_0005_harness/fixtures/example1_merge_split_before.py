"""Example 1 baseline: one Triton program per ``(token, head)``.

Extracted from ``example1/merge_split_origin.py``.  ``NEGATIVE_INFINITY`` is
included here because the original snippet relied on that symbol being defined
by its containing merged module.
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
    head_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Combine per-split attention states for one token/head row."""
    token = tl.program_id(0)
    head = tl.program_id(1)

    splits = tl.arange(0, BLOCK_S)
    dims = tl.arange(0, BLOCK_D)
    split_ok = splits < num_splits
    dim_ok = dims < head_dim

    partial_lse = tl.load(
        partial_lse_ptr + splits * stride_pl_split + token * stride_pl_token + head,
        mask=split_ok,
        other=NEGATIVE_INFINITY,
    )
    peak = tl.max(partial_lse)
    empty = peak == NEGATIVE_INFINITY
    shift = tl.where(empty, 0.0, peak)
    weights = tl.where(split_ok, tl.exp(partial_lse - shift), 0.0)
    total = tl.sum(weights)

    states = tl.load(
        partial_out_ptr + splits[:, None] * stride_po_split + token * stride_po_token + head * stride_po_head +
        dims[None, :],
        mask=split_ok[:, None] & dim_ok[None, :],
        other=0.0,
    ).to(tl.float32)
    merged = tl.sum(states * weights[:, None], axis=0) / tl.maximum(total, 1.0e-20)

    tl.store(
        out_ptr + token * stride_out_token + head * stride_out_head + dims,
        merged.to(out_ptr.dtype.element_ty),
        mask=dim_ok,
    )
    tl.store(
        lse_ptr + token * stride_lse_token + head,
        tl.where(empty, NEGATIVE_INFINITY, shift + tl.log(total)),
    )


def merge_split_states(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``[splits,tokens,heads,dim]`` states to token/head outputs."""
    num_splits, tokens, num_heads, head_dim = partial_out.shape
    out = torch.empty(
        (tokens, num_heads, head_dim),
        device=partial_out.device,
        dtype=partial_out.dtype,
    )
    lse = torch.empty((tokens, num_heads), device=partial_out.device, dtype=torch.float32)
    _merge_split_states_kernel[(tokens, num_heads)](
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
        head_dim=head_dim,
        BLOCK_S=triton.next_power_of_2(num_splits),
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return out, lse
