"""Independent PyTorch references for the three frozen before kernels."""

from __future__ import annotations

import math

import torch


def ref_merge_split_states(partial_out: torch.Tensor, partial_lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable split-state merge matching the before kernel's mathematical contract."""
    if partial_out.ndim != 4 or partial_lse.ndim != 3:
        raise ValueError("expected partial_out[S,T,H,D] and partial_lse[S,T,H]")
    splits, tokens, heads, _ = partial_out.shape
    if tuple(partial_lse.shape) != (splits, tokens, heads):
        raise ValueError("partial_lse shape does not match partial_out")

    lse = partial_lse.float()
    peak = lse.amax(dim=0)
    empty = torch.isneginf(peak)
    shift = torch.where(empty, torch.zeros_like(peak), peak)
    weights = torch.exp(lse - shift.unsqueeze(0))
    weights = torch.where(empty.unsqueeze(0), torch.zeros_like(weights), weights)
    total = weights.sum(dim=0)
    merged = (partial_out.float() * weights.unsqueeze(-1)).sum(dim=0)
    merged = merged / total.clamp_min(1.0e-20).unsqueeze(-1)
    output_lse = torch.where(empty, peak, shift + torch.log(total))
    return merged.to(partial_out.dtype), output_lse


def _bf16_materialize(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Model the required pre-RoPE materialization boundary."""
    return value.to(dtype).float()


def _normalize_and_rotate(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    weight_bias: float,
    subtract_mean: bool,
) -> torch.Tensor:
    if x.shape != (x.shape[0], num_heads * head_dim):
        raise ValueError("input shape is incompatible with head geometry")
    if not 0 <= 2 * rotary_dim <= head_dim:
        raise ValueError("rotary span must fit in head_dim")

    rows = x.view(x.shape[0], num_heads, head_dim).float()
    if subtract_mean:
        mean = rows.mean(dim=-1, keepdim=True)
        variance = (rows - mean).square().mean(dim=-1, keepdim=True)
    else:
        mean = torch.zeros_like(rows[..., :1])
        variance = rows.square().mean(dim=-1, keepdim=True)
    normalized = (rows - mean) * torch.rsqrt(variance + eps)
    normalized = normalized * (weight.float() + weight_bias)
    if bias is not None:
        normalized = normalized + bias.float()

    # The before DSL widens a BF16/FP16 result before it applies the rotation.
    # Keeping that round-trip here is essential: an all-FP32 reference would be
    # a different numerical contract.
    materialized = _bf16_materialize(normalized, x.dtype)
    result = materialized.clone()
    table_index = positions.to(torch.long)
    selected_cos = cos[table_index].float().unsqueeze(1)
    selected_sin = sin[table_index].float().unsqueeze(1)
    real = materialized[..., :rotary_dim]
    imaginary = materialized[..., rotary_dim:2 * rotary_dim]
    result[..., :rotary_dim] = _bf16_materialize(real * selected_cos - imaginary * selected_sin, x.dtype)
    result[..., rotary_dim:2 * rotary_dim] = _bf16_materialize(real * selected_sin + imaginary * selected_cos, x.dtype)
    return result.to(x.dtype).reshape_as(x)


def ref_indexer_norm_rope(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int = 256,
    num_q_heads: int = 16,
    num_k_heads: int = 1,
    rotary_dim: int = 16,
    eps: float = 1.0e-6,
    q_norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q RMSNorm/RoPE, K LayerNorm/RoPE, and exact Z passthrough."""
    if index_q.shape[0] != index_k.shape[0] or index_q.shape[0] != index_z.shape[0]:
        raise ValueError("Q, K, and Z token counts must agree")
    q = _normalize_and_rotate(
        index_q,
        q_norm_weight,
        None,
        cos,
        sin,
        positions,
        num_heads=num_q_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        weight_bias=q_norm_weight_bias,
        subtract_mean=False,
    )
    k = _normalize_and_rotate(
        index_k,
        k_norm_weight,
        k_norm_bias,
        cos,
        sin,
        positions,
        num_heads=num_k_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        weight_bias=0.0,
        subtract_mean=True,
    )
    return q, k, index_z.contiguous()


def round_activations_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """Match the baseline's E4M3 activation rounding and BF16 widening."""
    return tensor.to(torch.float8_e4m3fn).to(torch.bfloat16)


def ref_indexer_logits(index_q: torch.Tensor, weights: torch.Tensor, index_k: torch.Tensor) -> torch.Tensor:
    """Weighted ReLU score matrix, group-major in the output rows."""
    if index_q.ndim != 4 or weights.ndim != 3 or index_k.ndim != 3:
        raise ValueError("expected Q[S,G,H,D], weights[S,G,H], K[R,1,D]")
    seq_q, groups, heads_per_group, proxy_dim = index_q.shape
    if tuple(weights.shape) != (seq_q, groups, heads_per_group):
        raise ValueError("weights shape does not match Q")
    if tuple(index_k.shape[1:]) != (1, proxy_dim):
        raise ValueError("K must be MQA [seq_k, 1, proxy_dim]")

    q = round_activations_e4m3(index_q).float()
    k = round_activations_e4m3(index_k).float()
    scores = torch.einsum("sghd,rd->sghr", q, k[:, 0])
    scores = torch.relu(scores) * weights.float().unsqueeze(-1)
    return scores.sum(dim=2).permute(1, 0, 2).reshape(groups * seq_q, index_k.shape[0])


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
    return float((actual.float() - expected.float()).abs().max().item())


def bf16_peak_ulp(tensor: torch.Tensor) -> float:
    """Spacing of the BF16 bin containing the largest finite magnitude."""
    peak = float(tensor.detach().float().abs().nan_to_num().max().item())
    if peak == 0.0:
        return math.ldexp(1.0, -133)
    exponent = math.floor(math.log2(peak))
    return math.ldexp(1.0, exponent - 7)
