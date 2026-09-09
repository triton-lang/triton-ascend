"""Example 2 Stage 0 baseline: one program per ``(token, head)``.

Extracted from ``example2/kernel.py`` with only the dependencies required by
the kernel and its launcher.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _affine_normalize(
    raw,
    mean,
    rstd,
    weight_ptr,
    bias_ptr,
    offsets,
    mask,
    weight_bias,
    HAS_BIAS: tl.constexpr,
):
    """Apply ``(x-mean)*rstd*(weight+weight_bias) [+ bias]``."""
    scale = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = (raw - mean) * rstd * (scale + weight_bias)
    if HAS_BIAS:
        result += tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    return result


@triton.jit
def _indexer_norm_rope_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    stride_x_token,
    stride_out_token,
    stride_cos,
    eps,
    weight_bias,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Normalize one head of one token, then rotate its leading span."""
    token = tl.program_id(0)
    head = tl.program_id(1)
    base = token * stride_x_token + head * head_dim
    out_base = token * stride_out_token + head * head_dim

    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < head_dim
    values = tl.load(x_ptr + base + dims, mask=dim_ok, other=0.0).to(tl.float32)

    sum_sq = tl.sum(values * values)
    if SUBTRACT_MEAN:
        mean = tl.sum(values) / head_dim
        variance = sum_sq / head_dim - mean * mean
    else:
        mean = 0.0
        variance = sum_sq / head_dim
    rstd = tl.rsqrt(variance + eps)

    # Only the non-rotary tail is stored here, avoiding overlapping stores.
    tl.store(
        out_ptr + out_base + dims,
        _affine_normalize(
            values,
            mean,
            rstd,
            weight_ptr,
            bias_ptr,
            dims,
            dim_ok,
            weight_bias,
            HAS_BIAS=HAS_BIAS,
        ).to(out_ptr.dtype.element_ty),
        mask=dim_ok & (dims >= 2 * rotary_dim),
    )

    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim
    position = tl.load(positions_ptr + token).to(tl.int32)
    cos = tl.load(
        cos_ptr + position * stride_cos + half,
        mask=half_ok,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + position * stride_cos + half,
        mask=half_ok,
        other=0.0,
    ).to(tl.float32)

    # The BF16 materialization before RoPE is part of the baseline contract.
    real = _affine_normalize(
        tl.load(x_ptr + base + half, mask=half_ok, other=0.0).to(tl.float32),
        mean,
        rstd,
        weight_ptr,
        bias_ptr,
        half,
        half_ok,
        weight_bias,
        HAS_BIAS=HAS_BIAS,
    ).to(out_ptr.dtype.element_ty).to(tl.float32)
    imaginary = _affine_normalize(
        tl.load(
            x_ptr + base + rotary_dim + half,
            mask=half_ok,
            other=0.0,
        ).to(tl.float32),
        mean,
        rstd,
        weight_ptr,
        bias_ptr,
        rotary_dim + half,
        half_ok,
        weight_bias,
        HAS_BIAS=HAS_BIAS,
    ).to(out_ptr.dtype.element_ty).to(tl.float32)

    tl.store(
        out_ptr + out_base + half,
        (real * cos - imaginary * sin).to(out_ptr.dtype.element_ty),
        mask=half_ok,
    )
    tl.store(
        out_ptr + out_base + rotary_dim + half,
        (real * sin + imaginary * cos).to(out_ptr.dtype.element_ty),
        mask=half_ok,
    )


def _norm_rope(
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
    """Launch the baseline norm+RoPE kernel."""
    out = torch.empty_like(x)
    _indexer_norm_rope_kernel[(x.shape[0], num_heads)](
        x,
        out,
        weight,
        bias if bias is not None else weight,
        cos,
        sin,
        positions,
        x.stride(0),
        out.stride(0),
        cos.stride(0),
        eps,
        weight_bias,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        SUBTRACT_MEAN=subtract_mean,
        HAS_BIAS=bias is not None,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=triton.next_power_of_2(rotary_dim),
    )
    return out


def indexer_norm_rope(
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
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run baseline Q RMSNorm/RoPE, K LayerNorm/RoPE and Z passthrough."""
    return (
        _norm_rope(
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
        ),
        _norm_rope(
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
        ),
        index_z.contiguous(),
    )
