"""Example 2 Stage 0 active optimized norm+RoPE implementation.

Extracted from the active ``_norm_rope_opt`` path in
``example2/optimized_kernels.py``.  The uploaded implementation is preserved,
including its BF16-boundary and overlapping-store behavior, so this remains a
faithful before/after comparison artifact rather than a repaired variant.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension


def _npu_vec_core_num() -> int:
    import triton.runtime.driver as driver

    properties = driver.active.utils.get_device_properties(torch.npu.current_device())
    return properties["num_vectorcore"]


@triton.jit
def _indexer_norm_rope_kernel_opt(
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
    tokens,
    NUM_HEADS: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_T: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    num_tiles = (tokens + BLOCK_T - 1) // BLOCK_T

    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < head_dim
    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim

    # Head-independent vectors are loaded once per persistent program.
    w_full = tl.load(weight_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
    w_real = tl.load(weight_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
    w_imag = tl.load(weight_ptr + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b_full = tl.load(bias_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
        b_real = tl.load(bias_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
        b_imag = tl.load(bias_ptr + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32)

    # Flatten [BLOCK_T, NUM_HEADS] into independent norm rows.
    gr = tl.arange(0, BLOCK_T * NUM_HEADS)
    tok_idx = gr // NUM_HEADS
    head_idx = gr % NUM_HEADS

    for tile_id in tl.range(pid, num_tiles, nprog, num_stages=2):
        token_start = tile_id * BLOCK_T
        tok_abs = token_start + tok_idx
        row_ok = tok_abs < tokens

        x_bp = tl.make_block_ptr(
            base=x_ptr,
            shape=(tokens, NUM_HEADS * head_dim),
            strides=(stride_x_token, 1),
            offsets=(token_start, 0),
            block_shape=(BLOCK_T, NUM_HEADS * head_dim),
            order=(1, 0),
        )
        x_tile = tl.load(x_bp, boundary_check=(0, ), padding_option="zero").to(tl.float32)
        values = tl.reshape(x_tile, (BLOCK_T * NUM_HEADS, head_dim))

        sum_sq = tl.sum(values * values, axis=1)
        if SUBTRACT_MEAN:
            mean = tl.sum(values, axis=1) / head_dim
            variance = sum_sq / head_dim - mean * mean
        else:
            mean = tl.zeros((BLOCK_T * NUM_HEADS, ), dtype=tl.float32)
            variance = sum_sq / head_dim
        rstd = tl.rsqrt(variance + eps)

        normalized = ((values - mean[:, None]) * rstd[:, None] * (w_full[None, :] + weight_bias))
        if HAS_BIAS:
            normalized = normalized + b_full[None, :]

        # This full store is followed by stores that overwrite the rotary span.
        out_bp = tl.make_block_ptr(
            base=out_ptr,
            shape=(tokens, NUM_HEADS * head_dim),
            strides=(stride_out_token, 1),
            offsets=(token_start, 0),
            block_shape=(BLOCK_T, NUM_HEADS * head_dim),
            order=(1, 0),
        )
        tl.store(
            out_bp,
            tl.reshape(
                normalized.to(out_ptr.dtype.element_ty),
                (BLOCK_T, NUM_HEADS * head_dim),
            ),
            boundary_check=(0, ),
        )

        token_offsets = token_start + tl.arange(0, BLOCK_T)
        token_mask = token_offsets < tokens
        position = tl.load(positions_ptr + token_offsets, mask=token_mask, other=0).to(tl.int32)
        cos_t = tl.load(
            cos_ptr + position[:, None] * stride_cos + half[None, :],
            mask=token_mask[:, None] & half_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        extension.compile_hint(cos_t, "mayDiscretememaccess")
        sin_t = tl.load(
            sin_ptr + position[:, None] * stride_cos + half[None, :],
            mask=token_mask[:, None] & half_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        extension.compile_hint(sin_t, "mayDiscretememaccess")

        cos = tl.reshape(
            tl.broadcast_to(cos_t[:, None, :], (BLOCK_T, NUM_HEADS, BLOCK_R)),
            (BLOCK_T * NUM_HEADS, BLOCK_R),
        )
        sin = tl.reshape(
            tl.broadcast_to(sin_t[:, None, :], (BLOCK_T, NUM_HEADS, BLOCK_R)),
            (BLOCK_T * NUM_HEADS, BLOCK_R),
        )

        if NUM_HEADS == 1:
            real_raw = tl.load(
                x_ptr + tok_abs[:, None] * stride_x_token + half[None, :],
                mask=row_ok[:, None] & half_ok[None, :],
                other=0.0,
            ).to(tl.float32)
            imag_raw = tl.load(
                x_ptr + tok_abs[:, None] * stride_x_token + rotary_dim + half[None, :],
                mask=row_ok[:, None] & half_ok[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
            real_raw = extension.extract_slice(
                values,
                [0, 0],
                [BLOCK_T * NUM_HEADS, BLOCK_R],
                [1, 1],
            )
            imag_raw = extension.extract_slice(
                values,
                [0, rotary_dim],
                [BLOCK_T * NUM_HEADS, BLOCK_R],
                [1, 1],
            )

        real = ((real_raw - mean[:, None]) * rstd[:, None] * (w_real[None, :] + weight_bias))
        imag = ((imag_raw - mean[:, None]) * rstd[:, None] * (w_imag[None, :] + weight_bias))
        if HAS_BIAS:
            real = real + b_real[None, :]
            imag = imag + b_imag[None, :]

        out_base = tok_abs * stride_out_token + head_idx * head_dim
        mask_half = row_ok[:, None] & half_ok[None, :]
        tl.store(
            out_ptr + out_base[:, None] + half[None, :],
            (real * cos - imag * sin).to(out_ptr.dtype.element_ty),
            mask=mask_half,
        )
        tl.store(
            out_ptr + out_base[:, None] + rotary_dim + half[None, :],
            (real * sin + imag * cos).to(out_ptr.dtype.element_ty),
            mask=mask_half,
        )


def _pick_block_t(num_heads: int, head_dim: int) -> int:
    """Choose tokens per tile using the uploaded 196-KiB heuristic."""
    ub_budget = 196 * 1024
    bytes_per_token = num_heads * head_dim * 4 * 3
    block_t = max(2, min(64, ub_budget // bytes_per_token))
    return triton.next_power_of_2(block_t)


def _norm_rope_opt_flat(
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
    """Launch the flat persistent optimized kernel."""
    out = torch.empty_like(x)
    tokens = x.shape[0]
    block_t = _pick_block_t(num_heads, head_dim)
    grid = (min(triton.cdiv(tokens, block_t), _npu_vec_core_num()), )
    _indexer_norm_rope_kernel_opt[grid](
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
        tokens,
        NUM_HEADS=num_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        SUBTRACT_MEAN=subtract_mean,
        HAS_BIAS=bias is not None,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=triton.next_power_of_2(rotary_dim),
        BLOCK_T=block_t,
    )
    return out


def _norm_rope_opt(
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
    """Public optimized entry matching the source integration point."""
    return _norm_rope_opt_flat(
        x,
        weight,
        bias,
        cos,
        sin,
        positions,
        num_heads=num_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        weight_bias=weight_bias,
        subtract_mean=subtract_mean,
    )


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
    """Run optimized Q/K norm+RoPE paths and preserve Z passthrough."""
    return (
        _norm_rope_opt(
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
        _norm_rope_opt(
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
