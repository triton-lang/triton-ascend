# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Triton-Ascend implementation for KvRmsNormRopeCache."""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _update_inplace_64_kernel(
    kv,
    gamma,
    cos,
    sin,
    index,
    k_cache,
    ckv_cache,
    bsz: tl.constexpr,
    heads: tl.constexpr,
    skv: tl.constexpr,
    scache: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0)
    b = tl.program_id(1)

    d = tl.arange(0, BLOCK_D)
    dh = tl.arange(0, 32)
    safe_b = tl.where(b < bsz, b, 0)
    gamma_v = tl.load(gamma + d).to(tl.float32)
    for ss in range(0, skv):
        idx = tl.load(index + safe_b * skv + ss, mask=b < bsz, care_padding=False)
        valid = (b < bsz) & (idx >= 0) & (idx < scache)
        safe_idx = tl.where(valid, idx, 0)
        cache_base = ((b * heads + n) * scache + safe_idx) * 64
        kv_base = ((safe_b * heads + n) * skv + ss) * 128

        rope_lo = tl.load(kv + kv_base + dh, mask=valid, care_padding=False).to(tl.float32)
        rope_hi = tl.load(kv + kv_base + 32 + dh, mask=valid, care_padding=False).to(tl.float32)
        value = tl.load(kv + kv_base + 64 + d, mask=valid, care_padding=False).to(tl.float32)
        cos_lo = tl.load(cos + safe_b * 64 + dh, mask=valid, care_padding=False).to(tl.float32)
        cos_hi = tl.load(cos + safe_b * 64 + 32 + dh, mask=valid, care_padding=False).to(tl.float32)
        sin_lo = tl.load(sin + safe_b * 64 + dh, mask=valid, care_padding=False).to(tl.float32)
        sin_hi = tl.load(sin + safe_b * 64 + 32 + dh, mask=valid, care_padding=False).to(tl.float32)

        k_lo = rope_lo * cos_lo - rope_hi * sin_lo
        k_hi = rope_hi * cos_hi + rope_lo * sin_hi

        variance = tl.sum(value * value, axis=0) / 64.0
        ckv_val = value * tl.rsqrt(variance + epsilon) * gamma_v

        tl.store(k_cache + cache_base + dh, k_lo, mask=valid)
        tl.store(k_cache + cache_base + 32 + dh, k_hi, mask=valid)
        tl.store(ckv_cache + cache_base + d, ckv_val, mask=valid)


@triton.jit
def _update_inplace_dynamic_split_kernel(
    kv,
    gamma,
    cos,
    sin,
    index,
    k_cache,
    ckv_cache,
    bsz: tl.constexpr,
    heads: tl.constexpr,
    skv: tl.constexpr,
    scache: tl.constexpr,
    epsilon: tl.constexpr,
    d_rope: tl.constexpr,
    d_value: tl.constexpr,
    d_total: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    n = tl.program_id(0)
    b = tl.program_id(1)

    half = d_rope // 2
    dh = tl.arange(0, BLOCK_KH)
    dv = tl.arange(0, BLOCK_V)
    half_mask = dh < half
    value_mask = dv < d_value
    safe_b = tl.where(b < bsz, b, 0)
    gamma_v = tl.load(gamma + dv, mask=value_mask, other=0.0).to(tl.float32)
    for ss in range(0, skv):
        idx = tl.load(index + safe_b * skv + ss, mask=b < bsz, care_padding=False)
        valid = (b < bsz) & (idx >= 0) & (idx < scache)
        safe_idx = tl.where(valid, idx, 0)
        kv_base = ((safe_b * heads + n) * skv + ss) * d_total
        k_cache_base = ((b * heads + n) * scache + safe_idx) * d_rope
        ckv_cache_base = ((b * heads + n) * scache + safe_idx) * d_value

        rope_lo = tl.load(kv + kv_base + dh, mask=valid & half_mask, care_padding=False).to(tl.float32)
        rope_hi = tl.load(kv + kv_base + half + dh, mask=valid & half_mask, care_padding=False).to(tl.float32)
        cos_lo = tl.load(cos + safe_b * d_rope + dh, mask=valid & half_mask, care_padding=False).to(tl.float32)
        cos_hi = tl.load(cos + safe_b * d_rope + half + dh, mask=valid & half_mask, care_padding=False).to(tl.float32)
        sin_lo = tl.load(sin + safe_b * d_rope + dh, mask=valid & half_mask, care_padding=False).to(tl.float32)
        sin_hi = tl.load(sin + safe_b * d_rope + half + dh, mask=valid & half_mask, care_padding=False).to(tl.float32)
        k_lo = rope_lo * cos_lo - rope_hi * sin_lo
        k_hi = rope_hi * cos_hi + rope_lo * sin_hi

        value = tl.load(kv + kv_base + d_rope + dv, mask=valid & value_mask, other=0.0).to(tl.float32)
        variance = tl.sum(value * value, axis=0) / d_value
        ckv_val = value * tl.rsqrt(variance + epsilon) * gamma_v

        tl.store(k_cache + k_cache_base + dh, k_lo, mask=valid & half_mask)
        tl.store(k_cache + k_cache_base + half + dh, k_hi, mask=valid & half_mask)
        tl.store(ckv_cache + ckv_cache_base + dv, ckv_val, mask=valid & value_mask)


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def kv_rms_norm_rope_cache(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    index: torch.Tensor,
    kCacheRef: torch.Tensor,
    ckvCacheRef: torch.Tensor,
    epsilon: float = 1.0e-5,
    cache_mode: str = "Norm",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if str(cache_mode) != "Norm":
        raise ValueError('kv_rms_norm_rope_cache supports only cache_mode="Norm"')
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")
    if kv.dim() != 4:
        raise ValueError("kv must have shape [B, N, Skv, Dk + Dv]")
    if gamma.dim() != 1:
        raise ValueError("gamma must have shape [Dv]")
    if cos.dim() != 4 or sin.dim() != 4 or tuple(cos.shape) != tuple(sin.shape):
        raise ValueError("cos and sin must have matching shape [B, 1, 1, Dk]")
    if index.dim() != 2:
        raise ValueError("index must have shape [B, Skv]")
    if kCacheRef.dim() != 4 or ckvCacheRef.dim() != 4:
        raise ValueError("cache tensors must have shape [Bcache, N, Scache, Dk/Dv]")
    bsz, heads, skv, dim = [int(v) for v in kv.shape]
    d_rope = int(cos.shape[-1])
    d_value = int(gamma.shape[0])
    if d_rope <= 0 or d_value <= 0:
        raise ValueError("Dk and Dv must be positive")
    if d_rope % 2 != 0:
        raise ValueError("RoPE dimension Dk must be even")
    if dim != d_rope + d_value:
        raise ValueError("kv last dimension must equal Dk + Dv")
    if tuple(index.shape) != (bsz, skv):
        raise ValueError("index shape must match [B, Skv]")
    if tuple(cos.shape) != (bsz, 1, 1, d_rope):
        raise ValueError("cos/sin shape must match [B, 1, 1, Dk]")
    if int(kCacheRef.shape[0]) < bsz or int(ckvCacheRef.shape[0]) < bsz:
        raise ValueError("Bcache must be at least B")
    if int(kCacheRef.shape[1]) != heads or int(ckvCacheRef.shape[1]) != heads:
        raise ValueError("cache head count must match kv N")
    if int(kCacheRef.shape[2]) != int(ckvCacheRef.shape[2]):
        raise ValueError("cache Scache dimensions must match")
    if int(kCacheRef.shape[-1]) != d_rope or int(ckvCacheRef.shape[-1]) != d_value:
        raise ValueError("kCacheRef last dimension must equal Dk and ckvCacheRef last dimension must equal Dv")
    for name, tensor, dtype in (
        ("kv", kv, torch.bfloat16),
        ("gamma", gamma, torch.bfloat16),
        ("cos", cos, torch.bfloat16),
        ("sin", sin, torch.bfloat16),
        ("index", index, torch.int64),
        ("kCacheRef", kCacheRef, torch.bfloat16),
        ("ckvCacheRef", ckvCacheRef, torch.bfloat16),
    ):
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    scache = int(kCacheRef.shape[2])
    if d_rope == 64 and d_value == 64:
        _update_inplace_64_kernel[(heads, bsz)](
            kv,
            gamma,
            cos,
            sin,
            index,
            kCacheRef,
            ckvCacheRef,
            bsz,
            heads,
            skv,
            scache,
            float(epsilon),
            BLOCK_D=64,
        )
    else:
        block_kh = _next_power_of_2(d_rope // 2)
        block_v = _next_power_of_2(d_value)
        if _next_power_of_2(max(d_rope, d_value)) > 1024:
            raise ValueError("Dk/Dv split is too large for this Triton-Ascend kernel")
        _update_inplace_dynamic_split_kernel[(heads, bsz)](
            kv,
            gamma,
            cos,
            sin,
            index,
            kCacheRef,
            ckvCacheRef,
            bsz,
            heads,
            skv,
            scache,
            float(epsilon),
            d_rope,
            d_value,
            dim,
            BLOCK_KH=block_kh,
            BLOCK_V=block_v,
        )
    return kCacheRef, ckvCacheRef


__all__ = ["kv_rms_norm_rope_cache"]
