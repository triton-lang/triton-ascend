from __future__ import annotations

from typing import Sequence, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _mrope_apply_kernel(
    positions,
    x,
    cache,
    out,
    total: tl.constexpr,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    head_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_dim: tl.constexpr,
    max_seq: tl.constexpr,
    pos_rows: tl.constexpr,
    sec0: tl.constexpr,
    sec1: tl.constexpr,
    sec2: tl.constexpr,
    sec3: tl.constexpr,
    IS_ROPE: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    CACHE_INTERLEAVED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    d = offs % head_size
    head_token = offs // head_size
    token = head_token // heads
    base = head_token * head_size
    x_val = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)
    rotate_mask = mask & (d < rotary_dim)

    if ROTARY_INTERLEAVED:
        cos_idx = d // 2
        pair_d = tl.where((d % 2) == 0, d + 1, d - 1)
        pair_val = tl.load(x + base + pair_d, mask=rotate_mask, other=0.0).to(tl.float32)
    else:
        cos_idx = tl.where(d < half_dim, d, d - half_dim)
        pair_d = tl.where(d < half_dim, d + half_dim, d - half_dim)
        pair_val = tl.load(x + base + pair_d, mask=rotate_mask, other=0.0).to(tl.float32)

    if IS_ROPE:
        pos = tl.load(positions + token, mask=rotate_mask, other=0)
    else:
        cut0 = sec0
        cut1 = sec0 + sec1
        cut2 = sec0 + sec1 + sec2
        row = tl.full((BLOCK, ), 0, dtype=tl.int64)
        row = tl.where(cos_idx >= cut0, 1, row)
        row = tl.where(cos_idx >= cut1, 2, row)
        row = tl.where(cos_idx >= cut2, 3, row)
        row = tl.minimum(row, pos_rows - 1)
        pos = tl.load(positions + row * tokens + token, mask=rotate_mask, other=0)
    pos = tl.minimum(tl.maximum(pos, 0), max_seq - 1)

    if CACHE_INTERLEAVED:
        cos_val = tl.load(cache + pos * rotary_dim + cos_idx * 2, mask=rotate_mask, other=1.0).to(tl.float32)
        sin_val = tl.load(cache + pos * rotary_dim + cos_idx * 2 + 1, mask=rotate_mask, other=0.0).to(tl.float32)
    else:
        cos_val = tl.load(cache + pos * rotary_dim + cos_idx, mask=rotate_mask, other=1.0).to(tl.float32)
        sin_val = tl.load(cache + pos * rotary_dim + half_dim + cos_idx, mask=rotate_mask, other=0.0).to(tl.float32)

    if ROTARY_INTERLEAVED:
        even = (d % 2) == 0
        rotated_even = x_val * cos_val - pair_val * sin_val
        rotated_odd = x_val * cos_val + pair_val * sin_val
        rotated = tl.where(even, rotated_even, rotated_odd)
    else:
        first_half = d < half_dim
        rotated_first = x_val * cos_val - pair_val * sin_val
        rotated_second = x_val * cos_val + pair_val * sin_val
        rotated = tl.where(first_half, rotated_first, rotated_second)

    out_val = tl.where(rotate_mask, rotated, x_val)
    tl.store(out + offs, out_val, mask=mask)


@triton.jit
def _mrope_full_half_default_2d_kernel(
    positions,
    x,
    cache,
    out,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    max_seq: tl.constexpr,
    pos_rows: tl.constexpr,
    sec0: tl.constexpr,
    sec1: tl.constexpr,
    sec2: tl.constexpr,
    head_blocks: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // head_blocks
    head_block_id = pid - token * head_blocks

    cols_1d = tl.arange(0, 64)
    rows = tl.full((64, ), 0, dtype=tl.int32)
    cut0 = sec0
    cut1 = sec0 + sec1
    cut2 = sec0 + sec1 + sec2
    rows = tl.where(cols_1d >= cut0, 1, rows)
    rows = tl.where(cols_1d >= cut1, 2, rows)
    rows = tl.where(cols_1d >= cut2, 3, rows)
    rows = tl.minimum(rows, pos_rows - 1)
    pos = tl.load(positions + rows * tokens + token).to(tl.int32)
    pos = tl.minimum(tl.maximum(pos, 0), max_seq - 1)

    cos_vec = tl.load(cache + pos * 128 + cols_1d).to(tl.float32)
    sin_vec = tl.load(cache + pos * 128 + 64 + cols_1d).to(tl.float32)

    h = tl.arange(0, HEAD_BLOCK)[:, None]
    cols = tl.arange(0, 64)[None, :]
    head = head_block_id * HEAD_BLOCK + h
    mask = head < heads
    base = (token * heads + head) * 128
    first = tl.load(x + base + cols, mask=mask, other=0.0).to(tl.float32)
    second = tl.load(x + base + 64 + cols, mask=mask, other=0.0).to(tl.float32)
    cos_val = cos_vec[None, :]
    sin_val = sin_vec[None, :]
    tl.store(out + base + cols, first * cos_val - second * sin_val, mask=mask)
    tl.store(out + base + 64 + cols, first * sin_val + second * cos_val, mask=mask)


@triton.jit
def _mrope_full_half_default_2d_pair_kernel(
    positions,
    query,
    key,
    cache,
    query_out,
    key_out,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    max_seq: tl.constexpr,
    pos_rows: tl.constexpr,
    sec0: tl.constexpr,
    sec1: tl.constexpr,
    sec2: tl.constexpr,
    head_blocks: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // head_blocks
    head_block_id = pid - token * head_blocks

    cols_1d = tl.arange(0, 64)
    rows = tl.full((64, ), 0, dtype=tl.int32)
    cut0 = sec0
    cut1 = sec0 + sec1
    cut2 = sec0 + sec1 + sec2
    rows = tl.where(cols_1d >= cut0, 1, rows)
    rows = tl.where(cols_1d >= cut1, 2, rows)
    rows = tl.where(cols_1d >= cut2, 3, rows)
    rows = tl.minimum(rows, pos_rows - 1)
    pos = tl.load(positions + rows * tokens + token).to(tl.int32)
    pos = tl.minimum(tl.maximum(pos, 0), max_seq - 1)

    cos_vec = tl.load(cache + pos * 128 + cols_1d).to(tl.float32)
    sin_vec = tl.load(cache + pos * 128 + 64 + cols_1d).to(tl.float32)

    h = tl.arange(0, HEAD_BLOCK)[:, None]
    cols = tl.arange(0, 64)[None, :]
    head = head_block_id * HEAD_BLOCK + h
    mask = head < heads
    base = (token * heads + head) * 128
    cos_val = cos_vec[None, :]
    sin_val = sin_vec[None, :]

    query_first = tl.load(query + base + cols, mask=mask, other=0.0).to(tl.float32)
    query_second = tl.load(query + base + 64 + cols, mask=mask, other=0.0).to(tl.float32)
    key_first = tl.load(key + base + cols, mask=mask, other=0.0).to(tl.float32)
    key_second = tl.load(key + base + 64 + cols, mask=mask, other=0.0).to(tl.float32)

    tl.store(query_out + base + cols, query_first * cos_val - query_second * sin_val, mask=mask)
    tl.store(query_out + base + 64 + cols, query_first * sin_val + query_second * cos_val, mask=mask)
    tl.store(key_out + base + cols, key_first * cos_val - key_second * sin_val, mask=mask)
    tl.store(key_out + base + 64 + cols, key_first * sin_val + key_second * cos_val, mask=mask)


@triton.jit
def _rope_full_half_default_2d_pair_kernel(
    positions,
    query,
    key,
    cache,
    query_out,
    key_out,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    max_seq: tl.constexpr,
    head_blocks: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // head_blocks
    head_block_id = pid - token * head_blocks

    cols_1d = tl.arange(0, 64)
    pos = tl.load(positions + token).to(tl.int32)
    pos = tl.minimum(tl.maximum(pos, 0), max_seq - 1)
    cos_vec = tl.load(cache + pos * 128 + cols_1d).to(tl.float32)
    sin_vec = tl.load(cache + pos * 128 + 64 + cols_1d).to(tl.float32)

    h = tl.arange(0, HEAD_BLOCK)[:, None]
    cols = tl.arange(0, 64)[None, :]
    head = head_block_id * HEAD_BLOCK + h
    mask = head < heads
    base = (token * heads + head) * 128
    cos_val = cos_vec[None, :]
    sin_val = sin_vec[None, :]

    query_first = tl.load(query + base + cols, mask=mask, other=0.0).to(tl.float32)
    query_second = tl.load(query + base + 64 + cols, mask=mask, other=0.0).to(tl.float32)
    key_first = tl.load(key + base + cols, mask=mask, other=0.0).to(tl.float32)
    key_second = tl.load(key + base + 64 + cols, mask=mask, other=0.0).to(tl.float32)

    tl.store(query_out + base + cols, query_first * cos_val - query_second * sin_val, mask=mask)
    tl.store(query_out + base + 64 + cols, query_first * sin_val + query_second * cos_val, mask=mask)
    tl.store(key_out + base + cols, key_first * cos_val - key_second * sin_val, mask=mask)
    tl.store(key_out + base + 64 + cols, key_first * sin_val + key_second * cos_val, mask=mask)


@triton.jit
def _mrope_half_interleave64_2d_pair_kernel(
    positions,
    query,
    key,
    cache,
    query_out,
    key_out,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    max_seq: tl.constexpr,
    pos_rows: tl.constexpr,
    sec0: tl.constexpr,
    sec1: tl.constexpr,
    sec2: tl.constexpr,
    head_blocks: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // head_blocks
    head_block_id = pid - token * head_blocks

    cols_1d = tl.arange(0, 32)
    rows = tl.full((32, ), 0, dtype=tl.int32)
    cut0 = sec0
    cut1 = sec0 + sec1
    cut2 = sec0 + sec1 + sec2
    rows = tl.where(cols_1d >= cut0, 1, rows)
    rows = tl.where(cols_1d >= cut1, 2, rows)
    rows = tl.where(cols_1d >= cut2, 3, rows)
    rows = tl.minimum(rows, pos_rows - 1)
    pos = tl.load(positions + rows * tokens + token).to(tl.int32)
    pos = tl.minimum(tl.maximum(pos, 0), max_seq - 1)

    cos_vec = tl.load(cache + pos * 64 + cols_1d * 2).to(tl.float32)
    sin_vec = tl.load(cache + pos * 64 + cols_1d * 2 + 1).to(tl.float32)

    h = tl.arange(0, HEAD_BLOCK)[:, None]
    rot_cols = tl.arange(0, 32)[None, :]
    tail_cols = tl.arange(0, 64)[None, :]
    head = head_block_id * HEAD_BLOCK + h
    mask = head < heads
    base = (token * heads + head) * 128
    cos_val = cos_vec[None, :]
    sin_val = sin_vec[None, :]

    query_first = tl.load(query + base + rot_cols, mask=mask, other=0.0).to(tl.float32)
    query_second = tl.load(query + base + 32 + rot_cols, mask=mask, other=0.0).to(tl.float32)
    key_first = tl.load(key + base + rot_cols, mask=mask, other=0.0).to(tl.float32)
    key_second = tl.load(key + base + 32 + rot_cols, mask=mask, other=0.0).to(tl.float32)

    tl.store(query_out + base + rot_cols, query_first * cos_val - query_second * sin_val, mask=mask)
    tl.store(query_out + base + 32 + rot_cols, query_first * sin_val + query_second * cos_val, mask=mask)
    tl.store(key_out + base + rot_cols, key_first * cos_val - key_second * sin_val, mask=mask)
    tl.store(key_out + base + 32 + rot_cols, key_first * sin_val + key_second * cos_val, mask=mask)

    query_tail = tl.load(query + base + 64 + tail_cols, mask=mask, other=0.0)
    key_tail = tl.load(key + base + 64 + tail_cols, mask=mask, other=0.0)
    tl.store(query_out + base + 64 + tail_cols, query_tail, mask=mask)
    tl.store(key_out + base + 64 + tail_cols, key_tail, mask=mask)


@triton.jit
def _mrope_interleaved_default_2d_pair_kernel(
    positions,
    query,
    key,
    cache,
    query_out,
    key_out,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    max_seq: tl.constexpr,
    pos_rows: tl.constexpr,
    sec0: tl.constexpr,
    sec1: tl.constexpr,
    sec2: tl.constexpr,
    head_blocks: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // head_blocks
    head_block_id = pid - token * head_blocks

    cols_1d = tl.arange(0, 64)
    rows = tl.full((64, ), 0, dtype=tl.int32)
    cut0 = sec0
    cut1 = sec0 + sec1
    cut2 = sec0 + sec1 + sec2
    rows = tl.where(cols_1d >= cut0, 1, rows)
    rows = tl.where(cols_1d >= cut1, 2, rows)
    rows = tl.where(cols_1d >= cut2, 3, rows)
    rows = tl.minimum(rows, pos_rows - 1)
    pos = tl.load(positions + rows * tokens + token).to(tl.int32)
    pos = tl.minimum(tl.maximum(pos, 0), max_seq - 1)

    cos_vec = tl.load(cache + pos * 128 + cols_1d).to(tl.float32)
    sin_vec = tl.load(cache + pos * 128 + 64 + cols_1d).to(tl.float32)

    h = tl.arange(0, HEAD_BLOCK)[:, None]
    cols = tl.arange(0, 64)[None, :]
    head = head_block_id * HEAD_BLOCK + h
    mask = head < heads
    base = (token * heads + head) * 128
    even_cols = cols * 2
    odd_cols = even_cols + 1
    cos_val = cos_vec[None, :]
    sin_val = sin_vec[None, :]

    query_even = tl.load(query + base + even_cols, mask=mask, other=0.0).to(tl.float32)
    query_odd = tl.load(query + base + odd_cols, mask=mask, other=0.0).to(tl.float32)
    key_even = tl.load(key + base + even_cols, mask=mask, other=0.0).to(tl.float32)
    key_odd = tl.load(key + base + odd_cols, mask=mask, other=0.0).to(tl.float32)

    tl.store(query_out + base + even_cols, query_even * cos_val - query_odd * sin_val, mask=mask)
    tl.store(query_out + base + odd_cols, query_odd * cos_val + query_even * sin_val, mask=mask)
    tl.store(key_out + base + even_cols, key_even * cos_val - key_odd * sin_val, mask=mask)
    tl.store(key_out + base + odd_cols, key_odd * cos_val + key_even * sin_val, mask=mask)


def _as_section(mrope_section) -> list[int]:
    if mrope_section is None:
        return [0, 0, 0]
    section = [int(v) for v in list(mrope_section)]
    return section or [0, 0, 0]


def _is_rope(section: Sequence[int]) -> bool:
    return not section or all(int(v) == 0 for v in section)


def _launch_apply(
    positions: torch.Tensor,
    x: torch.Tensor,
    cache: torch.Tensor,
    out: torch.Tensor,
    head_size: int,
    section: Sequence[int],
    rotary_mode: str,
    cache_mode: str,
) -> None:
    tokens = int(x.shape[0])
    heads = int(x.shape[1]) // int(head_size)
    rotary_dim = int(cache.shape[1])
    half_dim = rotary_dim // 2
    is_rope = _is_rope(section)
    pos_rows = 1 if positions.dim() == 1 else int(positions.shape[0])
    padded = list(section) + [0, 0, 0, 0]
    if (not is_rope and int(head_size) == 128 and rotary_dim == 128 and half_dim == 64 and str(rotary_mode) == "half"
            and str(cache_mode) == "default" and len(section) in (3, 4) and sum(int(v) for v in section) == 64):
        head_block = 4
        head_blocks = triton.cdiv(heads, head_block)
        _mrope_full_half_default_2d_kernel[(tokens * head_blocks, )](
            positions,
            x,
            cache,
            out,
            tokens,
            heads,
            int(cache.shape[0]),
            pos_rows,
            int(padded[0]),
            int(padded[1]),
            int(padded[2]),
            head_blocks,
            HEAD_BLOCK=head_block,
        )
        return

    block = 256
    total = int(x.numel())
    _mrope_apply_kernel[(triton.cdiv(total, block), )](
        positions,
        x,
        cache,
        out,
        total,
        tokens,
        heads,
        int(head_size),
        rotary_dim,
        half_dim,
        int(cache.shape[0]),
        pos_rows,
        int(padded[0]),
        int(padded[1]),
        int(padded[2]),
        int(padded[3]),
        IS_ROPE=is_rope,
        ROTARY_INTERLEAVED=(str(rotary_mode) == "interleaved"),
        CACHE_INTERLEAVED=(str(cache_mode) == "interleave"),
        BLOCK=block,
    )


def _launch_apply_pair_2d(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cache: torch.Tensor,
    query_out: torch.Tensor,
    key_out: torch.Tensor,
    head_size: int,
    section: Sequence[int],
) -> None:
    tokens = int(query.shape[0])
    heads = int(query.shape[1]) // int(head_size)
    pos_rows = int(positions.shape[0])
    padded = list(section) + [0, 0, 0, 0]
    head_block = 16 if tokens <= 8 else 64
    head_blocks = triton.cdiv(heads, head_block)
    _mrope_full_half_default_2d_pair_kernel[(tokens * head_blocks, )](
        positions,
        query,
        key,
        cache,
        query_out,
        key_out,
        tokens,
        heads,
        int(cache.shape[0]),
        pos_rows,
        int(padded[0]),
        int(padded[1]),
        int(padded[2]),
        head_blocks,
        HEAD_BLOCK=head_block,
    )


def _launch_apply_pair_2d_rope_full_half_default(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cache: torch.Tensor,
    query_out: torch.Tensor,
    key_out: torch.Tensor,
    head_size: int,
) -> None:
    tokens = int(query.shape[0])
    heads = int(query.shape[1]) // int(head_size)
    head_block = 4
    head_blocks = triton.cdiv(heads, head_block)
    _rope_full_half_default_2d_pair_kernel[(tokens * head_blocks, )](
        positions,
        query,
        key,
        cache,
        query_out,
        key_out,
        tokens,
        heads,
        int(cache.shape[0]),
        head_blocks,
        HEAD_BLOCK=head_block,
    )


def _launch_apply_pair_2d_half_interleave64(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cache: torch.Tensor,
    query_out: torch.Tensor,
    key_out: torch.Tensor,
    head_size: int,
    section: Sequence[int],
) -> None:
    tokens = int(query.shape[0])
    heads = int(query.shape[1]) // int(head_size)
    pos_rows = int(positions.shape[0])
    padded = list(section) + [0, 0, 0, 0]
    head_block = 4
    head_blocks = triton.cdiv(heads, head_block)
    _mrope_half_interleave64_2d_pair_kernel[(tokens * head_blocks, )](
        positions,
        query,
        key,
        cache,
        query_out,
        key_out,
        tokens,
        heads,
        int(cache.shape[0]),
        pos_rows,
        int(padded[0]),
        int(padded[1]),
        int(padded[2]),
        head_blocks,
        HEAD_BLOCK=head_block,
    )


def _launch_apply_pair_2d_interleaved_default(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cache: torch.Tensor,
    query_out: torch.Tensor,
    key_out: torch.Tensor,
    head_size: int,
    section: Sequence[int],
) -> None:
    tokens = int(query.shape[0])
    heads = int(query.shape[1]) // int(head_size)
    pos_rows = int(positions.shape[0])
    padded = list(section) + [0, 0, 0, 0]
    head_block = 4
    head_blocks = triton.cdiv(heads, head_block)
    _mrope_interleaved_default_2d_pair_kernel[(tokens * head_blocks, )](
        positions,
        query,
        key,
        cache,
        query_out,
        key_out,
        tokens,
        heads,
        int(cache.shape[0]),
        pos_rows,
        int(padded[0]),
        int(padded[1]),
        int(padded[2]),
        head_blocks,
        HEAD_BLOCK=head_block,
    )


def mrope(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int = 128,
    mrope_section=None,
    rotary_mode: str = "half",
    cache_mode: str = "default",
) -> Tuple[torch.Tensor, torch.Tensor]:
    head = int(head_size)
    section = _as_section(mrope_section)
    if head <= 0 or head % 32 != 0:
        raise ValueError("head_size must be a positive multiple of 32")
    if query.dim() != 2 or key.dim() != 2:
        raise ValueError("query and key must be rank-2")
    if int(query.shape[0]) != int(key.shape[0]):
        raise ValueError("query and key must have the same token count")
    if int(query.shape[1]) % head != 0 or int(key.shape[1]) % head != 0:
        raise ValueError("query/key hidden dimensions must be multiples of head_size")
    if cos_sin_cache.dim() != 2:
        raise ValueError("cos_sin_cache must be rank-2")
    rotary_dim = int(cos_sin_cache.shape[1])
    if rotary_dim <= 0 or rotary_dim > head or rotary_dim % 32 != 0:
        raise ValueError("rotary_dim must be a positive multiple of 32 and <= head_size")
    if str(rotary_mode) not in {"half", "interleaved"}:
        raise ValueError("rotary_mode must be half or interleaved")
    if str(cache_mode) not in {"default", "interleave"}:
        raise ValueError("cache_mode must be default or interleave")
    if section == [16, 16, 16, 16] and str(cache_mode) != "default":
        raise ValueError("section [16,16,16,16] requires default cache mode")
    is_rope = _is_rope(section)
    tokens = int(query.shape[0])
    if is_rope:
        if positions.dim() != 1 or int(positions.shape[0]) != tokens:
            raise ValueError("RoPE mode expects positions shape [num_tokens]")
    else:
        if len(section) not in (3, 4):
            raise ValueError("MRoPE mrope_section length must be 3 or 4")
        if any(int(v) < 0 for v in section):
            raise ValueError("MRoPE mrope_section entries must be non-negative")
        if positions.dim() != 2 or int(positions.shape[0]) not in (3, 4) or int(
                positions.shape[0]) != len(section) or int(positions.shape[1]) != tokens:
            raise ValueError(
                "MRoPE positions shape must be [3, num_tokens] or [4, num_tokens] and match mrope_section length")
        if sum(section) != rotary_dim // 2:
            raise ValueError("sum(mrope_section) must equal rotary_dim / 2")
    for name, tensor, dtype in (
        ("positions", positions, torch.int64),
        ("query", query, torch.bfloat16),
        ("key", key, torch.bfloat16),
        ("cos_sin_cache", cos_sin_cache, torch.bfloat16),
    ):
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)
    query_heads = int(query.shape[1]) // head
    key_heads = int(key.shape[1]) // head
    half_dim = rotary_dim // 2
    if (not is_rope and query_heads == key_heads and head == 128 and rotary_dim == 128 and half_dim == 64
            and str(rotary_mode) == "half" and str(cache_mode) == "default" and len(section) in (3, 4)
            and sum(int(v) for v in section) == 64):
        _launch_apply_pair_2d(positions, query, key, cos_sin_cache, query_out, key_out, head, section)
        return query_out, key_out
    if (is_rope and query_heads == key_heads and head == 128 and rotary_dim == 128 and half_dim == 64
            and str(rotary_mode) == "half" and str(cache_mode) == "default"):
        _launch_apply_pair_2d_rope_full_half_default(positions, query, key, cos_sin_cache, query_out, key_out, head)
        return query_out, key_out
    if (not is_rope and query_heads == key_heads and head == 128 and rotary_dim == 64 and half_dim == 32
            and str(rotary_mode) == "half" and str(cache_mode) == "interleave" and len(section) == 3
            and sum(int(v) for v in section) == 32):
        _launch_apply_pair_2d_half_interleave64(positions, query, key, cos_sin_cache, query_out, key_out, head, section)
        return query_out, key_out
    if (not is_rope and query_heads == key_heads and head == 128 and rotary_dim == 128 and half_dim == 64
            and str(rotary_mode) == "interleaved" and str(cache_mode) == "default" and len(section) in (3, 4)
            and sum(int(v) for v in section) == 64):
        _launch_apply_pair_2d_interleaved_default(positions, query, key, cos_sin_cache, query_out, key_out, head,
                                                  section)
        return query_out, key_out

    _launch_apply(positions, query, cos_sin_cache, query_out, head, section, rotary_mode, cache_mode)
    _launch_apply(positions, key, cos_sin_cache, key_out, head, section, rotary_mode, cache_mode)
    return query_out, key_out
