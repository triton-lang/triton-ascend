#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Self-contained baseline, precision, and timing flow for MRoPE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

try:
    import torch_npu
except Exception as exc:  # pragma: no cover - depends on Ascend runtime.
    torch_npu = None
    _TORCH_NPU_IMPORT_ERROR = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None

from mrope import mrope
from profiler_timing import ProfileResult, profile_kernel_details

BF16_THRESHOLD = 0.02
RANDOM_GENERALIZATION_POLICY = "seeded_mrope_metadata_v1"

DOC_MODEL_ROWS = [
    (1, 3584, 28, 128),
    (1, 4096, 32, 128),
    (1, 5120, 40, 128),
    (1, 8192, 64, 128),
    (8, 3584, 28, 128),
    (8, 4096, 32, 128),
    (8, 5120, 40, 128),
    (8, 8192, 64, 128),
    (16, 3584, 28, 128),
    (16, 4096, 32, 128),
    (16, 5120, 40, 128),
    (16, 8192, 64, 128),
    (32, 3584, 28, 128),
    (32, 4096, 32, 128),
    (32, 5120, 40, 128),
    (32, 8192, 64, 128),
    (64, 3584, 28, 128),
    (64, 4096, 32, 128),
    (64, 5120, 40, 128),
    (64, 8192, 64, 128),
]

PUBLIC_CASE_DATA = [
    ("custom/mrope_1", [[1], [1, 3584], [1, 3584], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [0, 0, 0],
        "rotary_mode": "half",
        "cache_mode": "default",
    }, [[0, 2047], [-0.01, 0.01], [-0.01, 0.01], [-1.0, 1.0]], "bf16-rope-half-default-rotary128-near_zero-B1-H3584"),
    ("custom/mrope_2", [[3, 1], [1, 4096], [1, 4096], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [16, 24, 24],
        "rotary_mode": "half",
        "cache_mode": "default",
    }, [[0, 2047], [-1.0, 1.0], [-1.0, 1.0],
        [-1.0, 1.0]], "bf16-mrope-3row-half-default-16_24_24-ordinary_activation-B1-H4096"),
    ("custom/mrope_3", [[3, 1], [1, 5120], [1, 5120], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [24, 20, 20],
        "rotary_mode": "interleaved",
        "cache_mode": "default",
    }, [[0, 2047], [-8.0, 8.0], [-8.0, 8.0],
        [-1.0, 1.0]], "bf16-mrope-3row-interleaved-default-24_20_20-large_activation-B1-H5120"),
    ("custom/mrope_4", [[4, 1], [1, 8192], [1, 8192], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [16, 16, 16, 16],
        "rotary_mode": "half",
        "cache_mode": "default",
    }, [[0, 2047], [-64.0, 16.0], [-16.0, 64.0],
        [-1.0, 1.0]], "bf16-mrope-4row-half-default-16x4-asymmetric_finite-B1-H8192"),
    ("custom/mrope_5", [[3, 8], [8, 3584], [8, 3584], [2048, 64]], {
        "head_size": 128,
        "mrope_section": [8, 12, 12],
        "rotary_mode": "half",
        "cache_mode": "interleave",
    }, [[0, 2047], [-256.0, 256.0], [-256.0, 256.0],
        [-1.0, 1.0]], "bf16-mrope-3row-half-cache_interleave-8_12_12-extreme_finite-B8-H3584"),
    ("custom/mrope_6", [[8], [8, 4096], [8, 4096], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [0, 0, 0],
        "rotary_mode": "half",
        "cache_mode": "default",
    }, [[0, 2047], [-0.01, 0.01], [-0.01, 0.01], [-1.0, 1.0]], "bf16-rope-half-default-rotary128-near_zero-B8-H4096"),
    ("custom/mrope_7", [[3, 8], [8, 5120], [8, 5120], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [16, 24, 24],
        "rotary_mode": "half",
        "cache_mode": "default",
    }, [[0, 2047], [-1.0, 1.0], [-1.0, 1.0],
        [-1.0, 1.0]], "bf16-mrope-3row-half-default-16_24_24-ordinary_activation-B8-H5120"),
    ("custom/mrope_8", [[3, 8], [8, 8192], [8, 8192], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [24, 20, 20],
        "rotary_mode": "interleaved",
        "cache_mode": "default",
    }, [[0, 2047], [-8.0, 8.0], [-8.0, 8.0],
        [-1.0, 1.0]], "bf16-mrope-3row-interleaved-default-24_20_20-large_activation-B8-H8192"),
    ("custom/mrope_9", [[4, 16], [16, 3584], [16, 3584], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [16, 16, 16, 16],
        "rotary_mode": "half",
        "cache_mode": "default",
    }, [[0, 2047], [-64.0, 16.0], [-16.0, 64.0],
        [-1.0, 1.0]], "bf16-mrope-4row-half-default-16x4-asymmetric_finite-B16-H3584"),
    ("custom/mrope_10", [[3, 16], [16, 4096], [16, 4096], [2048, 64]], {
        "head_size": 128,
        "mrope_section": [8, 12, 12],
        "rotary_mode": "half",
        "cache_mode": "interleave",
    }, [[0, 2047], [-256.0, 256.0], [-256.0, 256.0],
        [-1.0, 1.0]], "bf16-mrope-3row-half-cache_interleave-8_12_12-extreme_finite-B16-H4096"),
    ("custom/mrope_11", [[16], [16, 5120], [16, 5120], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [0, 0, 0],
        "rotary_mode": "half",
        "cache_mode": "default",
    }, [[0, 2047], [-0.01, 0.01], [-0.01, 0.01], [-1.0, 1.0]], "bf16-rope-half-default-rotary128-near_zero-B16-H5120"),
    ("custom/mrope_12", [[3, 16], [16, 8192], [16, 8192], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [16, 24, 24],
        "rotary_mode": "half",
        "cache_mode": "default",
    }, [[0, 2047], [-1.0, 1.0], [-1.0, 1.0],
        [-1.0, 1.0]], "bf16-mrope-3row-half-default-16_24_24-ordinary_activation-B16-H8192"),
    ("custom/mrope_13", [[3, 32], [32, 3584], [32, 3584], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [24, 20, 20],
        "rotary_mode": "interleaved",
        "cache_mode": "default",
    }, [[0, 2047], [-8.0, 8.0], [-8.0, 8.0],
        [-1.0, 1.0]], "bf16-mrope-3row-interleaved-default-24_20_20-large_activation-B32-H3584"),
    ("custom/mrope_14", [[4, 32], [32, 4096], [32, 4096], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [16, 16, 16, 16],
        "rotary_mode": "half",
        "cache_mode": "default",
    }, [[0, 2047], [-64.0, 16.0], [-16.0, 64.0],
        [-1.0, 1.0]], "bf16-mrope-4row-half-default-16x4-asymmetric_finite-B32-H4096"),
    ("custom/mrope_15", [[3, 32], [32, 5120], [32, 5120], [2048, 64]], {
        "head_size": 128,
        "mrope_section": [8, 12, 12],
        "rotary_mode": "half",
        "cache_mode": "interleave",
    }, [[0, 2047], [-256.0, 256.0], [-256.0, 256.0],
        [-1.0, 1.0]], "bf16-mrope-3row-half-cache_interleave-8_12_12-extreme_finite-B32-H5120"),
    ("custom/mrope_16", [[32], [32, 8192], [32, 8192], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [0, 0, 0],
        "rotary_mode": "half",
        "cache_mode": "default",
    }, [[0, 2047], [-0.01, 0.01], [-0.01, 0.01], [-1.0, 1.0]], "bf16-rope-half-default-rotary128-near_zero-B32-H8192"),
    ("custom/mrope_17", [[3, 64], [64, 3584], [64, 3584], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [16, 24, 24],
        "rotary_mode": "half",
        "cache_mode": "default",
    }, [[0, 2047], [-1.0, 1.0], [-1.0, 1.0],
        [-1.0, 1.0]], "bf16-mrope-3row-half-default-16_24_24-ordinary_activation-B64-H3584"),
    ("custom/mrope_18", [[3, 64], [64, 4096], [64, 4096], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [24, 20, 20],
        "rotary_mode": "interleaved",
        "cache_mode": "default",
    }, [[0, 2047], [-8.0, 8.0], [-8.0, 8.0],
        [-1.0, 1.0]], "bf16-mrope-3row-interleaved-default-24_20_20-large_activation-B64-H4096"),
    ("custom/mrope_19", [[4, 64], [64, 5120], [64, 5120], [2048, 128]], {
        "head_size": 128,
        "mrope_section": [16, 16, 16, 16],
        "rotary_mode": "half",
        "cache_mode": "default",
    }, [[0, 2047], [-64.0, 16.0], [-16.0, 64.0],
        [-1.0, 1.0]], "bf16-mrope-4row-half-default-16x4-asymmetric_finite-B64-H5120"),
    ("custom/mrope_20", [[3, 64], [64, 8192], [64, 8192], [2048, 64]], {
        "head_size": 128,
        "mrope_section": [8, 12, 12],
        "rotary_mode": "half",
        "cache_mode": "interleave",
    }, [[0, 2047], [-256.0, 256.0], [-256.0, 256.0],
        [-1.0, 1.0]], "bf16-mrope-3row-half-cache_interleave-8_12_12-extreme_finite-B64-H8192"),
]


@dataclass(frozen=True)
class Case:
    case_id: str
    kind: str
    input_shape: list[list[int]]
    attrs: dict[str, object]
    value_range: list[list[float]]
    note: str = ""
    random_category: str = ""


def _as_section(mrope_section) -> list[int]:
    if mrope_section is None:
        return [0, 0, 0]
    section = [int(v) for v in list(mrope_section)]
    return section or [0, 0, 0]


def _is_rope(section: Sequence[int]) -> bool:
    return not section or all(int(v) == 0 for v in section)


def _decode_cache(cache: torch.Tensor, cache_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if str(cache_mode) == "interleave":
        return cache[:, 0::2].to(torch.float32), cache[:, 1::2].to(torch.float32)
    half = int(cache.shape[1]) // 2
    return cache[:, :half].to(torch.float32), cache[:, half:].to(torch.float32)


def _assemble_cos_sin(
    positions: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    section: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    max_seq = int(cos_table.shape[0])
    if positions.numel() and (int(positions.min().item()) < 0 or int(positions.max().item()) >= max_seq):
        raise ValueError("positions values must be within cos_sin_cache max_seq_len")
    if _is_rope(section):
        pos = positions.to(torch.long)
        return cos_table[pos], sin_table[pos]
    cos_parts = []
    sin_parts = []
    start = 0
    for row, length in enumerate(section):
        end = start + int(length)
        pos = positions[row].to(torch.long)
        cos_parts.append(cos_table[pos, start:end])
        sin_parts.append(sin_table[pos, start:end])
        start = end
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def mrope_reference(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int = 128,
    mrope_section=None,
    rotary_mode: str = "half",
    cache_mode: str = "default",
) -> tuple[torch.Tensor, torch.Tensor]:
    section = _as_section(mrope_section)
    head = int(head_size)
    rotary_dim = int(cos_sin_cache.shape[1])
    half = rotary_dim // 2
    if not _is_rope(section) and sum(section) != half:
        raise ValueError("sum(mrope_section) must equal rotary_dim / 2")
    cos_table, sin_table = _decode_cache(cos_sin_cache, cache_mode)
    cos, sin = _assemble_cos_sin(positions, cos_table, sin_table, section)
    return (
        _apply_to_tensor(query, cos, sin, head, rotary_dim, rotary_mode),
        _apply_to_tensor(key, cos, sin, head, rotary_dim, rotary_mode),
    )


def _apply_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    rotary_mode: str,
) -> torch.Tensor:
    token_count = int(x.shape[0])
    heads = int(x.shape[1]) // int(head_size)
    out = x.to(torch.float32).reshape(token_count, heads, int(head_size)).clone()
    rotary = out[:, :, :rotary_dim]
    half = rotary_dim // 2
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    if str(rotary_mode) == "interleaved":
        even = rotary[..., 0::2]
        odd = rotary[..., 1::2]
        rotated = torch.empty_like(rotary)
        rotated[..., 0::2] = even * cos - odd * sin
        rotated[..., 1::2] = odd * cos + even * sin
    else:
        first = rotary[..., :half]
        second = rotary[..., half:]
        rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    out[:, :, :rotary_dim] = rotated
    return out.reshape_as(x).to(dtype=x.dtype)


def _seed_from_case_id(case_id: str, seed: int = 0) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    deterministic_hash = int.from_bytes(digest[:8], byteorder="big") % (2**31)
    return (int(seed) + deterministic_hash) % (2**31)


def _doc_metadata(input_shape: list[list[int]], attrs: dict[str, object]) -> dict[str, int]:
    batch = int(input_shape[1][0])
    hidden = int(input_shape[1][1])
    head_dim = int(attrs.get("head_size", 128))
    if hidden % head_dim != 0:
        raise ValueError(f"hidden size {hidden} is not divisible by HeadDim {head_dim}")
    head_num = hidden // head_dim
    if (batch, hidden, head_num, head_dim) not in set(DOC_MODEL_ROWS):
        raise ValueError(f"shape {(batch, hidden, head_num, head_dim)} is not in docs/mrope.md test standard")
    return {
        "Batch": batch,
        "HiddenSize": hidden,
        "HeadNum": head_num,
        "HeadDim": head_dim,
    }


def _doc_attrs(input_shape: list[list[int]], attrs: dict[str, object]) -> dict[str, object]:
    _doc_metadata(input_shape, attrs)
    return dict(attrs)


def _gen_bf16_uniform(shape: Sequence[int], value_range: Sequence[float], gen: torch.Generator) -> torch.Tensor:
    low, high = float(value_range[0]), float(value_range[1])
    tensor = torch.rand(tuple(int(v) for v in shape), dtype=torch.float64, generator=gen)
    return (tensor * (high - low) + low).to(torch.bfloat16)


def _make_inputs(case: Case, device: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    shapes = case.input_shape
    attrs = case.attrs
    rotary_dim = int(shapes[3][1])
    half = rotary_dim // 2
    token_count = int(shapes[1][0])
    max_seq = int(shapes[3][0])

    if len(shapes[0]) == 1:
        base = torch.arange(token_count, dtype=torch.int64) % max(1, max_seq)
        positions_cpu = base
    else:
        rows = int(shapes[0][0])
        base = torch.arange(token_count, dtype=torch.int64) % max(1, max_seq)
        positions_cpu = torch.stack([(base + 7 * row) % max(1, max_seq) for row in range(rows)], dim=0)

    query_cpu = _gen_bf16_uniform(shapes[1], case.value_range[1], gen)
    key_cpu = _gen_bf16_uniform(shapes[2], case.value_range[2], gen)
    idx = torch.arange(max_seq, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(half, dtype=torch.float32).unsqueeze(0)
    angles = idx / torch.pow(torch.tensor(10000.0), (2.0 * freqs) / max(1, rotary_dim))
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    if str(attrs.get("cache_mode", "default")) == "interleave":
        cache_cpu = torch.empty((max_seq, rotary_dim), dtype=torch.float32)
        cache_cpu[:, 0::2] = cos
        cache_cpu[:, 1::2] = sin
    else:
        cache_cpu = torch.cat((cos, sin), dim=-1)

    return (
        positions_cpu.to(device=device, dtype=torch.int64).contiguous(),
        query_cpu.to(device=device).contiguous(),
        key_cpu.to(device=device).contiguous(),
        cache_cpu.to(device=device, dtype=torch.bfloat16).contiguous(),
    )


def public_cases() -> list[Case]:
    return [
        Case(case_id=case_id, kind="public", input_shape=[[int(v)
                                                           for v in shape]
                                                          for shape in input_shape],
             attrs=_doc_attrs([[int(v)
                                for v in shape]
                               for shape in input_shape], dict(attrs)), value_range=[[float(v)
                                                                                      for v in rng]
                                                                                     for rng in value_range], note=note)
        for case_id, input_shape, attrs, value_range, note in PUBLIC_CASE_DATA
    ]


def random_generalization_cases(count: int, seed: int) -> list[Case]:
    rng = random.Random(int(seed))
    templates = [
        ("rope_half_default", [0, 0, 0], "half", "default", 128, 1),
        ("mrope3_half_default", [16, 24, 24], "half", "default", 128, 3),
        ("mrope3_interleaved_default", [24, 20, 20], "interleaved", "default", 128, 3),
        ("mrope4_half_default", [16, 16, 16, 16], "half", "default", 128, 4),
        ("mrope3_half_interleave64", [8, 12, 12], "half", "interleave", 64, 3),
    ]
    cases: list[Case] = []
    seen: set[tuple[object, ...]] = set()
    public_signatures = {(
        tuple(tuple(v)
              for v in case.input_shape),
        tuple(str(case.attrs.get(key))
              for key in ("mrope_section", "rotary_mode", "cache_mode")),
    )
                         for case in public_cases()}
    attempts = 0
    while len(cases) < count and attempts < count * 100:
        attempts += 1
        name, section, rotary_mode, cache_mode, rotary_dim, rows = rng.choice(templates)
        tokens, hidden, heads, head_dim = rng.choice(DOC_MODEL_ROWS)
        if hidden != heads * head_dim:
            raise ValueError(f"bad docs row: {(tokens, hidden, heads, head_dim)}")
        max_seq = rng.choice([256, 512, 1024, 2048])
        shape_key = (name, tokens, hidden, max_seq)
        if shape_key in seen:
            continue
        pos_shape = [tokens] if rows == 1 else [rows, tokens]
        input_shape = [pos_shape, [tokens, hidden], [tokens, hidden], [max_seq, rotary_dim]]
        attrs = _doc_attrs(
            input_shape,
            {
                "head_size": head_dim,
                "mrope_section": list(section),
                "rotary_mode": rotary_mode,
                "cache_mode": cache_mode,
            },
        )
        signature = (
            tuple(tuple(v) for v in input_shape),
            tuple(str(attrs.get(key)) for key in ("mrope_section", "rotary_mode", "cache_mode")),
        )
        if signature in public_signatures:
            continue
        seen.add(shape_key)
        value_kind = rng.choice(["small", "ordinary", "wide"])
        if value_kind == "small":
            q_range = [-0.01, 0.01]
            k_range = [-0.01, 0.01]
        elif value_kind == "ordinary":
            q_range = [-1.0, 1.0]
            k_range = [-1.0, 1.0]
        else:
            q_range = [-128.0, 128.0]
            k_range = [-64.0, 64.0]
        idx = len(cases) + 1
        cases.append(
            Case(
                case_id=f"custom/mrope_random_{idx:03d}",
                kind="random_generalization",
                input_shape=input_shape,
                attrs=attrs,
                value_range=[[0, max_seq - 1], q_range, k_range, [-1.0, 1.0]],
                note="seeded non-public MRoPE metadata sample",
                random_category=name,
            ))
    if len(cases) != count:
        raise RuntimeError(f"generated {len(cases)} random cases, expected {count}")
    return cases


def run_torch_npu(inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                  attrs: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    if torch_npu is None:
        raise RuntimeError(f"torch_npu import failed: {_TORCH_NPU_IMPORT_ERROR}")
    if str(attrs.get("cache_mode", "default")) != "default":
        raise RuntimeError("torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered")
    op = getattr(torch_npu, "npu_mrope", None)
    if not callable(op):
        raise RuntimeError("torch_npu.npu_mrope is not available")
    positions, query, key, cache = inputs
    return op(
        positions,
        query,
        key,
        cache,
        int(attrs.get("head_size", 128)),
        mrope_section=list(attrs.get("mrope_section") or [0, 0, 0]),
        rotary_mode=str(attrs.get("rotary_mode", "half")),
    )


def compare_outputs(actual: tuple[torch.Tensor, torch.Tensor], expected: tuple[torch.Tensor,
                                                                               torch.Tensor]) -> dict[str, object]:
    output_results = []
    total_mismatch = 0
    max_diff = 0.0
    max_mare = 0.0
    total_abs_diff = 0.0
    total_sq_diff = 0.0
    total_rel_diff = 0.0
    total_count = 0
    for index, (name, out, ref) in enumerate(zip(["query_out", "key_out"], actual, expected)):
        diff = (out.float() - ref.float()).abs()
        rel = diff / (ref.float().abs() + 1.0e-6)
        allowed = BF16_THRESHOLD + BF16_THRESHOLD * ref.float().abs()
        mismatch = diff > allowed
        mismatch_count = int(mismatch.sum().item())
        total = int(diff.numel())
        mare = float(rel.max().item()) if total else 0.0
        mere = float(rel.mean().item()) if total else 0.0
        item_max = float(diff.max().item()) if total else 0.0
        item_sum = float(diff.sum().item()) if total else 0.0
        item_sq_sum = float((diff * diff).sum().item()) if total else 0.0
        item_mean = item_sum / max(1, total)
        item_rmse = math.sqrt(item_sq_sum / max(1, total))
        total_mismatch += mismatch_count
        max_diff = max(max_diff, item_max)
        max_mare = max(max_mare, mare)
        total_abs_diff += item_sum
        total_sq_diff += item_sq_sum
        total_rel_diff += float(rel.sum().item()) if total else 0.0
        total_count += total
        output_results.append({
            "index": index,
            "name": name,
            "dtype": str(out.dtype),
            "shape": list(out.shape),
            "criterion": f"BF16 mixed absolute/relative threshold, base={BF16_THRESHOLD}",
            "passed": mismatch_count == 0,
            "mismatch_count": mismatch_count,
            "total_count": total,
            "max_diff": item_max,
            "mean_diff": item_mean,
            "rmse": item_rmse,
            "mere": mere,
            "mare": mare,
        })
    return {
        "passed": total_mismatch == 0,
        "threshold": BF16_THRESHOLD,
        "mismatch_count": total_mismatch,
        "total_count": total_count,
        "max_diff": max_diff,
        "mean_diff": total_abs_diff / max(1, total_count),
        "rmse": math.sqrt(total_sq_diff / max(1, total_count)),
        "mere": total_rel_diff / max(1, total_count),
        "mare": max_mare,
        "error_msg": "" if total_mismatch == 0 else "BF16 mixed threshold exceeded",
        "output_results": output_results,
    }


def zero_accuracy(outputs: tuple[torch.Tensor, torch.Tensor]) -> dict[str, object]:
    output_results = []
    total = 0
    for index, (name, tensor) in enumerate(zip(["query_out", "key_out"], outputs)):
        total += int(tensor.numel())
        output_results.append({
            "index": index,
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "criterion": f"BF16 mixed absolute/relative threshold, base={BF16_THRESHOLD}",
            "passed": True,
            "mismatch_count": 0,
            "total_count": int(tensor.numel()),
            "max_diff": 0.0,
            "mean_diff": 0.0,
            "rmse": 0.0,
            "mere": 0.0,
            "mare": 0.0,
        })
    return {
        "passed": True,
        "threshold": BF16_THRESHOLD,
        "mismatch_count": 0,
        "total_count": total,
        "max_diff": 0.0,
        "mean_diff": 0.0,
        "rmse": 0.0,
        "mere": 0.0,
        "mare": 0.0,
        "error_msg": "",
        "output_results": output_results,
    }


def failed_accuracy(error: str) -> dict[str, object]:
    return {
        "passed": False,
        "threshold": BF16_THRESHOLD,
        "mismatch_count": 1,
        "total_count": 1,
        "max_diff": 0.0,
        "mean_diff": 0.0,
        "rmse": 0.0,
        "mere": 0.0,
        "mare": 0.0,
        "error_msg": error,
        "output_results": [],
    }


def fmt_us(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.3f} us"


def fmt_x(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.6f}x"


def speedup(base_us: object, cand_us: object) -> float | None:
    try:
        b = float(base_us) if base_us is not None else 0.0
        c = float(cand_us) if cand_us is not None else 0.0
    except (TypeError, ValueError):
        return None
    return b / c if b > 0.0 and c > 0.0 else None


def impl_record(
    name: str,
    role: str,
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    expected: tuple[torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
    case_id: str,
    *,
    profile_for_speed: bool,
) -> dict[str, object]:
    outputs = None
    error = ""
    profile = ProfileResult(None, None, None, None, None, None)
    try:
        with torch.inference_mode():
            outputs = fn()
            if args.device.startswith("npu"):
                torch.npu.synchronize()
        accuracy = zero_accuracy(outputs) if role == "pytorch_semantic_baseline" else compare_outputs(outputs, expected)
    except Exception as exc:  # pragma: no cover - runtime/API dependent.
        error = f"{type(exc).__name__}: {exc}"
        accuracy = failed_accuracy(error)
    if args.benchmark and profile_for_speed and outputs is not None:
        prof_outputs, profile = profile_kernel_details(name, case_id, fn, warmup=args.warmup, repeat=args.repeat)
        if prof_outputs is not None:
            outputs = prof_outputs
        if profile.error:
            error = profile.error
    elif error:
        profile.error = error
    return {
        "role": role,
        "accuracy": accuracy,
        "timing_strategy": "kernel_details" if profile_for_speed else "correctness_only",
        "perf_metric_strategy": profile.perf_metric_strategy,
        "measurement_scope": profile.measurement_scope,
        "elapsed_us_source": profile.elapsed_us_source,
        "primary_latency_us": profile.latency_us,
        "primary_latency": fmt_us(profile.latency_us),
        "latency_us": profile.latency_us,
        "latency": fmt_us(profile.latency_us),
        "active_window_us": profile.active_window_us,
        "active_window": fmt_us(profile.active_window_us),
        "kernel_sum_us": profile.kernel_sum_us,
        "kernel_sum": fmt_us(profile.kernel_sum_us),
        "window_gap_us": profile.window_gap_us,
        "window_gap": fmt_us(profile.window_gap_us),
        "kernel_count": profile.kernel_count,
        "step_count": profile.step_count,
        "device_kernels": profile.device_kernels,
        "device_timeline": profile.device_timeline,
        "timing_csv_path": profile.csv_path,
        "timing_trace_path": profile.trace_view_path,
        "profile_error": profile.error or error or None,
    }


def run_case(case: Case, index: int, total: int, args: argparse.Namespace) -> dict[str, object]:
    seed = _seed_from_case_id(case.case_id, args.random_seed)
    inputs = _make_inputs(case, args.device, seed)
    attrs = case.attrs
    with torch.inference_mode():
        expected = mrope_reference(*inputs, **attrs)
        if args.device.startswith("npu"):
            torch.npu.synchronize()
    impls = {
        "triton":
        impl_record("triton", "candidate", lambda: mrope(*inputs, **attrs), expected, args, case.case_id,
                    profile_for_speed=True),
        "torch":
        impl_record("torch", "pytorch_semantic_baseline", lambda: mrope_reference(*inputs, **attrs), expected, args,
                    case.case_id, profile_for_speed=bool(args.benchmark_torch)),
        "torch_npu":
        impl_record("torch_npu", "task_npu_baseline_probe", lambda: run_torch_npu(inputs, attrs), expected, args,
                    case.case_id, profile_for_speed=True),
    }
    cand_active_us = impls["triton"]["active_window_us"]
    cand_kernel_us = impls["triton"]["kernel_sum_us"]
    for name in ["torch", "torch_npu"]:
        active_vs_active = speedup(impls[name]["active_window_us"], cand_active_us)
        kernel_vs_kernel = speedup(impls[name]["kernel_sum_us"], cand_kernel_us)
        baseline_active_vs_candidate_kernel = speedup(impls[name]["active_window_us"], cand_kernel_us)
        baseline_kernel_vs_candidate_active = speedup(impls[name]["kernel_sum_us"], cand_active_us)
        s = active_vs_active
        impls[name]["speedup_vs_triton"] = s
        impls[name]["speedup_vs_triton_text"] = fmt_x(s)
        impls[name]["speedups_vs_triton"] = {
            "active_vs_active": active_vs_active,
            "active_vs_active_text": fmt_x(active_vs_active),
            "kernel_vs_kernel": kernel_vs_kernel,
            "kernel_vs_kernel_text": fmt_x(kernel_vs_kernel),
            "baseline_active_vs_candidate_kernel": baseline_active_vs_candidate_kernel,
            "baseline_active_vs_candidate_kernel_text": fmt_x(baseline_active_vs_candidate_kernel),
            "baseline_kernel_vs_candidate_active": baseline_kernel_vs_candidate_active,
            "baseline_kernel_vs_candidate_active_text": fmt_x(baseline_kernel_vs_candidate_active),
        }
    selected = "torch_npu" if impls["torch_npu"]["accuracy"]["passed"] else "torch"
    selected_speedup = impls[selected].get("speedup_vs_triton")
    triton_acc = impls["triton"]["accuracy"]
    status = "PASS" if triton_acc["passed"] else "FAIL"
    shape = case.input_shape
    print(
        f"[{status}] {index:03d}/{total:03d} {case.kind} case_id={case.case_id} "
        f"positions={shape[0]} query={shape[1]} cache={shape[3]} "
        f"rotary_mode={attrs.get('rotary_mode')} cache_mode={attrs.get('cache_mode')} "
        f"section={attrs.get('mrope_section')} triton_active={impls['triton']['active_window']} "
        f"torch_npu_active={impls['torch_npu']['active_window']} selected={selected} "
        f"mismatch={triton_acc['mismatch_count']} mare={triton_acc['mare']:.6e} "
        f"rmse={triton_acc['rmse']:.6e} max_diff={triton_acc['max_diff']:.6e} "
        f"torch_npu_accuracy={'PASS' if impls['torch_npu']['accuracy']['passed'] else 'FAIL'} "
        f"torch_npu_error={impls['torch_npu'].get('profile_error') or impls['torch_npu']['accuracy'].get('error_msg') or ''}"
    )
    return {
        "case": index,
        "case_id": case.case_id,
        "kind": case.kind,
        "note": case.note,
        "random_category": case.random_category,
        "input_shape": case.input_shape,
        "attrs": case.attrs,
        "case_detail": _doc_metadata(case.input_shape, case.attrs),
        "seed": seed,
        "dtype": "bfloat16",
        "timing_policy":
        "Candidate and torch_npu benchmark paths are timed by torch_npu.profiler kernel_details.csv active-window, matching OpForge/CANN-Bench; Torch semantic reference is timed only with --benchmark-torch as an auxiliary comparison and is not the main speed gate.",
        "implementations": impls,
        "selected_baseline": {
            "implementation": selected,
            "source": "task_npu_baseline_probe" if selected == "torch_npu" else "pytorch_semantic_baseline",
            "selection_rule":
            "Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline.",
            "active_window_us": impls[selected]["active_window_us"],
            "active_window": impls[selected]["active_window"],
            "kernel_sum_us": impls[selected]["kernel_sum_us"],
            "kernel_sum": impls[selected]["kernel_sum"],
            "latency_us": impls[selected]["latency_us"],
            "latency": impls[selected]["latency"],
            "speedup_vs_triton": selected_speedup,
            "speedup_vs_triton_text": fmt_x(selected_speedup),
            "speedups_vs_triton": impls[selected].get("speedups_vs_triton", {}),
            "perf_metric_strategy": impls[selected]["perf_metric_strategy"],
            "measurement_scope": impls[selected]["measurement_scope"],
            "elapsed_us_source": impls[selected]["elapsed_us_source"],
        },
    }


def geomean(values: list[float]) -> float | None:
    return math.exp(sum(math.log(max(v, 1.0e-9)) for v in values) / len(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true", help="run embedded public cases")
    parser.add_argument("--random-generalization", type=int, default=0, help="number of seeded random cases")
    parser.add_argument("--random-seed", type=int, default=20260617)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--benchmark", action="store_true",
                        help="collect OpForge-compatible kernel_details.csv active-window timing for all paths")
    parser.add_argument(
        "--benchmark-torch", action="store_true",
        help="also profile Torch semantic reference as auxiliary timing; not used for the main speed gate")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    cases: list[Case] = []
    if args.public:
        cases.extend(public_cases())
    if args.random_generalization:
        cases.extend(random_generalization_cases(args.random_generalization, args.random_seed))
    if not cases:
        cases.extend(public_cases())
    if args.max_cases is not None:
        cases = cases[:args.max_cases]

    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl.write_text("", encoding="utf-8")
    records = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        record = run_case(case, index, total, args)
        records.append(record)
        if args.jsonl:
            with args.jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    passed = sum(1 for r in records if r["implementations"]["triton"]["accuracy"]["passed"])
    triton_latencies = [
        float(r["implementations"]["triton"]["active_window_us"])
        for r in records
        if r["implementations"]["triton"].get("active_window_us") is not None
    ]
    selected_speedups = [
        float(r["selected_baseline"]["speedup_vs_triton"])
        for r in records
        if r["selected_baseline"].get("speedup_vs_triton") is not None
    ]
    torch_npu_runnable = [r for r in records if r["implementations"]["torch_npu"].get("speedup_vs_triton") is not None]
    torch_active_timed = [r for r in records if r["implementations"]["torch"].get("active_window_us") is not None]
    torch_speedup_sample = [r for r in records if r["implementations"]["torch"].get("speedup_vs_triton") is not None]
    torch_timed_candidate_active_geomean = geomean([
        float(r["implementations"]["triton"]["active_window_us"])
        for r in torch_active_timed
        if r["implementations"]["triton"].get("active_window_us") is not None
    ])
    torch_timed_baseline_active_geomean = geomean([
        float(r["implementations"]["torch"]["active_window_us"])
        for r in torch_active_timed
        if r["implementations"]["torch"].get("active_window_us") is not None
    ])
    torch_semantic_active_speedup_geomean = geomean([
        float(r["implementations"]["torch"]["speedup_vs_triton"])
        for r in torch_speedup_sample
        if r["implementations"]["torch"].get("speedup_vs_triton") is not None
    ])
    torch_npu_runnable_candidate_active_geomean = geomean([
        float(r["implementations"]["triton"]["active_window_us"])
        for r in torch_npu_runnable
        if r["implementations"]["triton"].get("active_window_us") is not None
    ])
    torch_npu_runnable_baseline_active_geomean = geomean([
        float(r["implementations"]["torch_npu"]["active_window_us"])
        for r in torch_npu_runnable
        if r["implementations"]["torch_npu"].get("active_window_us") is not None
    ])
    torch_npu_runnable_all_speedup_geomean = geomean([
        float(r["implementations"]["torch_npu"]["speedup_vs_triton"])
        for r in torch_npu_runnable
        if r["implementations"]["torch_npu"].get("speedup_vs_triton") is not None
    ])
    summary = {
        "schema_version":
        4,
        "source_jsonl":
        str(args.jsonl) if args.jsonl else "",
        "status":
        "PASS" if passed == total else "FAIL",
        "total_cases":
        total,
        "passed":
        passed,
        "failed":
        total - passed,
        "public_cases":
        sum(1 for r in records if r["kind"] == "public"),
        "random_generalization_cases":
        sum(1 for r in records if r["kind"] == "random_generalization"),
        "random_seed":
        args.random_seed,
        "random_policy":
        RANDOM_GENERALIZATION_POLICY,
        "benchmark":
        bool(args.benchmark),
        "benchmark_torch":
        bool(args.benchmark_torch),
        "timing_source":
        "kernel_details.csv.active_window_median" if args.benchmark else "",
        "candidate_active_geomean_us":
        geomean(triton_latencies),
        "selected_active_speedup_geomean":
        geomean(selected_speedups),
        "selected_baseline_counts": {
            name: sum(1
                      for r in records
                      if r["selected_baseline"]["implementation"] == name)
            for name in ["torch_npu", "torch"]
        },
        "torch_npu_runnable":
        len(torch_npu_runnable),
        "torch_npu_accuracy_passed":
        sum(1 for r in records if r["implementations"]["torch_npu"]["accuracy"]["passed"]),
        "torch_timed":
        len(torch_active_timed),
        "torch_speedup_sample":
        len(torch_speedup_sample),
        "torch_semantic_candidate_active_geomean_us":
        torch_timed_candidate_active_geomean,
        "torch_semantic_baseline_active_geomean_us":
        torch_timed_baseline_active_geomean,
        "torch_semantic_active_speedup_geomean":
        torch_semantic_active_speedup_geomean,
        "main_speed_sample":
        "torch_npu_runnable_all",
        "main_speed_case_count":
        len(torch_npu_runnable),
        "main_speed_candidate_active_geomean_us":
        torch_npu_runnable_candidate_active_geomean,
        "main_speed_torch_npu_active_geomean_us":
        torch_npu_runnable_baseline_active_geomean,
        "main_speed_active_speedup_geomean":
        torch_npu_runnable_all_speedup_geomean,
        "main_speed_gate":
        "N/A" if torch_npu_runnable_all_speedup_geomean is None else
        ("PASS" if torch_npu_runnable_all_speedup_geomean >= 1.2 else "FAIL"),
        "torch_npu_runnable_all_active_speedup_geomean":
        torch_npu_runnable_all_speedup_geomean,
        "torch_npu_runnable_all_speed_gate":
        "N/A" if torch_npu_runnable_all_speedup_geomean is None else
        ("PASS" if torch_npu_runnable_all_speedup_geomean >= 1.2 else "FAIL"),
        "max_diff":
        max(float(r["implementations"]["triton"]["accuracy"]["max_diff"]) for r in records),
        "max_mare":
        max(float(r["implementations"]["triton"]["accuracy"]["mare"]) for r in records),
        "max_rmse":
        max(float(r["implementations"]["triton"]["accuracy"].get("rmse") or 0.0) for r in records),
        "total_mismatch_count":
        sum(int(r["implementations"]["triton"]["accuracy"]["mismatch_count"]) for r in records),
        "notes": [
            "Public cases are embedded in validate_mrope.py; no external evidence manifest is required.",
            "Torch and torch_npu baselines are generated inside this operator directory.",
            "Torch semantic timing is collected only when --benchmark-torch is set and remains an auxiliary comparison.",
            "RMSE/MERE/MARE are emitted by validate_mrope.py.",
            "No external historical evaluation CSV/JSON is used by this self-validation flow.",
        ],
    }
    print("SUMMARY "
          f"status={summary['status']} total={total} passed={passed} failed={total - passed} "
          f"public={summary['public_cases']} random={summary['random_generalization_cases']} "
          f"torch_npu_runnable={summary['torch_npu_runnable']} "
          f"torch_npu_accuracy_passed={summary['torch_npu_accuracy_passed']} "
          f"torch_timed={summary['torch_timed']} "
          f"torch_speedup_sample={summary['torch_speedup_sample']} "
          f"torch_semantic_speedup={summary['torch_semantic_active_speedup_geomean']} "
          f"torch_npu_runnable_all_speedup={summary['torch_npu_runnable_all_active_speedup_geomean']} "
          f"torch_npu_runnable_all_gate={summary['torch_npu_runnable_all_speed_gate']} "
          f"max_diff={summary['max_diff']:.6e} max_mare={summary['max_mare']:.6e} "
          f"max_rmse={summary['max_rmse']:.6e} mismatch={summary['total_mismatch_count']}")
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
