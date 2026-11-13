#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Evaluate A5/Ascend950 pure-vector theoretical performance cases.

Inputs:
  --hardware-json references/hardware/ascend950pr.json
  --instruction-json references/vector_instruction_types.json
  --cases perf_cases.jsonl

Output is JSON by default; use --out-md for a Markdown table.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any


class EvalError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_eval(value: Any, variables: dict[str, float]) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str):
        tree = ast.parse(value, mode="eval")

        def visit(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.Name):
                if node.id not in variables:
                    raise EvalError(f"unknown variable in expression {value!r}: {node.id}")
                return float(variables[node.id])
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                inner = visit(node.operand)
                return inner if isinstance(node.op, ast.UAdd) else -inner
            if isinstance(node, ast.BinOp) and isinstance(node.op,
                                                          (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
                left = visit(node.left)
                right = visit(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if right == 0:
                    raise EvalError(f"division by zero in expression {value!r}")
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.FloorDiv):
                    return float(math.floor(left / right))
                return float(left - math.floor(left / right) * right)  # Mod
            # Whitelisted helpers for alignment/padding math (e.g. ceil(N/T)*T).
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords):
                fname = node.func.id
                fargs = [visit(arg) for arg in node.args]
                if fname == "ceil" and len(fargs) == 1:
                    return float(math.ceil(fargs[0]))
                if fname == "floor" and len(fargs) == 1:
                    return float(math.floor(fargs[0]))
                if fname == "min" and fargs:
                    return float(min(fargs))
                if fname == "max" and fargs:
                    return float(max(fargs))
                raise EvalError(
                    f"unsupported function call in expression {value!r}: {fname}(...) with {len(fargs)} args")
            raise EvalError(f"unsupported expression syntax: {value!r}")

        result = visit(tree)
    else:
        raise EvalError(f"cannot evaluate value of type {type(value).__name__}: {value!r}")

    if not math.isfinite(result) or result < 0:
        raise EvalError(f"value must be finite and non-negative: {value!r} -> {result}")
    return result


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            case = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise EvalError(f"invalid JSONL at {path}:{lineno}: {exc}") from exc
        if not isinstance(case, dict):
            raise EvalError(f"case at {path}:{lineno} must be an object")
        cases.append(case)
    if not cases:
        raise EvalError(f"no cases found in {path}")
    return cases


def dtype_bytes(dtype: str, instruction_db: dict[str, Any]) -> float:
    normalized = dtype.strip()
    table = instruction_db["dtype_bytes"]
    if normalized not in table:
        raise EvalError(f"unsupported dtype: {dtype}")
    return float(table[normalized])


def resolve_instruction_entry(inst: dict[str, Any], instruction_db: dict[str, Any]) -> tuple[dict[str, Any], str]:
    op = str(inst.get("op", ""))
    instructions = instruction_db["instructions"]
    if op not in instructions:
        raise EvalError(f"unsupported RegBase instruction: {op}")

    entry = instructions[op]
    default_model = (instruction_db.get("instruction_defaults") or {}).get("default_execution_model")
    model_name = entry.get("default_execution_model") or default_model
    if not model_name:
        raise EvalError(f"{op} missing default_execution_model")
    # dtype- and field-conditional models are applied later in resolve_execution_model.
    return entry, str(model_name)


# Aliases that spell the same logical dtype. Maps each spelling to one canonical
# token so dtype/pair validation treats fp16/float16/half (etc.) as equal. Names
# not listed (e.g. fp8_e4m3fn_t, fp4x2_e2m1_t, complex32, bool) are their own
# canonical token, preserving distinctions the instruction DB cares about.
_DTYPE_CANONICAL = {
    "float16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "bfloat16_t": "bf16",
    "float32": "fp32",
    "float": "fp32",
    "int8_t": "int8",
    "uint8_t": "uint8",
    "int16_t": "int16",
    "uint16_t": "uint16",
    "int32_t": "int32",
    "uint32_t": "uint32",
    "int64_t": "int64",
    "uint64_t": "uint64",
    "hi8": "hifloat8",
    "hifloat8_t": "hifloat8",
    "float8": "fp8",
    "float4": "fp4",
    "int4b_t": "int4x2_t",
}


def canonical_dtype(name: Any) -> str:
    token = str(name).strip()
    return _DTYPE_CANONICAL.get(token, token)


def validate_instruction_dtypes(inst: dict[str, Any], entry: dict[str, Any]) -> None:
    """Enforce the instruction DB's supported_dtypes / supported_dtype_pairs.

    Without this, an op/dtype combo the hardware does not support (e.g. Exp on
    int32, or an unsupported Cast src->dst pair) silently produces a confident
    but meaningless number. Comparisons are alias-normalized via canonical_dtype.
    """
    op = inst.get("op")
    src = inst.get("src_dtype")
    dst = inst.get("dst_dtype")

    # Conversion-style ops (Cast/Pack/UnPack/Truncate) are validated as src->dst
    # pairs whenever both ends are given, since a type being individually legal
    # does not mean a given conversion path is supported.
    pairs = entry.get("supported_dtype_pairs")
    conversions = entry.get("supported_conversions")
    if (pairs or conversions) and src and dst:
        allowed_pairs: set[tuple[str, str]] = set()
        for pair in pairs or []:
            allowed_pairs.add((
                canonical_dtype(pair.get("src_dtype")),
                canonical_dtype(pair.get("dst_dtype")),
            ))
        for conv in conversions or []:
            if "_to_" in str(conv):
                a, b = str(conv).split("_to_", 1)
                allowed_pairs.add((canonical_dtype(a), canonical_dtype(b)))
        if (canonical_dtype(src), canonical_dtype(dst)) not in allowed_pairs:
            raise EvalError(f"{op}: unsupported dtype conversion {src}->{dst} (not in supported pairs)")
        return

    supported = entry.get("supported_dtypes")
    if supported:
        allowed = {canonical_dtype(d) for d in supported}
        for field in ("dtype", "src_dtype", "dst_dtype"):
            value = inst.get(field)
            if value and canonical_dtype(value) not in allowed:
                raise EvalError(f"{op}: unsupported {field}={value} (not in {op} supported_dtypes)")


def dtype_keys_for_inst(inst: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    dtype = inst.get("dtype")
    src = inst.get("src_dtype")
    dst = inst.get("dst_dtype")
    if dtype:
        keys.append(str(dtype))
    if src and dst:
        keys.append(f"{src}->{dst}")
        keys.append(f"{src}_to_{dst}")
    for value in (src, dst):
        if value:
            keys.append(str(value))
    return keys


def condition_matches(cond: dict[str, Any], inst: dict[str, Any]) -> bool:
    field_equals = cond.get("when_field_equals") or {}
    if not field_equals:
        return False
    for field, expected in field_equals.items():
        actual = inst.get(str(field))
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def resolve_execution_model(
    inst: dict[str, Any],
    entry: dict[str, Any],
    model_name: str,
    instruction_db: dict[str, Any],
) -> dict[str, Any]:
    defaults = instruction_db.get("execution_model_defaults", {})
    if model_name not in defaults:
        raise EvalError(f"{inst.get('op')} references unknown execution model {model_name!r}")
    model = dict(defaults[model_name])
    model["_resolved_model_name"] = model_name

    for key in dtype_keys_for_inst(inst):
        override = (entry.get("dtype_execution_models") or {}).get(key)
        if override:
            override = dict(override)
            override_model_name = override.pop("default_execution_model", None)
            if override_model_name:
                if override_model_name not in defaults:
                    raise EvalError(f"{inst.get('op')} references unknown execution model {override_model_name!r}")
                model = dict(defaults[override_model_name])
                model["_resolved_model_name"] = override_model_name
            model.update(override)
            break

    for cond in entry.get("conditional_execution_models", []):
        if condition_matches(cond, inst):
            cond_model_name = cond.get("default_execution_model")
            if cond_model_name:
                if cond_model_name not in defaults:
                    raise EvalError(f"{inst.get('op')} references unknown execution model {cond_model_name!r}")
                model = dict(defaults[cond_model_name])
                model["_resolved_model_name"] = cond_model_name
            model.update(cond.get("execution_model", {}))
            break

    # bytes_per_instruction is a global constant (fixed SIMD width); a per-model
    # value, if any, overrides it, otherwise fall back to the global default.
    if model.get("bytes_per_instruction") is None:
        model["bytes_per_instruction"] = instruction_db.get("bytes_per_instruction")
    model.setdefault("dtype_byte_rule", "dtype_bytes")
    if model.get("bytes_per_instruction") is None:
        raise EvalError(f"{inst.get('op')} execution model missing bytes_per_instruction")
    if model.get("issue_slots") is None or model.get("throughput_cycles") is None:
        raise EvalError(f"{inst.get('op')} execution model missing issue_slots/throughput_cycles")
    return model


def instruction_dtype_width(inst: dict[str, Any], model: dict[str, Any], instruction_db: dict[str, Any]) -> float:
    rule = model.get("dtype_byte_rule")
    if rule == "max(src_dtype_bytes, dst_dtype_bytes)":
        src = inst.get("src_dtype") or inst.get("dtype")
        dst = inst.get("dst_dtype") or inst.get("dtype")
        if not src or not dst:
            raise EvalError(f"{inst.get('op')} requires src_dtype/dst_dtype or dtype")
        return max(dtype_bytes(str(src), instruction_db), dtype_bytes(str(dst), instruction_db))
    dtype = inst.get("dtype")
    if not dtype:
        src = inst.get("src_dtype")
        dst = inst.get("dst_dtype")
        if src and dst:
            return max(
                dtype_bytes(str(src), instruction_db),
                dtype_bytes(str(dst), instruction_db),
            )
        raise EvalError(f"{inst.get('op')} requires dtype")
    return dtype_bytes(str(dtype), instruction_db)


def effective_issue_per_cycle(inst: dict[str, Any], model: dict[str, Any]) -> float:
    issue = float(model["issue_slots"]) / float(model["throughput_cycles"])
    if str(inst.get("dependency", "independent")).lower() == "dependent":
        dependent_issue = model.get("dependent_issue_per_cycle")
        if dependent_issue is None:
            return min(issue, 1.0)
        return float(dependent_issue)
    return issue


def evaluate_instruction(
    inst: dict[str, Any],
    variables: dict[str, float],
    active_cores: float,
    vector_freq_hz: float,
    instruction_db: dict[str, Any],
) -> dict[str, Any]:
    entry, model_name = resolve_instruction_entry(inst, instruction_db)
    validate_instruction_dtypes(inst, entry)
    model = resolve_execution_model(inst, entry, model_name, instruction_db)
    width = instruction_dtype_width(inst, model, instruction_db)
    issue = effective_issue_per_cycle(inst, model)
    bytes_per_instruction = float(model["bytes_per_instruction"])
    if inst.get("elements") is None:
        raise EvalError(f"{inst.get('op')} instruction requires elements")
    elements = safe_eval(inst.get("elements"), variables)
    count = safe_eval(inst.get("count", 1), variables)
    efficiency = safe_eval(inst.get("efficiency", 1), variables)
    if not (0 < efficiency <= 1):
        raise EvalError(f"instruction efficiency must be in (0,1]: {inst}")

    elems_per_cycle_per_core = issue * bytes_per_instruction / width
    elems_per_second = (elems_per_cycle_per_core * active_cores * vector_freq_hz * efficiency)
    time_us = (elements * count / elems_per_second * 1e6 if elems_per_second > 0 else math.inf)
    return {
        "op": inst.get("op"),
        "execution_unit": model.get("unit"),
        "execution_pipe": str(model.get("pipe") or model.get("unit")),
        "execution_model": model.get("_resolved_model_name", model_name),
        "elements": elements,
        "count": count,
        "dtype_width_bytes": width,
        "bytes_per_instruction": bytes_per_instruction,
        "issue_slots": float(model["issue_slots"]),
        "throughput_cycles": float(model["throughput_cycles"]),
        "issue_per_cycle": issue,
        "elements_per_cycle_per_core": elems_per_cycle_per_core,
        "active_vector_cores": active_cores,
        "time_us": time_us,
    }


def memory_time_us(bytes_value: float, bandwidth_bytes_per_s: float) -> float:
    if bytes_value <= 0:
        return 0.0
    if bandwidth_bytes_per_s <= 0:
        return math.inf
    return bytes_value / bandwidth_bytes_per_s * 1e6


def resolve_gm_bandwidth(case: dict[str, Any], variant: dict[str, Any]) -> tuple[str, float]:
    model = str(case.get("gm_bandwidth_model", "chip_peak"))
    if model == "chip_peak":
        return model, float(variant["peak"]["gm_bandwidth_bytes_per_s"])
    if model == "vector_core_conservative":
        return model, float(variant["derived_from_ini"]["vector_core_ddr_bandwidth_bytes_per_s"])
    if model == "l2_aware":
        return (
            model,
            0.0,
        )  # effective BW is per-case (curve of working set); resolved in evaluate_case
    if model == "l2_capacity":
        return (
            model,
            0.0,
        )  # per-direction L2/HBM decided by working-set vs L2 capacity in evaluate_case
    raise EvalError(f"unsupported gm_bandwidth_model {model!r}")


def l2_aware_copy_bw(hbm_streamed_bytes: float, la: dict[str, Any]) -> float:
    """Effective combined (read+write) GM bandwidth as a piecewise-linear curve of the
    HBM-streamed working set. Calibrated from a same-box copy sweep (see hardware json
    common.gm_bandwidth_l2_aware.source): small working sets stay L2-resident (~2.8 TB/s),
    large ones fall to HBM-sustained (~1.45 TB/s). The curve already folds in read/write
    contention (it is the measured copy rate), so no separate alpha is applied.
    """
    curve = la["curve_combined_bytes_to_copy_bps"]
    pts = sorted((float(b), float(bw)) for b, bw in curve)
    if hbm_streamed_bytes <= pts[0][0]:
        return pts[0][1]
    if hbm_streamed_bytes >= pts[-1][0]:
        return pts[-1][1]
    for (b0, bw0), (b1, bw1) in zip(pts, pts[1:]):
        if b0 <= hbm_streamed_bytes <= b1:
            t = (hbm_streamed_bytes - b0) / (b1 - b0) if b1 > b0 else 0.0
            return bw0 + t * (bw1 - bw0)
    return pts[-1][1]


def resolve_fixed_overhead(case: dict[str, Any], variables: dict[str, float]) -> tuple[float, str]:
    """Use an explicitly supplied fixed overhead, or zero when omitted."""
    model = str(case.get("fixed_overhead_model", "none"))
    if model != "none":
        raise EvalError(
            f"unsupported fixed_overhead_model {model!r}; use fixed_overhead_us from a documented measurement")
    if case.get("fixed_overhead_us") is not None:
        return safe_eval(case["fixed_overhead_us"], variables), "explicit"
    return 0.0, "none"


def resolve_tiling(case: dict[str, Any], hardware_db: dict[str, Any],
                   variables: dict[str, float]) -> tuple[float | None, dict[str, Any]]:
    """Resolve the tile count used for load-balance and fill/drain modeling.

    Precedence: an explicit `tile_count` wins. Otherwise, if a `tiling` block is
    given, the tile count is derived from UB capacity: each tile must hold its
    working set (with double buffering) in UB, so

        max_units_per_tile = floor(ub_bytes / (ub_bytes_per_unit * double_buffer))
        tile_count         = ceil(total_tile_units / max_units_per_tile)

    A working set that cannot fit even one unit is a UB-infeasible design and
    raises. (Operator-level template routing is responsible for picking a feasible
    template — e.g. a row-split multi-pass template — before a case reaches here;
    the engine just evaluates the one case_dict it is given.) Returns (None, ...)
    when neither is given, preserving legacy behaviour (no balance, no fill/drain).
    """
    explicit = case.get("tile_count")
    if explicit is not None:
        tiles = safe_eval(explicit, variables)
        if tiles <= 0:
            raise EvalError(f"tile_count must be positive: {explicit!r}")
        return tiles, {"source": "explicit", "core_split_units": None}

    tiling = case.get("tiling")
    if not tiling:
        return None, {"source": "none", "core_split_units": None}

    # core_split_units = the work-unit count distributed ACROSS cores (elementwise:
    # elements `n`; row-based: `rows`). This — NOT the intra-core UB tile_count —
    # is what determines inter-core tail-load imbalance (core_balance). A kernel that
    # statically splits N units evenly over C cores has imbalance ceil(N/C)/(N/C),
    # which is ~1 at fine (element) granularity and only bites at coarse (few-row)
    # granularity. Feeding the UB tile_count here instead wrongly penalizes evenly
    # split kernels (e.g. vchain: 265 UB-tiles/64 → spurious 21%, though N split
    # evenly). Absent → core_balance falls back to tile_count (legacy).
    core_split_expr = tiling.get("core_split_units")
    core_split_units = (safe_eval(core_split_expr, variables) if core_split_expr is not None else None)
    if core_split_units is not None and core_split_units <= 0:
        raise EvalError(f"core_split_units must be positive: {core_split_expr!r}")

    total_units = safe_eval(tiling.get("total_tile_units"), variables)
    bytes_per_unit = safe_eval(tiling.get("ub_bytes_per_unit"), variables)
    double_buffer = safe_eval(tiling.get("double_buffer", 2), variables)
    if total_units <= 0 or bytes_per_unit <= 0 or double_buffer <= 0:
        raise EvalError(f"tiling requires positive total_tile_units/ub_bytes_per_unit/double_buffer: {tiling!r}")

    ub_bytes = float((hardware_db.get("common") or {})["memory_capacity_bytes_per_core"]["ub"])
    per_tile_budget = bytes_per_unit * double_buffer
    max_units = math.floor(ub_bytes / per_tile_budget)
    if max_units < 1:
        raise EvalError(f"UB overflow: one unit needs {per_tile_budget:.0f}B (double_buffer={double_buffer:g}) "
                        f"but UB is {ub_bytes:.0f}B; tiling infeasible (operator should route to a row-split template)")
    tiles = float(math.ceil(total_units / max_units))
    ub_used = min(max_units, total_units) * per_tile_budget
    return tiles, {
        "source": "ub_capacity",
        "max_units_per_tile": max_units,
        "ub_bytes_used": ub_used,
        "ub_utilization": ub_used / ub_bytes,
        "core_split_units": core_split_units,
    }


def core_balance_factor(split_units: float | None, active_cores: float) -> float:
    """Inter-core tail-load balance factor in (0, 1].

    `split_units` is the count of work-units the kernel hands to cores (elements
    for an even-split elementwise kernel, rows for a row-split one). When it is not
    divisible by active_vector_cores, the most-loaded core does ceil(avg) units
    while the average is avg, so the kernel waits on the slow core:
        eta = avg_units_per_core / max_units_per_core = (U/C) / ceil(U/C).
    At fine (element) granularity eta ~= 1 (even-split kernels are balanced); it
    only bites at coarse (few-row) granularity. Only the vector pipe is
    core-quantized (MTE pipes use chip-level aggregate bandwidth), so eta divides
    vector_time only. Returns 1.0 when split_units is None (legacy: no balance).
    """
    if split_units is None:
        return 1.0
    rounds = math.ceil(split_units / active_cores)
    avg = split_units / active_cores
    eta = avg / rounds if rounds > 0 else 1.0
    return min(1.0, eta)


def evaluate_case(case: dict[str, Any], hardware_db: dict[str, Any], instruction_db: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("case_id") or "")
    if not case_id:
        raise EvalError("case missing case_id")

    raw_variables = case.get("variables") or {}
    variables = {str(k): safe_eval(v, {}) for k, v in raw_variables.items()}
    variant_name = str(case.get("hardware_variant") or hardware_db["default_variant"])
    variants = hardware_db["variants"]
    if variant_name not in variants:
        raise EvalError(f"unsupported hardware_variant {variant_name!r}")
    variant = variants[variant_name]

    vector_cores = float(variant["core_counts"]["vector_core"])
    active_cores = safe_eval(case.get("active_vector_cores", vector_cores), variables)
    if not (0 < active_cores <= vector_cores):
        raise EvalError(f"active_vector_cores must be in (0,{int(vector_cores)}]: {active_cores}")
    vector_freq_hz = float(variant["frequency"]["vector_freq_mhz"]) * 1e6

    tile_count, tiling_info = resolve_tiling(case, hardware_db, variables)

    memory = case.get("memory") or {}
    gm_read = safe_eval(memory.get("gm_read_bytes", 0), variables)
    gm_write = safe_eval(memory.get("gm_write_bytes", 0), variables)
    gm_reread = safe_eval(memory.get("gm_reread_bytes", 0), variables)
    l2_read = safe_eval(memory.get("l2_read_bytes", 0), variables)
    l2_write = safe_eval(memory.get("l2_write_bytes", 0), variables)

    gm_bandwidth_model, gm_bw = resolve_gm_bandwidth(case, variant)
    l2_bw = float(variant["peak"]["l2_bandwidth_bytes_per_s"])
    gm_rampup_model = str(case.get("gm_rampup_model", "none"))
    if gm_rampup_model != "none":
        raise EvalError(f"unsupported gm_rampup_model {gm_rampup_model!r}; only 'none' is supported")
    gm_rw_contention_alpha = None
    combined_mem = False

    if gm_bandwidth_model == "l2_aware":
        # L2-residency-aware GM model (calibrated). Effective read+write bandwidth depends
        # on the HBM-streamed working set (L2-assisted for small, HBM-sustained for large),
        # and a tensor read more than once (gm_reread_bytes) is served by L2 on the re-read
        # so it overlaps the HBM stream rather than adding to it. The copy-rate curve already
        # includes read/write contention, so no separate alpha. Fixes both the medium-shape
        # over-prediction (L2) and the large-shape under-prediction (R+W don't overlap).
        la = (hardware_db.get("common") or {}).get("gm_bandwidth_l2_aware")
        if not la:
            raise EvalError("gm_bandwidth_model=l2_aware requires common.gm_bandwidth_l2_aware in hardware json")
        cold_read = max(0.0, gm_read - gm_reread)
        hbm_streamed = cold_read + gm_write
        gm_bw = l2_aware_copy_bw(hbm_streamed, la)  # effective combined copy BW
        l2_reread_bw = float(la["l2_reread_bandwidth_bytes_per_s"])
        mte2_time = max(memory_time_us(cold_read, gm_bw), memory_time_us(l2_read, l2_bw))
        mte3_time = max(memory_time_us(gm_write, gm_bw), memory_time_us(l2_write, l2_bw))
        combined_gm = memory_time_us(hbm_streamed, gm_bw)  # R+W share the copy BW
        reread_time = memory_time_us(gm_reread, l2_reread_bw)  # L2-served, overlaps HBM stream
        mem_pipes = [max(combined_gm, reread_time, mte2_time, mte3_time)]
        combined_mem = True
    elif gm_bandwidth_model == "l2_capacity":
        # 确定性 L2 容量模型:逐方向按"复用工作集 W vs L2 容量 C"判 L2 还是 HBM 带宽——不是曲线拟合。
        # 物理:循环 reread 在 LRU 下是阶跃,W≤C 全命中 L2、W>C 全 miss HBM(thrash);写同理(L2 吸收 vs 积压回压)。
        # 常数全部 membw 上板实测(common.gm_bandwidth_measured)。W 默认=reread 总量(global 复用跨度),
        # case 可显式 memory.reuse_working_set 覆盖(tile 级复用给 tile 字节)。
        m = (hardware_db.get("common") or {}).get("gm_bandwidth_measured") or {}
        cap = float(m.get("l2_residency_knee_bytes", 0))
        if cap <= 0 or "l2_read_bandwidth_bytes_per_s" not in m:
            raise EvalError(
                "gm_bandwidth_model=l2_capacity requires common.gm_bandwidth_measured.l2_residency_knee_bytes"
                " + l2_read/write_bandwidth_bytes_per_s")
        l2_rd = float(m["l2_read_bandwidth_bytes_per_s"])
        l2_wr = float(m["l2_write_bandwidth_bytes_per_s"])
        hbm_rd = float(m["hbm_sustained_read_bytes_per_s"])
        hbm_wr = float(m["hbm_sustained_write_bytes_per_s"])
        cold_read = max(0.0, gm_read - gm_reread)
        # 有效容量 C_eff = C / pass内并存流数:一个 pass 里同时占 L2 的大流(reread 读 + 写出)互相挤,
        # 单流能独占的驻留容量 = C/份数。上板实测(rms_norm 行级双 pass, reread x + write y 并存):
        # 满命中拐点 ~64MB = C/2 = 128MiB/2。流数由 flow 标(memory.l2_concurrent_streams,默认 1=独占)。
        streams = max(1.0, safe_eval(memory.get("l2_concurrent_streams", 1), variables))
        c_eff = cap / streams
        w_reuse = safe_eval(memory.get("reuse_working_set", gm_reread), variables)
        reread_bw = l2_rd if (gm_reread > 0 and w_reuse <= c_eff) else hbm_rd
        write_bw = l2_wr if (0 < gm_write <= c_eff) else hbm_wr
        # 读侧:冷读(HBM)+ 第二遍 reread(命中 L2 / miss HBM),两遍串行相加。l2_read/write 显式字节走 L2 实测带宽。
        mte2_time = (memory_time_us(cold_read, hbm_rd) + memory_time_us(gm_reread, reread_bw) +
                     memory_time_us(l2_read, l2_rd))
        mte3_time = memory_time_us(gm_write, write_bw) + memory_time_us(l2_write, l2_wr)
        mem_pipes = [mte2_time, mte3_time]
    else:
        mte2_time = max(memory_time_us(gm_read, gm_bw), memory_time_us(l2_read, l2_bw))
        mte3_time = max(memory_time_us(gm_write, gm_bw), memory_time_us(l2_write, l2_bw))

        # Optional HBM read/write contention. max_pipe treats mte2 (read) and mte3
        # (write) as fully overlapped, but they share HBM. membw-bench measured the
        # combined GM time ~= max(mte2,mte3) + alpha*min(mte2,mte3), alpha~0.3 (see
        # hardware common.gm_bandwidth_measured). Opt-in via case "gm_rw_contention"
        # (true -> hardware alpha; a number -> explicit alpha). Default off keeps the
        # two pipes independent, so existing cases are unchanged.
        rw = case.get("gm_rw_contention")
        if rw:
            if isinstance(rw, bool):
                measured = (hardware_db.get("common") or {}).get("gm_bandwidth_measured") or {}
                gm_rw_contention_alpha = float(measured.get("rw_contention_alpha", 0.3))
            else:
                gm_rw_contention_alpha = safe_eval(rw, variables)
            hi, lo = max(mte2_time, mte3_time), min(mte2_time, mte3_time)
            mem_pipes = [hi + gm_rw_contention_alpha * lo]
        else:
            mem_pipes = [mte2_time, mte3_time]

    vector = case.get("vector") or {}
    instructions = vector.get("instructions") or []
    if not isinstance(instructions, list) or not instructions:
        raise EvalError(f"{case_id}: vector.instructions must be a non-empty list")
    instruction_rows = [
        evaluate_instruction(inst, variables, active_cores, vector_freq_hz, instruction_db) for inst in instructions
    ]
    # Aggregate vector time by physical execution resource. Within one unit,
    # instructions on DIFFERENT execution sub-pipes (the multi-cycle
    # transcendental datapath vs the 1-cycle ALU datapath) run in parallel and
    # contend only for the shared issue port. So a unit's time is
    #   max(issue_port_time, each sub-pipe's summed time)
    # NOT the sum of all instruction times. Ascend950 microbench calibration
    # (exp + k*add, independent/ILP-rich) matches this max model within ~6%,
    # while the old sum model overestimated by 20-60% whenever transcendentals
    # mixed with cheap ALU ops. A `dependency="dependent"` correction still
    # inflates the individual instruction's time before aggregation, so it
    # rides on top of this floor rather than being baked in. See
    # tests/calibration/instruments/regbase-microbench/docs/overlap_calibration_findings.md.
    issue_port_times: dict[str, float] = {}
    subpipe_times: dict[str, float] = {}  # key "unit::pipe"
    for row in instruction_rows:
        unit = str(row.get("execution_unit") or "legacy_vector_pipe")
        t = float(row["time_us"])
        throughput = float(row.get("throughput_cycles", 1)) or 1.0
        # Issue-port occupancy is throughput-independent (one issue slot per
        # instruction). time_us already folds in throughput_cycles, so dividing
        # it back out recovers the pure issue-port time on this unit.
        issue_port_times[unit] = issue_port_times.get(unit, 0.0) + t / throughput
        pipe = str(row.get("execution_pipe") or unit)
        subpipe_times[f"{unit}::{pipe}"] = subpipe_times.get(f"{unit}::{pipe}", 0.0) + t
    units = set(issue_port_times) | {key.split("::", 1)[0] for key in subpipe_times}
    vector_resource_times: dict[str, float] = {}
    for unit in units:
        candidates = [issue_port_times.get(unit, 0.0)]
        candidates.extend(v for key, v in subpipe_times.items() if key.split("::", 1)[0] == unit)
        vector_resource_times[unit] = max(candidates)
    vector_time_balanced = (max(vector_resource_times.values()) if vector_resource_times else 0.0)

    # Inter-core tail-load balance (A) is driven by how the kernel splits work
    # ACROSS cores (core_split_units: elements for elementwise, rows for row-based),
    # NOT by the intra-core UB tile_count. Falls back to tile_count when a flow does
    # not declare core_split_units (legacy). The pipeline fill/drain term (B) below
    # still uses the UB tile_count (intra-core pipeline depth). Only the vector pipe
    # is core-quantized, so eta divides vector_time only.
    core_split_units = tiling_info.get("core_split_units")
    balance_units = core_split_units if core_split_units is not None else tile_count
    core_balance = core_balance_factor(balance_units, active_cores)
    vector_time = vector_time_balanced / max(core_balance, 1e-9)

    overlap_model = str(case.get("overlap_model", "max_pipe"))
    fill_drain = 0.0
    if overlap_model == "max_pipe":
        body = max(max(mem_pipes), vector_time)
        # Pipeline fill/drain: a finite tile count cannot reach perfect steady
        # state. First-order, the non-bottleneck per-tile stages of the first
        # and last tiles are not overlapped. Reduces to `serial` at 1 tile and
        # to pure max_pipe as tile_count -> inf. Disable with "fill_drain": false.
        if tile_count and tile_count > 0 and case.get("fill_drain", True):
            per_tile_sum = (sum(mem_pipes) + vector_time) / tile_count
            per_tile_bottleneck = max(max(mem_pipes), vector_time) / tile_count
            fill_drain = per_tile_sum - per_tile_bottleneck
            body += fill_drain
    elif overlap_model == "serial":
        body = sum(mem_pipes) + vector_time
    else:
        raise EvalError(f"{case_id}: unsupported overlap_model {overlap_model!r}")

    # Launch floor: a kernel cannot finish faster than its fixed launch/startup cost.
    # Applied as a FLOOR on the body (max, NOT additive) so it stays a valid lower bound:
    # tiny shapes are lifted to the floor; medium/large keep their body. Additive would
    # over-predict medium shapes whose body already ~= measured. Calibrated on Ascend950
    # (common.overheads.min_launch_floor_us); opt-in via case "launch_floor_model".
    launch_floor_model = str(case.get("launch_floor_model", "none"))
    launch_floor = 0.0
    if launch_floor_model == "hardware":
        lf = ((hardware_db.get("common") or {}).get("overheads") or {}).get("min_launch_floor_us")
        if lf is None:
            raise EvalError("launch_floor_model=hardware requires common.overheads.min_launch_floor_us")
        launch_floor = float(lf)
    elif launch_floor_model != "none":
        raise EvalError(f"{case_id}: unsupported launch_floor_model {launch_floor_model!r}")
    body = max(body, launch_floor)

    fixed_overhead, fixed_overhead_source = resolve_fixed_overhead(case, variables)
    total = fixed_overhead + body
    target_efficiency = safe_eval(case.get("target_efficiency", 0.60), variables)
    if not (0 < target_efficiency <= 1):
        raise EvalError(f"{case_id}: target_efficiency must be in (0,1]")

    # Bound is the bottleneck pipe. With gm_rw_contention on, mte2 (read) and
    # mte3 (write) are merged into one HBM-shared mem pipe, so compare that
    # combined time (the value that actually entered `body`) against vector and
    # attribute it to the dominant memory direction; otherwise compare the three
    # pipes independently.
    if gm_rw_contention_alpha is not None or combined_mem:
        mem_label = "mte2_bound" if mte2_time >= mte3_time else "mte3_bound"
        pipe_times = {mem_label: mem_pipes[0], "vector_bound": vector_time}
    else:
        pipe_times = {
            "mte2_bound": mte2_time,
            "vector_bound": vector_time,
            "mte3_bound": mte3_time,
        }
    bound = max(pipe_times, key=pipe_times.get)
    return {
        "case_id": case_id,
        "hardware_variant": variant_name,
        "active_vector_cores": active_cores,
        "gm_bandwidth_model": gm_bandwidth_model,
        "gm_bandwidth_bytes_per_s": gm_bw,
        "l2_bandwidth_bytes_per_s": l2_bw,
        "fixed_overhead_us": fixed_overhead,
        "fixed_overhead_source": fixed_overhead_source,
        "launch_floor_us": launch_floor,
        "core_balance": core_balance,
        "tile_count": tile_count,
        "tiling": tiling_info,
        "gm_rampup_model": gm_rampup_model,
        "gm_rw_contention_alpha": gm_rw_contention_alpha,
        "fill_drain_us": fill_drain,
        "overlap_model": overlap_model,
        "memory": {
            "gm_read_bytes": gm_read,
            "gm_write_bytes": gm_write,
            "gm_reread_bytes": gm_reread,
            "l2_read_bytes": l2_read,
            "l2_write_bytes": l2_write,
        },
        "mte2_time_us": mte2_time,
        "mte3_time_us": mte3_time,
        "vector_time_us": vector_time,
        "vector_time_pre_balance_us": vector_time_balanced,
        "vector_resource_times_us": vector_resource_times,
        "vector_issue_port_times_us": issue_port_times,
        "vector_subpipe_times_us": subpipe_times,
        "kernel_body_time_us": body,
        "theoretical_time_us": total,
        "target_efficiency": target_efficiency,
        "cp4_target_time_us": total / target_efficiency,
        "bound": bound,
        "instruction_breakdown": instruction_rows,
    }


def render_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# A5 Pure Vector Theoretical Performance",
        "",
        "| case_id | variant | cores | bound | mte2_us | vector_us | mte3_us | fixed_us | "
        "theoretical_us | cp4_target_us |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append("| {case_id} | {variant} | {cores:.0f} | {bound} | {mte2:.4g} | {vec:.4g} | "
                     "{mte3:.4g} | {fixed:.4g} | {total:.4g} | {target:.4g} |".format(
                         case_id=row["case_id"],
                         variant=row["hardware_variant"],
                         cores=row["active_vector_cores"],
                         bound=row["bound"],
                         mte2=row["mte2_time_us"],
                         vec=row["vector_time_us"],
                         mte3=row["mte3_time_us"],
                         fixed=row["fixed_overhead_us"],
                         total=row["theoretical_time_us"],
                         target=row["cp4_target_time_us"],
                     ))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-json", type=Path, required=True)
    parser.add_argument("--instruction-json", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    try:
        hardware_db = load_json(args.hardware_json)
        instruction_db = load_json(args.instruction_json)
        results = [evaluate_case(case, hardware_db, instruction_db) for case in load_cases(args.cases)]
        output = json.dumps({"schema_version": 1, "results": results}, ensure_ascii=False, indent=2)
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(output + "\n", encoding="utf-8")
        if args.out_md:
            args.out_md.parent.mkdir(parents=True, exist_ok=True)
            args.out_md.write_text(render_markdown(results), encoding="utf-8")
        if not args.out_json and not args.out_md:
            print(output)
        return 0
    except (OSError, EvalError, json.JSONDecodeError, SyntaxError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
