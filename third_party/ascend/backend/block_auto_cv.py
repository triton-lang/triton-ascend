# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Adapt ordinary compiled CV kernels to the packaged CPU selector."""
from .block_auto_models.native_model import reject

import math
from .block_auto_models.native_model import analyze as analyze_candidate

TERMINAL_EXCLUSIONS = frozenset(("COMPILE_REJECTED", "FRONTEND_REJECTED", "UB_REJECTED", "NO_MERGE_OPPORTUNITY"))


def merge_kind(metadata, factor, *, frontend=False):
    if factor == 1:
        return "unmerged"
    report = metadata if frontend else metadata.get("block_batching", {})
    if frontend:
        reject(report.get('compile_status') != 'REWRITTEN', 'frontend did not admit the CV rewrite', ValueError)
    if report.get("cv_mode") == "independent_batch_v1":
        return "batch" if frontend else "independent_batch"
    lowering = report.get("cv_lowering", "")
    if frontend:
        lowering = lowering.replace("_uniform_control", "")
    if report.get("cv_mode") == "spatial_concat_v1":
        role = {"spatial_m_v1": "M", "spatial_n_v1": "N"}.get(lowering)
        if role:
            return role
    raise ValueError("compiler CV axis role is not covered")


def candidate_plan(contract, *, probe, axis=-1):
    """Choose two candidates from a provider reporting proved frontend roles.

    probe(axis, factor) returns the rewrite report or raises ValueError
    for an unsupported candidate. This module does not rewrite any IR.
    """

    def plan(reason, candidates=()):
        return dict(candidates=list(candidates), reason=reason)

    grid = contract["grid"]
    admitted = []
    for selected_axis in (range(3) if axis == -1 else (axis, )):
        for factor in (2, 4):
            if grid[selected_axis] < factor or grid[selected_axis] % factor:
                continue
            try:
                report = probe(selected_axis, factor)
                reject(
                    report.get('selected_axis') != selected_axis or report.get('requested_factor') != factor,
                    'frontend selected a different axis or factor', ValueError)
                kind = merge_kind(report, factor, frontend=True)
                admitted.append(
                    dict(key=f"a{selected_axis}_f{factor}", axis=selected_axis, factor=factor, merge_kind=kind))
            except ValueError:
                continue
    for selected_axis in range(3):
        roles = {item["merge_kind"] for item in admitted if item["axis"] == selected_axis}
        if len(roles) > 1:
            return plan("CV_AXIS_ROLE_DEPENDS_ON_FACTOR")
    spatial = sorted((item for item in admitted if item["merge_kind"] in ("M", "N")), key=lambda item:
                     (item["merge_kind"], item["axis"], item["factor"]))
    by_role = {
        role: next((item
                    for item in spatial
                    if item["merge_kind"] == role and item["factor"] == 4), None)
        for role in ("M", "N")
    }
    if all(by_role.values()):
        return plan("CV_M4_N4", [by_role["M"], by_role["N"]])
    elif any(by_role.values()):
        preferred = by_role["M"] or by_role["N"]
        return plan("CV_ONE_SPATIAL_AXIS_F2_F4", [item for item in spatial if item["axis"] == preferred["axis"]])
    elif spatial:
        return plan("CV_SPATIAL_F2_ONLY", spatial[:2])
    batch = [item for item in admitted if item["merge_kind"] == "batch"]
    if batch:
        selected_axis = min(item["axis"] for item in batch)
        return plan("CV_BATCH_F2_F4", [item for item in batch if item["axis"] == selected_axis])
    return plan("CV_NO_ADMITTED_MERGE")


def _llvm_modules(item):
    # File roles are emitted by the backend. C++ verifies the LLVM entry and
    # the cube-only attribute for a single-module CV kernel.
    modules = {role: item["llvm_" + role] for role in ("aic", "aiv") if "llvm_" + role in item}
    if not modules and "llvm" in item:
        modules["aic"] = item["llvm"]
    return modules


def rank_cv(candidates, contract):
    """Rank compiled CV candidates, retaining the short independent-batch rule."""
    tensors = [v for v in contract["arguments"].values() if v["kind"] == "tensor"]
    records, incomplete = {}, []
    result = dict(selected_key="unmerged", model_complete=False, candidates=records, reason="CV_DTYPE_NOT_COVERED")
    if not tensors or any(v.get("dtype") not in ("fp32", "float32", "torch.float32") for v in tensors):
        return result
    for item in candidates:
        key, factor, axis = item["key"], item["factor"], item["axis"]
        row = dict(factor=factor, axis=axis, resource_checked=False,
                   status=item.get("status", "STATIC_ANALYSIS_UNCONFIRMED"))
        try:
            reject(not item.get("resource_checked"), item.get("error", "candidate resource evidence unavailable"),
                   ValueError)
            metadata = item["metadata"]
            trace = analyze_candidate(_llvm_modules(item), item["ttir"], item["final_ir"], metadata,
                                      item["launch_contract"], axis, factor)
            score, groups, cores = trace["score"], trace["launched_groups"], trace["cube_core_count"]
            reject(
                not trace["counts"].get("mma_calls") or not math.isfinite(score) or score <= 0 or groups <= 0
                or cores <= 0 or trace["auto_blockify"] and groups > cores, "incomplete Cube accounting")
            row.update(status="STATIC_FEATURES_READY", resource_checked=True, estimated_cycles=score,
                       merge_kind=merge_kind(metadata, factor), batch_grid=trace["logical_grid"], mode=trace["mode"],
                       useful_flops=trace["counts"]["useful_flops"], cube_core_count=cores)
        except (ValueError, KeyError, TypeError, IndexError, StopIteration, OverflowError) as error:
            row["error"] = str(error)
            if factor == 1 or row["status"] not in TERMINAL_EXCLUSIONS:
                incomplete.append(key)
        records[key] = row
    if incomplete:
        result.update(reason="CV_MODEL_INCOMPLETE", incomplete_candidates=incomplete)
        return result
    baseline, eligible = records["unmerged"], {}
    for key, row in records.items():
        if row["factor"] == 1 or not row["resource_checked"]:
            continue
        factor, axis, grid = row["factor"], row["axis"], list(contract["grid"])
        if axis not in (0, 1, 2) or grid[axis] < factor or grid[axis] % factor:
            continue
        grid[axis] //= factor
        if grid == row["batch_grid"] and all(row[k] == baseline[k]
                                             for k in ("mode", "cube_core_count", "useful_flops")):
            eligible[key] = row
    ranked = dict(eligible, unmerged=baseline)
    selected = min(ranked, key=lambda key:
                   (ranked[key]["estimated_cycles"], ranked[key]["factor"], ranked[key]["axis"]))
    reason = "LOWER_ESTIMATED_LATENCY" if selected != "unmerged" else "ORIGINAL_LOWEST_LATENCY"
    if not eligible:
        reason = "MERGED_RESOURCE_UNAVAILABLE"
    grid = contract["grid"]
    if (contract.get("cv_empirical_short_batch", True) and math.prod(grid) <= baseline["cube_core_count"]
            and not any(row.get("merge_kind") in ("M", "N") for row in records.values())):
        preferred = [
            key for key, row in eligible.items()
            if row["merge_kind"] == "independent_batch" and row["factor"] == 2 and all(
                n == 1 for axis, n in enumerate(grid) if axis != row["axis"])
        ]
        if preferred:
            selected = min(preferred, key=lambda key: (eligible[key]["axis"], key))
            reason = "EMPIRICAL_SINGLE_TILE_SHORT_BATCH_X2"
    result.update(selected_key=selected, model_complete=True, reason=reason)
    return result
