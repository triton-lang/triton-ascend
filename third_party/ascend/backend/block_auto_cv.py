# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Adapt ordinary compiled CV kernels to the packaged CPU selector."""
import hashlib
import json
import re

from .block_auto_models.cv_binary_selector import choose_factor, features
from .block_auto_models.cv_candidate_plan import merge_kind
from .block_auto_models.cv_cost_model import analyze_candidate, validated_assumptions
from .block_auto_models.cv_empirical_rule import select as empirical_select
from .block_batching import BatchingRejected

# The original CV selector uses only this target FP32 microtile from the
# hardware specification. Core counts and UB capacities come from final IR.
HARDWARE = {"compute_units": {"cube": {"fractal_sizes": {"fp32": {"m": 16, "k": 8, "n": 16}}}}}
TERMINAL_EXCLUSIONS = frozenset(("COMPILE_REJECTED", "FRONTEND_REJECTED", "UB_REJECTED", "NO_MERGE_OPPORTUNITY"))


def _probe_candidate(backend, frozen, options, axis, factor):
    """Run only frontend normalization/rewrite on a fresh frozen-IR clone."""
    from triton._C.libtriton import ir
    from .compiler import make_ttir
    variant = dict(options, block_batching_factor=factor, block_batching_axis=axis, block_auto_contract=None,
                   block_auto_collect=False, enable_graph_optimize=False, debug=False)
    normalized = backend.parse_options(variant)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = frozen.make_ir(backend.target, normalized, backend.get_codegen_implementation(normalized),
                            backend.get_module_map(), context)
    # make_ttir applies the exact normalization preceding the batching rewrite.
    # Disabling later GraphOptimize avoids modifying the frozen source or
    # attempting TTAdapter/native binary generation during role discovery.
    metadata = dict(normalized.__dict__)
    make_ttir(module, metadata, normalized)
    return metadata["block_batching"]


def _probe_kind(report):
    if report.get("compile_status") != "REWRITTEN":
        raise ValueError("frontend did not admit the CV rewrite")
    if report.get("cv_mode") == "independent_batch_v1":
        return "batch"
    lowering = report.get("cv_lowering", "")
    if report.get("cv_mode") == "spatial_concat_v1":
        if lowering in ("spatial_m_v1", "spatial_m_uniform_control_v1"):
            return "M"
        if lowering in ("spatial_n_v1", "spatial_n_uniform_control_v1"):
            return "N"
    raise ValueError("frontend CV axis role is not covered")


def candidate_plan(contract, baseline, *, backend=None, frozen=None, options=None, axis=-1):
    """Derive the published two-candidate CV shortlist from actual axis roles.

    Native frontend probes establish M/N/batch roles. Grid divisibility chooses
    at most two fully compiled candidates; names and timings are never inputs.
    """
    result = dict(candidates=[], axis_roles={}, front_end_probe_count=0, probes=[], reason="CV_AXIS_ROLES_UNAVAILABLE",
                  policy="cv-spatial-shortlist-v1")
    grid = contract.get("grid")
    if (not isinstance(grid,
                       (tuple, list)) or len(grid) != 3 or any(type(size) is not int or size <= 0 for size in grid)
            or type(axis) is not int or axis not in (-1, 0, 1, 2)):
        result["reason"] = "CV_INVALID_PLANNING_GRID_OR_AXIS"
        return result
    if backend is None or frozen is None or options is None:
        result["reason"] = "CV_FROZEN_FRONTEND_UNAVAILABLE"
        return result
    metadata = baseline.metadata._asdict()
    if metadata.get("mix_mode") not in ("aic", "mix") or metadata.get("parallel_mode") != "simd":
        result["reason"] = "CV_FRONTEND_MODE_NOT_COVERED"
        return result
    admitted = []
    for selected_axis in (range(3) if axis == -1 else (axis, )):
        for factor in (2, 4):
            if grid[selected_axis] < factor or grid[selected_axis] % factor:
                continue
            probe = dict(axis=selected_axis, factor=factor)
            result["front_end_probe_count"] += 1
            try:
                report = _probe_candidate(backend, frozen, options, selected_axis, factor)
                if report.get("selected_axis") != selected_axis or report.get("requested_factor") != factor:
                    raise ValueError("frontend selected a different axis or factor")
                kind = _probe_kind(report)
                probe.update(status="ADMITTED", merge_kind=kind)
                admitted.append(
                    dict(key=f"a{selected_axis}_f{factor}", axis=selected_axis, factor=factor, merge_kind=kind))
            except (BatchingRejected, ValueError) as error:
                probe.update(status="FRONTEND_REJECTED", reason=f"{type(error).__name__}: {error}")
            result["probes"].append(probe)
    for selected_axis in range(3):
        roles = {item["merge_kind"] for item in admitted if item["axis"] == selected_axis}
        if len(roles) > 1:
            result["reason"] = "CV_AXIS_ROLE_DEPENDS_ON_FACTOR"
            return result
        if roles:
            result["axis_roles"][str(selected_axis)] = roles.pop()
    spatial = sorted((item for item in admitted if item["merge_kind"] in ("M", "N")), key=lambda item:
                     (item["merge_kind"], item["axis"], item["factor"]))
    by_role = {
        role: next((item
                    for item in spatial
                    if item["merge_kind"] == role and item["factor"] == 4), None)
        for role in ("M", "N")
    }
    if all(by_role.values()):
        result.update(candidates=[by_role["M"], by_role["N"]], reason="CV_M4_N4")
    elif any(by_role.values()):
        preferred = by_role["M"] or by_role["N"]
        result.update(candidates=[item for item in spatial if item["axis"] == preferred["axis"]],
                      reason="CV_ONE_SPATIAL_AXIS_F2_F4")
    elif spatial:
        # The development shapes used four-tile or single-tile spatial grids.
        # For a grid supporting only pairs, retain the same spatial preference.
        result.update(candidates=spatial[:2], reason="CV_SPATIAL_F2_ONLY")
    else:
        batch = [item for item in admitted if item["merge_kind"] == "batch"]
        if batch:
            selected_axis = min(item["axis"] for item in batch)
            result.update(candidates=[item for item in batch if item["axis"] == selected_axis], reason="CV_BATCH_F2_F4")
        else:
            result["reason"] = "CV_NO_ADMITTED_MERGE"
    if len(result["candidates"]) > 2:
        raise AssertionError("CV shortlist exceeded two merged candidates")
    return result


def _check_merge(metadata, factor, axis):
    if factor == 1:
        return
    merge = metadata["block_batching"]
    if (merge["compile_status"] != "COMPILED_GUARDED" or merge["requested_factor"] != factor
            or merge["selected_axis"] != axis or merge.get("tail_policy") != "full_groups_only"):
        raise ValueError("compiled CV merging contract mismatch")
    if merge.get("cv_lowering") in ("rank3_masked_batch_v1", "rank3_uniform_control_batch_v1"):
        if (metadata.get("enable_hivm_batch_matmul") is not True
                or merge.get("cv_memory_lowering") != "full_reduction_guard_v3"
                or merge.get("guard", {}).get("require_same_fractal_bounds") is not True):
            raise ValueError("finalized masked-batch memory contract missing")
    transforms = metadata["program_grid_transforms"]["transforms"]
    if (len(transforms) != 1 or transforms[0]["kind"] != "ceil_div" or transforms[0]["factor"] != factor
            or transforms[0]["axis"] != axis):
        raise ValueError("compiled CV grid transform mismatch")


def _llvm_modules(item):
    functions = [line for line in item["final_ir"].splitlines() if re.match(r"\s*func\.func\s", line)]
    roles = []
    for line in functions:
        role = re.search(r"hivm\.func_core_type\s*=\s*#hivm\.func_core_type<(AIC|AIV)>", line)
        if role is None:
            raise ValueError("final CV function has no proved core role")
        roles.append(role.group(1).lower())
    final_roles = set(roles)
    if not functions or "aic" not in final_roles:
        raise ValueError("final CV module has no proved Cube function")
    modules = {role: item["llvm_" + role] for role in ("aic", "aiv") if "llvm_" + role in item}
    if modules:
        if set(modules) != final_roles:
            raise ValueError("split CV LLVM modules do not cover final core roles")
    elif "llvm" in item:
        # Early metadata may still say MIX after the backend eliminates all
        # Vector work. Require both final IR and the actual LLVM entry to prove
        # Cube-only execution; never infer it merely from a missing AIV file.
        name = item["metadata"]["name"]
        entries = [line for line in functions if "hacc.entry" in line]
        if (final_roles != {"aic"} or len(entries) != 1 or not re.search(r"@" + re.escape(name) + r"\(", entries[0])):
            raise ValueError("single CV LLVM module does not cover final core roles")
        definitions = re.findall(r"^define[^\n]*@" + re.escape(name) + r"\([^\n]*\)\s+#(\d+)\s*\{", item["llvm"], re.M)
        if len(definitions) != 1:
            raise ValueError("single CV LLVM entry identity is unavailable")
        attributes = re.findall(r"^attributes #" + definitions[0] + r"\s*=\s*\{([^\n]*)\}", item["llvm"], re.M)
        if len(attributes) != 1 or '"cube-only-kernel"' not in attributes[0]:
            raise ValueError("single LLVM entry is not explicitly Cube")
        modules["aic"] = item["llvm"]
    else:
        raise ValueError("CV LLVM modules are unavailable")
    return modules


def rank_cv(candidates, contract, table, model_identity):
    """Score F1 and at most two admitted candidates, without executing kernels.

    A compiler/resource rejection excludes a candidate. Incomplete CPU model
    coverage instead keeps F1, so absence of a score is never a good score.
    """
    values = list(candidates.values()) if isinstance(candidates, dict) else list(candidates)
    if len(values) > 3 or sum(item["factor"] == 1 for item in values) != 1:
        raise ValueError("CV shortlist requires F1 and at most two merged candidates")
    if any(item["factor"] not in (1, 2, 4) for item in values):
        raise ValueError("CV automatic selection only accepts F1/F2/F4")
    tensors = [value for value in contract["arguments"].values() if value["kind"] == "tensor"]
    fp32 = bool(tensors) and all(value.get("dtype") in ("fp32", "float32", "torch.float32") for value in tensors)
    if not fp32:
        return dict(selected_key="unmerged", reason="CV_DTYPE_NOT_COVERED", model_complete=False, candidates={},
                    device_executed=False, measured_performance_used=False)
    records, incomplete = {}, []
    spec_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    assumptions = validated_assumptions()
    context = hashlib.sha256(
        json.dumps(dict(model=model_identity, assumptions=assumptions, hardware=HARDWARE),
                   sort_keys=True).encode()).hexdigest()
    for item in values:
        factor, axis = item["factor"], item["axis"]
        key = item.get("key", "unmerged" if factor == 1 else f"a{axis}_f{factor}")
        record = dict(factor=factor, axis=axis, resource_checked=False, analysis_available=False,
                      status=item.get("status", "STATIC_ANALYSIS_UNCONFIRMED"))
        if key in records:
            raise ValueError("duplicate CV shortlist candidate")
        try:
            if not item.get("resource_checked"):
                raise ValueError(item.get("error", "candidate resource evidence unavailable"))
            metadata = item["metadata"]
            _check_merge(metadata, factor, axis)
            usage = item["ub_usage"]
            proof = dict(resource_checked=True, binary_sha256=item["binary_sha256"],
                         ub_bits=usage["address_extent_bits"], ub_capacity_bits=usage["capacity_bits"])
            from .block_auto_model import _contract_for_ttir
            bound_contract = _contract_for_ttir(contract, item["ttir"])
            trace = analyze_candidate(_llvm_modules(item), item["ttir"], item["final_ir"], metadata, bound_contract,
                                      axis, factor, table, assumptions=assumptions)
            record = features(trace, proof, factor, axis, contract["grid"], spec_hash, item["compiler_sha256"],
                              HARDWARE, "latency", context)
            record.update(status="STATIC_FEATURES_READY", merge_kind=merge_kind(metadata, factor))
        except (ValueError, KeyError, TypeError, IndexError, StopIteration, OverflowError) as error:
            record["error"] = str(error)
            if factor == 1 or record["status"] not in TERMINAL_EXCLUSIONS:
                incomplete.append(key)
        records[key] = record
    if incomplete:
        return dict(selected_key="unmerged", reason="CV_MODEL_INCOMPLETE", model_complete=False,
                    incomplete_candidates=incomplete, candidates=records, device_executed=False,
                    measured_performance_used=False)
    baseline = records.get("unmerged")
    if baseline is None:
        raise ValueError("CV baseline key must be unmerged")
    merged = {key: record for key, record in records.items() if key != "unmerged"}
    decision = choose_factor(baseline, merged, (2, 4), "latency")
    empirical = contract.get("cv_empirical_short_batch", True)
    if empirical:
        decision = empirical_select(baseline, merged, fp32)
    return dict(selected_key=decision.get("selected_candidate") or "unmerged", reason=decision["reason_code"],
                model_complete=True, decision=decision, candidates=records, device_executed=False,
                measured_performance_used=False, empirical_short_batch=bool(empirical))
