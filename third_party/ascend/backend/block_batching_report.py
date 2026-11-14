# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Finalize a candidate using the ordinary TA/backend compilation contract.

Call only after successful compilation and verification of the output binary.
UB numbers are optional diagnostics, not another resource admission gate.
No custom AscendNPU-IR report or transitive SIMD proof is required.
"""
from .block_batching import _is_cv_report
from .block_batching_worker import BatchCompileError, BatchDecline

VALIDATION_POLICY = "ta_only_v1"
CV_VALIDATION_POLICY = "ta_cv_integer_rank_two_v23"

CV_MASKED_BATCH_LOWERINGS = ("rank3_masked_batch_v1", "rank3_uniform_control_batch_v1")
CV_SPATIAL_LOWERINGS = (
    "spatial_m_v1",
    "spatial_n_v1",
    "spatial_m_uniform_control_v1",
    "spatial_n_uniform_control_v1",
)

# Both launch-policy selection and final validation must use the same set.
# These lowerings retain the hidden ORIGINAL grid while AutoBlockify iterates
# over the transformed grid. Eligibility still requires MIX, no blacklist,
# no persistent cap, and the owned v3 grid/effect contract below.
CV_AUTO_BLOCKIFY_LOWERINGS = (
    *CV_MASKED_BATCH_LOWERINGS,
    *CV_SPATIAL_LOWERINGS,
    "static_uniform_control_lanes_v1",
    "static_integer_lanes_v1",
)


def validate_vv_loop_contract(metadata):
    report = metadata.get('block_batching', {})
    guard = report.get('guard', {})
    grouped = report.get('vv_lowering') == 'sequential_iterations_v1'
    if not grouped and 'loop_schedule' not in guard:
        return False
    from .block_batching_guard import GuardFailure
    from .block_batching_guard_v3 import iteration_schedule
    try:
        loop = iteration_schedule(guard, report['lane_width'])
    except (GuardFailure, KeyError, TypeError) as error:
        raise BatchCompileError('VV_LOOP_SCHEDULE_MISMATCH', kind='InvalidArtifact', stage='validate') from error
    if (not grouped or loop is None or report.get('cv_mode') != 'disabled'
            or metadata.get('program_grid_transforms') is not None
            or metadata.get('program_grid_mapping_applied') is not False
            or metadata.get('row_coalescing_applied') is not False or metadata.get('ptsm_cap_authorized') is not False
            or metadata.get('coalesce_factor', 1) != 1):
        raise BatchCompileError('VV_LOOP_GRID_CONTRACT_MISMATCH', kind='InvalidArtifact', stage='validate')
    return True


def _validate_cv_contract(metadata):
    from .program_grid import normalize_program_grid_transforms
    report = metadata["block_batching"]
    full_tile_lowerings = ("rank3_full_tile_batch_v1", "rank3_full_tile_batch_v2")
    lowerings = {
        "independent_batch_v1": ("rank3_batch_v1", "static_masked_lanes_v1", "static_uniform_control_lanes_v1",
                                 "static_integer_lanes_v1", *CV_MASKED_BATCH_LOWERINGS, *full_tile_lowerings),
        "spatial_concat_v1":
        CV_SPATIAL_LOWERINGS,
    }
    if report.get("cv_lowering") not in lowerings.get(report.get("cv_mode"), ()):
        raise BatchCompileError("CV_LOWERING_CONTRACT_MISSING", kind='InvalidArtifact', stage='validate')
    if report.get("cv_lowering") == "static_integer_lanes_v1":
        expected = dict(version=1, input_bits=8, accumulator_bits=32, lane_semantics="original_program_clone",
                        dot_rank=2)
        proof = report.get("guard", {}).get("cv_integer_contract")
        if type(proof) is not dict or proof != expected or any(
                type(proof.get(k)) is not int for k in ('version', 'input_bits', 'accumulator_bits', 'dot_rank')):
            raise BatchCompileError("CV_INTEGER_LANE_CONTRACT_MISSING", kind='InvalidArtifact', stage='validate')
    uniform_contracts = {
        "static_uniform_control_lanes_v1": ("original_program_clone", 2),
        "spatial_m_uniform_control_v1": ("spatial_concatenation", 2),
        "spatial_n_uniform_control_v1": ("spatial_concatenation", 2),
        "rank3_uniform_control_batch_v1": ("independent_batch", 3),
    }
    if report.get("cv_lowering") in uniform_contracts:
        semantics, rank = uniform_contracts[report["cv_lowering"]]
        proof = report.get("guard", {}).get("cv_control_flow_contract", {})
        if (type(proof) is not dict or proof.get("version") != 1 or proof.get("lane_semantics") != semantics
                or proof.get("uniform_bounds") is not True or proof.get("dot_rank") != rank
                or type(proof.get("for_count")) is not int or proof["for_count"] < 0
                or type(proof.get("if_count")) is not int or proof["if_count"] < 0
                or proof["for_count"] + proof["if_count"] == 0):
            raise BatchCompileError("CV_UNIFORM_CONTROL_CONTRACT_MISSING", kind='InvalidArtifact', stage='validate')
    if report.get("cv_lowering") in full_tile_lowerings:
        if (metadata.get("enable_hivm_batch_matmul") is not True
                or report.get("guard", {}).get("require_full_masks") is not True):
            raise BatchCompileError("CV_FULL_TILE_BATCH_CONTRACT_MISSING", kind='InvalidArtifact', stage='validate')
    if report.get("cv_lowering") in CV_MASKED_BATCH_LOWERINGS:
        if (metadata.get("enable_hivm_batch_matmul") is not True or "require_full_masks" in report.get("guard", {})):
            raise BatchCompileError("CV_MASKED_BATCH_CONTRACT_MISSING", kind='InvalidArtifact', stage='validate')
        if (report.get("cv_memory_lowering") != "full_reduction_guard_v3"
                or report.get("guard", {}).get("require_same_fractal_bounds") is not True):
            raise BatchCompileError("CV_MASKED_BATCH_MEMORY_CONTRACT_MISSING", kind='InvalidArtifact', stage='validate')
        contract = report["guard"].get("cv_mask_contract")
        if (type(contract) is not dict or contract.get("version") != 1 or type(contract.get("inputs")) is not list
                or len(contract["inputs"]) != 2 or type(contract.get("outputs")) is not list
                or not contract["outputs"]):
            raise BatchCompileError("CV_MASKED_BATCH_DOT_CONTRACT_MISSING", kind='InvalidArtifact', stage='validate')
    if report.get("tail_policy") != "full_groups_only":
        raise BatchCompileError("CV_TAIL_CONTRACT_MISSING", kind='InvalidArtifact', stage='validate')
    contract = normalize_program_grid_transforms(metadata.get("program_grid_transforms"))
    expected = dict(order=0, kind="ceil_div", axis=report["selected_axis"], factor=report["lane_width"],
                    persistent_coverage=False, grid_stride_abi_verified=False)
    if (contract is None or contract["version"] != 3 or contract["hidden_extent_axes"] != [0, 1, 2]
            or contract["transforms"] != [expected]):
        raise BatchCompileError("CV_GRID_CONTRACT_MISMATCH", kind='InvalidArtifact', stage='validate')
    if (metadata.get("program_grid_mapping_applied") is not True
            or any(metadata.get(key) is not False for key in ("row_coalescing_applied", "ptsm_cap_authorized"))):
        raise BatchCompileError("CV_MAPPING_CONFLICT", kind='InvalidArtifact', stage='validate')
    auto_blockify = metadata.get("auto_blockify_enabled")
    if auto_blockify is not False and auto_blockify is not True:
        raise BatchCompileError("CV_MAPPING_CONFLICT", kind='InvalidArtifact', stage='validate')
    if auto_blockify and (report.get("cv_lowering") not in CV_AUTO_BLOCKIFY_LOWERINGS or metadata.get("mix_mode")
                          != "mix" or metadata.get("has_auto_blockify_blacklist_op") is not False):
        return BatchDecline("CV_AUTO_BLOCKIFY_NOT_PROVEN", 'npubin')
    # v1 validates independent programs, not cross-program synchronization.
    # Standard MIX/AIC workspace and per-core CV synchronization are retained.
    locks = metadata.get("sync_block_lock_layout", 0)
    if type(locks) is not int or locks != 0:
        return BatchDecline("CV_GLOBAL_SYNC_NOT_SUPPORTED", 'npubin')
    task = metadata.get("bs_task_type", 0)
    if type(task) is not int or (task != 0 and task // 10 not in (1, 2, 3, 4)):
        raise BatchCompileError("CV_INVALID_TASK_TYPE", kind='InvalidArtifact', stage='validate')


def validate_compilation_result(metadata):
    # Reuse the normal launcher's callback/default-zero workspace contract.
    # Explicit invalid values must not silently become the default.
    workspace = metadata.get("workspace_size", 0)
    if type(workspace) is not int or workspace < 0:
        raise BatchCompileError("INVALID_WORKSPACE_SIZE", kind='InvalidArtifact', stage='validate')
    cv = metadata["block_batching"].get("cv_mode") in ("independent_batch_v1", "spatial_concat_v1")
    if metadata.get("is_pure_simt") or metadata.get("mix_mode") not in (("mix", "aic") if cv else ("aiv", )):
        return BatchDecline("UNSUPPORTED_FINAL_EXECUTION_MODE", 'npubin')
    if cv:
        decline = _validate_cv_contract(metadata)
        if decline is not None:
            return decline
    else:
        validate_vv_loop_contract(metadata)
    metadata["block_batching"].update(
        compile_status="COMPILED_GUARDED",
        owner="block_batching",
        validation_policy=CV_VALIDATION_POLICY if cv else VALIDATION_POLICY,
        resource_validation="backend_compile_success",
        execution_mode_validation="standard_backend_metadata",
        internal_simt="not_inspected",
        workspace_bytes=workspace,
        workspace_source="callback" if "workspace_size" in metadata else "standard_backend_default",
    )


def validated_candidate_report(candidate, request, implementation_version):
    """Check the complete compiled candidate before publishing its report.

    Preserve the existing check order and shallow report-copy semantics.
    Runtime proof, lazy handle initialization and variant selection are separate.
    """
    import hashlib
    native_report = getattr(candidate.metadata, "block_batching", {})
    cv = _is_cv_report(native_report)
    if (native_report.get("cv_mode") not in ("disabled", "independent_batch_v1", "spatial_concat_v1")
            or native_report.get("compile_status") != "COMPILED_GUARDED"
            or native_report.get("validation_policy") != (CV_VALIDATION_POLICY if cv else VALIDATION_POLICY)):
        raise BatchCompileError("CANDIDATE_VALIDATION_MISSING", kind='InvalidArtifact', stage='validate')
    if (native_report.get("implementation_version") != implementation_version
            or native_report.get("lane_width") != request.factor):
        raise BatchCompileError("CANDIDATE_LAYOUT_MISMATCH", kind='InvalidArtifact', stage='validate')
    if cv and native_report.get("tail_policy") != "full_groups_only":
        raise BatchCompileError("CV_CONTRACT_MISMATCH", kind='InvalidArtifact', stage='validate')
    expected = getattr(candidate.metadata, "block_batching_binary_sha256", None)
    if expected != hashlib.sha256(candidate.kernel).hexdigest():
        raise BatchCompileError("CANDIDATE_BINARY_CHANGED", kind='InvalidArtifact', stage='validate')
    from .block_batching_guard_v3 import validate_plan_binding
    guard = native_report.get("guard", {})
    if (not validate_plan_binding(guard)
            or getattr(candidate.metadata, "block_batching_plan_sha256", None) != guard.get("plan_fingerprint")):
        raise BatchCompileError("CANDIDATE_PLAN_CHANGED", kind='InvalidArtifact', stage='validate')
    validate_vv_loop_contract(candidate.metadata._asdict())
    return dict(native_report)
