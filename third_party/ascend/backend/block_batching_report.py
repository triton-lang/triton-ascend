# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Compile-time admission, memory lowering policy and final artifact checks.

Source/layout checks precede backend code generation. Final contract checks
follow successful compilation and verification of the output binary.
UB numbers are optional diagnostics, not another resource admission gate.
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


def _has_dot(mod):
    """Classify actual IR operations, including dots inside nested regions."""
    found = False

    def visit(op):
        nonlocal found
        if op.get_name() == "tt.dot":
            found = True

    mod.walk(visit)
    return found


def _has_independent_row_layout(report):
    """Recognize top-level row accesses already covered by the source proof.

    This does not certify the backend memory form. The subsequent lowering
    check and per-launch byte-domain proof remain mandatory.
    """
    if any(report.get("operations", {}).get(op) for op in ("scf.for", "scf.if")):
        return False
    accesses = report.get("layout", {}).get("accesses", [])
    return bool(accesses) and all(
        access.get("control_flow_independent") is True and access.get("proof_scope") == "existing_guard_no_wrap"
        and access.get("action") == "Keep" and access.get("rewritten") is False
        and access.get("reason") == "ALREADY_STRUCTURED" and access.get("mask_class") == "independent_axes" and
        (shape := access.get("original_shape", [])) and all(type(extent) is int and extent > 0
                                                            for extent in shape) and sum(extent != 1
                                                                                         for extent in shape) <= 1
        for access in accesses)


def vv_layout_refusal(report):
    """Conservative structural matching for independent row pointwise bodies.

    Proven top-level rows may use the structured backend path. Incomplete
    layout evidence, shared accesses and control-flow bodies retain the
    existing refusal. No kernel names or measured timings enter this policy.
    """
    if (report.get("owner") != "block_batching" or report.get("cv_mode") != "disabled"
            or report.get("lane_width", 1) <= 1):
        return None
    # An unsupported symbolic bridge leaves coupled lane/local tail masks.
    # Structured lowering can implement their store as a full-tile read/select/
    # write, touching bytes owned by another program even when the source mask
    # proves disjoint writes. Require the layout proof before backend lowering.
    if any(
            access.get("reason") == "SYMBOLIC_LAYOUT_BRIDGE_UNSUPPORTED"
            for access in report.get("layout", {}).get("accesses", [])):
        return "VV_SYMBOLIC_LAYOUT_NOT_PROVEN"
    if _has_independent_row_layout(report):
        return None
    operations = report.get("operations", {})
    if (not operations.get("tt.load") or not operations.get("tt.store")
            or any(operations.get(name) for name in ("tt.reduce", "tt.scan", "tt.dot"))):
        return None
    accesses = report.get("layout", {}).get("accesses", [])
    if (any(operations.get(name) for name in ("scf.for", "scf.if"))
            and not all(access.get("control_flow_independent") is True for access in accesses)):
        return None

    def row_rank(access):
        shape = access.get("original_shape", [])
        if any(type(extent) is not int or extent <= 0 for extent in shape):
            return None
        # Retain the existing distinction between scalar accesses and a
        # one-element row while ignoring singleton coordinate dimensions.
        return max(1, sum(extent != 1 for extent in shape)) if shape else 0

    independent = [access for access in accesses if access.get("mask_class") == "independent_axes"]
    if independent and all(
            access.get("action") == "Keep" and access.get("rewritten") is False and (
                (access.get("mask_class") == "independent_axes" and row_rank(access) == 1) or
                (access.get("mask_class") == "shared" and row_rank(access) in (0, 1))) for access in accesses):
        return "VV_POINTWISE_MEMORY_VIEW_REQUIRED"
    return None


def vv_computation_refusal(report):
    """Require the native original-domain contract, independently of memory.

    The native verifier checks combiners, effects and loop-carried values.
    This boundary checks that their domain contract reached the compiler;
    neither operation counts nor a successful memory conversion replace it.
    """
    if (report.get("owner") != "block_batching" or report.get("cv_mode") != "disabled"
            or report.get("lane_width", 1) <= 1):
        return None
    computation = report.get("computation", {})
    schedule = "sequential_iterations" if report.get("vv_lowering") == "sequential_iterations_v1" else "program_lanes"
    if (computation.get("version") != 2 or computation.get("scope") != "normalized_original_program"
            or computation.get("stage") != "semantic_ttir_before_memory_views"
            or computation.get("schedule") != schedule or computation.get("lane_policy") != "separate_original_domains"
            or not isinstance(computation.get("aggregations"), list)):
        return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
    expected = {"reduce": "original_program_reduction", "scan": "independent_row_scan"}
    counts = {kind: 0 for kind in expected}
    target_ids = set()
    for source_id, aggregation in enumerate(computation["aggregations"]):
        if (not isinstance(aggregation, dict) or type(aggregation.get("id")) is not int
                or aggregation["id"] != source_id):
            return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
        kind = aggregation.get("kind")
        if kind not in expected or aggregation.get("domain") != expected[kind]:
            return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
        axis = aggregation.get("original_axis")
        shape = aggregation.get("original_shape")
        targets = aggregation.get("mapped_operations")
        if (type(axis) is not int or not isinstance(shape, list) or not 0 <= axis < len(shape)
                or any(type(d) is not int or d <= 0 for d in shape) or not isinstance(targets, list) or not targets
                or (schedule == "program_lanes" and len(targets) != 1)):
            return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
        for target in targets:
            if not isinstance(target, dict):
                return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
            prefix = target.get("batch_prefix")
            target_id = target.get("id")
            if (type(prefix) is not bool or (schedule == "sequential_iterations" and prefix)
                    or type(target_id) is not int or target_id < 0 or target_id in target_ids
                    or type(target.get("axis")) is not int or type(aggregation.get("lifted_axis")) is not int
                    or target.get("axis") != axis + int(prefix) or aggregation.get("lifted_axis") != target.get("axis")
                    or target.get("shape") != ([report["lane_width"]] + shape if prefix else shape)):
                return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
            target_ids.add(target_id)
        counts[kind] += 1
    if target_ids != set(range(len(target_ids))):
        return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
    if any(count != report.get("operations", {}).get("tt." + kind, 0) for kind, count in counts.items()):
        return "VV_COMPUTATION_DOMAIN_NOT_PROVEN"
    return None


def vv_memory_lowering_refusal(report):
    """Keep VV candidates on supported memory paths after layout lowering.

    Independent row reductions and scans remain supported. Their presence
    does not prove that a batched access became a structured memory view:
    reject residual indirect GM helpers regardless of the compute body.
    Internal tensor gathers and SIMT arithmetic are not memory refusals.
    """
    if (report.get("owner") != "block_batching" or report.get("cv_mode") != "disabled"
            or report.get("lane_width", 1) <= 1):
        return None
    observation = report.get("layout", {}).get("lowering_observation")
    if (not isinstance(observation, dict) or observation.get("stage") != "ttadapter"
            or observation.get("scope") != "module" or any(
                type(observation.get(key)) is not int or observation[key] < 0
                for key in ("indirect_load_ops", "indirect_store_ops"))):
        return "VV_MEMORY_LOWERING_NOT_PROVEN"
    if observation.get("indirect_load_ops", 0) or observation.get("indirect_store_ops", 0):
        return "VV_INDIRECT_MEMORY_NOT_SUPPORTED"
    return None


def memory_compile_options(metadata, compiler_path, supports_inlining):
    """Inline contiguous or proved independent-row DMA in VV candidates.

    This is a lowering policy, not a factor selector or a profitability test.
    Unsupported compiler versions retain their normal candidate compilation.
    """
    report = metadata.get("block_batching", {})
    lowering = report.get("memory_lowering", {})
    observation = report.get("layout", {}).get("lowering_observation", {})
    row_copy = (_has_independent_row_layout(report) and observation.get("stage") == "ttadapter"
                and observation.get("scope") == "module" and observation.get("indirect_load_ops") == 0
                and observation.get("indirect_store_ops") == 0 and type(observation.get("memref_copy_ops")) is int
                and observation["memref_copy_ops"] > 0)
    if (metadata.get("block_batching_factor", 1) <= 1 or report.get("owner") != "block_batching"
            or report.get("cv_mode") != "disabled" or not (lowering.get("dynamic_contiguous_copy") is True or row_copy)
            or not metadata["target"].arch.startswith("Ascend950") or metadata.get("mix_mode") != "aiv"
            or metadata.get("parallel_mode") != "simd"):
        return []
    lowering["structured_row_copy"] = row_copy
    lowering["libcall_inlining"] = bool(supports_inlining(compiler_path))
    return ["--enable-lib-call-no-inline=false"] if lowering["libcall_inlining"] else []


def validate_vv_loop_contract(metadata):
    report = metadata.get('block_batching', {})
    guard = report.get('guard', {})
    grouped = report.get('vv_lowering') == 'sequential_iterations_v1'
    if not grouped and 'loop_schedule' not in guard:
        return False
    from .block_batching_guard import GuardFailure, iteration_schedule
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
    from .block_batching_guard import validate_plan_binding
    guard = native_report.get("guard", {})
    if (not validate_plan_binding(guard)
            or getattr(candidate.metadata, "block_batching_plan_sha256", None) != guard.get("plan_fingerprint")):
        raise BatchCompileError("CANDIDATE_PLAN_CHANGED", kind='InvalidArtifact', stage='validate')
    validate_vv_loop_contract(candidate.metadata._asdict())
    return dict(native_report)
