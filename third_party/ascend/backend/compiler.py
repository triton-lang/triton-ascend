# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import ctypes
import copy
import functools
import hashlib
import glob
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple, Union
from .debug_line_rewriter import rewrite_debug_line

from triton._C.libtriton import ir, passes, ascend, buffer_ir
from triton._C.libtriton.ascend import ir as ascend_ir
try:
    from triton._C.libtriton import distributed
except ImportError:
    distributed = None
from triton.backends.ascend.utils import (
    _multibuffer_mode_to_tuple,
    _check_bishengir_api_change,
    _check_bishengir_able_save_ir,
    _check_bishengir_is_regbased,
    _check_bishengir_supports_optimized_metaop,
    _enable_print_ub_bits,
    _enable_dump_memory_info,
    _enable_msdebug,
    _get_kernel_target,
    _get_npucompiler_path,
    _get_triton_adapter_opt_path,
    _get_triton_mlir_opt_path,
    _get_triton_opt_path,
    _get_bishengir_opt_path,
    _is_ascend_sanitizer_enabled,
    _is_debug_line_info_disabled,
    _is_auto_map_parallel_blocks_enabled,
    _get_auto_blockify_blacklist_reasons,
    _get_modeled_superblock_factors,
    _get_current_scope_superblock_factors,
    _warn_auto_blockify_disabled,
    _remove_deprecated_npu_options,
    _warn_deprecated_npu_option,
    _warn_deprecated_ascend_env_vars,
    downgrade_llir,
    force_disable_ffts,
    graph_ub_budget_bytes_for_arch,
    ub_size_in_kbytes_for_arch,
    get_cann_version_file_hash,
    is_cann_version_at_least,
)
from triton.backends.ascend.driver import (NPUUtils)
from triton.backends.ascend.program_grid import (
    DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK,
    PROGRAM_GRID_TRANSFORMS_ATTR,
    ProgramGridContractError,
    get_persistent_transform,
    normalize_graph_optimization_rule_mask,
    normalize_program_grid_transforms,
)
from triton.backends.compiler import (
    BaseBackend,
    GPUTarget,
)
from triton.runtime.cache import get_dump_manager


# TODO: materialize the concrete min shape
def min_dot_size(target: GPUTarget):
    return lambda lhsType, rhsType: (1, 1, 1)


# Get result code saved in module {attr_name = rc}
def _get_then_remove_rc(mod, attr_name: str) -> int:
    get_int_attr = getattr(ascend.ir, "get_int_attr", None)
    remove_attr = getattr(ascend.ir, "remove_attr", None)

    if get_int_attr is None:
        return -1
    try:
        attr_value = get_int_attr(mod, attr_name)
    except TypeError:
        return -1

    if remove_attr:
        remove_attr(mod, attr_name)

    if not isinstance(attr_value, int):
        return -1

    return attr_value


def _get_then_remove_program_grid_transforms(mod):
    get_transforms = getattr(ascend.ir, "get_program_grid_transforms", None)
    remove_attr = getattr(ascend.ir, "remove_attr", None)
    if get_transforms is None:
        if PROGRAM_GRID_TRANSFORMS_ATTR in str(mod):
            raise RuntimeError("hacc.program_grid_transforms requires the matching Ascend C++ binding")
        return None
    try:
        raw = get_transforms(mod)
    except TypeError:
        if PROGRAM_GRID_TRANSFORMS_ATTR in str(mod):
            raise RuntimeError("hacc.program_grid_transforms requires an MLIR module accepted by "
                               "the Ascend C++ binding")
        return None
    if raw is not None and remove_attr:
        remove_attr(mod, PROGRAM_GRID_TRANSFORMS_ATTR)
    return raw


def _export_coalesce_metadata(mod, metadata, *, require_row_contract=False):
    # Tile/strided coalescing (TritonToLinalg) records the chosen coalesce factor
    # H and the grid axis it applies to as module attrs hacc.coalesce_factor /
    # hacc.coalesce_axis. In the full-TA design the *launcher* (driver.py) owns
    # the grid division: it divides grid[axis] by H before launch, and bishengir
    # no longer interprets the attrs. Read them into metadata here and strip them
    # from the module so the hacc.* attrs never reach hivmc (which rejects
    # unknown module attrs). Absent attrs -> factor 1 (no-op) / axis -1.
    # RowCoalescing also records whether the launcher should use ceil-div when
    # shrinking the grid, because its runtime valid-count guard handles tails.
    # A Row result is only valid as the complete triple.  The regular T2L
    # Axis/Chunk ABI predates ceil-div and deliberately remains compatible with
    # a missing ceil-div attr.
    factor = _get_then_remove_rc(mod, "hacc.coalesce_factor")
    axis = _get_then_remove_rc(mod, "hacc.coalesce_axis")
    ceil_div = _get_then_remove_rc(mod, "hacc.coalesce_grid_ceil_div")
    has_any_contract_attr = any(value != -1 for value in (factor, axis, ceil_div))
    valid_factor = isinstance(factor, int) and factor > 1
    valid_axis = isinstance(axis, int) and axis in (0, 1, 2)
    valid_ceil_div = isinstance(ceil_div, int) and ceil_div > 0
    if has_any_contract_attr and (not valid_factor or not valid_axis):
        raise RuntimeError("invalid hacc.coalesce launch contract")
    if require_row_contract and has_any_contract_attr and not valid_ceil_div:
        raise RuntimeError("RowCoalescing requires hacc.coalesce_grid_ceil_div")

    metadata["coalesce_factor"] = factor if valid_factor else 1
    metadata["coalesce_axis"] = axis if valid_axis else -1
    metadata["coalesce_grid_ceil_div"] = valid_ceil_div
    metadata["row_coalescing_applied"] = metadata["coalesce_factor"] > 1


def _export_program_grid_metadata(mod, metadata, *, require_row_contract=False):
    raw_transforms = _get_then_remove_program_grid_transforms(mod)
    if raw_transforms is not None:
        factor = _get_then_remove_rc(mod, "hacc.coalesce_factor")
        axis = _get_then_remove_rc(mod, "hacc.coalesce_axis")
        ceil_div = _get_then_remove_rc(mod, "hacc.coalesce_grid_ceil_div")
        has_legacy_attrs = any(value != -1 for value in (factor, axis, ceil_div))
        if has_legacy_attrs:
            raise RuntimeError("hacc.program_grid_transforms conflicts with legacy hacc.coalesce_* metadata")
        try:
            transforms = normalize_program_grid_transforms(raw_transforms)
        except ProgramGridContractError as error:
            raise RuntimeError(f"invalid hacc.program_grid_transforms: {error}") from error
        metadata["program_grid_transforms"] = transforms
        metadata["program_grid_mapping_applied"] = True
        metadata["coalesce_factor"] = 1
        metadata["coalesce_axis"] = -1
        metadata["coalesce_grid_ceil_div"] = False
        metadata["row_coalescing_applied"] = False
        return

    metadata["program_grid_transforms"] = None
    metadata["program_grid_mapping_applied"] = False
    _export_coalesce_metadata(
        mod,
        metadata,
        require_row_contract=require_row_contract,
    )


def _finalize_program_launch_policy(metadata, opt):
    required_fields = (
        "program_grid_transforms",
        "program_grid_mapping_applied",
        "row_coalescing_applied",
        "has_auto_blockify_blacklist_op",
        "mix_mode",
    )
    missing = [name for name in required_fields if name not in metadata]
    if missing:
        raise RuntimeError("cannot finalize program launch policy; missing metadata: " + ", ".join(missing))

    raw_transforms = metadata["program_grid_transforms"]
    mapping_applied = metadata["program_grid_mapping_applied"]
    row_coalescing_applied = metadata["row_coalescing_applied"]
    has_auto_blockify_blacklist_op = metadata["has_auto_blockify_blacklist_op"]
    if not isinstance(mapping_applied, bool):
        raise RuntimeError("program_grid_mapping_applied must be a boolean")
    if not isinstance(row_coalescing_applied, bool):
        raise RuntimeError("row_coalescing_applied must be a boolean")
    if not isinstance(has_auto_blockify_blacklist_op, bool):
        raise RuntimeError("has_auto_blockify_blacklist_op must be a boolean")

    transforms = None
    if raw_transforms is not None:
        try:
            transforms = normalize_program_grid_transforms(raw_transforms)
        except ProgramGridContractError as error:
            raise RuntimeError(f"invalid exported program_grid_transforms: {error}") from error
    if mapping_applied != (transforms is not None):
        raise RuntimeError("program_grid_mapping_applied disagrees with program_grid_transforms")
    if mapping_applied and row_coalescing_applied:
        raise RuntimeError("program-grid mapping conflicts with legacy RowCoalescing")

    persistent_transform = get_persistent_transform(transforms) if transforms is not None else None
    ptsm_cap_authorized = False
    if persistent_transform is not None:
        if metadata["mix_mode"] != "aiv":
            raise RuntimeError("persistent program-grid transform requires final mix_mode=aiv")
        if not (persistent_transform["persistent_coverage"] and persistent_transform["grid_stride_abi_verified"]):
            raise RuntimeError("persistent program-grid transform lacks coverage/ABI verification")
        ptsm_cap_authorized = True

    if opt.is_pure_simt:
        auto_blockify_enabled = _is_auto_map_parallel_blocks_enabled() and not row_coalescing_applied
    else:
        auto_blockify_enabled = (_is_auto_map_parallel_blocks_enabled() and not has_auto_blockify_blacklist_op
                                 and not ptsm_cap_authorized)

    # When the native Route Model ran it owns the V1 admission decision; honor
    # its published enabled/TA-materialized state instead of the generic policy.
    costmodel_v1_enabled = metadata.get("auto_blockify_v1_enabled")
    if costmodel_v1_enabled is not None:
        auto_blockify_enabled = (bool(costmodel_v1_enabled)
                                 and not metadata.get("ta_auto_blockify_v1_materialized", False)
                                 and not has_auto_blockify_blacklist_op and not ptsm_cap_authorized)

    if auto_blockify_enabled and ptsm_cap_authorized:
        raise RuntimeError("AutoBlockify and persistent-grid cap cannot both be enabled")

    metadata["auto_blockify_enabled"] = auto_blockify_enabled
    metadata["ptsm_cap_authorized"] = ptsm_cap_authorized


def _update_dynamic_cv_costmodel_report(metadata, opt, requested, rc, rc_readable):
    report = metadata.get("auto_simt_scope_report")
    if not report:
        return
    try:
        data = json.loads(report)
    except (TypeError, ValueError, json.JSONDecodeError):
        return

    effective = bool(metadata.get("enable_dynamic_cv_pipeline", False))
    if not requested:
        state = "not_requested"
    elif rc > 0:
        state = "fell_back"
    elif rc_readable and getattr(opt, "compile_on_910_95", False):
        # The pass removes the return-code attribute on entry and only restores
        # a positive value when it falls back. An absent readable attribute
        # after the scheduled 910/950 pass therefore means it completed.
        state = "pass_completed"
    else:
        state = "unverified"

    data["dynamic_cv_pipeline_backend"] = {
        "state": state,
        "requested": requested,
        "effective_option": effective,
        "fallback_rc": rc if rc > 0 else None,
        "physical_hir_certified": False,
    }
    confirmed = state == "pass_completed" and effective
    candidate = data.get("stage_dependency_elapsed_prediction")
    if isinstance(candidate, dict) and candidate.get("dynamic_cv_requested") is True:
        candidate["backend_state"] = state
        candidate["backend_effective_dynamic_cv_confirmed"] = confirmed
        promotion_allowed = (confirmed and candidate.get("offline_schedule_calibration_certified") is True
                             and isinstance(candidate.get("schedule_contract"), str)
                             and bool(candidate["schedule_contract"])
                             and isinstance(candidate.get("full_kernel_validation_sha256"), str)
                             and len(candidate["full_kernel_validation_sha256"]) == 64)
        candidate["promotion_allowed"] = promotion_allowed
        if promotion_allowed:
            candidate["status"] = "qualified_on_full_tile_candidate"
        elif state == "fell_back":
            candidate["status"] = "requested_on_rejected_after_backend_fallback"

    updated = json.dumps(data, sort_keys=True, separators=(",", ":"))
    metadata["auto_simt_scope_report"] = updated
    if dump_path := metadata.get("auto_simt_scope_dump"):
        Path(dump_path).write_text(updated, encoding="utf-8", newline="\n")


def _adjust_metadata_by_module_result(mod, metadata, opt, **kwargs):
    requested = bool(metadata.get("enable_dynamic_cv_pipeline", False))
    rc_readable = getattr(ascend.ir, "get_int_attr", None) is not None
    rc = _get_then_remove_rc(mod, "triton_ascend.dynamic_cv_pipeline.rc")
    if rc == 3:
        metadata["disable_vf_operand_substitution"] = True
        return
    if rc != -1 and rc > 0:
        # When the option dynamic_cv_pipeline is set to False,
        # these options should also reverted.
        metadata["enable_dynamic_cv_pipeline"] = False
        metadata["enable_mixed_cv"] = kwargs["enable_mixed_cv"]
        metadata["disable_auto_inject_block_sync"] = kwargs["disable_auto_inject_block_sync"]
        metadata["set_workspace_multibuffer"] = kwargs["set_workspace_multibuffer"]
        if opt.debug:
            print(f"SSBUFFER return code={rc}, will fallback to enable_dynamic_cv_pipeline=False")
    _update_dynamic_cv_costmodel_report(metadata, opt, requested, rc, rc_readable)


def _get_dump_paths(hash_key: str, src_path: str, dst_path: str) -> Tuple[str, str]:
    dump_manager = get_dump_manager(hash_key)
    return (dump_manager._make_path(os.path.basename(src_path)), dump_manager._make_path(os.path.basename(dst_path)))


def _with_debug_line(npubin_stage, options):
    """Wrap an npubin-producing stage so the emitted kernel binary gets
    .debug_line cleanup for msdebug stepping. No-op unless
    LLVM_EXTRACT_DI_LOCAL_VARIABLES is set; never raises (failure returns the
    artifact unchanged), so it cannot break a build."""

    def stage(src, metadata):
        artifact = npubin_stage(src, metadata)
        return rewrite_debug_line(artifact, metadata=metadata, options=options)

    return stage


def _graph_optimize_kwargs(opt):
    kwargs = {
        "ub_capacity_bytes": graph_ub_budget_bytes_for_arch(opt.target_arch),
        "compile_mode": opt.compile_mode,
    }
    rule_mask = getattr(opt, "rule_mask", DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK)
    if rule_mask != DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK:
        kwargs["rule_mask"] = rule_mask
        kwargs["ub_safety_percent"] = 80
        kwargs["reserved_ub_bytes"] = 0
        kwargs["mapping_ub_capacity_bytes"] = (ub_size_in_kbytes_for_arch(opt.target_arch) * 1024)
    return kwargs


def _serialize_module(mod, opt):
    if getattr(opt, "debug", False) or not _is_debug_line_info_disabled() or _enable_msdebug():
        return str(mod)
    return mod.str_nodebug()


def make_ttir(mod, metadata, opt):
    if "hash" not in metadata:
        metadata["hash"] = hashlib.sha256(f"{mod}-{metadata}".encode()).hexdigest()
    # the same optimize pass for triton-ir as all other backends
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    # passes.ttir.add_rewrite_tensor_pointer(pm)
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    if opt.enable_graph_optimize:
        ascend.passes.ttir.add_graph_optimize(pm, **_graph_optimize_kwargs(opt))
    pm.run(mod, 'make_ttir')
    if opt.debug:
        dump_manager = get_dump_manager(metadata["hash"])
        print(f"Dumping intermediate results to {dump_manager.cache_dir}")
        dump_manager.put(str(mod), "kernel.ttir.mlir", binary=False)

    return mod


# --- Native SIMD/SIMT Route Model (compile_mode="simd_simt") ---------------
# Python only schedules the C++ selection/materialization passes here; the
# route decision itself is owned by the native cost model.  Every helper is a
# no-op unless the user asks for compile_mode="simd_simt" and opts into
# auto_simt_scope_mode, so the default pipeline is unchanged.


def _costmodel_profiles_dir() -> Path:
    """Locate runtime assets without requiring backend-specific setup.py code."""
    compiler = Path(__file__)

    # Ascend CMake copies canonical profiles beside the native extension.
    native_packaged = (compiler.resolve().parents[2] / "_C" / "ascend" / "costmodel_profiles")
    if native_packaged.is_dir():
        return native_packaged

    # Compatibility with packages produced before the Cost Model ownership
    # cleanup.  Check this only after the canonical native package location so
    # incremental build directories cannot shadow freshly packaged profiles.
    legacy_packaged = compiler.with_name("costmodel_profiles")
    if legacy_packaged.is_dir():
        return legacy_packaged

    # Editable/source-tree builds resolve compiler.py back into
    # third_party/ascend/backend.
    source = Path(__file__).resolve().parent.parent / "costmodel" / "profiles"
    return source if source.is_dir() else native_packaged


def _normalize_auto_simt_scope_mode(mode: Optional[str]) -> str:
    if mode is None:
        return "off"
    mode = str(mode).strip().lower()
    if mode in ("1", "true", "on", "yes", "auto"):
        return "auto"
    if mode in ("report", "dry_run", "dry-run", "dump"):
        return "report"
    return "off"


VALID_PARTITION_AND_BIND_SUB_BLOCK = (
    "off",
    "default-pin",
    "load-balanced",
)


def _validate_partition_and_bind_sub_block(partition_mode: str) -> str:
    if partition_mode not in VALID_PARTITION_AND_BIND_SUB_BLOCK:
        valid_values = ", ".join(VALID_PARTITION_AND_BIND_SUB_BLOCK)
        raise ValueError("Invalid enable_partition_and_bind_sub_block="
                         f"{partition_mode!r}. Expected one of: {valid_values}.")
    return partition_mode


def _auto_simt_asset_hash(path: str, default_name: str) -> str:
    asset = Path(path) if path else _costmodel_profiles_dir() / default_name
    try:
        selection_bytes = asset.read_bytes()
    except OSError:
        return f"missing:{asset}"

    digest = hashlib.sha256()
    digest.update(b"selection-profile\0")
    digest.update(selection_bytes)
    try:
        profile = json.loads(selection_bytes)
    except (TypeError, ValueError, UnicodeError):
        profile = {}
    for key, label in [("microbenchmark_profile", b"shared-microbenchmark"),
                       ("indirect_memory_profile", b"indirect-memory"), ("reduction_profile", b"reduction-profile"),
                       ("prefixscan_profile", b"prefixscan-profile")]:
        reference = profile.get(key) if isinstance(profile, dict) else None
        if reference is None:
            continue
        digest.update(b"\0" + label + b"\0")
        if not isinstance(reference, str) or not reference:
            digest.update(b"invalid-reference\0" + repr(reference).encode())
            continue
        dependency = Path(reference)
        if not dependency.is_absolute():
            dependency = asset.parent / dependency
        try:
            digest.update(dependency.read_bytes())
        except OSError:
            digest.update(f"missing:{dependency}".encode())
    return digest.hexdigest()


_CCEC_RESOURCE_USAGE_RE = re.compile(r"\[BISHENG\]\s+(?:(?P<execution_model>SIMT VF)\s+)?Function properties for\s+"
                                     r"(?P<function>[^:\r\n]+):\s+Stack size:\s*(?P<stack_bytes>\d+)\s+bytes,\s+"
                                     r"Used register number:\s*(?P<used_registers>\d+)")
_CCEC_REGALLOC_SIZE_RE = re.compile(
    r"Greedy Register Allocator:\s+Function:\s+(?P<function>[^:\r\n]+):\s+"
    r"MI Instruction count changed from\s+\d+\s+to\s+\d+;\s+Delta:\s*(?P<delta>[+-]?\d+)")
_OBJDUMP_SYMBOL_RE = re.compile(r"^([0-9a-fA-F]+) <([^>]+)>:", re.MULTILINE)
_OBJDUMP_INSTRUCTION_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+(?:[0-9a-fA-F]+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*)$")
_OBJDUMP_BRANCH_TARGET_RE = re.compile(r"<([^>+]+)\+0x([0-9a-fA-F]+)>")


def _bisheng_options_with_resource_usage(bisheng_options):
    """Request allocator facts without changing CCEC optimization policy."""
    option = "--cce-res-usage"
    if not bisheng_options:
        return option
    if option in shlex.split(bisheng_options):
        return bisheng_options
    return f"{bisheng_options} {option}"


def _publish_backend_resource_usage(stdout, stderr, metadata):
    """Publish CCEC allocator facts as structured, kernel-independent data.

    Stack bytes are deliberately not converted to spill cycles here: the
    dynamic LDK/STK count depends on allocator placement and use frequency.
    The raw stack/register facts are nevertheless useful for detecting a
    SuperBlock candidate whose register allocation has crossed a cliff.
    """
    text = b"\n".join(part for part in (stdout, stderr) if part).decode("utf-8", errors="replace")
    regalloc_generated = {}
    for match in _CCEC_REGALLOC_SIZE_RE.finditer(text):
        function = match.group("function").strip()
        regalloc_generated[function] = (regalloc_generated.get(function, 0) + max(0, int(match.group("delta"))))
    resources = []
    for match in _CCEC_RESOURCE_USAGE_RE.finditer(text):
        function = match.group("function").strip()
        resource = {
            "function": function,
            "execution_model": "simt_vf" if match.group("execution_model") else "cce",
            "stack_size_bytes": int(match.group("stack_bytes")),
            "used_registers": int(match.group("used_registers")),
        }
        if function in regalloc_generated:
            resource["regalloc_generated_instructions"] = regalloc_generated[function]
        resources.append(resource)
    if not resources:
        return []
    metadata["backend_function_resources"] = resources
    metadata["backend_max_stack_size_bytes"] = max(row["stack_size_bytes"] for row in resources)
    metadata["backend_min_used_registers"] = min(row["used_registers"] for row in resources)
    return resources


def _parse_simt_private_stack_disassembly(disassembly):
    """Extract actual LDK/STK placement from one CCEC SIMT entry.

    AutoBlockify's persistent scheduling loop is the widest unconditional
    backward branch in the SIMT entry.  Keeping instructions before, inside,
    and after that loop separate is essential: only the body repeats with the
    route's runtime loop count.  The result is a backend fact, not a cost; its
    conversion to cycles is kept explicit in backend feedback below.
    """
    symbols = list(_OBJDUMP_SYMBOL_RE.finditer(disassembly or ""))
    for index, symbol in enumerate(symbols):
        function = symbol.group(2)
        if not function.endswith("_vf_simt_simt_entry"):
            continue
        base = int(symbol.group(1), 16)
        end = symbols[index + 1].start() if index + 1 < len(symbols) else len(disassembly)
        instructions = []
        for line in disassembly[symbol.end():end].splitlines():
            match = _OBJDUMP_INSTRUCTION_RE.match(line)
            if match:
                instructions.append((int(match.group(1), 16), match.group(2), match.group(3)))
        if not instructions:
            continue

        backedges = []
        for address, opcode, operands in instructions:
            if opcode != "BRANCH":
                continue
            target_match = _OBJDUMP_BRANCH_TARGET_RE.search(operands)
            if not target_match or target_match.group(1) != function:
                continue
            target = base + int(target_match.group(2), 16)
            if target >= address:
                continue
            # A predicate appears between the immediate and the wait field.
            # AutoBlockify's outer scheduling backedge is unconditional.
            prefix = operands[:target_match.start()]
            unconditional = not re.search(r"(?:^|[,\s])!?P\d+(?:[,\s]|$)", prefix)
            backedges.append((address, target, unconditional))
        unconditional = [edge for edge in backedges if edge[2]]
        if not unconditional:
            continue
        backedge, loop_start, _ = max(unconditional, key=lambda edge: edge[0] - edge[1])

        counts = {
            "once_before": {"load": 0, "store": 0},
            "per_persistent_iteration": {"load": 0, "store": 0},
            "once_after": {"load": 0, "store": 0},
        }
        for address, opcode, _ in instructions:
            if opcode.startswith("LDK"):
                kind = "load"
            elif opcode.startswith("STK"):
                kind = "store"
            else:
                continue
            if address < loop_start:
                region = "once_before"
            elif address <= backedge:
                region = "per_persistent_iteration"
            else:
                region = "once_after"
            counts[region][kind] += 1
        return {
            "schema_version": 1,
            "function": function,
            "instruction_count": len(instructions),
            "persistent_loop_detection": "widest_unconditional_backward_branch",
            "persistent_loop_start_offset": loop_start - base,
            "persistent_loop_backedge_offset": backedge - base,
            "static_private_stack_instructions": counts,
            "static_private_stack_totals":
            {kind: sum(region[kind]
                       for region in counts.values())
             for kind in ("load", "store")},
        }
    return None


def _find_backend_llvm_objdump(npu_compiler_path, env):
    """Find the vendor LLVM objdump installed with bishengir-compile."""
    candidates = []
    override = (env or {}).get("TRITON_ASCEND_LLVM_OBJDUMP") or os.getenv("TRITON_ASCEND_LLVM_OBJDUMP")
    if override:
        candidates.append(Path(override))
    if npu_compiler_path:
        candidates.append(Path(npu_compiler_path).with_name("llvm-objdump"))
    llvm_root = (env or {}).get("LLVM_ROOT") or os.getenv("LLVM_ROOT")
    if llvm_root:
        candidates.append(Path(llvm_root) / "bin" / "llvm-objdump")
    from_path = shutil.which("llvm-objdump", path=(env or {}).get("PATH"))
    if from_path:
        candidates.append(Path(from_path))
    return next((str(path) for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)


def _publish_backend_private_stack_disassembly(binary_path, metadata, npu_compiler_path, env):
    """Disassemble a generated binary and publish exact static LDK/STK facts."""
    objdump = _find_backend_llvm_objdump(npu_compiler_path, env)
    if not objdump or not Path(binary_path).is_file():
        return None
    completed = subprocess.run(
        [objdump, "-d", "--triple=hiipu64", str(binary_path)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        return None
    facts = _parse_simt_private_stack_disassembly(completed.stdout)
    if facts:
        facts["objdump"] = objdump
        metadata["backend_simt_private_stack_disassembly"] = facts
    return facts


def _private_stack_dependent_load_group_cycles(arch):
    """Read the architecture-specific private-load slope from microbench data."""
    normalized = (arch or "").lower().replace("-", "_")
    if "950pr" not in normalized:
        return 0.0
    profile_path = _costmodel_profiles_dir() / "microbench" / "ascend_950pr_v1.json"
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        metric = profile["metrics"]["simt.private_stack.dependent_access.system_cycles"]
        return max(0.0, float(metric["value"]))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return 0.0


def _backend_private_stack_load_charge(resource, route, factor, num_warps, group_cycles, calibration_warps=8):
    """Price actual static private loads with their measured warp-group slope."""
    facts = resource.get("simt_private_stack_disassembly", {}) if resource else {}
    regions = facts.get("static_private_stack_instructions", {})
    if not regions or group_cycles <= 0.0:
        return None
    try:
        iterations = max(1, int(route.get("runtime_loop_iteration_count", 1)))
        before = float(regions.get("once_before", {}).get("load", 0))
        body = float(regions.get("per_persistent_iteration", {}).get("load", 0))
        after = float(regions.get("once_after", {}).get("load", 0))
    except (AttributeError, TypeError, ValueError):
        return None
    static_warp_instructions = before + body * iterations + after
    active_warps = max(1, int(num_warps)) * max(1, int(factor))
    charge = static_warp_instructions * active_warps / max(1, calibration_warps) * group_cycles
    return {
        "formula":
        "(once_before + per_iteration * runtime_iterations + once_after) * active_warps / calibration_warps * group_cycles",
        "once_before_load_instructions_per_warp": before,
        "per_iteration_load_instructions_per_warp": body,
        "once_after_load_instructions_per_warp": after,
        "runtime_loop_iteration_count": iterations,
        "static_load_instructions_per_warp": static_warp_instructions,
        "active_warp_count": active_warps,
        "calibration_active_warp_count": calibration_warps,
        "group_cycles_per_load_instruction": group_cycles,
        "system_cycles": charge,
    }


def _compile_options_with_regalloc_size_info(options):
    """Request machine-instruction deltas from the allocator remark stream."""
    result = list(options)
    remark = "-Rpass-analysis=size-info"
    for index, option in enumerate(result):
        if not option.startswith("--append-bisheng-options="):
            continue
        if remark not in option:
            result[index] = f"{option} {remark}"
        return result
    return result


def _representative_simt_resource(resources):
    """Return the allocator row that owns the pure-SIMT kernel body."""
    if not resources:
        return None
    simt_resources = [row for row in resources if row.get("execution_model") == "simt_vf"]
    candidates = simt_resources or resources
    return max(candidates, key=lambda row: (int(row.get("stack_size_bytes", 0)), int(row.get("used_registers", 0))))


def _select_backend_resource_aware_factor(report, factor_resources, selected_factor, num_warps=1,
                                          private_load_group_cycles=0.0):
    """Reject SuperBlock factors that cross a backend allocation cliff.

    The StageCostModel runs before register allocation, so it cannot know that
    a wider SuperBlock made the allocator halve the register budget and grow
    the private stack.  Compare independently compiled legal factors here and
    keep the StageCostModel score as the ordering among resource-stable ones.
    The test is deliberately relative: it does not identify a kernel, shape,
    absolute stack size, or a fitted spill-cycle constant.
    """
    try:
        report_data = json.loads(report) if isinstance(report, str) else dict(report)
    except (TypeError, ValueError, json.JSONDecodeError):
        return int(selected_factor), {}, report

    routes = report_data.get("stage_model", {}).get("all_simt_factor_routes", {})
    legal_scores = {}
    routes_by_factor = {}
    for factor_text, route in routes.items():
        try:
            factor = int(factor_text)
            score = float(route["total_system_cycles"])
        except (KeyError, TypeError, ValueError):
            continue
        if route.get("legal", False) and factor in factor_resources:
            legal_scores[factor] = score
            routes_by_factor[factor] = route
    if not legal_scores:
        return int(selected_factor), {}, report

    adjusted_scores = dict(legal_scores)
    private_load_charges = {}
    for factor, score in legal_scores.items():
        charge = _backend_private_stack_load_charge(factor_resources[factor], routes_by_factor[factor], factor,
                                                    num_warps, private_load_group_cycles)
        if charge is None:
            continue
        private_load_charges[factor] = charge
        adjusted_scores[factor] = score + charge["system_cycles"]

    dominated = set()
    comparisons = []
    ordered_factors = sorted(legal_scores)
    # The allocator remark reports the delta of the complete machine
    # instruction stream.  It does not identify LDK/STK instructions or tell
    # whether an insertion is inside the persistent loop.  Keep the delta as
    # diagnostic evidence, but do not convert it to an absolute spill cost.
    # Doing so would price moves/address/control instructions as private-stack
    # traffic and multiply loop-invariant insertions by Q.
    for factor in ordered_factors:
        current = factor_resources[factor]
        current_stack = int(current.get("stack_size_bytes", 0))
        current_registers = int(current.get("used_registers", 0))
        for lower_factor in (candidate for candidate in ordered_factors if candidate < factor):
            lower = factor_resources[lower_factor]
            lower_stack = int(lower.get("stack_size_bytes", 0))
            lower_registers = int(lower.get("used_registers", 0))
            # A simultaneous >=25% register collapse, >=50% stack growth and
            # at least one 64-byte stack region is a backend allocation cliff,
            # rather than ordinary factor-to-factor noise.
            register_cliff = lower_registers > 0 and current_registers * 4 <= lower_registers * 3
            stack_cliff = (current_stack >= lower_stack + 64 and current_stack * 2 >= max(1, lower_stack) * 3)
            if register_cliff and stack_cliff:
                dominated.add(factor)
                comparisons.append({
                    "factor": factor,
                    "dominated_by_factor": lower_factor,
                    "reason": "register_allocation_cliff",
                    "used_registers": current_registers,
                    "reference_used_registers": lower_registers,
                    "stack_size_bytes": current_stack,
                    "reference_stack_size_bytes": lower_stack,
                })
                break

    admissible = [factor for factor in ordered_factors if factor not in dominated]
    chosen = min(admissible or ordered_factors, key=lambda factor: (adjusted_scores[factor], factor))
    resource_report = {str(factor): factor_resources[factor] for factor in ordered_factors}
    feedback = {
        "policy":
        "relative_register_and_stack_allocation_cliff_v2",
        "factor_resources":
        resource_report,
        "rejected_factors":
        sorted(dominated),
        "comparisons":
        comparisons,
        "base_stage_model_scores": {str(factor): legal_scores[factor]
                                    for factor in ordered_factors},
        "backend_adjusted_scores": {str(factor): adjusted_scores[factor]
                                    for factor in ordered_factors},
        "private_stack_load_charges":
        {str(factor): private_load_charges[factor]
         for factor in sorted(private_load_charges)},
        "absolute_spill_cost_applied":
        bool(private_load_charges),
        "absolute_spill_cost_reason": ("actual_static_ldk_count_with_persistent_loop_placement"
                                       if private_load_charges else "actual_ldk_count_and_loop_placement_unavailable"),
        "pre_backend_selected_factor":
        int(selected_factor),
        "selected_factor":
        chosen,
    }

    chosen_route = routes.get(str(chosen), {})
    report_data["backend_resource_feedback"] = feedback
    report_data["selected_superblock_factor"] = chosen
    report_data["effective_runtime_factor"] = (chosen if chosen_route.get("runtime_full_group_count", 0) else 1)
    report_data["selection_source"] = "stage_cost_model_with_backend_resource_feedback"
    report_data["application_reason"] = ("backend_resource_cliff_reselection" if chosen != int(selected_factor) else
                                         report_data.get("application_reason", "minimum_cost_candidate"))
    if chosen_route:
        report_data["full_group_count"] = chosen_route.get("runtime_full_group_count", 0)
        report_data["tail_count"] = chosen_route.get("runtime_tail_program_count", 0)
        for key in ("runtime_loop_iteration_count", "runtime_physical_program_count", "runtime_scheduling_slot_count",
                    "runtime_logical_programs_per_scheduling_slot"):
            if key in chosen_route:
                top_key = ("logical_programs_per_scheduling_slot"
                           if key == "runtime_logical_programs_per_scheduling_slot" else key)
                report_data[top_key] = chosen_route[key]
        chosen_score = adjusted_scores[chosen]
        candidate_costs = report_data.get("candidate_costs")
        if isinstance(candidate_costs, dict):
            candidate_costs["all_simt_only"] = chosen_score
            selectable = report_data.get("selectable_candidates", [])
            if not isinstance(selectable, list) or not selectable:
                selectable = ["all_simt_only"]
            post_backend_scores = {}
            for candidate in selectable:
                try:
                    score = float(candidate_costs[candidate])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(score) and score >= 0.0:
                    post_backend_scores[candidate] = score
            if post_backend_scores:
                candidate_order = {
                    "all_simd": 0,
                    "all_simt_only": 1,
                    "mixed_simd_simt": 2,
                }
                post_backend_selected = min(
                    post_backend_scores,
                    key=lambda candidate: (
                        post_backend_scores[candidate],
                        candidate_order.get(candidate, len(candidate_order)),
                    ),
                )
                feedback["post_backend_candidate_scores"] = post_backend_scores
                feedback["post_backend_selected_candidate"] = post_backend_selected
                report_data["backend_recommended_decision_kind"] = post_backend_selected
        elapsed = report_data.get("selected_route_elapsed_prediction")
        if (isinstance(elapsed, dict) and elapsed.get("candidate") == "all_simt_only"
                and feedback.get("post_backend_selected_candidate", "all_simt_only") == "all_simt_only"):
            elapsed["pre_backend_base_selection_score"] = legal_scores[chosen]
            elapsed["base_selection_score"] = chosen_score
            elapsed["superblock_factor"] = chosen
            if chosen in private_load_charges:
                elapsed["backend_private_stack_load_system_cycles"] = (private_load_charges[chosen]["system_cycles"])
            fixed = float(elapsed.get("fixed_system_cycles", 0.0))
            frequency = float(elapsed.get("system_counter_frequency_mhz", 1000.0))
            elapsed["predicted_system_counter_cycles"] = fixed + chosen_score
            elapsed["predicted_microseconds"] = (fixed + chosen_score) / max(1.0, frequency)
    updated_report = json.dumps(report_data, sort_keys=True, separators=(",", ":"))
    return chosen, feedback, updated_report


def _metadata_for_backend_cross_route_rerun(metadata, rejected_factors, opt, forced_candidate=None):
    """Recreate route-neutral metadata after backend evidence changes the winner."""
    reroute = copy.deepcopy(dict(metadata))
    generated_prefixes = (
        "auto_blockify_v1_",
        "auto_simt_",
        "backend_forced_",
        "backend_resource_",
        "backend_simt_",
        "backend_superblock_",
        "route_transform_",
        "scope_pure_simt_",
        "ta_auto_blockify_",
        "ttir_layout_",
    )
    for key in list(reroute):
        if key.startswith(generated_prefixes) or key in ("force_simt_only", "is_pure_simt"):
            reroute.pop(key, None)
    reroute["compile_mode"] = "simd_simt"
    reroute["parallel_mode"] = opt.parallel_mode
    reroute["shared_mem_dynamic_size"] = opt.shared_mem_dynamic_size
    reroute["force_simt_only"] = False
    reroute["is_pure_simt"] = False
    reroute["backend_rejected_whole_kernel_factors"] = sorted(
        {int(factor)
         for factor in rejected_factors
         if int(factor) > 1})
    if forced_candidate in ("all_simd", "all_simt_only", "mixed_simd_simt"):
        reroute["backend_forced_route_candidate"] = forced_candidate
    return reroute


def _attach_backend_cross_route_feedback(metadata, feedback, adjusted_report, pre_backend_candidate="all_simt_only"):
    """Keep backend evidence in the report produced by the materialized reroute."""
    try:
        report_data = json.loads(metadata.get("auto_simt_scope_report", ""))
        adjusted_data = json.loads(adjusted_report)
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return
    candidate_costs = report_data.get("candidate_costs")
    adjusted_costs = adjusted_data.get("candidate_costs", {})
    if isinstance(candidate_costs, dict) and isinstance(adjusted_costs, dict):
        if "all_simt_only" in adjusted_costs:
            candidate_costs["all_simt_only"] = adjusted_costs["all_simt_only"]
    report_data["backend_resource_feedback"] = feedback
    report_data["pre_backend_decision_kind"] = pre_backend_candidate
    report_data["backend_recommended_decision_kind"] = feedback.get("post_backend_selected_candidate",
                                                                    report_data.get("effective_decision_kind"))
    report_data["selection_source"] = "stage_cost_model_with_backend_resource_feedback"
    report_data["application_reason"] = "backend_resource_cross_route_reselection"
    updated = json.dumps(report_data, sort_keys=True, separators=(",", ":"))
    metadata["auto_simt_scope_report"] = updated
    dump_path = metadata.get("auto_simt_scope_dump")
    if dump_path:
        Path(dump_path).write_text(updated, encoding="utf-8", newline="\n")


def _default_auto_simt_profile_name(arch: str) -> str:
    """Select the hardware-specific StageCostModel profile by target arch.

    The route model is evaluated before lowering, but its calibration is still
    target-specific.  Falling back to the historical David-V100 profile for
    every target silently discards the 950PR/950DT layout and local-memory
    envelopes, so keep the generic profile only for unknown architectures.
    """
    normalized = (arch or "").lower().replace("-", "_")
    if "950pr" in normalized:
        return "simd_simt/ascend_950pr_simd_simt_v1.json"
    if "950dt" in normalized:
        return "simd_simt/ascend_950dt_simd_simt_v1.json"
    return "simd_simt/david_v100_simd_simt_v1.json"


def _apply_cpp_simd_simt_decision(metadata, effective: str, superblock_factor: int, report: str,
                                  base_num_warps: int = 1) -> None:
    """Translate the native decision into the backend execution contract."""
    metadata["auto_simt_scope_report"] = report
    metadata["auto_simt_superblock_factor"] = superblock_factor
    metadata["auto_simt_effective_kind"] = effective
    if effective == "all_simd":
        # ``simd_simt`` is the user's request to run the Route Model, not the
        # backend mode after selection.  Keeping it here makes later compiler
        # stages add mixed-runtime flags even though no SIMT scope exists.
        metadata["compile_mode"] = "simd"
        metadata["parallel_mode"] = "simd"
        # All-SIMD can still execute AutoBlockify V1 as a logical-program loop.
        # Keep the policy resolved before the cost model instead of disabling it.
        v1_materializable = bool(metadata.get("route_transform_v1_materializable", False))
        metadata["auto_blockify_v1_enabled"] = v1_materializable
        metadata["auto_blockify_v1_runtime_cap"] = v1_materializable
    elif effective == "mixed_simd_simt":
        metadata["auto_simt_requested_kind"] = effective
        # Mixed execution still needs V1 at factor one: its persistent
        # logical-program loop distributes the launch grid across physical
        # cores and is independent of whether F2/F4 SuperBlock aggregation is
        # profitable.  F2/F4 remain legal candidates, but must not be used as
        # the admission condition for V1 itself.
        scope_factor = max(1, int(superblock_factor))
        v1_materializable = bool(metadata.get("route_transform_v1_materializable", False))
        metadata["auto_blockify_v1_enabled"] = v1_materializable
        metadata["auto_blockify_v1_runtime_cap"] = v1_materializable
        metadata["auto_simt_scope_superblock_factor"] = scope_factor
    elif effective == "all_simt_only":
        # Auto route selection owns this decision.  When the user did not
        # explicitly disable V1, make the selected pure-SIMT plan
        # executable instead of depending on a separate environment switch.
        if metadata.get("auto_simt_whole_kernel_superblock_materializable", False):
            metadata["auto_blockify_v1_enabled"] = True


def _preserve_all_simd_auto_blockify_contract(metadata) -> None:
    """Keep the executable V1 loop that the all-SIMD candidate was scored on.

    The Route Model analyzes the post-AutoBlockify V1 clone.  Clearing this
    flag after it selects all-SIMD makes the final BiShengIR command omit
    ``--enable-auto-blockify-loop`` and materializes a different, slower
    binary than the candidate that won.  Preserve the already-resolved shared
    policy; this helper never enables V1 when analysis rejected it.
    """
    enabled = bool(metadata.get("auto_blockify_v1_enabled", False))
    metadata["auto_blockify_v1_enabled"] = enabled
    metadata["auto_blockify_v1_runtime_cap"] = enabled


def _selected_npuir_superblock_factor(metadata, opt) -> int:
    """Return the factor owned by the active route.

    CostModel-selected routes publish their factor in metadata. Explicit
    manual routes do not run the CostModel and therefore must retain the
    launch option. Reading ``auto_simt_scope_superblock_factor``
    unconditionally made an old/default F1 value shadow explicit F2/F4.
    """
    effective = metadata.get("auto_simt_effective_kind")
    if (effective in ("all_simd", "mixed_simd_simt")
            or (effective is None and metadata.get("compile_mode") == "simd_simt")):
        # The process-wide factor owns the surrounding SIMD/AIC scheduling
        # graph and must stay F1. The selected F2/F4 value is carried on the
        # scope itself and restored only in its outlined SIMT module.
        selected = 1
    elif effective == "all_simt_only":
        selected = metadata.get("auto_simt_superblock_factor", opt.superblock_factor)
    else:
        selected = opt.superblock_factor
    return max(1, int(selected or 1))


def _append_pure_simt_auto_blockify_options(options, metadata, opt) -> None:
    """Pass the executable V1 schedule contract to bishengir-compile."""
    if (not metadata.get("auto_blockify_v1_enabled", False) or not metadata.get("auto_blockify_v1_runtime_cap", False)):
        return
    if not opt.enable_ta_auto_blockify_v1:
        options.append("--enable-auto-blockify-loop")
    selected_factor = _selected_npuir_superblock_factor(metadata, opt)
    # F1 is the compiler default.  Older 950PR BiShengIR packages expose the
    # blockify pass but not the optional SuperBlock-factor flag, so emitting
    # ``--super-block-factor=1`` makes otherwise ordinary SIMT compilation
    # fail before IR lowering starts.
    if selected_factor > 1:
        options.append(f"--super-block-factor={selected_factor}")


def _can_materialize_scope_superblock(metadata, opt, whole_kernel_materializable: bool) -> bool:
    """Whether the mixed backend can execute a selected F2/F4 plan.

    The selected Stage is materialized as ``scope.scope<vector_mode=simt>``.
    NPUIR batches F logical programs around that scope after bufferization,
    while the process-wide V1 factor remains one.
    """
    del metadata
    return bool(whole_kernel_materializable and getattr(opt, "compile_on_910_95", False)
                and int(getattr(opt, "num_warps", 0) or 0) > 0)


def _publish_route_transform_capability(metadata, opt) -> str:
    """Publish the single capability fact consumed by scoring and lowering.

    Layout transforms and the V1 policy have already run when this function is
    called.  The returned JSON is passed unchanged to the native Route Model so
    the report, candidate legality, and backend materialization describe the
    same transformed TTIR.
    """
    target_supported = bool(getattr(opt, "compile_on_910_95", False))
    num_warps = max(1, int(getattr(opt, "num_warps", 1) or 1))
    v1_enabled = bool(metadata.get("auto_blockify_v1_enabled", False))
    v1_materializable = target_supported and v1_enabled

    disable_reasons = metadata.get("auto_blockify_v1_disable_reasons", [])
    if not isinstance(disable_reasons, list):
        disable_reasons = [str(disable_reasons)]
    if not v1_enabled and not disable_reasons:
        disable_reasons = ["not_requested_or_explicitly_disabled"]
    if not target_supported:
        disable_reasons.append("target_does_not_support_auto_blockify_v1")

    modeled_factors = _get_modeled_superblock_factors()
    current_scope_factors = _get_current_scope_superblock_factors()
    legal_factors = [factor for factor in modeled_factors if num_warps * factor <= 64]
    if not legal_factors:
        legal_factors = [1]

    rejected_whole_kernel_factors = {
        int(factor)
        for factor in metadata.get("backend_rejected_whole_kernel_factors", [])
        if str(factor).isdigit()
    }
    whole_kernel_superblock_factors = (
        [factor for factor in legal_factors
         if factor not in rejected_whole_kernel_factors] if v1_materializable else [1]) or [1]
    superblock_factors = ([factor for factor in legal_factors
                           if factor in current_scope_factors] if v1_materializable else [1]) or [1]
    coalesce_factor = max(1, int(metadata.get("ttir_layout_coalesce_factor", 1) or 1))

    coalesce_axis = metadata.get("ttir_layout_coalesce_axis", -1)
    if not isinstance(coalesce_axis, int):
        coalesce_axis = -1
    physical_vector_cores = max(0, int(getattr(opt, "physical_vector_core_count_hint", 0) or 0))
    physical_aic_cores = max(0, int(getattr(opt, "physical_aicore_count_hint", 0) or 0))
    if not physical_vector_cores or not physical_aic_cores:
        try:
            npu_utils = NPUUtils()
            if not physical_vector_cores:
                physical_vector_cores = int(npu_utils.get_aivector_core_num())
            if not physical_aic_cores:
                physical_aic_cores = int(npu_utils.get_aicore_num())
        except Exception:
            pass
    if not physical_vector_cores:
        try:
            physical_vector_cores = int(NPUUtils().get_aivector_core_num())
        except Exception:
            physical_vector_cores = 0
    source_logical_program_count = max(0, int(getattr(opt, "logical_program_count_hint", 0) or 0))
    transformed_logical_program_count = ((source_logical_program_count + coalesce_factor - 1) //
                                         coalesce_factor if source_logical_program_count else 0)
    capability = {
        "schema_version":
        2,
        "layout_merge_applied":
        bool(metadata.get("ttir_layout_merge_applied", False)),
        "layout_coalescing_applied":
        coalesce_factor > 1,
        "layout_coalescing_factor":
        coalesce_factor,
        "layout_coalescing_axis":
        coalesce_axis,
        # Compatibility aliases for the pre-layout terminology consumed by
        # older Python guards.  Both names describe the same resolved fact;
        # neither is an independent route decision.
        "row_coalescing_applied":
        coalesce_factor > 1,
        "row_coalescing_factor":
        coalesce_factor,
        "row_coalescing_axis":
        coalesce_axis,
        "auto_blockify_v1_requested":
        bool(metadata.get("auto_blockify_v1_requested", False)),
        "auto_blockify_v1_materializable":
        v1_materializable,
        "auto_blockify_v1_disable_reasons":
        sorted(set(disable_reasons)),
        "whole_kernel_superblock_factors":
        whole_kernel_superblock_factors,
        "scope_superblock_factors":
        superblock_factors,
        "source_logical_program_count_hint":
        source_logical_program_count,
        "logical_program_count_hint":
        transformed_logical_program_count,
        # Legacy readers still consume these two derived fields.  Keep them
        # as compatibility views of the facts above; route scheduling remains
        # owned by the native RouteSolver.
        "modeled_superblock_factors":
        modeled_factors,
        "superblock_runtime_groups": {
            str(factor): {
                "full_group_count": transformed_logical_program_count // factor,
                "tail_count": transformed_logical_program_count % factor,
            }
            for factor in modeled_factors
        },
        "physical_vector_core_count_hint":
        physical_vector_cores,
        "physical_aicore_count_hint":
        physical_aic_cores,
        "runtime_scalar_values": [{
            "argument_index": int(argument_index),
            "value": int(value),
        } for argument_index, value in getattr(opt, "auto_simt_runtime_scalar_values", ())],
        "cv_aiv_subblock_count": (2 if physical_aic_cores > 0 and physical_vector_cores == 2 * physical_aic_cores
                                  and getattr(opt, "enable_auto_bind_sub_block", None) is not False
                                  and getattr(opt, "auto_tile_and_bind_subblock", True) else 1),
    }
    capability["dynamic_cv_pipeline_requested"] = bool(getattr(opt, "enable_dynamic_cv_pipeline", False))
    forced_candidate = metadata.get("backend_forced_route_candidate")
    if forced_candidate in ("all_simd", "all_simt_only", "mixed_simd_simt"):
        capability["backend_forced_route_candidate"] = forced_candidate
    # Keep this contract factual. Route-specific Q/group/tail composition has
    # one implementation in the native RouteSolver; duplicating that formula
    # in Python previously let the two sides drift apart.
    capability_json = json.dumps(capability, sort_keys=True, separators=(",", ":"))
    metadata["route_transform_capability"] = capability_json
    metadata["route_transform_v1_materializable"] = v1_materializable
    metadata["route_transform_whole_kernel_factors"] = ",".join(map(str, whole_kernel_superblock_factors))
    metadata["route_transform_scope_factors"] = ",".join(map(str, superblock_factors))
    return capability_json


def _run_cpp_simd_simt_costmodel(mod, metadata, opt, analysis_ttir_code: str = "") -> str:
    """Run native selection/materialization; Python only schedules the passes."""
    mode = opt.auto_simt_scope_mode
    if mode == "off" or metadata.get("compile_mode") != "simd_simt":
        return "backend_default"

    profile = opt.auto_simt_model_profile or str(_costmodel_profiles_dir() / _default_auto_simt_profile_name(opt.arch))
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    capability_json = metadata.get("route_transform_capability")
    if not capability_json:
        capability_json = _publish_route_transform_capability(metadata, opt)
    capability = json.loads(capability_json)
    whole_kernel_factors = capability.get("whole_kernel_superblock_factors", [1])
    scope_factors = capability.get("scope_superblock_factors", [1])
    whole_kernel_superblock_materializable = bool(metadata.get("route_transform_v1_materializable", False))
    metadata["auto_simt_whole_kernel_superblock_materializable"] = (whole_kernel_superblock_materializable)
    # Mixed F2/F4 requires an outer factor-one V1 loop that NPUIR can refine
    # around the selected local scope.
    scope_superblock_materializable = _can_materialize_scope_superblock(metadata, opt,
                                                                        whole_kernel_superblock_materializable)
    metadata["auto_simt_scope_superblock_materializable"] = scope_superblock_materializable
    with tempfile.TemporaryDirectory() as tmpdir:
        analysis_path = ""
        if analysis_ttir_code:
            analysis_path = os.path.join(tmpdir, "post_auto_blockify_v1.ttir.mlir")
            Path(analysis_path).write_text(analysis_ttir_code, encoding="utf-8", newline="\n")
        ascend.passes.ttir.add_select_simd_simt_costmodel(
            pm,
            mode,
            profile,
            str(opt.arch),
            int(opt.num_warps),
            bool(opt.compile_on_910_95),
            whole_kernel_superblock_materializable,
            scope_superblock_materializable,
            max(whole_kernel_factors, default=1),
            max(scope_factors, default=1),
            int(capability.get("logical_program_count_hint", 0)),
            analysis_path,
            capability_json,
            str(opt.auto_simt_scope_dump),
        )
        ascend.passes.ttir.add_materialize_simt_scopes(pm)
        pm.run(mod, "select_simd_simt_costmodel")

    report = ascend.ir.get_string_attr(mod, "ascend.simt_costmodel.report_json")
    effective = ascend.ir.get_string_attr(mod, "ascend.simt_costmodel.effective")
    get_int_attr = getattr(ascend.ir, "get_int_attr", None)
    superblock_factor = (get_int_attr(mod, "ascend.simt_costmodel.superblock_factor") if get_int_attr else 1)
    if not isinstance(superblock_factor, int) or superblock_factor < 1:
        superblock_factor = 1
    if not report or effective not in {"all_simd", "all_simt_only", "mixed_simd_simt", "backend_default"}:
        raise RuntimeError("invalid native SIMD/SIMT costmodel result")
    _apply_cpp_simd_simt_decision(metadata, effective, superblock_factor, report, int(opt.num_warps))
    return effective


def _run_ttir_layout_merge(mod, metadata) -> None:
    """Materialize TTIR memory-layout merges before route scoring."""
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    ascend.passes.ttir.add_ttir_layout_merge(pm)
    pm.run(mod, "ttir_layout_merge_before_costmodel")
    get_int_attr = getattr(ascend.ir, "get_int_attr", None)
    factor = get_int_attr(mod, "hacc.coalesce_factor") if get_int_attr else -1
    axis = get_int_attr(mod, "hacc.coalesce_axis") if get_int_attr else -1
    ceil_div = get_int_attr(mod, "hacc.coalesce_grid_ceil_div") if get_int_attr else -1
    metadata["ttir_layout_merge_applied"] = True
    metadata["ttir_layout_coalesce_factor"] = factor if isinstance(factor, int) and factor > 1 else 1
    metadata["ttir_layout_coalesce_axis"] = axis if isinstance(axis, int) and axis >= 0 else -1
    metadata["ttir_layout_coalesce_grid_ceil_div"] = bool(isinstance(ceil_div, int) and ceil_div > 0)


def _resolve_auto_blockify_v1_policy(ttir_code: str, metadata, opt) -> bool:
    """Resolve the shared admission policy for NPUIR and TA V1.

    ``enable_auto_blockify`` keeps its existing tri-state contract:
      * ``None`` follows ``TRITON_ALL_BLOCKS_PARALLEL``;
      * ``True`` explicitly enables V1;
      * ``False`` explicitly disables V1.

    An explicitly supplied ``has_auto_blockify_blacklist_op`` remains the
    final blacklist override.  Otherwise the same TTIR blacklist is evaluated
    for either implementation.  This function decides whether V1 is legal;
    ``enable_ta_auto_blockify_v1`` only selects where the legal transform runs.
    """
    env_enabled = _is_auto_map_parallel_blocks_enabled()
    requested = opt.enable_auto_blockify
    requested_enabled = env_enabled if requested is None else bool(requested)
    metadata["auto_blockify_v1_requested"] = requested_enabled

    blacklist_reasons = []
    has_blacklist_op = metadata.get("has_auto_blockify_blacklist_op")
    if has_blacklist_op is None and requested_enabled:
        blacklist_reasons = _get_auto_blockify_blacklist_reasons(ttir_code)
        has_blacklist_op = bool(blacklist_reasons)
    elif has_blacklist_op is None:
        has_blacklist_op = False

    metadata["has_auto_blockify_blacklist_op"] = bool(has_blacklist_op)
    if has_blacklist_op and blacklist_reasons:
        match = re.search(r"tt\.func\s+public\s+@(\w+)", ttir_code)
        _warn_auto_blockify_disabled(match.group(1) if match else "<unknown>", blacklist_reasons)

    enabled = requested_enabled and not has_blacklist_op
    metadata["auto_blockify_v1_enabled"] = enabled
    if blacklist_reasons:
        metadata["auto_blockify_v1_disable_reasons"] = list(blacklist_reasons)
    elif has_blacklist_op:
        metadata["auto_blockify_v1_disable_reasons"] = ["explicit_blacklist_override"]
    elif not requested_enabled:
        metadata["auto_blockify_v1_disable_reasons"] = ["not_requested_or_explicitly_disabled"]
    else:
        metadata["auto_blockify_v1_disable_reasons"] = []
    metadata["auto_blockify_v1_selection_source"] = ("option"
                                                     if requested is not None else "TRITON_ALL_BLOCKS_PARALLEL")
    return enabled


def _run_ta_auto_blockify_v1(mod, metadata, opt, *, super_block_factor=None) -> bool:
    """Materialize the TA port of AutoBlockify V1.

    V1 keeps one logical program's tile tensor shapes unchanged.  The launcher
    is capped to the same physical vector-core count passed to this transform;
    NPUIR's copy of V1 is disabled later to prevent applying the same schedule
    twice.  AutoBlockify V2 remains a separate, disabled pass.
    """
    physical_vector_cores = int(NPUUtils().get_aivector_core_num())
    if super_block_factor is None:
        super_block_factor = opt.superblock_factor
    super_block_factor = max(1, int(super_block_factor))
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    ascend.passes.ttir.add_auto_blockify_v1(
        pm,
        physical_vector_cores,
        super_block_factor,
    )
    pm.run(mod, "ta_auto_blockify_v1")
    materialized = _get_then_remove_rc(mod, "ta.auto_blockify_v1.materialized") == 1
    metadata["ta_auto_blockify_v1_materialized"] = materialized
    metadata["ta_auto_blockify_v1_physical_core_count"] = physical_vector_cores
    metadata["ta_auto_blockify_v1_super_block_factor"] = super_block_factor
    return materialized


# Compatibility alias for callers predating the TA pass rename.
_run_ta_simt_auto_blockify_v1 = _run_ta_auto_blockify_v1


def _parse_ttir_text(ttir_code: str, context=None):
    """Recreate a TTIR module in ``context`` from an immutable snapshot."""
    if context is None:
        context = ir.context()
        ir.load_dialects(context)
        buffer_ir.load_dialects(context)
        ascend_ir.load_dialects(context)
        ascend.load_dialects(context)
        if distributed is not None:
            distributed.ir.load_dialects(context)
    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "route_neutral.ttir.mlir"
        source.write_text(ttir_code, encoding="utf-8", newline="\n")
        module = ir.parse_mlir_module(str(source), context)
        module.context = context
        return module


def _build_costmodel_analysis_ttir(mod, metadata, opt) -> str:
    """Build a disposable post-AutoBlockify V1-F1 analysis view.

    The real module stays post-layout and route-neutral.  The native Route
    Model scores this clone, then materializes its decision on the real
    module.  Pure-SIMT and mixed lowering apply their executable V1 schedule
    only after selection, so all-SIMD never inherits SIMT dispatch IR.
    """
    if not metadata.get("auto_blockify_v1_enabled", False):
        metadata["auto_simt_costmodel_analysis_ir"] = "post_layout_ttir"
        return ""

    analysis_mod = _parse_ttir_text(str(mod), getattr(mod, "context", None))
    analysis_metadata = {}
    materialized = _run_ta_simt_auto_blockify_v1(
        analysis_mod,
        analysis_metadata,
        opt,
        super_block_factor=1,
    )
    metadata["auto_simt_costmodel_analysis_v1_materialized"] = materialized
    metadata["auto_simt_costmodel_analysis_ir"] = ("post_auto_blockify_v1_f1_ttir"
                                                   if materialized else "post_layout_ttir")
    return str(analysis_mod) if materialized else ""


def _refine_ta_auto_blockify_v1_superblock(mod, metadata, super_block_factor) -> None:
    """Apply a pure-SIMT SuperBlock factor to the factor-1 V1 schedule."""
    super_block_factor = max(1, int(super_block_factor))
    if super_block_factor == 1:
        return
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    ascend.passes.ttir.add_refine_auto_blockify_v1_superblock(pm, super_block_factor)
    pm.run(mod, "refine_ta_auto_blockify_v1_superblock")
    metadata["ta_auto_blockify_v1_super_block_factor"] = super_block_factor


def ttir_to_linalg(mod, metadata, opt, *, named_ops=True):
    # use triton_adapter to lower Triton-MLIR to linalg
    metadata["compile_mode"] = opt.compile_mode
    # Native SIMD/SIMT Route Model. When the user asks for compile_mode
    # "simd_simt" the C++ cost model runs before the downstream lowering and may
    # materialize SIMT scopes / publish the effective route. This is a no-op for
    # every other mode (auto_simt_scope_mode defaults to "off").
    route_neutral_ttir = str(mod)
    analysis_ttir_code = ""
    if metadata["compile_mode"] == "simd_simt" and opt.auto_simt_scope_mode != "off":
        if opt.enable_ttir_layout_merge:
            _run_ttir_layout_merge(mod, metadata)
        else:
            metadata["ttir_layout_merge_applied"] = False
            metadata["ttir_layout_coalesce_factor"] = 1
            metadata["ttir_layout_coalesce_axis"] = -1
            metadata["ttir_layout_coalesce_grid_ceil_div"] = False
        _resolve_auto_blockify_v1_policy(str(mod), metadata, opt)
        _publish_route_transform_capability(metadata, opt)
        analysis_ttir_code = _build_costmodel_analysis_ttir(mod, metadata, opt)
    cpp_decision = _run_cpp_simd_simt_costmodel(mod, metadata, opt, analysis_ttir_code)
    cpp_all_simt = cpp_decision == "all_simt_only"
    if metadata["compile_mode"] == "simd_simt" and (cpp_all_simt or ascend.ir.is_whole_body_void_simt_scope(mod)):
        # The native Route Model selected pure SIMT (or the kernel body is a
        # single void SIMT scope). Carry TTIR forward to the pure-SIMT compiler;
        # the npubin stage dispatches on metadata["force_simt_only"].
        if cpp_all_simt:
            # Backend register allocation is the first point where a whole
            # SIMT factor's resource cliff is observable. Preserve the input
            # needed to materialize a different legal route if that evidence
            # changes the global candidate ordering.
            metadata["auto_simt_route_neutral_ttir"] = route_neutral_ttir
        metadata["scope_pure_simt_auto"] = True
        metadata["force_simt_only"] = True
        # Mirror NPUOptions.__post_init__ so the runtime launcher observes the
        # pure-SIMT ABI even though the route model flipped the effective mode
        # after options normalization.
        metadata["is_pure_simt"] = True
        metadata["parallel_mode"] = "simt"
        metadata["shared_mem_dynamic_size"] = 122880
        metadata["compile_mode"] = "simt_only"
        ascend.ir.inline_void_simt_scopes_for_pure_simt(mod)
        ascend.ir.clear_simd_simt_costmodel_attrs(mod)
        auto_blockify_v1_enabled = metadata.get("auto_blockify_v1_enabled")
        if auto_blockify_v1_enabled is None:
            auto_blockify_v1_enabled = _resolve_auto_blockify_v1_policy(str(mod), metadata, opt)
        selected_superblock_factor = metadata.get("auto_simt_superblock_factor", opt.superblock_factor)
        if (opt.enable_ta_auto_blockify_v1 and auto_blockify_v1_enabled
                and metadata.get("ta_auto_blockify_v1_materialized", False)):
            _refine_ta_auto_blockify_v1_superblock(mod, metadata, selected_superblock_factor)
            metadata["auto_blockify_v1_runtime_cap"] = True
        elif (opt.enable_ta_auto_blockify_v1 and auto_blockify_v1_enabled
              and not metadata.get("ta_auto_blockify_v1_materialized", False)):
            # A TA silent-skip must not cap the launch grid. It also must not
            # fall through to NPUIR and hide a parity bug in the TA port.
            materialized = _run_ta_auto_blockify_v1(mod, metadata, opt, super_block_factor=selected_superblock_factor)
            metadata["auto_blockify_v1_runtime_cap"] = materialized
        elif not metadata.get("ta_auto_blockify_v1_materialized", False):
            metadata["auto_blockify_v1_runtime_cap"] = auto_blockify_v1_enabled
        # The auto-selected pure-SIMT path returns TTIR directly and therefore
        # bypasses the normal linalg metadata export. Publish the launch
        # contract here, exactly as the ordinary ttir-to-linalg path does.
        _export_program_grid_metadata(mod, metadata)
        # This analysis marker is useful to the Route Model but is not part of
        # the public TTIR contract consumed by bishengir-compile.
        _get_then_remove_rc(mod, "ta.ttir_layout_merge.applied")
        return _serialize_module(mod, opt)

    # A hand-written mixed kernel already owns its scope boundary, so no Route
    # Model pass runs to attach the requested factor. Publish the same local
    # scope ABI here; automatic routes attach it during C++ materialization.
    if cpp_decision == "backend_default" and metadata["compile_mode"] == "simd_simt":
        manual_scope_factor = max(1, int(opt.superblock_factor or 1))
        manual_scope_count = ascend.ir.set_simt_scope_superblock_factor(mod, manual_scope_factor)
        if manual_scope_count:
            metadata["auto_simt_scope_superblock_factor"] = manual_scope_factor
            metadata["manual_simt_scope_count"] = manual_scope_count

    # Get Triton-MLIR as string
    ttir_code = _serialize_module(mod, opt)
    auto_map_parallel_blocks_enabled = _is_auto_map_parallel_blocks_enabled()
    # This is compiler-derived safety metadata, never a user compile option.
    # Derive it even when the feature is currently disabled so a later runtime
    # environment change cannot enable AutoBlockify for unsafe TTIR.
    blacklist_reasons = _get_auto_blockify_blacklist_reasons(ttir_code)
    has_auto_blockify_blacklist_op = bool(blacklist_reasons)
    metadata["has_auto_blockify_blacklist_op"] = has_auto_blockify_blacklist_op
    if auto_map_parallel_blocks_enabled and has_auto_blockify_blacklist_op and blacklist_reasons:
        kernel_name = re.search(r"tt\.func\spublic\s+@(\w+)", ttir_code).group(1)
        _warn_auto_blockify_disabled(kernel_name or "<unknown>", blacklist_reasons)
    if metadata.get("auto_simt_effective_kind") == "all_simd":
        # Preserve the decision-owned V1 policy from
        # _apply_cpp_simd_simt_decision. Re-running the option policy here
        # could change the transform contract after candidate scoring.
        # All-SIMD also runs NPUIR AutoBlockify when the shared policy allows it.
        # Preserve the values set by _apply_cpp_simd_simt_decision; do not
        # re-enable or force-disable here.
        _preserve_all_simd_auto_blockify_contract(metadata)
    elif metadata.get("auto_simt_effective_kind") == "mixed_simd_simt":
        # Preserve the mixed SuperBlock contract established above.  NPUIR V1
        # creates the persistent logical-program loop even for F1; the
        # selected factor independently controls logical-program aggregation.
        metadata["auto_blockify_v1_enabled"] = True
        metadata["auto_blockify_v1_runtime_cap"] = True
    else:
        _resolve_auto_blockify_v1_policy(ttir_code, metadata, opt)
    has_auto_blockify_blacklist_op = metadata["has_auto_blockify_blacklist_op"]
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        dst_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        if opt.debug:
            Path(src_path).write_text(ttir_code)
        triton_adapter_opt_path = _get_triton_adapter_opt_path()

        # Select analysis is a fixed lowering policy, not a user compile option.
        enable_select_analysis = True
        compile_on_910_95 = metadata["compile_on_910_95"]
        # The native Route Model may have rewritten the effective mode (for
        # example all_simd or mixed). Honor that decision here so the lowering
        # passes implement it via the ordinary compile_mode string interface.
        compile_mode = metadata.get("compile_mode") or opt.compile_mode
        metadata["compile_mode"] = compile_mode
        enable_mixed_cv = metadata.get("enable_mixed_cv")
        disable_auto_inject_block_sync = metadata.get("disable_auto_inject_block_sync")
        set_workspace_multibuffer = metadata.get("set_workspace_multibuffer")

        # Inject grid tile-count hint for ChunkCoalescing. When the kernel
        # has no boundary mask but grid[axis] is known at compile time (e.g.
        # from constexpr nchunks), the pass uses this to safely choose H.
        grid_num_tiles = metadata.get("grid_num_tiles")
        if isinstance(grid_num_tiles, int) and grid_num_tiles > 0:
            try:
                _builder = ascend.ir.ascendnpu_ir_builder(mod.context, opt.target_arch)
                mod.set_attr("hacc.grid_num_tiles", _builder.parse_attr(f"{grid_num_tiles} : i32"))
            except Exception:
                pass  # graceful fallback: pass runs without hint

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        if distributed is not None:
            distributed.ascend_passes.ttgpuir.add_convert_triton_distributed_to_hivm(pm)

        ascend.passes.ttir.add_triton_control_flow_opt(pm)
        ascend.passes.ttir.add_triton_to_structure(pm, False, False)
        ascend.passes.ttir.add_discrete_mask_access_conversion(pm, compile_on_910_95, compile_mode)
        ascend.passes.ttir.add_triton_to_annotation(pm)
        ascend.passes.ttir.add_triton_to_unstructure(pm, compile_on_910_95, compile_mode)
        ascend.passes.ttir.add_triton_to_hivm(pm)
        ascend.passes.ttir.add_triton_to_hfusion(pm, compile_on_910_95)
        ascend.passes.ttir.add_triton_to_llvm(pm)
        ascend.passes.ttir.add_bubble_up_operation(pm, compile_on_910_95)
        ascend.passes.ttir.add_triton_to_structure(pm, False, False)
        ascend.passes.ttir.add_triton_to_linalg(pm, False, named_ops, False, enable_select_analysis, compile_on_910_95,
                                                compile_mode)
        # Restricted to 910_95/950. The merged buffer is written by two disjoint
        # memref.copy ops, and on 910_9362 the generated code only makes the
        # copy next to the surviving to_tensor visible, so the other half of the
        # tile reads back the padding value. The IR is unchanged through
        # bishengir-opt, so the loss happens in code generation.
        if compile_on_910_95:
            ascend.passes.ttir.add_merge_concat_load_buffer(pm)
        if metadata["enable_dynamic_cv_pipeline"]:
            metadata["set_workspace_multibuffer"] = 0
            metadata["enable_mixed_cv"] = True
            metadata["disable_auto_inject_block_sync"] = True

            # Must run before add_dynamic_cv_pipeline because the driven
            # AddMultiBufferInnerScope pass reads the module-level
            # `ssbuffer.insertionOptimization` attribute (set here) at run time.
            # Keep the existing default-on buffer insertion behavior.
            ascend.passes.ttir.set_enable_buffer_insert_optimization(mod)
            ascend.passes.ttir.add_dynamic_cv_pipeline(pm, compile_on_910_95)

        if _enable_msdebug():
            ascend.passes.ttir.add_normalize_debug_line_locations(pm)

        _intra_val = metadata.get("buf_slot_num_of_veccore")
        if _intra_val is not None:
            ascend.passes.ttir.set_buffer_count(mod, "INTRA", _intra_val)

        _inter_val = metadata.get("buf_slot_num_of_crosscore")
        if _inter_val is not None:
            ascend.passes.ttir.set_buffer_count(mod, "INTER", _inter_val)

        _load_val = metadata.get("buf_slot_num_of_gm")
        if _load_val is not None:
            ascend.passes.ttir.set_buffer_count(mod, "LOAD", _load_val)

        if opt.debug:
            # Print the equivalent triton-opt command line so the pass
            # pipeline can be reproduced and debugged outside of Python.
            print_src_path, print_dst_path = _get_dump_paths(metadata["hash"], src_path, dst_path)
            cmd = [
                _get_triton_opt_path(),
                print_src_path,
                f"--pass-pipeline={pm.get_pipeline_str()}",
                "--mlir-print-debuginfo",
                "-o",
                print_dst_path,
            ]
            print(f"[DEBUG] cmd list: {shlex.join(cmd)}")

        pm.run(mod, 'ttir_to_linalg')
        _adjust_metadata_by_module_result(mod, metadata, opt, enable_mixed_cv=enable_mixed_cv,
                                          disable_auto_inject_block_sync=disable_auto_inject_block_sync,
                                          set_workspace_multibuffer=set_workspace_multibuffer)
        _export_program_grid_metadata(mod, metadata)

        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(str(mod), "kernel.ttadapter.mlir", binary=False)

        return _serialize_module(mod, opt)


def linalg_to_bc_by_triton_mlir_opt(linalg: str, metadata, opt):
    """
    Convert Linalg IR to MLIR Bytecode format using triton-mlir-opt.
    This function supports both MLIR and BishengIR ops.
    Args:
        linalg: Linalg IR in text format
        metadata: Compilation metadata
        opt: Compilation options
    Returns:
        Bytecode data as bytes (not file path, to avoid temp directory cleanup issues)
    """
    # The Stage graph is fixed before the C++ Route Model runs. When auto
    # selects all-SIMT, ttadapter deliberately carries TTIR forward to the
    # pure-SIMT compiler; the optional Linalg bytecode round-trip must become
    # an identity stage rather than trying to parse TTIR as Linalg/BishengIR.
    if metadata.get("force_simt_only", False):
        return linalg

    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        bc_path = os.path.join(tmpdir, "kernel.mlirbc")
        Path(ttadapter_path).write_text(linalg)

        triton_mlir_opt_path = _get_triton_mlir_opt_path()

        # The --emit-bytecode flag ensures output is in BC format
        subprocess.run([
            triton_mlir_opt_path,
            ttadapter_path,
            "--emit-bytecode",
            "-o",
            bc_path,
        ], check=True, capture_output=True, text=True)

        # Read bytecode as binary before temp directory is cleaned up
        with open(bc_path, "rb") as f:
            bc_data = f.read()

        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(bc_data, "kernel.mlirbc", binary=True)

        return bc_data


def bc_to_linalg_by_bishengir_opt(bc_data: bytes, metadata, opt):
    """
    Convert MLIR Bytecode to MLIR text format using bishengir-opt.
    Args:
        bc_data: Bytecode data as bytes
        metadata: Compilation metadata
        opt: Compilation options
    Returns:
        MLIR text as string
    """
    if metadata.get("force_simt_only", False):
        return bc_data

    with tempfile.TemporaryDirectory() as tmpdir:
        bc_path = os.path.join(tmpdir, "kernel.mlirbc")
        mlir_path = os.path.join(tmpdir, "kernel.mlir")

        # Write bytecode data to temporary file
        with open(bc_path, "wb") as f:
            f.write(bc_data)

        bishengir_opt_path, env = _get_bishengir_opt_path()

        cmd = [
            bishengir_opt_path,
            bc_path,
        ]
        if opt.debug:
            cmd += ["--mlir-print-debuginfo"]
        cmd += ["-o", mlir_path]
        subprocess.run(cmd, env=env, capture_output=True, check=True, text=True)

        # Read the generated MLIR text
        linalg_text = Path(mlir_path).read_text()

        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(linalg_text, "kernel.mlir", binary=False)

        return linalg_text


def __get_metadata_attr_by_callback(lib, postfix: str, metadata, meta_key: str):
    func_symbol = metadata["kernel_name"] + postfix
    if hasattr(lib, func_symbol):
        callback_func = getattr(lib, func_symbol)
        callback_func.restype = ctypes.c_int64
        callback_func.argtypes = []
        metadata[meta_key] = callback_func()


def _parse_linalg_metadata(linalg: str, metadata: dict):
    """
    Parse Linalg IR to extract metadata required for NPU compilation.
    Extracts and updates the following fields in metadata:
      - mix_mode
      - kernel_name
      - tensor_kinds
      - shared (currently hardcoded)
      - name (kernel_name)

    Additionally, removes the mix_mode attribute from the IR.
    """
    # --- Regular expressions and examples ---

    DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX = r'hivm.disable_auto_tile_and_bind_subblock'

    # Inserted by DiscreteMaskAccessConversionPass / MemOpConverter when discrete
    # masked stores need cross-block exclusion (e.g. hivm.sync_block_lock).
    SYNC_BLOCK_LOCK_REGEX = r'\bsync_block_lock\b'

    # Example: mix_mode = "aiv" -> aiv
    MIX_MODE_REGEX = r'mix_mode\s*=\s*"([^"]+)"'

    # Example: parallel_mode = "mix_simd_simt" -> mix_simd_simt
    PARALLEL_MODE_REGEX = r'parallel_mode\s*=\s*"([^"]+)"'

    # Example: func.func @gather_sorted_kernel(%arg0: ...) -> gather_sorted_kernel
    KERNEL_NAME_REGEX = r"func\.func\s+@(\w+)"

    # Example: %arg1: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} -> ('1', '0')
    TENSOR_KIND_REGEX = r'%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}'

    # Example: bitcode = "a.bc"
    BITCODES_REGEX = r'bitcode\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|(\w+))'

    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, NPU backend does not limit on shared memory
    metadata["shared"] = 1
    module_attrs = re.search(r'\bmodule(?:\s+@\w+)?\s+attributes\s*\{([^}]*)\}', linalg)
    metadata["has_full_row_copy"] = bool(module_attrs and re.search(r'\btt\.full_row_copy\b', module_attrs.group(1)))
    # Force disable auto tile and bind subblock if attribute is present in module
    metadata["auto_tile_and_bind_subblock"] = not re.search(DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX, linalg)
    # Turn off auto-blockify only for the ORDERED (token-ring) sync_block_lock:
    if re.search(SYNC_BLOCK_LOCK_REGEX, linalg) and not re.search(r"sync_block_lock_unordered", linalg):
        metadata["has_auto_blockify_blacklist_op"] = True
    # Mixed kernels use (block, subblock) as the unordered-lock participant
    # identity. Pure AIV kernels keep subblock tiling disabled: they neither
    # need it nor necessarily have the FFTS runtime resource it requires.
    has_unordered_sync_block_lock = re.search(r"sync_block_lock_unordered", linalg) is not None
    metadata["has_unordered_sync_block_lock"] = has_unordered_sync_block_lock
    if has_unordered_sync_block_lock:
        mix_mode = re.search(MIX_MODE_REGEX, linalg).group(1)
        if mix_mode != "mix":
            metadata["auto_tile_and_bind_subblock"] = False
        # One metadata cache line for runtime participant_num, plus one
        # choosing and one ticket cache line per participant. Each cache line is
        # 8 i64. This fallback is for one lock; the bishengir callback supplies
        # the exact total after lowering.
        metadata["sync_block_lock_layout"] = 1 << 32
        metadata["lock_init_val"] = 0
    # the mix mode is also encoded into metadata['name'] for runtime to distinguish
    metadata["mix_mode"] = re.search(MIX_MODE_REGEX, linalg).group(1)
    metadata["parallel_mode"] = re.search(PARALLEL_MODE_REGEX, linalg).group(1)
    if metadata.get("auto_simt_requested_kind") == "mixed_simd_simt":
        applied = metadata["parallel_mode"] == "mix_simd_simt"
        metadata["auto_simt_scope_applied"] = applied
        if not applied and os.environ.get("TRITON_ASCEND_AUTO_SIMT_STRICT_VERIFY",
                                          "").strip().lower() in {"1", "true", "yes", "on"}:
            raise RuntimeError("C++ costmodel selected mixed SIMD/SIMT, but lowering "
                               f"reported {metadata['parallel_mode']!r}")
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, linalg).group(1)
    # Check the function load_binary in npu_driver.py.
    metadata["name"] = metadata["kernel_name"]
    # Parse all tensor kinds from arguments
    metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, linalg)]
    # init the ub bits of triton kernel for inductor autotune using
    metadata["required_ub_bits"] = 0

    # Parse all bitcode paths
    bitcodes = re.findall(BITCODES_REGEX, linalg)
    metadata["bitcodes"] = [val for group in bitcodes for val in group if val]
    return linalg, metadata


def _parse_ttir_metadata(ttir: str, metadata: dict):
    """
    Parse TTIR to extract metadata required for NPU compilation.
    Extracts and updates the following fields in metadata:
      - kernel_name
      - shared (currently hardcoded)
    """
    # --- Regular expressions and examples ---
    # Example: tt.func @gather_sorted_kernel(%arg0: ...) -> gather_sorted_kernel
    KERNEL_NAME_REGEX = r"tt\.func\spublic\s+@(\w+)"

    # Example: %arg1: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} -> ('1', '0')
    TENSOR_KIND_REGEX = r'%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}'

    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, NPU backend does not limit on shared memory
    metadata["shared"] = 1
    # Note: Currently, for TTIR inputs, we only support vector kernels.
    metadata["mix_mode"] = "aiv"
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, ttir).group(1)
    metadata["name"] = metadata["kernel_name"]
    # Keep this as compiler-derived safety metadata.  In particular, do not
    # trust a caller-provided False value to override an unsafe TTIR pattern.
    metadata["has_auto_blockify_blacklist_op"] = bool(_get_auto_blockify_blacklist_reasons(ttir))
    # Parse all tensor kinds from arguments
    metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, ttir)]
    return metadata


def _parse_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_common_bishengir_compile_options(metadata):
    bishengir_target = metadata['target'].arch
    bishengir_target_opt = f"--target={bishengir_target}"
    return [bishengir_target_opt]


def _needs_lib_call_no_inline(metadata):
    """Return whether the target needs the CANN 9.1 hacc.noinline workaround."""
    arch = metadata['target'].arch
    return arch.startswith("Ascend950")


@functools.lru_cache()
def _npu_compiler_supports_option(compiler_path: str, option: str) -> bool:
    """Check an optional BiShengIR flag instead of assuming toolchain parity."""
    try:
        result = subprocess.run([compiler_path, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return option in result.stdout


@functools.lru_cache()
def _supports_full_row_copy_inlining(compiler_path):
    try:
        result = subprocess.run([compiler_path, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "--enable-lib-call-no-inline" in result.stdout


def _full_row_copy_compile_options(metadata, compiler_path):
    # A shared noinline load helper loses constant-width specialization when
    # the masked tail also calls it. Scope this to compiler-versioned A5 SIMD
    # loads; other kernels retain the NPU compiler's default inlining policy.
    if (metadata.get("has_full_row_copy", False) and metadata["target"].arch.startswith("Ascend950")
            and metadata.get("mix_mode") == "aiv" and metadata.get("parallel_mode") == "simd"
            and _supports_full_row_copy_inlining(compiler_path)):
        return ["--enable-lib-call-no-inline=false"]
    return []


def get_auto_bind_sub_block_option(metadata):
    # auto_tile_and_bind_subblock is read from the module.
    # enable_auto_bind_sub_block is set by the user and has a higher priority.
    enable_auto_bind_sub_block = metadata["enable_auto_bind_sub_block"]
    return (metadata["auto_tile_and_bind_subblock"]
            if enable_auto_bind_sub_block is None else enable_auto_bind_sub_block)


def _save_npuir_debug_output(stdout_bytes: bytes, stderr_bytes: bytes, tmpdir: str, metadata_hash: str):
    stdout = stdout_bytes.decode('utf-8') if stdout_bytes else ''
    stderr = stderr_bytes.decode('utf-8') if stderr_bytes else ''
    combined = stdout + stderr
    if not combined.strip():
        combined = "No output captured."
    output_path = os.path.join(tmpdir, "kernel.npuir.mlir")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined)

    dump_manager = get_dump_manager(metadata_hash)
    dump_manager.put(Path(output_path).read_text(encoding='utf-8'), "kernel.npuir.mlir", binary=False)


def _dump_kernel_binary(metadata_hash: str, bin_path: str):
    """Copy the compiled kernel object into the TRITON_DEBUG dump directory."""
    dump_manager = get_dump_manager(metadata_hash)
    dump_manager.put(Path(bin_path).read_bytes(), os.path.basename(bin_path), binary=True)


def try_compile_with_config(linalg: str, ub_config: Dict[str, Any], metadata: dict, opt) -> Tuple[bool, str]:
    """
    Try to compile with given UB config, return (success, error_msg).
    If compilation fails due to UB overflow or other errors, returns (False, error_msg).
    If compilation succeeds, returns (True, "").
    """
    # Must import from ubtuner.py - this is the single source of truth
    from triton.backends.ascend.runtime.ubtuner import UB_TO_NPU_OPTION_MAP
    option_mapping = UB_TO_NPU_OPTION_MAP

    metadata = dict(metadata)

    for ub_key, meta_key in option_mapping.items():
        if ub_key in ub_config:
            metadata[meta_key] = ub_config[ub_key]

    # Choose compile function based on options
    if opt.compile_on_910_95:
        compile_func = linalg_to_bin_enable_npu_compile_910_95
    else:
        compile_func = linalg_to_bin_enable_npu_compile_A2_A3

    try:
        compile_func(linalg, metadata, opt)
        return (True, "")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        return (False, error_msg)
    except Exception as e:
        return (False, str(e))


def linalg_to_bin_enable_npu_compile_910_95(linalg: str, metadata, opt):
    linalg, metadata = _parse_linalg_metadata(linalg, metadata)
    _finalize_program_launch_policy(metadata, opt)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file_name = "kernel.mlir"
        ttadapter_path = os.path.join(tmpdir, tmp_file_name)
        Path(ttadapter_path).write_text(linalg)
        bin_file = os.path.join(tmpdir, "kernel")
        if _check_bishengir_api_change():
            bin_file_with_ext = "kernel.o"
        else:
            bin_file_with_ext = "kernel_reloc.o"
        bin_path = os.path.join(tmpdir, bin_file_with_ext)
        callback_path = os.path.join(tmpdir, "libkernel.so")
        _compile_option_list = get_common_bishengir_compile_options(metadata)

        multibuffer_mode = metadata.get("multibuffer_mode")
        if multibuffer_mode is not None:
            mode_arg = "[" + ",".join(f"({level},{count})" for level, count in multibuffer_mode) + "]"
            _compile_option_list.append(f"--multibuffer-mode={mode_arg}")

        multibuffer = metadata.get("multibuffer")
        num_stages = metadata.get("num_stages")

        if multibuffer is not None or num_stages is not None:
            multi_buffer_value = True
            if multibuffer is not None and not multibuffer:
                multi_buffer_value = False
            elif num_stages is not None and num_stages == 1:
                multi_buffer_value = False
            _compile_option_list += [
                f"--enable-auto-multi-buffer={multi_buffer_value}",
            ]

        vf_fusion_mode = metadata["vf_fusion_mode"]
        if vf_fusion_mode is not None:
            _compile_option_list += [
                f"--vf-fusion-mode={vf_fusion_mode}",
            ]

        enable_preload = metadata["enable_preload"]
        if enable_preload is not None:
            _compile_option_list += [
                f"--enable-preload={enable_preload}",
            ]

        disable_tightly_coupled_buffer_reuse = metadata["disable_tightly_coupled_buffer_reuse"]
        if disable_tightly_coupled_buffer_reuse:
            _compile_option_list += ["--disable-tightly-coupled-buffer-reuse"]

        _compile_option_list += [
            f"--enable-auto-bind-sub-block={get_auto_bind_sub_block_option(metadata)}",
        ]

        if _check_bishengir_supports_optimized_metaop():
            _compile_option_list += [f"--enable-optimized-metaop={is_cann_version_at_least(9, 2, 0)}"]
        _compile_option_list += _custom_core_count_options()

        if force_disable_ffts(opt.target_arch):
            _compile_option_list += ["--disable-ffts"]
        if _is_ascend_sanitizer_enabled():
            _compile_option_list += ["--enable-sanitizer=true"]
        if not _is_debug_line_info_disabled():
            _compile_option_list += ["--enable-debug-info=true"]
        if _enable_msdebug():
            _compile_option_list += ["--enable-debug-variables=true"]
        if _enable_print_ub_bits():
            _compile_option_list += ["--enable-print-memory-allocated-size"]

        enable_hivm_auto_cv_balance = metadata["enable_hivm_auto_cv_balance"]
        if enable_hivm_auto_cv_balance is not None:
            _compile_option_list += \
                [f"--enable-hivm-auto-cv-balance={enable_hivm_auto_cv_balance}"]

        sync_solver = metadata["sync_solver"]
        if sync_solver is not None:
            _compile_option_list += \
                [f"--enable-hivm-graph-sync-solver={sync_solver}"]

        unit_flag = metadata["unit_flag"]
        if unit_flag is not None:
            _compile_option_list += \
                [f"--enable-hivm-unit-flag-sync={unit_flag}"]

        inject_barrier_all = metadata["inject_barrier_all"]
        if inject_barrier_all is not None:
            _compile_option_list += \
                [f"--enable-hivm-inject-barrier-all-sync={inject_barrier_all}"]

        inject_block_all = metadata["inject_block_all"]
        if inject_block_all is not None:
            _compile_option_list += \
                [f"--enable-hivm-inject-block-all-sync={inject_block_all}"]

        limit_auto_multi_buffer_only_for_local_buffer = metadata["limit_auto_multi_buffer_only_for_local_buffer"]
        if limit_auto_multi_buffer_only_for_local_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-only-for-local-buffer={limit_auto_multi_buffer_only_for_local_buffer}"]

        set_workspace_multibuffer = metadata["set_workspace_multibuffer"]
        if set_workspace_multibuffer is not None:
            _compile_option_list += \
                [f"--set-workspace-multibuffer={set_workspace_multibuffer}"]

        auto_multi_buffer = metadata["limit_auto_multi_buffer_of_local_buffer"]
        if auto_multi_buffer is None:
            auto_multi_buffer = "no-limit"
        _compile_option_list += \
            [f"--limit-auto-multi-buffer-of-local-buffer={auto_multi_buffer}"]
        auto_multi_buffer_buffer = metadata["limit_auto_multi_buffer_buffer"]
        if auto_multi_buffer_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-buffer={auto_multi_buffer_buffer}"]

        enable_mixed_cv = metadata["enable_mixed_cv"]
        if enable_mixed_cv is not None:
            _compile_option_list += \
                [f"--enable-mixed-cv={enable_mixed_cv}"]

        enable_dynamic_cv_pipeline = metadata["enable_dynamic_cv_pipeline"]
        if enable_dynamic_cv_pipeline == True and not metadata.get("disable_vf_operand_substitution", False):
            _compile_option_list += [f"--enable-vf-operand-substitution=True"]

        enable_flatten = metadata["enable_flatten"]
        if enable_flatten is not None:
            _compile_option_list += \
                [f"--enable-flatten={enable_flatten}"]

        enable_auto_vectorize_v2 = metadata["enable_auto_vectorize_v2"]
        if enable_auto_vectorize_v2 is not None:
            _compile_option_list += \
                [f"--enable-auto-vectorize-v2={enable_auto_vectorize_v2}"]
        auto_vectorize_v2_max_fused_ops_num = metadata["auto_vectorize_v2_max_fused_ops_num"]
        if auto_vectorize_v2_max_fused_ops_num is not None:
            _compile_option_list += \
                [f"--hfusion-max-fused-ops-in-auto-vectorize-v2={auto_vectorize_v2_max_fused_ops_num}"]
        prevec_max_fused_ops_num = metadata["prevec_max_fused_ops_num"]
        if prevec_max_fused_ops_num is not None:
            _compile_option_list += \
                [f"--hfusion-max-fused-elementwise-ops={prevec_max_fused_ops_num}"]

        disable_auto_inject_block_sync = metadata["disable_auto_inject_block_sync"]
        if disable_auto_inject_block_sync is not None:
            _compile_option_list += \
                [f"--disable-auto-inject-block-sync={disable_auto_inject_block_sync}"]

        bitcodes = metadata["bitcodes"]
        if bitcodes is not None:
            for bitcode in bitcodes:
                _compile_option_list += \
                    [f"--link-aicore-bitcode={bitcode}"]

        if metadata["auto_blockify_enabled"]:
            _compile_option_list += ["--enable-auto-blockify-loop"]
            # Only mixed/local-scope routes carry a SuperBlock factor; all-SIMD
            # and ordinary SIMD keep the historical F1 autoblockify behavior.
            if metadata.get("parallel_mode") == "mix_simd_simt":
                _compile_option_list += [f"--super-block-factor={_selected_npuir_superblock_factor(metadata, opt)}"]
        npu_compiler_path, env = _get_npucompiler_path()
        if npu_compiler_path.endswith("bishengir-compile"):
            _compile_option_list += [
                "--enable-hfusion-compile=true",
                "--enable-triton-kernel-compile=true",
            ]
            if (_needs_lib_call_no_inline(metadata)
                    and _npu_compiler_supports_option(npu_compiler_path, "--enable-lib-call-no-inline")):
                _compile_option_list += ["--enable-lib-call-no-inline=false"]
            if metadata.get("parallel_mode") == "mix_simd_simt":
                num_warps = metadata.get("num_warps") or opt.num_warps
                _compile_option_list += [f"--num-warps={num_warps}"]
                warp_size = metadata.get("warp_size") or opt.warp_size
                _compile_option_list += [f"--threads-per-warp={warp_size}"]
            partition_mode = _validate_partition_and_bind_sub_block(
                metadata.get(
                    "enable_partition_and_bind_sub_block",
                    opt.enable_partition_and_bind_sub_block,
                ) or "off")
            if partition_mode != "off":
                _compile_option_list += [f"--partition-and-bind-sub-block={partition_mode}"]
            _compile_option_list += _full_row_copy_compile_options(metadata, npu_compiler_path)
            # Temporary until the NPU compiler enables batch matmul by default in Q4.
            if metadata.get("enable_hivm_batch_matmul"):
                _compile_option_list += ["--enable-hivm-batch-matmul"]
            if metadata.get("enable_vf_stack_limit"):
                _compile_option_list += ["--enable-vf-stack-limit"]
        bisheng_options = _bisheng_options_with_resource_usage(metadata["bisheng_options"])
        _compile_option_list += [f"--append-bisheng-options={bisheng_options}"]
        if metadata.get("mix_mode") in ["aic"]:
            _compile_option_list += ["--disable-hfusion-vectorize=true"]
        if opt.debug:
            _compile_option_list += ["--mlir-print-ir-after-failure"]
            _compile_option_list += ["--mlir-print-stacktrace-on-diagnostic"]
            _compile_option_list += ["--bishengir-print-ir-after=hivm-graph-sync-solver"]

        compile_mode = metadata.get("compile_mode", "simd")
        if compile_mode == "simd_simt":
            _compile_option_list += ["--enable-simd-simt-mix-compile"]
        if compile_mode == "simd_simt" and metadata.get("parallel_mode") != "mix_simd_simt":
            num_warps = metadata.get("num_warps", opt.num_warps)
            _compile_option_list += [f"--num-warps={num_warps}"]
            warp_size = metadata.get("warp_size", opt.warp_size)
            _compile_option_list += [f"--threads-per-warp={warp_size}"]
            if opt.shared_mem_dynamic_size is not None:
                _compile_option_list += [f"--shared-mem-dynamic-size={opt.shared_mem_dynamic_size}"]

        vf_merge_level = metadata["vf_merge_level"]
        if vf_merge_level is not None:
            _compile_option_list += [f"--enable-vf-merge-level={vf_merge_level}"]

        hfusion_enable_multiple_consumer_fusion = metadata["hfusion_enable_multiple_consumer_fusion"]
        if hfusion_enable_multiple_consumer_fusion:
            _compile_option_list += [
                f"--hfusion-enable-multiple-consumer-fusion={hfusion_enable_multiple_consumer_fusion}"
            ]

        enable_cross_if_fusion = metadata["enable_cross_if_fusion"]
        if enable_cross_if_fusion:
            _compile_option_list += [f"--hfusion-enable-cross-if-fusion={enable_cross_if_fusion}"]

        plan_memory_strategy = metadata["plan_memory_strategy"]
        if plan_memory_strategy is not None:
            _compile_option_list += [f"--plan-memory-strategy={plan_memory_strategy}"]

        cmd_list = ([npu_compiler_path, ttadapter_path] + _compile_option_list + ["-o", bin_file])

        if opt.debug or os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
            print_cmd_list = cmd_list.copy()
            print_cmd_list[1], print_cmd_list[-1] = _get_dump_paths(metadata["hash"], ttadapter_path, bin_path)
            print(f"[DEBUG] cmd_list: {shlex.join(print_cmd_list)}")

        try:
            ret = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            if opt.debug:
                _save_npuir_debug_output(e.stdout, e.stderr, tmpdir, metadata["hash"])
            raise

        if opt.debug:
            _save_npuir_debug_output(ret.stdout, ret.stderr, tmpdir, metadata["hash"])

        _publish_backend_resource_usage(ret.stdout, ret.stderr, metadata)

        stdout_str = ret.stdout.decode('utf-8') if ret.stdout else ''
        match = re.search(r'UB\s+size\s*=\s*(\d+)\s*bits', stdout_str)
        if match:
            # get the ub bits of triton kernel from bisheng for inductor autotune using
            metadata["required_ub_bits"] = int(match.group(1))

        if not Path(bin_path).exists():
            error_msg = ret.stderr.decode('utf-8') if ret.stderr else ''
            print(f"[DEBUG] {bin_path} is not found")
            print(f"[DEBUG] Stderr:\n{error_msg}")
            raise subprocess.CalledProcessError(ret.returncode, cmd_list, ret.stdout, ret.stderr)

        if opt.debug:
            _dump_kernel_binary(metadata["hash"], bin_path)

        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            __get_metadata_attr_by_callback(lib, "_infer_task_type_function", metadata, "bs_task_type")
            __get_metadata_attr_by_callback(lib, "_infer_workspace_shape_function", metadata, "workspace_size")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_num_function", metadata,
                                            "sync_block_lock_layout")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_init_function", metadata, "lock_init_val")

        return Path(bin_path).read_bytes()


def linalg_to_bin_enable_npu_compile_A2_A3(linalg: str, metadata, opt):
    linalg, metadata = _parse_linalg_metadata(linalg, metadata)
    _finalize_program_launch_policy(metadata, opt)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file_name = "kernel.mlir"
        ttadapter_path = os.path.join(tmpdir, tmp_file_name)
        Path(ttadapter_path).write_text(linalg)
        bin_file = os.path.join(tmpdir, "kernel")
        if _check_bishengir_api_change():
            bin_file_with_ext = "kernel.o"
        else:
            bin_file_with_ext = "kernel_reloc.o"
        if _check_bishengir_is_regbased():
            bishengir_hivm_opt = "--reg-based=true"
        else:
            bishengir_hivm_opt = "--enable-hivm-compile=true"
        bin_path = os.path.join(tmpdir, bin_file_with_ext)
        callback_path = os.path.join(tmpdir, "libkernel.so")
        _compile_option_list = [
            f"--target={NPUUtils().get_arch()}",
        ]

        multibuffer_mode = metadata.get("multibuffer_mode")
        if multibuffer_mode is not None:
            mode_arg = "[" + ",".join(f"({level},{count})" for level, count in multibuffer_mode) + "]"
            _compile_option_list.append(f"--multibuffer-mode={mode_arg}")

        multibuffer = metadata.get("multibuffer")
        num_stages = metadata.get("num_stages")

        if multibuffer is not None or num_stages is not None:
            multi_buffer_value = True
            if multibuffer is not None and not multibuffer:
                multi_buffer_value = False
            elif num_stages is not None and num_stages == 1:
                multi_buffer_value = False
            _compile_option_list.append(f"--enable-auto-multi-buffer={multi_buffer_value}")

        enable_ubuf_saving = metadata["enable_ubuf_saving"]
        if enable_ubuf_saving is not None:
            _compile_option_list += [
                f"--enable-ubuf-saving={enable_ubuf_saving}",
            ]

        disable_size_align_for_cast = metadata["disable_size_align_for_cast"]
        if disable_size_align_for_cast is not None:
            _compile_option_list += [
                f"--disable-size-align-for-cast={disable_size_align_for_cast}",
            ]

        enable_preload = metadata["enable_preload"]
        if enable_preload is not None:
            _compile_option_list += [
                f"--enable-preload={enable_preload}",
            ]

        _compile_option_list += [
            f"--enable-auto-bind-sub-block={get_auto_bind_sub_block_option(metadata)}",
        ]

        if _check_bishengir_supports_optimized_metaop():
            _compile_option_list += [f"--enable-optimized-metaop=false"]

        if _is_ascend_sanitizer_enabled():
            _compile_option_list += ["--enable-sanitizer=true"]
        if not _is_debug_line_info_disabled():
            _compile_option_list += ["--enable-debug-info=true"]

        if _enable_print_ub_bits():
            _compile_option_list += ["--enable-print-memory-allocated-size"]

        if _enable_dump_memory_info():
            _compile_option_list += ["--enable-memory-display=true"]

        if _enable_msdebug():
            _compile_option_list += ["--enable-debug-variables=true"]

        enable_hivm_auto_cv_balance = metadata["enable_hivm_auto_cv_balance"]
        if enable_hivm_auto_cv_balance is not None:
            _compile_option_list += \
                [f"--enable-hivm-auto-cv-balance={enable_hivm_auto_cv_balance}"]

        sync_solver = metadata["sync_solver"]
        if sync_solver is not None:
            _compile_option_list += [
                f"--enable-hivm-graph-sync-solver={sync_solver}",
                f"--enable-hivm-cross-core-gss={sync_solver}",
            ]

        unit_flag = metadata["unit_flag"]
        if unit_flag is not None:
            _compile_option_list += \
                [f"--enable-hivm-unit-flag-sync={unit_flag}"]

        enable_flatten = metadata["enable_flatten"]
        if enable_flatten is not None:
            _compile_option_list += \
                [f"--enable-flatten={enable_flatten}"]

        enable_auto_vectorize_v2 = metadata["enable_auto_vectorize_v2"]
        if enable_auto_vectorize_v2 is not None:
            _compile_option_list += \
                [f"--enable-auto-vectorize-v2={enable_auto_vectorize_v2}"]

        inject_barrier_all = metadata["inject_barrier_all"]
        if inject_barrier_all is not None:
            _compile_option_list += \
                [f"--enable-hivm-inject-barrier-all-sync={inject_barrier_all}"]

        inject_block_all = metadata["inject_block_all"]
        if inject_block_all is not None:
            _compile_option_list += \
                [f"--enable-hivm-inject-block-all-sync={inject_block_all}"]

        limit_auto_multi_buffer_only_for_local_buffer = metadata["limit_auto_multi_buffer_only_for_local_buffer"]
        if limit_auto_multi_buffer_only_for_local_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-only-for-local-buffer={limit_auto_multi_buffer_only_for_local_buffer}"]

        set_workspace_multibuffer = metadata["set_workspace_multibuffer"]
        if set_workspace_multibuffer is not None:
            _compile_option_list += \
                [f"--set-workspace-multibuffer={set_workspace_multibuffer}"]

        tile_mix_vector_loop = metadata["tile_mix_vector_loop"]
        if tile_mix_vector_loop is not None:
            _compile_option_list += \
                [f"--tile-mix-vector-loop={tile_mix_vector_loop}"]

        tile_mix_cube_loop = metadata["tile_mix_cube_loop"]
        if tile_mix_cube_loop is not None:
            _compile_option_list += \
                [f"--tile-mix-cube-loop={tile_mix_cube_loop}"]

        auto_multi_buffer = metadata["limit_auto_multi_buffer_of_local_buffer"]
        if auto_multi_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-of-local-buffer={auto_multi_buffer}"]

        disable_auto_inject_block_sync = metadata["disable_auto_inject_block_sync"]
        if disable_auto_inject_block_sync is not None:
            _compile_option_list += \
                [f"--disable-auto-inject-block-sync={disable_auto_inject_block_sync}"]

        bitcodes = metadata["bitcodes"]
        if bitcodes is not None:
            for bitcode in bitcodes:
                _compile_option_list += \
                    [f"--link-aicore-bitcode={bitcode}"]

        enable_libdevice = os.getenv("TRITON_ENABLE_LIBDEVICE", False)
        if enable_libdevice:
            _compile_option_list += [f"--link-aicore-bitcode={get_libdevice()}"]

        if metadata["auto_blockify_enabled"]:
            _compile_option_list += ["--enable-auto-blockify-loop"]
            # Only mixed/local-scope routes carry a SuperBlock factor; all-SIMD
            # and ordinary SIMD keep the historical F1 autoblockify behavior.
            if metadata.get("parallel_mode") == "mix_simd_simt":
                _compile_option_list += [f"--super-block-factor={_selected_npuir_superblock_factor(metadata, opt)}"]
        npu_compiler_path, env = _get_npucompiler_path()
        if npu_compiler_path.endswith("bishengir-compile"):
            _compile_option_list += [
                "--enable-hfusion-compile=true",
                bishengir_hivm_opt,
                "--enable-triton-kernel-compile=true",
            ]

        if opt.debug:
            _compile_option_list += ["--mlir-print-ir-after-failure"]
            _compile_option_list += ["--mlir-print-stacktrace-on-diagnostic"]
            _compile_option_list += ["--bishengir-print-ir-after=hivm-graph-sync-solver"]

        cmd_list = ([npu_compiler_path, ttadapter_path] + _compile_option_list + ["-o", bin_file])

        if opt.debug or os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
            print_cmd_list = cmd_list.copy()
            print_cmd_list[1], print_cmd_list[-1] = _get_dump_paths(metadata["hash"], ttadapter_path, bin_path)
            print(f"[DEBUG] cmd_list: {shlex.join(print_cmd_list)}")

        try:
            ret = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            if opt.debug:
                _save_npuir_debug_output(e.stdout, e.stderr, tmpdir, metadata["hash"])
            raise

        if opt.debug:
            _save_npuir_debug_output(ret.stdout, ret.stderr, tmpdir, metadata["hash"])

        stdout_str = ret.stdout.decode('utf-8') if ret.stdout else ''
        match = re.search(r'UB\s+size\s*=\s*(\d+)\s*bits', stdout_str)
        if match:
            metadata["required_ub_bits"] = int(match.group(1))

        if not Path(bin_path).exists():
            error_msg = ret.stderr.decode('utf-8') if ret.stderr else ''
            print(f"[DEBUG] {bin_path} is not found")
            print(f"[DEBUG] Stderr:\n{error_msg}")
            raise subprocess.CalledProcessError(ret.returncode, cmd_list, ret.stdout, ret.stderr)

        if opt.debug:
            _dump_kernel_binary(metadata["hash"], bin_path)

        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            __get_metadata_attr_by_callback(lib, "_infer_task_type_function", metadata, "bs_task_type")
            __get_metadata_attr_by_callback(lib, "_infer_workspace_shape_function", metadata, "workspace_size")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_num_function", metadata,
                                            "sync_block_lock_layout")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_init_function", metadata, "lock_init_val")

        return Path(bin_path).read_bytes()


def get_libdevice():
    current = os.path.dirname(__file__)
    return os.path.join(current, "lib/libdevice.10.bc")


def _is_a5_target_arch(arch: str) -> bool:
    return isinstance(arch, str) and arch.startswith(("Ascend910_95", "Ascend950"))


def _get_libdevice_compile_state(arch: str) -> tuple[bool, bool]:
    return (
        bool(os.getenv("TRITON_ENABLE_LIBDEVICE", False)),
        bool(os.getenv("TRITON_ENABLE_LIBDEVICE_SIMT", False)) and _is_a5_target_arch(arch),
    )


_DEFAULT_COMPILE_MODE = "simd_simt_template"
_CANONICAL_COMPILE_MODES = ("simd", "simd_simt", "simd_simt_template", "simt_only")
_COMPILE_MODE_ALIASES = {
    "simt_template": "simd_simt_template",
    "unstructured_in_simt": "simd_simt_template",
}


def _normalize_compile_mode(compile_mode, arch: str) -> str:
    """Validate and canonicalize the public compile-mode contract.

    ``simd_simt_template`` is the portable default: it retains the ordinary
    SIMD pipeline on A2/A3 and enables the existing template-SIMT subpaths on
    Ascend 950.  ``unstructured_in_simt`` and ``simt_template`` are equivalent
    spellings of that mode.  ``simd_simt`` runs the native SIMD/SIMT Route
    Model, which selects the effective mode per kernel.  Pure-SIMT remains an
    Ascend 950-only explicit compile mode.
    """
    if compile_mode is None:
        # TorchNPU's Inductor path forwards None when the kernel config does
        # not record a compile mode; "unspecified" resolves to the default.
        compile_mode = _DEFAULT_COMPILE_MODE
    elif not isinstance(compile_mode, str):
        # Some TorchNPU builds forward the NPUKernelType enum member instead
        # of its string value, so unwrap the underlying value when present.
        enum_value = getattr(compile_mode, "value", None)
        if isinstance(enum_value, str):
            compile_mode = enum_value
    if not isinstance(compile_mode, str):
        raise ValueError("compile_mode must be a string; expected one of: " + ", ".join(_CANONICAL_COMPILE_MODES))

    canonical_mode = _COMPILE_MODE_ALIASES.get(compile_mode, compile_mode)
    if canonical_mode not in _CANONICAL_COMPILE_MODES:
        raise ValueError(f"invalid compile_mode={compile_mode!r}; expected one of: " +
                         ", ".join(_CANONICAL_COMPILE_MODES))

    if canonical_mode == "simt_only" and not _is_a5_target_arch(arch):
        raise ValueError('compile_mode="simt_only" is supported only on Ascend 950 targets.')
    return canonical_mode


@dataclass(frozen=True)
class NPUOptions:
    debug: bool = False
    sanitize_overflow: bool = True
    # Backend-only construction input.  AscendBackend.parse_options injects
    # GPUTarget.arch and never forwards a user-supplied compile option.
    arch: InitVar[str] = ""
    rule_mask: InitVar[int] = DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK
    # This becomes compiler metadata, so its name must also be valid for the
    # namedtuple constructed by CompiledKernel on Python 3.10.
    target_arch: str = field(init=False, repr=False)

    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = 32
    num_ctas: int = 1
    # Keep an omitted stage count distinct from an explicit mode.
    num_stages: Optional[int] = None
    # Ascend threads-per-warp is a backend capability, not a compile option.
    warp_size: int = field(default=32, init=False)
    ir_override: Optional[str] = None  # filename of a user-defined IR (*.{ttir|ttadapter|mlirbc|bcmlir|npubin})

    # Deprecated constructor-only compatibility input.  The supplied value is
    # ignored and replaced with the lowering selector derived from GPUTarget.arch.
    compile_on_910_95: Optional[bool] = field(default=None, repr=False, kw_only=True)
    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    backend_name: str = 'cann'
    instrumentation_mode: str = ""
    enable_graph_optimize: bool = True
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15", "fp8e4nv", "fp8e4b8", "fp8e5b16")
    deprecated_fp8_dtypes: Tuple[str] = ()
    vf_merge_level: int = 1
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "hf32")
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = None
    # Backend-managed BiSheng arguments. Keep the name discoverable so legacy
    # launch keywords are accepted, but never forward a user-supplied value.
    bisheng_options: str = field(default="-cce-link-aicore-ll-module " + get_libdevice(), init=False)
    multibuffer: bool = True
    # Public dictionaries are stored as sorted pairs for hashing and JSON caches.
    multibuffer_mode: Optional[Union[Dict[str, int], Tuple[Tuple[str, int], ...]]] = None
    vf_fusion_mode: str = None
    enable_ubuf_saving: bool = None
    disable_size_align_for_cast: bool = None
    enable_preload: bool = None
    enable_auto_bind_sub_block: bool = None
    # Experimental BiShengIR policy for assigning work to the two AIV
    # sub-blocks. Off preserves the existing backend behavior.
    enable_partition_and_bind_sub_block: str = "off"
    disable_tightly_coupled_buffer_reuse: bool = False
    enable_hivm_auto_cv_balance: bool = None
    # Temporary 910_95 switch; the NPU compiler plans to make this default in Q4.
    enable_hivm_batch_matmul: bool = False
    # Only takes effect on the Ascend 950 non-pure-SIMT BiShengIR compilation path.
    enable_vf_stack_limit: bool = False
    sync_solver: bool = None
    unit_flag: bool = None
    enable_flatten: bool = None
    enable_auto_vectorize_v2: bool = None
    auto_vectorize_v2_max_fused_ops_num: int = None
    prevec_max_fused_ops_num: int = None
    inject_barrier_all: bool = None
    inject_block_all: bool = None
    limit_auto_multi_buffer_only_for_local_buffer: bool = None
    limit_auto_multi_buffer_of_local_buffer: str = None
    limit_auto_multi_buffer_buffer: str = None
    set_workspace_multibuffer: int = None
    tile_mix_vector_loop: int = None
    tile_mix_cube_loop: int = None
    disable_auto_inject_block_sync: bool = None
    enable_mixed_cv: bool = None
    enable_dynamic_cv_pipeline: bool = None
    enable_cube_block_merge: bool = False
    hfusion_enable_multiple_consumer_fusion: bool = None
    buf_slot_num_of_veccore: int = None
    buf_slot_num_of_crosscore: int = None
    buf_slot_num_of_gm: int = None

    # plan memory strategy: "default" (default) or "largest-first"
    plan_memory_strategy: str = None
    # Internal launch metadata.  The mode is initially derived from
    # compile_mode, then replaced with the mode emitted by TritonToLinalg.
    parallel_mode: str = field(default="simd", init=False)
    # Internal pure-SIMT state, derived from the effective lowering mode.
    is_pure_simt: bool = field(default=False, init=False)
    # Only takes effect on the pure-SIMT path.
    shared_mem_dynamic_size: int = None
    # Ascend 950 pure-SIMT-only option passed as -simt-optimization-mode
    # to bishengir-compile. Its value grammar belongs to the toolchain.
    # Individual digits are passed to various passes to control behavior,
    # and are parsed right-to-left.
    # For example, a value of 101 is interpreted as 0000101.
    # If left as 0, bishengir-compile sets this to 900101
    simt_optimization_mode: int = 0000000
    # Canonical modes: SIMD (D), SIMD with template-SIMT (P), and pure-SIMT
    # (T). ``unstructured_in_simt`` is an equivalent P spelling.
    compile_mode: str = _DEFAULT_COMPILE_MODE
    # disable simt fma optimization to get high precision
    disable_fma: bool = False

    # superblocking factor
    superblock_factor: int = 1

    # --- Native SIMD/SIMT Route Model (compile_mode="simd_simt") -----------
    # Select the TA implementation of SIMT AutoBlockify V1. This does not enable
    # V1 by itself: both TA and NPUIR honor enable_auto_blockify, the
    # environment default, and the common blacklist policy. V2 is independent.
    enable_ta_auto_blockify_v1: bool = False
    # Apply implicit-permute and row/tile/strided coalescing before Route Model
    # analysis. Exposed primarily for controlled compiler ablation; normal
    # compilation should retain the default.
    enable_ttir_layout_merge: bool = True
    # Optional runtime launch facts used by SuperBlock costing. Zero means the
    # frontend cannot provide a stable value for this compile.
    logical_program_count_hint: int = 0
    physical_vector_core_count_hint: int = 0
    physical_aicore_count_hint: int = 0
    # Concrete non-constexpr integer arguments used only by the native route
    # model.  The JIT backend hook populates (TTIR argument index, value)
    # pairs before cache lookup, so a different loop bound cannot reuse a
    # cost-model decision compiled for another bound.
    auto_simt_runtime_scalar_values: Tuple[Tuple[int, int], ...] = ()
    # Native AscendModel selection: Python only schedules the C++ passes.
    # Empty/"off" disables the route model; "auto" selects, "report" dumps.
    auto_simt_scope_mode: str = ""
    auto_simt_scope_dump: str = ""
    auto_simt_model_profile: str = ""
    # Content digest is populated in __post_init__ so profile edits invalidate
    # the Triton compile cache even when the path itself is unchanged.
    auto_simt_model_assets_hash: str = ""

    # ChunkCoalescing: number of tiles along the outermost grid axis.
    # Auto-injected from static grid tuples; enables safe coalescing for
    # unmasked kernels whose grid dims are compile-time known.
    grid_num_tiles: int = None

    # Deprecated names remain visible to callers that use parse_options({})
    # to distinguish compile options from kernel arguments. Their public
    # values are ignored (or routed to canonical options) before construction.
    add_auto_scheduling: bool = field(default=False, init=False)
    allow_fp8e4nv: bool = field(default=False, init=False)
    auto_blockify_size: int = field(default=1, init=False)
    auto_tile_and_bind_subblock: bool = field(default=True, init=False)
    code_motion: Optional[bool] = field(default=None, init=False)
    enable_auto_blockify: Optional[bool] = field(default=None, init=False)
    enable_bishengir_simt_optimization: int = field(default=0, init=False)
    enable_buffer_insert_optimization: bool = field(default=True, init=False)
    enable_cce_vf_auto_sync: Optional[bool] = field(default=None, init=False)
    enable_cce_vf_remove_membar: Optional[bool] = field(default=None, init=False)
    enable_cross_if_fusion: bool = field(default=False, init=False)
    enable_drop_unit_dims: Optional[bool] = field(default=None, init=False)
    enable_linearize: Optional[bool] = field(default=None, init=False)
    enable_mask_fallback_conversion: bool = field(default=False, init=False)
    enable_nd2nz_on_vector: bool = field(default=False, init=False)
    enable_select_analysis: bool = field(default=True, init=False)
    enable_simt_reorder_instruction: bool = field(default=False, init=False)
    enable_sync_block_lock: Optional[bool] = field(default=None, init=False)
    enable_ub_refine_opt: bool = field(default=False, init=False)
    enable_vf_fusion: Optional[bool] = field(default=None, init=False)
    force_simt_only: bool = field(default=False, init=False)
    force_simt_template: bool = field(default=False, init=False)
    graph_optimize_emit_remarks: bool = field(default=False, init=False)
    graph_optimize_max_rewrites_per_function: int = field(default=64, init=False)
    graph_optimize_rule_mask: int = field(default=DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK, init=False)
    graph_optimize_ub_capacity_bytes: Optional[int] = field(default=None, init=False)
    has_auto_blockify_blacklist_op: Optional[bool] = field(default=None, init=False)
    inter_cache_num: Optional[int] = field(default=None, init=False)
    intra_cache_num: Optional[int] = field(default=None, init=False)
    kernel_name: str = field(default="triton_", init=False)
    llvm_version: int = field(default=15, init=False)
    load_cache_num: Optional[int] = field(default=None, init=False)
    mix_mode: str = field(default="", init=False)
    ops_reorder: Optional[bool] = field(default=None, init=False)
    optimize_dynamic_offset: bool = field(default=False, init=False)
    simt_reorder_instruction: bool = field(default=False, init=False)
    storage_align: Optional[bool] = field(default=None, init=False)
    stream: Optional[int] = field(default=None, init=False)
    use_bytecode: bool = field(default=True, init=False)

    def __post_init__(self, arch, rule_mask):
        from triton.backends.ascend import _apply_ascend_patch

        _apply_ascend_patch()
        object.__setattr__(self, "target_arch", arch)
        # Plain init=False defaults live on the class. Materialize them in
        # the instance because both Inductor and JIT inspect options.__dict__.
        for name, option_field in self.__dataclass_fields__.items():
            if not option_field.init and name not in self.__dict__:
                object.__setattr__(self, name, option_field.default)
        # Keep the legacy name discoverable while its property and all
        # compiler decisions continue to use the injected target architecture.
        self.__dict__["arch"] = arch
        object.__setattr__(self, "multibuffer_mode", _multibuffer_mode_to_tuple(self.multibuffer_mode))
        if self.multibuffer_mode is not None and self.num_stages is not None:
            raise ValueError("num_stages and multibuffer_mode cannot be specified together")
        if self.multibuffer_mode is None and self.num_stages is None:
            object.__setattr__(self, "num_stages", 2)
        if self.compile_on_910_95 is not None:
            _warn_deprecated_npu_option("compile_on_910_95")
        object.__setattr__(
            self,
            "compile_on_910_95",
            isinstance(arch, str) and arch.startswith(("Ascend910_95", "Ascend950")),
        )
        try:
            normalized_rule_mask = normalize_graph_optimization_rule_mask(rule_mask)
        except ProgramGridContractError as error:
            raise ValueError(f"invalid GraphOptimize rule_mask: {error}") from error
        if normalized_rule_mask != DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK:
            object.__setattr__(self, "rule_mask", normalized_rule_mask)

        compile_mode = str(_normalize_compile_mode(self.compile_mode, arch))
        object.__setattr__(self, "compile_mode", compile_mode)

        if compile_mode == "simt_only":
            object.__setattr__(self, "is_pure_simt", True)
            object.__setattr__(self, "parallel_mode", "simt")
        else:
            object.__setattr__(self, "parallel_mode", "simd")

        if self.is_pure_simt:
            if self.shared_mem_dynamic_size is None:
                object.__setattr__(self, "shared_mem_dynamic_size", 122880)
        else:
            object.__setattr__(self, "shared_mem_dynamic_size", 221184)

        # Normalize the native Route Model options. The default empty mode
        # resolves to "off", so the route model is opt-in and the profile digest
        # stays out of the compile cache key until a profile is actually used.
        mode = self.auto_simt_scope_mode or os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE", "")
        object.__setattr__(self, "auto_simt_scope_mode", _normalize_auto_simt_scope_mode(mode))
        dump = self.auto_simt_scope_dump or os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP", "")
        profile = self.auto_simt_model_profile or os.environ.get("TRITON_ASCEND_AUTO_SIMT_PROFILE", "")
        if self.auto_simt_scope_mode == "off":
            dump, profile, asset_hash = "", "", "disabled"
        else:
            asset_hash = _auto_simt_asset_hash(profile, _default_auto_simt_profile_name(self.arch))
        object.__setattr__(self, "auto_simt_scope_dump", dump)
        object.__setattr__(self, "auto_simt_model_profile", profile)
        object.__setattr__(self, "auto_simt_model_assets_hash", asset_hash)

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        key = "_".join([key, get_cann_version_file_hash()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _get_npu_options_arch(options: NPUOptions) -> str:
    """Expose the injected target to the established lowering builder API.

    ``arch`` is an ``InitVar`` rather than a user compile option. Its legacy
    name is also serialized for option discovery, while this view always
    returns the target injected by ``AscendBackend.parse_options``.
    """
    return options.target_arch


NPUOptions.arch = property(_get_npu_options_arch)

_INTERNAL_NPU_OPTION_MARKERS = frozenset({
    "target_arch",
    "compile_on_910_95",
    "parallel_mode",
    "is_pure_simt",
})


def _is_internal_npu_options(options, target_arch: str) -> bool:
    """Recognize an NPUOptions metadata round trip by backend-only fields."""
    return (_INTERNAL_NPU_OPTION_MARKERS <= options.keys() and options.get("target_arch") == target_arch
            and options.get("compile_on_910_95") == _is_a5_target_arch(target_arch))


def _normalize_bishengir_simt_optimization_for_context(options: NPUOptions, raw_options) -> None:
    """Restrict the vendor SIMT optimization switch to its Ascend 950 pure-SIMT path."""
    option_name = "simt_optimization_mode"
    if option_name not in raw_options:
        return

    if _is_a5_target_arch(options.target_arch) and options.is_pure_simt:
        # The BiShengIR toolchain owns the option's value grammar; keep the
        # Python boundary free of numeric or range validation.
        return

    warnings.warn(
        "simt_optimization_mode only takes effect for Ascend 950 "
        "pure-SIMT compilation; ignoring the explicit value.",
        UserWarning,
        stacklevel=3,
    )
    object.__setattr__(options, option_name, 0)


def _custom_core_count_options():
    # Scoring, launch capping and both A5 compiler entries must agree on the
    # scheduling-slot count. In particular, pure-SIMT V1 must not silently
    # keep the target's full AIV count when NPU_DEVICE_LIMIT is set.
    npu_utils = NPUUtils()
    if not npu_utils.has_device_limit():
        return []
    return [
        f"--custom-aic-number={npu_utils.get_aicore_num()}", f"--custom-aiv-number={npu_utils.get_aivector_core_num()}"
    ]


def ttir_to_npubin(mod, metadata, opt):
    force_simt_only = bool(getattr(opt, "force_simt_only", False) or metadata.get("force_simt_only", False))
    if force_simt_only:
        metadata["force_simt_only"] = True
        # The runtime launcher keys its launch ABI off is_pure_simt, so publish
        # it here as well when the route model flipped the effective mode.
        metadata["is_pure_simt"] = True
        metadata["parallel_mode"] = "simt"
        metadata.setdefault("shared_mem_dynamic_size", opt.shared_mem_dynamic_size)
        if not isinstance(mod, str):
            # In-process TTIR (explicit simt_only or an IR override): run the
            # layout merge and V1 policy here, mirroring the SIMD route model.
            ascend.ir.inline_void_simt_scopes_for_pure_simt(mod)
            # Pure-SIMT must observe the same post-layout TTIR as the Route
            # Model so the independent-row merge precedes AutoBlockify V1.
            if opt.enable_ttir_layout_merge:
                _run_ttir_layout_merge(mod, metadata)
            else:
                metadata["ttir_layout_merge_applied"] = False
                metadata["ttir_layout_coalesce_factor"] = 1
                metadata["ttir_layout_coalesce_axis"] = -1
                metadata["ttir_layout_coalesce_grid_ceil_div"] = False
            auto_blockify_v1_enabled = _resolve_auto_blockify_v1_policy(str(mod), metadata, opt)
            if opt.enable_ta_auto_blockify_v1 and auto_blockify_v1_enabled:
                materialized = _run_ta_auto_blockify_v1(mod, metadata, opt)
                metadata["auto_blockify_v1_runtime_cap"] = materialized
            else:
                metadata["auto_blockify_v1_runtime_cap"] = auto_blockify_v1_enabled
            _export_program_grid_metadata(mod, metadata)
            _get_then_remove_rc(mod, "ta.ttir_layout_merge.applied")
        elif not metadata.get("scope_pure_simt_auto", False):
            auto_blockify_v1_enabled = _resolve_auto_blockify_v1_policy(mod, metadata, opt)
            if opt.enable_ta_auto_blockify_v1 and auto_blockify_v1_enabled:
                raise RuntimeError("TA AutoBlockify V1 requires an in-process TTIR module; "
                                   "textual TTIR overrides must use the NPUIR V1 implementation")
            metadata["auto_blockify_v1_runtime_cap"] = auto_blockify_v1_enabled
    else:
        _export_program_grid_metadata(mod, metadata, require_row_contract=True)

    ttir_code = mod if isinstance(mod, str) else _serialize_module(mod, opt)
    metadata = _parse_ttir_metadata(ttir_code, metadata)
    _finalize_program_launch_policy(metadata, opt)

    with tempfile.TemporaryDirectory() as tmpdir:
        # prepare input
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        Path(src_path).write_text(ttir_code)
        # prepare output
        bin_file = os.path.join(tmpdir, "kernel")
        bin_path = os.path.join(tmpdir, "kernel.o")
        # build compile options
        _compile_option_list = get_common_bishengir_compile_options(metadata)
        if force_simt_only or opt.is_pure_simt:
            _compile_option_list += ["--enable-hivm-compile=false"]
            _compile_option_list += ["--enable-triton-ir-compile"]
            _compile_option_list += ["--pure-simt"]
            _compile_option_list += _custom_core_count_options()
            _compile_option_list += [f"--num-warps={opt.num_warps}"]
            _compile_option_list += [f"--threads-per-warp={opt.warp_size}"]
            if opt.simt_optimization_mode != 0000000:
                _compile_option_list += [f"--simt-optimization-mode={opt.simt_optimization_mode}"]
            _compile_option_list += [f"--simt-stack-limit={get_simt_stack_limit()}"]
            if opt.shared_mem_dynamic_size is not None:
                _compile_option_list += [f"--shared-mem-dynamic-size={opt.shared_mem_dynamic_size}"]
            if opt.disable_fma:
                _compile_option_list += [f"--disable-fma"]
            if opt.compile_on_910_95:
                npu_utils = NPUUtils()
                if npu_utils.has_device_limit():
                    _compile_option_list += [
                        f"--custom-aic-number={npu_utils.get_aicore_num()}",
                        f"--custom-aiv-number={npu_utils.get_aivector_core_num()}",
                    ]

            bisheng_options = _bisheng_options_with_resource_usage(metadata["bisheng_options"])
            _compile_option_list += [f"--append-bisheng-options={bisheng_options}"]

            if metadata.get("auto_blockify_v1_enabled") is not None:
                # The native Route Model published the executable V1 schedule
                # (and, for mixed routes, the selected SuperBlock factor).
                _append_pure_simt_auto_blockify_options(_compile_option_list, metadata, opt)
            elif _is_auto_map_parallel_blocks_enabled() and not metadata.get("row_coalescing_applied", False):
                _compile_option_list += ["--enable-auto-blockify-loop"]
                if opt.superblock_factor > 1:
                    _compile_option_list += [f"--super-block-factor={opt.superblock_factor}"]

        npu_compiler_path, env = _get_npucompiler_path()
        selected_factor = _selected_npuir_superblock_factor(metadata, opt)
        feedback_factors = []
        if (force_simt_only and metadata.get("scope_pure_simt_auto", False) and not opt.enable_ta_auto_blockify_v1
                and any(option.startswith("--super-block-factor=") for option in _compile_option_list)):
            try:
                report_data = json.loads(metadata.get("auto_simt_scope_report", ""))
                factor_routes = report_data.get("stage_model", {}).get("all_simt_factor_routes", {})
                feedback_factors = sorted(
                    int(factor)
                    for factor, route in factor_routes.items()
                    if route.get("legal", False) and int(factor) <= selected_factor)
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                feedback_factors = []

        if len(feedback_factors) > 1:
            # Compile the selected factor first, then only narrower fallbacks.
            # A narrower factor cannot introduce the widening allocation cliff
            # that this feedback resolves, and factors already more expensive
            # than the native minimum do not need speculative compilation.
            compile_order = [selected_factor] + [factor for factor in feedback_factors if factor != selected_factor]
            compiled = {}
            factor_resources = {}
            failures = {}
            for factor in compile_order:
                factor_options = [
                    option for option in _compile_option_list if not option.startswith("--super-block-factor=")
                ]
                factor_options.append(f"--super-block-factor={factor}")
                factor_options = _compile_options_with_regalloc_size_info(factor_options)
                factor_bin_file = os.path.join(tmpdir, f"kernel_f{factor}")
                factor_bin_path = factor_bin_file + ".o"
                factor_cmd = [npu_compiler_path, src_path] + factor_options + ["-o", factor_bin_file]
                try:
                    factor_ret = subprocess.run(factor_cmd, env=env, capture_output=True, check=True)
                except subprocess.CalledProcessError as error:
                    failures[str(factor)] = {
                        "returncode": error.returncode,
                        "stderr": (error.stderr or b"").decode("utf-8", errors="replace")[-4096:],
                    }
                    if factor == selected_factor:
                        raise
                    continue
                if not Path(factor_bin_path).exists():
                    if factor == selected_factor:
                        raise subprocess.CalledProcessError(factor_ret.returncode, factor_cmd, factor_ret.stdout,
                                                            factor_ret.stderr)
                    failures[str(factor)] = {
                        "returncode": factor_ret.returncode, "stderr": "compiler did not produce kernel.o"
                    }
                    continue
                resource_metadata = {}
                resources = _publish_backend_resource_usage(factor_ret.stdout, factor_ret.stderr, resource_metadata)
                representative = _representative_simt_resource(resources)
                if representative is not None:
                    representative = dict(representative)
                    private_stack = _publish_backend_private_stack_disassembly(factor_bin_path, resource_metadata,
                                                                               npu_compiler_path, env)
                    if private_stack:
                        representative["simt_private_stack_disassembly"] = private_stack
                    factor_resources[factor] = representative
                compiled[factor] = (Path(factor_bin_path).read_bytes(), factor_ret)

            chosen_factor, feedback, updated_report = _select_backend_resource_aware_factor(
                metadata.get("auto_simt_scope_report", ""), factor_resources, selected_factor, num_warps=opt.num_warps,
                private_load_group_cycles=_private_stack_dependent_load_group_cycles(opt.arch))
            if chosen_factor not in compiled:
                chosen_factor = selected_factor
            chosen_binary, chosen_ret = compiled[chosen_factor]
            _publish_backend_resource_usage(chosen_ret.stdout, chosen_ret.stderr, metadata)
            chosen_resource = factor_resources.get(chosen_factor, {})
            if chosen_resource.get("simt_private_stack_disassembly"):
                metadata["backend_simt_private_stack_disassembly"] = (chosen_resource["simt_private_stack_disassembly"])
            metadata["backend_superblock_factor_resources"] = {
                str(factor): resource
                for factor, resource in sorted(factor_resources.items())
            }
            if failures:
                metadata["backend_superblock_factor_compile_failures"] = failures
            if feedback:
                feedback["compile_failures"] = failures
                metadata["backend_resource_feedback"] = feedback
                metadata["auto_simt_superblock_factor"] = chosen_factor
                metadata["auto_simt_scope_report"] = updated_report
                if opt.auto_simt_scope_dump:
                    Path(opt.auto_simt_scope_dump).write_text(updated_report, encoding="utf-8", newline="\n")
            post_backend_candidate = feedback.get("post_backend_selected_candidate", "all_simt_only")
            if post_backend_candidate != "all_simt_only":
                route_neutral_ttir = metadata.get("auto_simt_route_neutral_ttir", "")
                if not route_neutral_ttir:
                    raise RuntimeError("backend resource feedback changed the route winner, but the "
                                       "route-neutral TTIR snapshot is unavailable")
                reroute_metadata = _metadata_for_backend_cross_route_rerun(metadata,
                                                                           feedback.get("rejected_factors", []), opt,
                                                                           post_backend_candidate)
                reroute_mod = _parse_ttir_text(route_neutral_ttir)
                rerouted_linalg = ttir_to_linalg(reroute_mod, reroute_metadata, opt, named_ops=True)
                reroute_dump = os.environ.get("TRITON_ASCEND_BACKEND_REROUTE_DUMP", "")
                if reroute_dump:
                    Path(reroute_dump).write_text(rerouted_linalg, encoding="utf-8", newline="\n")
                materialized_candidate = reroute_metadata.get("auto_simt_effective_kind")
                if materialized_candidate != post_backend_candidate:
                    raise RuntimeError("backend resource reroute selected "
                                       f"{post_backend_candidate!r}, but route materialization produced "
                                       f"{materialized_candidate!r}")
                rerouted_compile_input = rerouted_linalg
                if opt.use_bytecode:
                    rerouted_bytecode = linalg_to_bc_by_triton_mlir_opt(rerouted_linalg, reroute_metadata, opt)
                    rerouted_compile_input = bc_to_linalg_by_bishengir_opt(rerouted_bytecode, reroute_metadata, opt)
                rerouted_binary = linalg_to_bin_enable_npu_compile_910_95(rerouted_compile_input, reroute_metadata, opt)
                reroute_metadata["backend_superblock_factor_resources"] = {
                    str(factor): resource
                    for factor, resource in sorted(factor_resources.items())
                }
                if failures:
                    reroute_metadata["backend_superblock_factor_compile_failures"] = failures
                reroute_metadata["backend_resource_feedback"] = feedback
                _attach_backend_cross_route_feedback(reroute_metadata, feedback, updated_report)
                reroute_metadata.pop("auto_simt_route_neutral_ttir", None)
                metadata.clear()
                metadata.update(reroute_metadata)
                if opt.auto_simt_scope_dump:
                    Path(opt.auto_simt_scope_dump).write_text(metadata["auto_simt_scope_report"], encoding="utf-8",
                                                              newline="\n")
                return rerouted_binary
            metadata.pop("auto_simt_route_neutral_ttir", None)
            return chosen_binary

        cmd_list = ([npu_compiler_path, src_path] + _compile_option_list + ["-o", bin_file])
        ret = subprocess.run(cmd_list, env=env, capture_output=True, check=True)
        _publish_backend_resource_usage(ret.stdout, ret.stderr, metadata)
        if not Path(bin_path).exists():
            error_msg = ret.stderr.decode('utf-8')
            print(f"[DEBUG] {bin_path} is not found")
            print(f"[DEBUG] Stderr:\n{error_msg}")
            raise subprocess.CalledProcessError(ret.returncode, cmd_list, ret.stdout, ret.stderr)
        _publish_backend_private_stack_disassembly(bin_path, metadata, npu_compiler_path, env)
        metadata.pop("auto_simt_route_neutral_ttir", None)
        return Path(bin_path).read_bytes()


def get_simt_stack_limit():
    # simt_stack_limit resolution precedence:
    #  1. torch_npu's acl_default.json "StackSize":{"simt_stack_size":N}.
    #  2. The kernel-time default simt_stack_limit=1152.
    _simt_stack_limit = 1152
    try:
        import torch_npu
        torch_npu_basic_path = os.path.dirname(torch_npu.__file__)
        _acl_cfg_path = os.path.join(torch_npu_basic_path, "acl_default.json")
        with open(_acl_cfg_path, "r") as f:
            _acl_cfg = json.load(f)
        _cfg_stack = _acl_cfg.get("StackSize", {}).get("simt_stack_size", None)
        if isinstance(_cfg_stack, int) and not isinstance(_cfg_stack, bool) and _cfg_stack > 0:
            _simt_stack_limit = _cfg_stack
    except Exception as e:
        print(f"[DEBUG] read acl_default.json failed: {e}")
    return _simt_stack_limit


class AscendBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "npu"

    @staticmethod
    def use_alignment_specialization(options: dict) -> bool:
        # The binder runs before parse_options. Normalize its option copy so
        # legacy selectors affect alignment specialization and the cache key.
        normalized = _remove_deprecated_npu_options(options, in_place=True)
        # Only suppress alignment specialization for pure SIMD mode. simd_simt
        # is a mixed mode whose per-kernel route (SIMT kernels become
        # simt_only) is decided later inside the compiler pipeline; gating on
        # the module-level compile_mode would strip tt.divisibility from those
        # SIMT kernels and regress their memory access.
        return normalized.get("compile_mode") in ("simt_only", "simd_simt")

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        if target.backend == "npu":
            from triton.backends.ascend import _apply_ascend_patch
            _apply_ascend_patch()
            self.binary_ext = "npubin"
            # Include all binary file extensions (mlirbc is always emitted for normal kernels).
            self.binary_extensions = {"npubin", "mlirbc"}

    def get_runtime_compile_options(self, bound_args, specialization, options):
        """Expose concrete integer scalar values to the native route model.

        TTIR intentionally keeps ordinary integer parameters dynamic.  The
        route model nevertheless needs their concrete values when they form
        loop bounds or affine strides.  Record every non-constexpr integer
        scalar by its compact TTIR argument index.  This is a generic launch
        fact: it does not inspect a kernel name, argument name, source path, or
        tensor shape.
        """
        compile_mode = options.get("compile_mode") or os.environ.get("TRITON_ASCEND_COMPILE_MODE", "")
        if compile_mode != "simd_simt":
            return {}
        explicit = options.get("auto_simt_runtime_scalar_values")
        # parse_options defaults this field to an empty tuple and some launch
        # paths feed that default back through the binder.  Empty therefore
        # means "not supplied"; only a non-empty value is an explicit override.
        if explicit:
            return {"auto_simt_runtime_scalar_values": tuple(tuple(pair) for pair in explicit)}

        values = []
        ttir_argument_index = 0
        for (_, value), spec in zip(bound_args.items(), specialization):
            if spec[0] == "constexpr":
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                values.append((ttir_argument_index, int(value)))
            ttir_argument_index += 1
        return {"auto_simt_runtime_scalar_values": tuple(values)}

    def parse_options(self, opts) -> Any:
        # TODO: get available targets when building options?
        if self.target.backend == "npu":
            _warn_deprecated_ascend_env_vars()
            option_names = {
                name
                for name, option_field in NPUOptions.__dataclass_fields__.items()
                if option_field.init and name not in {"arch", "compile_on_910_95"}
            }
            # Serialized NPUOptions include backend-only derived fields.  Use
            # those provenance markers instead of depending on every public
            # field being present, which changes whenever the dataclass evolves.
            internal_options = _is_internal_npu_options(opts, self.target.arch)
            # Normalize a copy, except for the legacy launch keyword below.
            # NPUOptions retains compatibility defaults for legacy names so
            # JIT and Inductor can still recognize them as compile options.
            normalized_opts = opts if internal_options else _remove_deprecated_npu_options(opts)
            args = {k: normalized_opts[k] for k in option_names if k in normalized_opts}
            options = NPUOptions(arch=self.target.arch, **args)
            # Lazy init enable_dynamic_cv_pipeline if not provided.
            # compile_on_910_95 is already resolved from the requested target.
            if options.enable_dynamic_cv_pipeline is None:
                object.__setattr__(options, "enable_dynamic_cv_pipeline", options.compile_on_910_95)
            if not internal_options:
                _normalize_bishengir_simt_optimization_for_context(options, normalized_opts)
                # Community JIT rejects the legacy launch keyword "stream"
                # even when options.__dict__ recognizes it. Consume it at
                # the Ascend boundary; the returned compatibility field stays
                # None and the runtime still selects the current stream.
                opts.pop("stream", None)
        else:
            raise NotImplementedError(f"Backend '{self.target.backend}' is not supported. "
                                      "Please ensure the target backend is set to 'npu'.")
        return options

    def pack_metadata(self, metadata):
        # collect necessary metadata to launch kernels
        # TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 could set unique name.
        # Get this name as the kernel_name to CANN runtime.
        # kernel_name is unique to Ascend backend and should not be public.
        # CANN runtime limits the length of kernel name <= 50.
        # Considering '\n' is appended, thus the real kernel name <= 49.
        KERNEL_NAME_MAX_LEN = 49
        kernel_name_orig = metadata.kernel_name
        if len(kernel_name_orig) > KERNEL_NAME_MAX_LEN:
            kernel_name = kernel_name_orig[-KERNEL_NAME_MAX_LEN:]
        else:
            kernel_name = kernel_name_orig
        return {
            "kernel_name": kernel_name,
            "hash": metadata.hash,
            "debug": metadata.debug,
            "tensor_kinds": metadata.tensor_kinds,
        }

    def get_codegen_implementation(self, options):
        # Note: a dict of functions is required to generate vendor-specific code piecies
        #       e.g. convert custom types like fp8e4b15
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def load_dialects(self, ctx):
        from triton._C.libtriton import buffer_ir
        from triton._C.libtriton.ascend import ir as ascend_ir
        if distributed is not None:
            distributed.ir.load_dialects(ctx)
        buffer_ir.load_dialects(ctx)
        ascend_ir.load_dialects(ctx)
        ascend.load_dialects(ctx)

    def add_stages(self, stages, options, language):
        if self.target.backend == "npu":
            stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
            if options.is_pure_simt:
                stages["npubin"] = (lambda src, metadata: ttir_to_npubin(src, metadata, options))
                return
            stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options)
            # Normal kernels always convert Linalg IR to bytecode and back to MLIR text.
            stages["mlirbc"] = lambda src, metadata: linalg_to_bc_by_triton_mlir_opt(src, metadata, options)
            stages["bcmlir"] = lambda src, metadata: bc_to_linalg_by_bishengir_opt(src, metadata, options)
            # The native Route Model may select all-SIMT during the ttadapter
            # stage, in which case ttadapter carries TTIR forward and the npubin
            # stage must dispatch to the pure-SIMT compiler at run time.
            if options.compile_on_910_95:
                stages["npubin"] = (lambda src, metadata: ttir_to_npubin(src, metadata, options) if metadata.get(
                    "force_simt_only", False) else linalg_to_bin_enable_npu_compile_910_95(src, metadata, options))
            else:
                stages["npubin"] = (lambda src, metadata: ttir_to_npubin(src, metadata, options) if metadata.get(
                    "force_simt_only", False) else linalg_to_bin_enable_npu_compile_A2_A3(src, metadata, options))
            stages["npubin"] = _with_debug_line(stages["npubin"], options)
        else:
            raise NotImplementedError(f"Backend '{self.target.backend}' is not supported. "
                                      "Please ensure the target backend is set to 'npu'.")

    @functools.lru_cache()
    def hash(self):
        # TODO fetch compiler version
        version_key = (self.target, _get_libdevice_compile_state(self.target.arch))
        return str(version_key)

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cann import libdevice

        return {"triton.language.extra.libdevice": libdevice}
